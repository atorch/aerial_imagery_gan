import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tensorflow as tf
import time

from constants import (
    BATCH_SIZE,
    LEARNING_RATE_DISCRIMINATOR,
    LEARNING_RATE_GENERATOR,
    PATCH_SHAPE,
    STEPS_PER_EPOCH,
    LABEL_SMOOTHING_ALPHA,
)
from generator import get_naip_patch_generator
from models import get_discriminator_model, get_generator_model


## Based on https://www.tensorflow.org/tutorials/generative/dcgan


def discriminator_loss(real_output, fake_output):

    # "Real" refers to the discriminator's output on real images
    # "Fake" refers to the discriminator's output on the generator's images

    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(LABEL_SMOOTHING_ALPHA * tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss


def generator_loss(fake_output):

    # TODO Try feature matching loss

    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(
    image_batch, discriminator, generator, discriminator_optimizer, generator_optimizer
):

    # TODO Try GP noise with spatial correlation
    # TODO Do results change with a single band of noise (instead of 4 bands of noise)?
    noise = tf.random.normal(image_batch.shape)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(noise, training=True)

        real_output = discriminator(image_batch, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )


def save_real_images(image_batch):

    for image_index in range(image_batch.shape[0]):
        filename = f"./examples/real_image_{image_index}.png"

        real_image_rgb = image_batch[image_index, :, :, :3].astype(int)
        plt.imsave(filename, real_image_rgb)


def train(naip_patch_generator, epochs=200):

    discriminator = get_discriminator_model(PATCH_SHAPE)
    generator = get_generator_model(PATCH_SHAPE)

    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE_GENERATOR, beta_1=0.5
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE_DISCRIMINATOR, beta_1=0.5
    )

    # Generate images using the same noise (fixed input) at the end of each epoch
    noise_shape = (2,) + PATCH_SHAPE
    fixed_noise = np.random.normal(size=noise_shape)

    for epoch in range(epochs):

        start = time.time()
        for _ in range(STEPS_PER_EPOCH):

            image_batch = next(naip_patch_generator)

            train_step(
                image_batch,
                discriminator,
                generator,
                discriminator_optimizer,
                generator_optimizer,
            )

        print("Time for epoch {} is {} sec".format(epoch, time.time() - start))

        if epoch == 0:

            # Save a few real NAIP image patches for inspection
            save_real_images(image_batch)

        # Generate an image with the generator (same noise every time)
        fake_image = generator(fixed_noise)
        prediction = discriminator(fake_image)
        for image_index in range(noise_shape[0]):
            fake_image_rgb = fake_image[image_index, :, :, :3].numpy().astype(int)
            filename = (
                f"./examples/generated_image_noise_{image_index}_epoch_{epoch}.png"
            )
            plt.imsave(filename, fake_image_rgb)
            # The discriminator wants to push these predictions toward 0
            print(
                f"discriminator's prediction for {filename}: {prediction.numpy()[image_index]}"
            )

        prediction_on_real_images = discriminator(image_batch)

        print(
            f"discriminator's prediction on some real images (discriminator wants to push these towards 1):\n{prediction_on_real_images}"
        )


def get_naip_scenes():

    # TODO Config
    # TODO Have a look at these scenes in qgis
    naip_paths = [
        "./naip/m_4109258_se_15_1_20170723.tif",
        "./naip/m_4009202_se_15_1_20170811.tif",
        "./naip/m_4109007_ne_15_1_20170823.tif",
        "./naip/m_4109326_ne_15_1_20170701.tif",
        "./naip/m_4209638_nw_14_1_20170716.tif",
        "./naip/m_4209141_sw_15_1_20170908.tif",
        "./naip/m_4209256_nw_15_1_20170908.tif",
    ]

    Xs = []

    for naip_path in naip_paths:

        print(f"Reading {naip_path}")
        with rasterio.open(naip_path) as naip:

            # Note: band order is (x, y, band) after call to np.swapaxes
            X = np.swapaxes(naip.read(), 0, 2)

            Xs.append(X)

    return Xs


def main():

    naip_scenes = get_naip_scenes()

    naip_patch_generator = get_naip_patch_generator(
        naip_scenes, PATCH_SHAPE, batch_size=BATCH_SIZE
    )

    # Array of shape (BATCH_SIZE, 256, 256, 4)
    sample_batch = next(naip_patch_generator)

    train(naip_patch_generator)


if __name__ == "__main__":
    main()
