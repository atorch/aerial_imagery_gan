import glob
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import tensorflow as tf
import time

from constants import (
    BATCH_SIZE,
    LEARNING_RATE_DISCRIMINATOR,
    LEARNING_RATE_GENERATOR,
    NOISE_SHAPE,
    DISCRIMINATOR_INPUT_NOISE_STD_DEV,
    PATCH_SHAPE,
    STEPS_PER_EPOCH,
    LABEL_SMOOTHING_ALPHA,
)
from generator import get_naip_patch_generator, inverse_transform
from models import get_discriminator_model, get_generator_model


## Based on https://www.tensorflow.org/tutorials/generative/dcgan


def discriminator_loss(real_output, fake_output):

    # "Real" refers to the discriminator's output on real images
    # "Fake" refers to the discriminator's output on the generator's images

    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(LABEL_SMOOTHING_ALPHA * tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)

    return real_loss + fake_loss


def generator_loss(fake_output, generated_images):

    # "Fake" refers to the discriminator's output on the generator's images

    mse = tf.keras.losses.MeanSquaredError()
    cross_entropy = tf.keras.losses.BinaryCrossentropy()

    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(
    image_batch, discriminator, generator, discriminator_optimizer, generator_optimizer
):

    # TODO Try GP noise with spatial correlation
    # TODO What about mixture of normals with different means?
    batch_size = image_batch.shape[0]
    noise_shape = (batch_size, ) + NOISE_SHAPE
    input_noise = tf.random.normal(noise_shape)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_images = generator(input_noise, training=True)

        # TODO Decrease noise std dev as the number of iterations increases?
        discriminator_input_noise = tf.random.normal(image_batch.shape, stddev=DISCRIMINATOR_INPUT_NOISE_STD_DEV)

        real_output = discriminator(image_batch + discriminator_input_noise, training=True)
        fake_output = discriminator(generated_images + discriminator_input_noise, training=True)

        gen_loss = generator_loss(fake_output, generated_images)
        disc_loss = discriminator_loss(real_output, fake_output)

        # TODO Track losses over time, make a graph at the end

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

    return gen_loss, disc_loss


def save_real_images(image_batch):

    for image_index in range(image_batch.shape[0]):
        filename = f"./naip_patches/naip_patch_{image_index}.png"

        real_image_rgb = inverse_transform(image_batch[image_index, :, :, :3]).astype(int)
        plt.imsave(filename, real_image_rgb)


def save_generator_output(noise, generator, discriminator, epoch):

    # Generate an image with the generator (same noise every time)
    fake_images = generator(noise)

    prediction = discriminator(fake_images)
    for image_index in range(noise.shape[0]):
        fake_image_rgb = inverse_transform(fake_images[image_index, :, :, :3].numpy()).astype(int)
        filename = (
            f"./generator_output/generated_image_noise_{image_index}_epoch_{epoch}.png"
        )
        plt.imsave(filename, fake_image_rgb)
        # The discriminator wants to push these predictions toward 0
        print(
            f"discriminator's prediction for {filename}: {prediction.numpy()[image_index]}"
        )


def train(naip_patch_generator, epochs=300):

    discriminator = get_discriminator_model(PATCH_SHAPE)
    generator = get_generator_model(PATCH_SHAPE)

    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE_GENERATOR, beta_1=0.5
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE_DISCRIMINATOR, beta_1=0.5
    )

    # Generate images using the same noise (fixed input) at the end of each epoch
    noise_shape = (3,) + NOISE_SHAPE
    fixed_noise = np.random.normal(size=noise_shape)

    gen_losses, disc_losses = [], []

    for epoch in range(epochs):

        gen_losses_current_epoch = []
        disc_losses_current_epoch = []

        start = time.time()
        for _ in range(STEPS_PER_EPOCH):

            image_batch = next(naip_patch_generator)

            # TODO Keep history of some recent iterations of the generator, replay
            gen_loss, disc_loss = train_step(
                image_batch,
                discriminator,
                generator,
                discriminator_optimizer,
                generator_optimizer,
            )

            gen_losses_current_epoch.append(gen_loss.numpy())
            disc_losses_current_epoch.append(disc_loss.numpy())

        gen_losses.append(np.mean(gen_losses_current_epoch))
        disc_losses.append(np.mean(disc_losses_current_epoch))

        print("Time for epoch {} is {} sec".format(epoch, time.time() - start))

        if epoch == 0:

            # Save a few real NAIP image patches for inspection
            save_real_images(image_batch)

        save_generator_output(fixed_noise, generator, discriminator, epoch)

        # TODO Put this in a function
        prediction_on_real_images = discriminator(image_batch)

        description = f"discriminator wants to push these towards {LABEL_SMOOTHING_ALPHA}"
        print(
            f"discriminator's prediction on real images ({description}):\n{prediction_on_real_images}"
        )

    print(f"generator losses {gen_losses}")
    print(f"discriminator losses {disc_losses}")
    # TODO Also plot norm of gradients?
    # TODO Plot


def get_naip_mean_and_std(naip_scenes):

    # Compute means and std deviations per band (R G B NIR)
    naip_sizes = [naip_scene.size for naip_scene in naip_scenes]
    naip_means = [naip_scene.mean(axis=(0, 1)) for naip_scene in naip_scenes]
    naip_vars = [naip_scene.var(axis=(0, 1)) for naip_scene in naip_scenes]

    # Note: this produces the same result as
    #  np.hstack((X.flatten() for X, Y in naip_scenes)).mean()
    # but uses less memory
    naip_mean = np.average(naip_means, weights=naip_sizes, axis=0)
    naip_var = np.average(naip_vars, weights=naip_sizes, axis=0)
    naip_std = np.sqrt(naip_var)

    return naip_mean.astype(np.float32), naip_std.astype(np.float32)


def read_naip_values(naip_path):

    print(f"Reading {naip_path}")
    with rasterio.open(naip_path) as naip:

        # Note: band order is (x, y, band) after call to np.swapaxes
        X = np.swapaxes(naip.read(), 0, 2)

    return X


def get_naip_scenes():

    naip_paths = glob.glob("./naip/*tif")
    return [read_naip_values(naip_path) for naip_path in naip_paths]


def main():

    # TODO float16 to save memory?
    tf.keras.backend.set_floatx("float32")

    naip_scenes = get_naip_scenes()

    naip_mean, naip_std = get_naip_mean_and_std(naip_scenes)

    print(f"naip means: {naip_mean}")
    print(f"naip std deviations: {naip_std}")

    naip_patch_generator = get_naip_patch_generator(
        naip_scenes, PATCH_SHAPE, batch_size=BATCH_SIZE
    )

    # Array of shape (BATCH_SIZE, 256, 256, 4)
    sample_batch = next(naip_patch_generator)

    train(naip_patch_generator)

    # TODO Print mean and std of generated output (on each iteration?)

    # TODO Plots of gradient norms over time
    # TODO Plots of losses over time
    # TODO Plots of discriminator accuracies over time


if __name__ == "__main__":
    main()
