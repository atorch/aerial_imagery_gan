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
    STEPS_PER_EPOCH,
    LABEL_SMOOTHING_ALPHA,
    N_CDL_CLASSES,
)
from generator import get_naip_patch_generator, inverse_transform, transform
from models import get_discriminator_model, get_generator_model


def discriminator_loss(real_output, fake_output):

    # "Real" refers to the discriminator's output on real images
    # "Fake" refers to the discriminator's output on the generator's images

    cross_entropy = tf.keras.losses.BinaryCrossentropy()
    real_loss = cross_entropy(
        LABEL_SMOOTHING_ALPHA * tf.ones_like(real_output), real_output
    )
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

    batch_size = image_batch.shape[0]
    noise_shape = (batch_size,) + NOISE_SHAPE
    input_noise = tf.random.normal(noise_shape)

    # TODO This needs to be consistent with the number of CDL bands in the image batch
    # TODO Might be much cleaner to have multiple inputs for generator and discriminator...
    cdl = image_batch[:, :, :, -N_CDL_CLASSES:]
    input_noise_and_cdl = tf.concat([input_noise, cdl], axis=-1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        generated_pixels = generator(input_noise_and_cdl, training=True)
        generated_pixels_and_cdl = tf.concat([generated_pixels, cdl], axis=-1)

        # TODO Decrease noise std dev as the number of iterations increases?
        discriminator_input_noise = tf.random.normal(
            image_batch.shape, stddev=DISCRIMINATOR_INPUT_NOISE_STD_DEV
        )

        real_output = discriminator(
            image_batch + discriminator_input_noise, training=True
        )
        fake_output = discriminator(
            generated_pixels_and_cdl + discriminator_input_noise, training=True
        )

        gen_loss = generator_loss(fake_output, generated_pixels)
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

    return gen_loss, disc_loss


def save_real_images(step, image_batch):

    for image_index in range(image_batch.shape[0]):
        filename = f"./naip_patches/naip_patch_step_{step}_{image_index}.png"

        real_image_rgb = inverse_transform(image_batch[image_index, :, :, :3]).astype(
            int
        )
        plt.imsave(filename, real_image_rgb)


def save_generator_output(generator_input, generator, discriminator, epoch):

    fake_images = generator(generator_input)

    # TODO This would be much less messy with multi input models
    cdl = generator_input[:, :, :, -N_CDL_CLASSES:]
    prediction = discriminator(np.concatenate([fake_images, cdl], axis=-1))

    for image_index in range(generator_input.shape[0]):
        fake_image_rgb = inverse_transform(
            fake_images[image_index, :, :, :3].numpy()
        ).astype(int)
        filename = (
            f"./generator_output/generated_image_noise_{image_index}_epoch_{epoch}.png"
        )
        plt.imsave(filename, fake_image_rgb)
        # The discriminator wants to push these predictions toward 0
        print(
            f"discriminator's prediction for {filename}: {prediction.numpy()[image_index]}"
        )


def train(naip_patch_generator, epochs=400):

    discriminator = get_discriminator_model()
    generator = get_generator_model()

    # TODO Firm this up.  Instead of starting with random weights, start
    # with a saved generator that is already working reasonably well
    # TODO Separate saved models by patch size?
    # generator.load_weights(f"saved_models/generator_epoch_199")

    generator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE_GENERATOR, beta_1=0.5
    )
    discriminator_optimizer = tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE_DISCRIMINATOR, beta_1=0.5
    )

    # Generate images using the same noise (fixed input) at the end of each epoch
    noise_shape = (3,) + NOISE_SHAPE
    fixed_noise = np.random.normal(size=noise_shape)

    cdl_one_hot = np.zeros((3, NOISE_SHAPE[0], NOISE_SHAPE[1], N_CDL_CLASSES))
    cdl_one_hot[0, :10, :10, 0] = 1
    cdl_one_hot[1, :, :, 0] = 1
    cdl_one_hot[2, :, :, 1] = 1
    fixed_generator_input = np.concatenate([fixed_noise, cdl_one_hot], axis=-1)

    gen_losses, disc_losses = [], []

    for epoch in range(epochs):

        gen_losses_current_epoch = []
        disc_losses_current_epoch = []

        start = time.time()
        for step in range(STEPS_PER_EPOCH):

            image_batch = next(naip_patch_generator)

            # During the beginning of the first epoch, save a few real NAIP image patches for inspection
            if epoch == 0 and step < 20:
                save_real_images(step, image_batch)

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

        # Generate an image with the generator (same noise every time)
        save_generator_output(fixed_generator_input, generator, discriminator, epoch)

        prediction_on_real_images = discriminator(image_batch)

        description = (
            f"discriminator wants to push these towards {LABEL_SMOOTHING_ALPHA}"
        )
        print(
            f"discriminator's prediction on real images ({description}):\n{prediction_on_real_images}"
        )

        generator.save_weights(f"saved_models/generator_epoch_{epoch}")

    # TODO Plot losses
    # TODO Also plot norm of gradients?
    print(f"generator losses {gen_losses}")
    print(f"discriminator losses {disc_losses}")

    # TODO Put this in a function
    plt.plot(range(epochs), gen_losses)

    return generator, discriminator


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


def read_raster_values(raster_path):

    print(f"Reading {raster_path}")
    with rasterio.open(raster_path) as raster:

        # Note: band order is (x, y, band) after call to np.swapaxes
        X = np.swapaxes(raster.read(), 0, 2)

    return X


def read_labeled_naip_values(naip_path):

    naip_X = read_raster_values(naip_path)

    # Note: we map NAIP pixel values in [0, 255] to [-1, 1]
    # We _do not_ apply the transform to one hot encoded CDL
    # Careful: the NAIP values are originally unsigned integers!
    naip_X = transform(naip_X.astype(np.float32))

    # Convention: for each .tif file in naip/, there is a corresponding CDL annotation (label)
    # raster in cdl_annotations/, of the exact same width and height (spatial dimensions)
    cdl_annotation_path = naip_path.replace("naip/", "cdl_annotations/")
    cdl_X = read_raster_values(cdl_annotation_path).astype(int)

    building_annotation_path = naip_path.replace("naip/", "building_annotations/")
    building_X = read_raster_values(building_annotation_path).astype(np.float32)

    # TODO One hot encode several CDL classes, generalize, use a label encoder
    cdl_forest_codes = [63, 141, 142, 143]
    cdl_forest_one_hot = np.isin(cdl_X, cdl_forest_codes)

    cdl_developed_codes = [82, 121, 122, 123, 124]
    cdl_developed_one_hot = np.isin(cdl_X, cdl_developed_codes)

    # Note: class labels (CDL annotations) are added as extra bands
    # TODO This needs to be consistent with N_CDL_CLASSES, messy
    return np.concatenate(
        [naip_X, cdl_forest_one_hot, cdl_developed_one_hot, building_X], axis=-1
    )


def get_labeled_naip_scenes():

    naip_paths = glob.glob("./naip/*tif")
    return [read_labeled_naip_values(naip_path) for naip_path in naip_paths]


def main():

    # TODO float16 to save memory?
    tf.keras.backend.set_floatx("float32")

    naip_scenes = get_labeled_naip_scenes()

    naip_mean, naip_std = get_naip_mean_and_std(naip_scenes)

    print(f"naip means: {naip_mean}")
    print(f"naip std deviations: {naip_std}")

    naip_patch_generator = get_naip_patch_generator(naip_scenes)

    # TODO Try training sequentially on larger and larger patches,
    # reusing weights from previous patch size (except the FC part of the discriminator)
    # For example, start on 64-by-64 pixel patches, then 128, then 256
    generator, discriminator = train(naip_patch_generator)

    # TODO Plots of gradient norms over time
    # TODO Plots of losses over time
    # TODO Plots of discriminator accuracies over time


if __name__ == "__main__":
    main()
