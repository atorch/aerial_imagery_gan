import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from generator import inverse_transform
from models import get_generator_model
from constants import NOISE_SHAPE, N_CDL_CLASSES, PATCH_SHAPE


def main(generator_epoch=368):

    generator = get_generator_model()
    generator.load_weights(f"saved_models/generator_epoch_{generator_epoch}")

    # Note: this can be larger than the PATCH_SHAPE the model was trained on
    # TODO How is this possible if the generator's input shape doesn't have Nones in it?
    large_noise_shape = (1, 1024, 1024, NOISE_SHAPE[2])
    large_noise = np.random.normal(size=large_noise_shape)

    cdl = np.zeros((1, 1024, 1024, N_CDL_CLASSES))

    # Set the first CDL class (forest)
    cdl[0, :512, :512, 0] = 1
    cdl[0, :128, :, 0] = 1

    # In another corner of the image, set the second CDL class (developed)
    cdl[0, -512:, -512:, 1] = 1

    # TODO Add some buildings (next "CDL" band, rename it to labels)
    cdl[0, -20:-10, -20:-10, 2] = 1
    cdl[0, -50:-40, -50:-40, 2] = 1
    cdl[0, -200:-170, -200:-170, 2] = 1
    cdl[0, -250:-235, -200:-170, 2] = 1
    cdl[0, -300:-270, -300:-270, 2] = 1

    generator_input = np.concatenate([large_noise, cdl], axis=-1)
    large_image = generator(generator_input)
    large_image_rgb = inverse_transform(large_image[0, :, :, :3].numpy()).astype(int)
    filename = f"./generator_output/large_image_epoch_{generator_epoch}.png"
    plt.imsave(filename, large_image_rgb)


if __name__ == "__main__":
    main()
