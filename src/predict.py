import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from generator import inverse_transform
from models import get_generator_model
from constants import NOISE_SHAPE, PATCH_SHAPE


def main(generator_epoch=328):

    generator = get_generator_model(PATCH_SHAPE)
    generator.load_weights(f"saved_models/generator_epoch_{generator_epoch}")

    # Note: we generate a large image (512 by 512 pixels),
    # which is larger than the PATCH_SHAPE the model was trained on
    large_noise_shape = (1, 1024, 1024, NOISE_SHAPE[2])
    large_noise = np.random.normal(size=large_noise_shape)
    large_image = generator(large_noise)
    large_image_rgb = inverse_transform(large_image[0, :, :, :3].numpy()).astype(int)
    filename = f"./generator_output/large_image_epoch_{generator_epoch}.png"
    plt.imsave(filename, large_image_rgb)


if __name__ == "__main__":
    main()
