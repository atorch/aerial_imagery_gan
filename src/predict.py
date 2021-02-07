import matplotlib.pyplot as plt
import numpy as np

import tensorflow as tf

from fit_gan import read_labeled_naip_values
from generator import inverse_transform
from models import get_generator_model
from constants import NOISE_SHAPE, N_LABEL_CLASSES, PATCH_SHAPE


def main(generator_epoch=397, image_width=1024):

    generator = get_generator_model()
    generator.load_weights(f"saved_models/generator_epoch_{generator_epoch}")

    # Note: this can be larger than the PATCH_SHAPE the model was trained on
    # TODO How is this possible if the generator's input shape doesn't have Nones in it?
    large_noise_shape = (1, image_width, image_width, NOISE_SHAPE[2])
    large_noise = np.random.normal(size=large_noise_shape)

    naip_and_labels = read_labeled_naip_values("naip/m_4209141_sw_15_1_20170908.tif")
    labels_one_hot = naip_and_labels[0:image_width, 0:image_width, -N_LABEL_CLASSES:]

    labels_one_hot = np.expand_dims(labels_one_hot, 0)

    # Generator input is random noise plus labels from a real naip scene
    generator_input = np.concatenate([large_noise, labels_one_hot], axis=-1)
    large_image = generator(generator_input)
    large_image_rgb = inverse_transform(large_image[0, :, :, :3].numpy()).astype(int)
    filename = f"./generator_output/large_image_epoch_{generator_epoch}.png"
    plt.imsave(filename, large_image_rgb)


if __name__ == "__main__":
    main()
