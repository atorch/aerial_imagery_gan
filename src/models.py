from tensorflow.keras import losses, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    Input,
    LeakyReLU,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
    concatenate,
)

from constants import (
    ADDITIONAL_FILTERS_PER_BLOCK_DISCRIMINATOR,
    ADDITIONAL_FILTERS_PER_BLOCK_GENERATOR,
    BASE_N_FILTERS_DISCRIMINATOR,
    BASE_N_FILTERS_GENERATOR,
    DROPOUT_RATE,
    N_BLOCKS_DISCRIMINATOR,
    N_BLOCKS_GENERATOR,
    NOISE_SHAPE,
)


def add_discriminator_downsampling_block(input_layer, block_index):

    n_filters = BASE_N_FILTERS_DISCRIMINATOR + ADDITIONAL_FILTERS_PER_BLOCK_DISCRIMINATOR * block_index

    conv1 = Conv2D(n_filters, kernel_size=3, padding="same", activation=LeakyReLU())(
        input_layer
    )
    dropout = Dropout(rate=DROPOUT_RATE)(conv1)
    conv2 = Conv2D(n_filters, kernel_size=3, padding="same", activation=LeakyReLU())(dropout)

    batchnorm = BatchNormalization()(conv2)

    # Note: strided conv (instead of maxpool) to downsample
    return Conv2D(n_filters, kernel_size=2, strides=2)(batchnorm)


def add_generator_block(input_layer, block_index):

    n_filters = BASE_N_FILTERS_GENERATOR + ADDITIONAL_FILTERS_PER_BLOCK_GENERATOR * block_index

    conv = Conv2D(n_filters, kernel_size=3, padding="same", activation=LeakyReLU())(
        input_layer
    )

    return BatchNormalization()(conv)


def get_discriminator_model(patch_shape):

    # Note: the discriminator is not fully conv (needs a specific input shape)
    input_layer = Input(shape=(patch_shape))

    current_last_layer = input_layer

    for index in range(N_BLOCKS_DISCRIMINATOR):

        current_last_layer = add_discriminator_downsampling_block(current_last_layer, index)

    flat = Flatten()(current_last_layer)

    probabilities = Dense(1, activation="sigmoid")(flat)

    model = Model(inputs=input_layer, outputs=[probabilities])

    print(model.summary())

    return model


def get_generator_model(patch_shape):

    # Note: during training, input has the same width and height
    # as the NAIP image patches, but has only one band
    # TODO Replace NOISE_SHAPE[:2] with Nones to make generator fully conv
    input_layer = Input(shape=NOISE_SHAPE)

    current_last_layer = input_layer

    for index in range(N_BLOCKS_GENERATOR):

        current_last_layer = add_generator_block(
            current_last_layer, index
        )

    # Note: NAIP pixel values are in [0, 255],
    # so we force the generator to output values in the same range
    # TODO Rescale pixel values to [-1, 1] instead?
    n_bands = patch_shape[2]
    pixel_values = 255 * Conv2D(n_bands, 1, activation="sigmoid")(current_last_layer)

    model = Model(inputs=input_layer, outputs=[pixel_values])

    print(model.summary())

    return model
