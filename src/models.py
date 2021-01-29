from tensorflow.keras import losses, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
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
    N_CDL_CLASSES,
    NOISE_SHAPE,
    PATCH_SHAPE,
)


def add_discriminator_downsampling_block(input_layer, block_index):

    n_filters = (
        BASE_N_FILTERS_DISCRIMINATOR
        + ADDITIONAL_FILTERS_PER_BLOCK_DISCRIMINATOR * block_index
    )

    # TODO Put discriminator kernal size in constants
    conv = Conv2D(n_filters, kernel_size=4, padding="same", activation=LeakyReLU())(
        input_layer
    )

    # Note: strided conv (instead of maxpool) to downsample
    return Conv2D(n_filters, kernel_size=2, strides=2)(conv)


def add_generator_downsampling_block(input_layer, block_index):

    n_filters = (
        BASE_N_FILTERS_GENERATOR + ADDITIONAL_FILTERS_PER_BLOCK_GENERATOR * block_index
    )

    # TODO Put generator kernel size in constants
    conv = Conv2D(n_filters, kernel_size=5, padding="same", activation=LeakyReLU())(
        input_layer
    )

    batch_norm = BatchNormalization()(conv)

    # Note: strided conv (instead of maxpool) to downsample
    # TODO Larger kernel size here?
    return Conv2D(n_filters, kernel_size=2, strides=2)(batch_norm)


def add_generator_upsampling_block(input_layer, block_index, downsampling_layers):

    n_filters = (
        BASE_N_FILTERS_GENERATOR + ADDITIONAL_FILTERS_PER_BLOCK_GENERATOR * block_index
    )

    conv_transpose = Conv2DTranspose(
        n_filters, kernel_size=2, strides=2, padding="same"
    )(input_layer)

    conv = Conv2D(n_filters, kernel_size=5, padding="same", activation=LeakyReLU())(
        conv_transpose
    )

    # TODO Does it help to have BatchNormalization here too?

    # Note: this assumes that we initialize
    # downsampling_layers = [input_layer] in the generator code
    return concatenate([conv, downsampling_layers[block_index]])


def get_discriminator_model():

    # Note: the discriminator is not fully conv (needs a specific input shape)
    # The discriminator sees NAIP pixel values with CDL labels concatenated as an extra band
    input_shape = (PATCH_SHAPE[0], PATCH_SHAPE[1], PATCH_SHAPE[2] + N_CDL_CLASSES)
    input_layer = Input(shape=input_shape)

    current_last_layer = input_layer

    for index in range(N_BLOCKS_DISCRIMINATOR):

        current_last_layer = add_discriminator_downsampling_block(
            current_last_layer, index
        )

    dropout = Dropout(rate=DROPOUT_RATE)(current_last_layer)
    flat = Flatten()(dropout)

    # TODO Constants
    dense = Dense(512, activation=LeakyReLU())(flat)

    probabilities = Dense(1, activation="sigmoid")(dense)

    model = Model(inputs=input_layer, outputs=[probabilities])

    print(model.summary())

    return model


def get_generator_model():

    # Note: during training, input has the same width and height
    # as the NAIP image patches, but may have a different number of bands
    # TODO Replace NOISE_SHAPE[:2] with Nones to make generator fully conv
    input_shape = (NOISE_SHAPE[0], NOISE_SHAPE[1], NOISE_SHAPE[2] + N_CDL_CLASSES)
    input_layer = Input(shape=input_shape)

    current_last_layer = input_layer

    # Note: including the input layer lets us concat
    # with the input noise shortly before the output layer
    downsampling_layers = [input_layer]

    for index in range(N_BLOCKS_GENERATOR):

        current_last_layer = add_generator_downsampling_block(current_last_layer, index)

        downsampling_layers.append(current_last_layer)

    for index in range(N_BLOCKS_GENERATOR - 1, -1, -1):

        current_last_layer = add_generator_upsampling_block(
            current_last_layer, index, downsampling_layers
        )

    # TODO BatchNorm before the final convs?
    final_conv5 = Conv2D(
        BASE_N_FILTERS_GENERATOR, kernel_size=5, padding="same", activation=LeakyReLU()
    )(current_last_layer)

    final_conv3 = Conv2D(
        BASE_N_FILTERS_GENERATOR, kernel_size=3, padding="same", activation=LeakyReLU()
    )(final_conv5)

    # Note: NAIP pixel values are in [0, 255],
    # but we map them to [-1, 1] and then use a tanh activation
    # Note: the generator outputs only NAIP pixel values, not CDL classes
    # (they are inputs to the generator, not outputs)
    n_bands = PATCH_SHAPE[2]
    pixel_values = Conv2D(n_bands, kernel_size=3, padding="same", activation="tanh")(
        final_conv3
    )

    model = Model(inputs=input_layer, outputs=[pixel_values])

    print(model.summary())

    return model
