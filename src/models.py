from tensorflow.keras import losses, optimizers
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    BatchNormalization,
    Conv2D,
    Dense,
    Dropout,
    Flatten,
    GlobalAveragePooling2D,
    Input,
    MaxPooling2D,
    Reshape,
    UpSampling2D,
    concatenate,
)

from constants import (
    ADDITIONAL_FILTERS_PER_BLOCK,
    BASE_N_FILTERS,
    DROPOUT_RATE,
    N_BLOCKS,
)


def add_downsampling_block(input_layer, block_index):

    n_filters = BASE_N_FILTERS + ADDITIONAL_FILTERS_PER_BLOCK * block_index

    # TODO Try leaky relu?
    conv1 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(
        input_layer
    )
    dropout = Dropout(rate=DROPOUT_RATE)(conv1)
    conv2 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(dropout)

    batchnorm = BatchNormalization()(conv2)

    # Note: Don't MaxPool in last downsampling block
    if block_index == N_BLOCKS - 1:

        return batchnorm, conv2

    return MaxPooling2D()(batchnorm), conv2


def add_upsampling_block(input_layer, block_index, downsampling_conv2_layers):

    n_filters = BASE_N_FILTERS + ADDITIONAL_FILTERS_PER_BLOCK * block_index

    upsample = UpSampling2D()(input_layer)

    concat = concatenate([upsample, downsampling_conv2_layers[block_index - 1]])

    conv1 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(concat)
    dropout = Dropout(rate=DROPOUT_RATE)(conv1)
    conv2 = Conv2D(n_filters, kernel_size=3, padding="same", activation="relu")(dropout)

    return BatchNormalization()(conv2)


def get_discriminator_model(patch_shape):

    # Note: the discriminator is not fully conv (needs a specific input shape)
    input_layer = Input(shape=(patch_shape))

    current_last_layer = input_layer

    for index in range(N_BLOCKS):

        current_last_layer, _ = add_downsampling_block(current_last_layer, index)

    # TODO Try leaky relu for discriminator
    flat = Flatten()(current_last_layer)
    fc = Dense(16, activation="relu")(flat)

    probabilities = Dense(1, activation="sigmoid")(fc)

    model = Model(inputs=input_layer, outputs=[probabilities])

    print(model.summary())

    return model


def get_generator_model(patch_shape):

    # TODO Make input None, None, 1?  Single band of noise instead of 4?
    input_layer = Input(shape=(None, None, patch_shape[2]))

    current_last_layer = input_layer

    # Note: Keep track of conv2 layers so that they can be connected to the upsampling blocks
    downsampling_conv2_layers = []

    # TODO Could have its own N_BLOCKS tuning parameter instead of sharing with multi-output model
    for index in range(N_BLOCKS):

        # TODO Strided conv instead of maxpool?
        current_last_layer, conv2_layer = add_downsampling_block(
            current_last_layer, index
        )

        downsampling_conv2_layers.append(conv2_layer)

    for index in range(N_BLOCKS - 1, 0, -1):

        current_last_layer = add_upsampling_block(
            current_last_layer, index, downsampling_conv2_layers
        )

    n_bands = patch_shape[2]

    # Note: NAIP pixel values are in [0, 255],
    # so we force the generator to output values in the same range
    pixel_values = 255 * Conv2D(n_bands, 1, activation="sigmoid")(current_last_layer)

    model = Model(inputs=input_layer, outputs=[pixel_values])

    print(model.summary())

    return model
