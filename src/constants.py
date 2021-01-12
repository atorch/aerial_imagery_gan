# Note: I'm trying a lower learning rate for the generator
LEARNING_RATE_DISCRIMINATOR = 0.0002
LEARNING_RATE_GENERATOR = 0.00001

# TODO Kernel
ADDITIONAL_FILTERS_PER_BLOCK_DISCRIMINATOR = 16
BASE_N_FILTERS_DISCRIMINATOR = 128

ADDITIONAL_FILTERS_PER_BLOCK_GENERATOR = 16
BASE_N_FILTERS_GENERATOR = 128

DROPOUT_RATE = 0.3

N_BLOCKS_DISCRIMINATOR = 4
N_BLOCKS_GENERATOR = 4

# Note: NAIP scenes have 4 bands (R B G NIR),
# but we want to generate images with only 3 (R G B) bands
PATCH_SHAPE = (128, 128, 3)

# Generator input noise
NOISE_SHAPE = (128, 128, 1)

# Add continuous noise to the inputs of the discriminator
DISCRIMINATOR_INPUT_NOISE_STD_DEV = 0.05

BATCH_SIZE = 4
STEPS_PER_EPOCH = 100

# TODO Back to 1.0 if adding instance noise?
LABEL_SMOOTHING_ALPHA = 1.0
