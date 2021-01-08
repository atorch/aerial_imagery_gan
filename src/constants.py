# Note: I'm trying a lower learning rate for the generator
LEARNING_RATE_DISCRIMINATOR = 0.0002
LEARNING_RATE_GENERATOR = 0.00002

ADDITIONAL_FILTERS_PER_BLOCK_DISCRIMINATOR = 16
BASE_N_FILTERS_DISCRIMINATOR = 64

ADDITIONAL_FILTERS_PER_BLOCK_GENERATOR = 0
BASE_N_FILTERS_GENERATOR = 128

DROPOUT_RATE = 0.3

N_BLOCKS_DISCRIMINATOR = 4
N_BLOCKS_GENERATOR = 3

# Note: NAIP scenes have 4 bands (R B G NIR),
# but we want to generate images with only 3 (R G B) bands
PATCH_SHAPE = (96, 96, 3)

# TODO NOISE_SHAPE and N_BLOCKS_GENERATOR need to be consistent with PATCH_SHAPE,
# only two degrees of freedom
NOISE_SHAPE = (96, 96, 1)

# TODO Distinguish between generator input noise and instance noise
# (added to images before they are seen by generator)
NOISE_STD_DEV = 10

BATCH_SIZE = 4
STEPS_PER_EPOCH = 96

# TODO Back to 1.0 if adding instance noise?
LABEL_SMOOTHING_ALPHA = 1.0
