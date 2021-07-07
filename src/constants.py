# Note: I'm trying a lower learning rate for the generator
LEARNING_RATE_DISCRIMINATOR = 0.0002
LEARNING_RATE_GENERATOR = 0.00001

# TODO Put kernel sizes here

ADDITIONAL_FILTERS_PER_BLOCK_DISCRIMINATOR = 16
BASE_N_FILTERS_DISCRIMINATOR = 128

ADDITIONAL_FILTERS_PER_BLOCK_GENERATOR = 16
BASE_N_FILTERS_GENERATOR = 128

DROPOUT_RATE = 0.2

# TODO Increase this as patch shape increases?
N_BLOCKS_DISCRIMINATOR = 4
N_BLOCKS_GENERATOR = 5

# Note: NAIP scenes have 4 bands (R B G NIR),
# and we add one extra band with CDL class information (labels)
PATCH_SHAPE = (128, 128, 4)

# Generator input noise
NOISE_SHAPE = (128, 128, 2)

# Add continuous noise to the inputs of the discriminator
DISCRIMINATOR_INPUT_NOISE_STD_DEV = 0.02

BATCH_SIZE = 4
STEPS_PER_EPOCH = 100

LABEL_SMOOTHING_ALPHA = 1.0

# TODO Could put this in a little yaml file
# Add another level so that "cdl_annotations" doesn't get repeated so many times
LABEL_CONFIG = [
    ("cdl_forest", "cdl_annotations/", [63, 141, 142, 143],),
    ("cdl_developed", "cdl_annotations/", [82, 121, 122, 123, 124],),
    ("cdl_water", "cdl_annotations/", [83, 111],),
    ("cdl_pasture", "cdl_annotations/", [176],),
    ("cdl_corn_soy", "cdl_annotations/", [1, 5, 26],),
    ("buildings", "building_annotations/", [1],),
    ("roads", "road_annotations/", [1],),
]

N_LABEL_CLASSES = len(LABEL_CONFIG)
