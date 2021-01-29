import numpy as np

from constants import BATCH_SIZE, N_CDL_CLASSES, PATCH_SHAPE


def transform(x):

    # TODO Map [0, 255] to [-1, 1]
    return 2*x/255 - 1


def inverse_transform(z):

    return (z + 1) * 255 / 2


def get_naip_patch_generator(naip_scenes):

    while True:

        # Note: we add the number of CDL classes to the number of NAIP bands
        # TODO Would it be cleaner to set up a multi-input generator and discriminator instead?
        # TODO Could give the generator inputs names ("noise" and "CDL"),
        # and the discriminator inputs could have names like "image_pixels" and "CDL"
        batch_shape = (BATCH_SIZE,) + (PATCH_SHAPE[:2]) + (PATCH_SHAPE[2] + N_CDL_CLASSES, )
        batch_X = np.empty(batch_shape, dtype=np.float32)

        scene_indices = np.random.choice(range(len(naip_scenes)), size=BATCH_SIZE)

        for batch_index, scene_index in enumerate(scene_indices):

            naip_X = naip_scenes[scene_index]

            # Note: PATCH_SHAPE is band last, i.e. (dim0, dim1, band)
            dim0_start = np.random.choice(range(naip_X.shape[0] - PATCH_SHAPE[0]))
            dim1_start = np.random.choice(range(naip_X.shape[1] - PATCH_SHAPE[1]))

            dim0_end = dim0_start + PATCH_SHAPE[0]
            dim1_end = dim1_start + PATCH_SHAPE[1]

            naip_patch = naip_X[dim0_start:dim0_end, dim1_start:dim1_end]

            batch_X[batch_index] = naip_patch

        yield batch_X
