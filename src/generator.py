import numpy as np


def get_naip_patch_generator(naip_scenes, patch_shape, batch_size):

    while True:

        batch_X = np.empty((batch_size,) + patch_shape)

        scene_indices = np.random.choice(range(len(naip_scenes)), size=batch_size)

        for batch_index, scene_index in enumerate(scene_indices):

            naip_X = naip_scenes[scene_index]

            # Note: patch_shape is band last, i.e. (dim0, dim1, band)
            dim0_start = np.random.choice(range(naip_X.shape[0] - patch_shape[0]))
            dim1_start = np.random.choice(range(naip_X.shape[1] - patch_shape[1]))

            dim0_end = dim0_start + patch_shape[0]
            dim1_end = dim1_start + patch_shape[1]

            naip_patch = naip_X[dim0_start:dim0_end, dim1_start:dim1_end]

            batch_X[batch_index] = naip_patch

        yield batch_X
