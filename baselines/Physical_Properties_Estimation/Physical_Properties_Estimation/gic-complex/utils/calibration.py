"""Convert shared OpenGL camera poses to the GIC/3DGS OpenCV convention."""
import numpy as np


def camera_extrinsics(c2w):
    matrix = np.asarray(c2w, dtype=float)
    if matrix.shape == (3, 4):
        matrix = np.vstack([matrix, [0, 0, 0, 1]])
    if matrix.shape != (4, 4):
        raise ValueError('Camera c2w must be 3x4 or 4x4')
    world_to_camera = np.linalg.inv(matrix)
    world_to_camera[1:3, :] *= -1  # OpenGL: up/-Z; rasterizer: down/+Z.
    return world_to_camera[:3, :3].T, world_to_camera[:3, 3]
