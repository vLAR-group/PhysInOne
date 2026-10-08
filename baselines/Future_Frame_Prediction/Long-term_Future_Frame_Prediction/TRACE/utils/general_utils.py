#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
import open3d as o3d


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def ArrayToTorch(array, resolution):
    # resized_image = np.resize(array, resolution)
    resized_image_torch = torch.from_numpy(array)

    if len(resized_image_torch.shape) == 3:
        return resized_image_torch.permute(2, 0, 1)
    else:
        return resized_image_torch.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r, normalize=True):
    if normalize:
        norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])
        q = r / norm[:, None]
    else:
        q = r

    # R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[..., 0]
    x = q[..., 1]
    y = q[..., 2]
    z = q[..., 3]

    R00 = 1 - 2 * (y * y + z * z)
    R01 = 2 * (x * y - r * z)
    R02 = 2 * (x * z + r * y)
    R10 = 2 * (x * y + r * z)
    R11 = 1 - 2 * (x * x + z * z)
    R12 = 2 * (y * z - r * x)
    R20 = 2 * (x * z - r * y)
    R21 = 2 * (y * z + r * x)
    R22 = 1 - 2 * (x * x + y * y)

    R0 = torch.stack((R00, R01, R02), dim=-1)
    R1 = torch.stack((R10, R11, R12), dim=-1)
    R2 = torch.stack((R20, R21, R22), dim=-1)

    R = torch.stack((R0, R1, R2), dim=-2)

    # R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    # R[:, 0, 1] = 2 * (x * y - r * z)
    # R[:, 0, 2] = 2 * (x * z + r * y)
    # R[:, 1, 0] = 2 * (x * y + r * z)
    # R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    # R[:, 1, 2] = 2 * (y * z - r * x)
    # R[:, 2, 0] = 2 * (x * z - r * y)
    # R[:, 2, 1] = 2 * (y * z + r * x)
    # R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def quaternion_multiply(q1, q2):
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)


def rotation_to_quaternion(R):
    q = torch.zeros((R.shape[0], 4), device=R.device)
    q[:, 0] = 0.5 * torch.sqrt(1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2])
    q[:, 1] = (R[:, 2, 1] - R[:, 1, 2]) / (4 * q[:, 0])
    q[:, 2] = (R[:, 0, 2] - R[:, 2, 0]) / (4 * q[:, 0])
    q[:, 3] = (R[:, 1, 0] - R[:, 0, 1]) / (4 * q[:, 0])
    return q

def draw_cameras(R, t, focal, img_w, img_h, color=[1., 0., 0.], scale=1):
    """
    :param poses: (Nv, 4, 4).
    """
    color = np.array(color)
    intrinsic = np.array([[focal, 0., -img_w / 2],
                          [0., -focal, -img_h / 2],
                          [0., 0., -1.]])

    w2c = np.eye(4)
    w2c[:3, :3] = R.T
    w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)
    c2w[:3, 1:3] *= -1

    extrinsic = np.eye(4)
    R, t = c2w[:3, :3], c2w[:3, 3]
    extrinsic[:3, :3] = R.T
    extrinsic[:3, 3] = - np.dot(R.T, t)

    cam_pcd = o3d.geometry.LineSet()
    cam_pcd = cam_pcd.create_camera_visualization(view_width_px=int(img_w),
                                                  view_height_px=int(img_h),
                                                  intrinsic=intrinsic,
                                                  extrinsic=extrinsic)
    cam_pcd.paint_uniform_color(color)
    cam_pcd.colors[4] = 0.5 * color
    cam_pcd.scale(scale=scale, center=t)
    return cam_pcd
