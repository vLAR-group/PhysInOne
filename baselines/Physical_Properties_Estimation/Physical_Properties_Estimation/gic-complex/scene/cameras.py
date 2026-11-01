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
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda", fid=None, depth=None):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device")
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.fid = torch.Tensor(np.array([fid])).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.depth = torch.Tensor(depth).to(self.data_device) if depth is not None else None
        gt_alpha_mask = torch.from_numpy(gt_alpha_mask)
        self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(
            self.data_device)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx,
                                                     fovY=self.FoVy).transpose(0, 1).to(self.data_device)
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def reset_extrinsic(self, R, T):
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, self.trans, self.scale)).transpose(0, 1).cuda()
        self.full_proj_transform = (
            self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def load2device(self, data_device='cuda'):
        self.original_image = self.original_image.to(data_device)
        self.world_view_transform = self.world_view_transform.to(data_device)
        self.projection_matrix = self.projection_matrix.to(data_device)
        self.full_proj_transform = self.full_proj_transform.to(data_device)
        self.camera_center = self.camera_center.to(data_device)
        self.fid = self.fid.to(data_device)
    
    def pw2pix(self, pw):
        assert pw.shape[1] == 3
        intrinsic = torch.from_numpy(self.intrinsic).to(pw)
        pc = self.pw2pc(pw)
        pix_coord = pc @ intrinsic.T
        pix_d = pix_coord[:, 2]
        pix_coord = pix_coord / pix_coord[:, 2:3]
        pix_coord = torch.round(pix_coord)
        pix_w, pix_h = pix_coord[:, 0].to(torch.int64), pix_coord[:, 1].to(torch.int64)
        return pix_w, pix_h, pix_d

    def pw2pc(self, pw):
        R_w2c = torch.from_numpy(self.R).to(pw)
        t_w2c = torch.from_numpy(self.T).to(pw)
        R_w2c = R_w2c.T
        t_w2c = t_w2c
        pc = pw @ R_w2c.T + t_w2c.reshape(1, 3)
        return pc

    def is_in_view(self, pix_w, pix_h):
        in_mask = torch.logical_and(
            torch.logical_and(pix_w >= 0, pix_w < self.image_width),
            torch.logical_and(pix_h >= 0, pix_h < self.image_height),
        )
        return in_mask

    @property
    def intrinsic(self):
        fx = 0.5 * self.image_width / np.tan(self.FoVx / 2)
        fy = 0.5 * self.image_height / np.tan(self.FoVy / 2)
        cx = self.image_width / 2
        cy = self.image_height / 2
        intrinsic = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ])
        return intrinsic


class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
