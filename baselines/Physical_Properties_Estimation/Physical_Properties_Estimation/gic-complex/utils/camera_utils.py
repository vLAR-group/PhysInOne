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

from scene.cameras import Camera
from utils.general_utils import PILtoTorch, ArrayToTorch
from utils.graphics_utils import fov2focal
from PIL import Image
import numpy as np
import taichi as ti
import torch
import json

WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w / (resolution_scale * args.resolution)), round(
            orig_h / (resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                          "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    elif cam_info.alpha is not None:
        alpha = Image.fromarray((cam_info.alpha[..., 0] * 255).astype(np.uint8)).resize(resolution)
        alpha = np.asarray(alpha).astype(float) / 255
        loaded_mask = alpha[None]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=cam_info.uid,
                  data_device=args.data_device if not args.load2gpu_on_the_fly else 'cpu', fid=cam_info.fid,
                  depth=cam_info.depth)


def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera: Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id': id,
        'img_name': camera.image_name,
        'width': camera.width,
        'height': camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy': fov2focal(camera.FovY, camera.height),
        'fx': fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def camera_nerfies_from_JSON(path, scale):
    """Loads a JSON camera into memory."""
    with open(path, 'r') as fp:
        camera_json = json.load(fp)

    # Fix old camera JSON.
    if 'tangential' in camera_json:
        camera_json['tangential_distortion'] = camera_json['tangential']

    return dict(
        orientation=np.array(camera_json['orientation']),
        position=np.array(camera_json['position']),
        focal_length=camera_json['focal_length'] * scale,
        principal_point=np.array(camera_json['principal_point']) * scale,
        skew=camera_json['skew'],
        pixel_aspect_ratio=camera_json['pixel_aspect_ratio'],
        radial_distortion=np.array(camera_json['radial_distortion']),
        tangential_distortion=np.array(camera_json['tangential_distortion']),
        image_size=np.array((int(round(camera_json['image_size'][0] * scale)),
                             int(round(camera_json['image_size'][1] * scale)))),
    )


def K_dpt2cld(dpt, cam_scale, K):
    dpt = dpt.astype(np.float32)
    dpt /= cam_scale

    Kinv = np.linalg.inv(K)

    h, w = dpt.shape[0], dpt.shape[1]

    x, y = np.meshgrid(np.arange(w), np.arange(h))
    ones = np.ones((h, w), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(w*h, 3)

    # backproj
    R = np.dot(Kinv, x2d.transpose())

    # compute 3D points
    X = R * np.tile(dpt.reshape(1, w*h), (3, 1))
    X = np.array(X).transpose()

    X = X.reshape(h, w, 3)
    return X


def get_intrinsic(fovx, fovy, w, h):
    fx = 0.5 * w / np.tan(fovx / 2)
    fy = 0.5 * h / np.tan(fovy / 2)
    cx = w / 2
    cy = h / 2
    intrinsic = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ]).astype(np.float32)
    return intrinsic


def transform_w2c(pw: torch.Tensor, 
                  R_w2c: np.ndarray, t_w2c: np.ndarray, intrinsic: np.ndarray) -> torch.Tensor:
    R_w2c = torch.from_numpy(R_w2c).to(pw)
    t_w2c = torch.from_numpy(t_w2c).to(pw)
    intrinsic = torch.from_numpy(intrinsic).to(pw)
    R_w2c = R_w2c.T
    t_w2c = t_w2c
    init_inner_points_c = pw @ R_w2c.T + t_w2c.reshape(1, 3)
    pix_coord = init_inner_points_c @ intrinsic.T
    return pix_coord


def gen_surround_cam(center, r, num_angle, FoVx, FoVy, image):
    camera = ti.ui.Camera()
    c_x, c_y, c_z = center
    camera.lookat(*center)
    camera.up(0, 1, 0)
    d_theta = np.pi / num_angle
    d_phi = 2 * np.pi / num_angle
    views = []
    for i in range(num_angle):
        theta = i * d_theta
        for j in range(num_angle):
            phi = j * d_phi
            x=c_x + r * np.sin(theta) * np.cos(phi)
            y=c_y + r * np.sin(theta) * np.sin(phi)
            z=c_z + r * np.cos(theta) 
            camera.position(x,y,z)
            matrix=camera.get_view_matrix().T
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]
            if np.isnan(R).any() or np.isnan(T).any():
                continue
            cam = Camera(-1, R, T, FoVx, FoVy, image, None, None, -1, fid=0.0)
            views.append(cam)
    return views
