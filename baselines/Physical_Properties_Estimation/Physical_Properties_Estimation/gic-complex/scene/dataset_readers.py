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

import os
import sys
from PIL import Image
from typing import NamedTuple, Optional
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text, \
    read_cameras_binary, read_images_binary
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.calibration import camera_extrinsics
import numpy as np
import json
from tqdm import tqdm
from glob import glob
import cv2
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.camera_utils import camera_nerfies_from_JSON


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    fid: float
    depth: Optional[np.array] = None
    alpha: Optional[np.array] = None


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    num_frames = len(cam_extrinsics)
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(
            "Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        fid = int(image_name) / (num_frames - 1)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, fid=fid)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'],
                       vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(
            cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(tqdm(frames)):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            frame_time = frame['time']

            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3, :3])
            R[:, 0] = -R[:, 0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]

            arr = norm_data[:, :, :3] * norm_data[:, :,
                                                  3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[
                                            0],
                                        height=image.size[1], fid=frame_time))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readSpringGausCaptureRealInfo(path, white_background, eval, extension=".png"):
    root_dir = path.split('/')
    obj_name = root_dir[-1]
    root_dir = '/'.join(root_dir[:-1])
    # initialize pcd based on colmap pcd
    num_pts = 100_000
    colmap_pcd_path = os.path.join(root_dir, f'static/colmap/{obj_name}/colmap_pcd.ply')
    plydata = PlyData.read(colmap_pcd_path)
    vertices = plydata['vertex']
    colmap_pcd = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    xyz_max = np.max(colmap_pcd, axis=0)
    xyz_min = np.min(colmap_pcd, axis=0)
    xyz = np.random.random((num_pts, 3)) * (xyz_max - xyz_min) + xyz_min
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    ply_path = os.path.join(path, "points3d.ply")
    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    static_cam_infos, dynamic_cam_infos = readSpringGausCaptureRealCameras(root_dir, obj_name, white_background)
    train_cam_infos = static_cam_infos + dynamic_cam_infos
    test_cam_infos = train_cam_infos if eval else []
    nerf_normalization = getNerfppNorm(static_cam_infos)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readSpringGausCaptureRealCameras(root_dir, obj_name, white_background):
    
    # static scene
    root_dir = Path(root_dir)
    static_dir = root_dir/'static'
    
    camdata = read_cameras_binary(static_dir/'colmap'/obj_name/'cameras.bin')
    imdata = read_images_binary(static_dir/'colmap'/obj_name/'images.bin')
    
    imdata = sorted(imdata.items(), reverse=False)
    
    # K_static = np.array([[camdata[1].params[0], 0, camdata[1].params[2]],
    #                          [0, camdata[1].params[1], camdata[1].params[3]], [0, 0, 1]])

    static_cams = []
    for i, (_, im) in enumerate(tqdm(imdata)):
        R = im.qvec2rotmat()
        t = im.tvec
        
        image_path = static_dir/'images'/obj_name/im.name
        image = Image.open(image_path)
        mask_path = Path(str(image_path).replace('.JPG', '.png').replace('/images/', '/masks/'))
        mask = Image.open(mask_path)
        if mask.size != image.size:
            image = image.resize(mask.size)
        im_data = np.array(image)
        height, width = im_data.shape[:2]
        mask = np.array(mask)[:, :, np.newaxis] / 255.0
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        arr = (im_data / 255.0) * mask + bg * (1 - mask)
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")
        
        static_cams.append(CameraInfo(
            uid=i+3, 
            R=np.transpose(R), 
            T=np.array(t), 
            FovY=focal2fov(camdata[1].params[1], height), 
            FovX=focal2fov(camdata[1].params[0], width), 
            image=image, 
            image_path=image_path, 
            image_name=im.name, 
            width=image.size[0], height=image.size[1], 
            fid=-1, alpha=mask, 
        ))
    
    # dynamic scene
    dynamic_dir = root_dir/'dynamic'
    cam_name = ['C0733', 'C0787', 'C0801']
    H, W, H_S, W_S = 1080, 1920, 2672, 4752
    K = np.array([
        [camdata[1].params[0] * W / W_S, 0, W / 2],
        [0, camdata[1].params[1] * H / H_S, H / 2],
        [0, 0, 1],
    ])
    
    FovY = focal2fov(K[0][0], H)
    FovX = focal2fov(K[1][1], W)
    
    with open(dynamic_dir/'cameras_calib.json', 'r') as f:
        cam_calib = json.load(f)
    # with open(dynamic_dir/'sync.json', 'r') as ff:
    #     sync = json.load(ff)

    with open(dynamic_dir/'sequences'/obj_name/'0.json', 'r') as fff:
        seq_info = json.load(fff)
    
    hit_frame = seq_info['hit_frame']
    n_frames = len(seq_info[cam_name[0]])
    dynamic_cams = []
    
    for frame_id in tqdm(range(n_frames)):
        for cam_id, camera in enumerate(cam_name):
            rvecs = cam_calib[camera]['rvecs']
            tvecs = cam_calib[camera]['tvecs']
            rot_mat, _ = cv2.Rodrigues(np.array(rvecs))
            R = np.transpose(rot_mat)

            image_path = dynamic_dir/'videos_images'/camera/seq_info[camera][frame_id]
            mask_path = Path(str(image_path).replace('/videos_images/', '/videos_masks/').replace('.jpg', '.png'))

            image = Image.open(image_path)
            im_data = np.array(image)
            mask = Image.open(mask_path)
            mask = np.array(mask)[:, :, np.newaxis] / 255.0
            arr = (im_data / 255.0) * mask + bg * (1 - mask)
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            dynamic_cams.append(
                CameraInfo(
                    uid=cam_id,
                    fid=frame_id / 120, 
                    R=R,
                    T=np.array(tvecs).reshape(3),
                    FovY=FovY,
                    FovX=FovX,
                    image=image,
                    image_path=image_path,
                    image_name=seq_info[camera][frame_id],
                    width=W,
                    height=H, 
                    alpha=mask, 
                ))
    
    return static_cams, dynamic_cams


def readPACNeRFInfo(path, config_path, white_background, eval_cam_id=0, load_fix_pcd=False, read_cam=True, max_frames=-1):
    with open(os.path.join(config_path), 'r') as f:
        cfg = json.load(f)
    print("Reading data")
    if read_cam:
        cam_infos = readCamerasFromAllData(path, white_background, max_frames=max_frames)
        eval_cam_infos = [cam_info for cam_info in cam_infos if  cam_info.uid == eval_cam_id]
        nerf_normalization = getNerfppNorm(cam_infos)
    else:
        eval_cam_infos = []
        cam_infos = []
        nerf_normalization = {}

    ply_path = os.path.join(path, "points3d.ply")
    # We create random points inside the bounds of the synthetic Blender scenes
    if not load_fix_pcd:
        xyz_min = np.asarray(cfg['data']['xyz_min']).reshape(1, 3)
        xyz_max = np.asarray(cfg['data']['xyz_max']).reshape(1, 3)
        bounds = xyz_max[0] - xyz_min[0]
        num_pts = np.prod(bounds*50)
        num_pts = int(min(max(num_pts, 1e4), 2e5))
        # Since this data set has no colmap data, we start with random points
        print(f"Generating random point cloud ({num_pts})...")
        xyz = np.random.random((num_pts, 3)) * (xyz_max - xyz_min) + xyz_min
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
            shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd = None
        print(f"Load fixed pcd.")

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=cam_infos,
                           test_cameras=eval_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromAllData(path, white_background, max_frames=-1):
    cam_infos = []
    with open(os.path.join(path, "all_data.json")) as stream:
        frames = json.load(stream)
    for frame in tqdm(frames):
        _, cam_id, frame_id = Path(frame['file_path'].replace('\\', '/')).stem.rsplit('_', 2)
        if int(frame_id) < 0 or (max_frames > 0 and int(frame_id) >= max_frames):
            continue
        relative = frame['file_path'].replace('r_', 'm_').replace('.jpg', '.png')
        image_path = os.path.join(path, relative)
        foreground = np.asarray(Image.open(image_path).convert('RGB'))
        mask = (foreground.astype(int).sum(-1, keepdims=True) != 255 * 3).astype(float)
        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])
        image = Image.fromarray(np.rint(255 * (foreground / 255.0 * mask + bg * (1 - mask))).astype(np.uint8))
        R, T = camera_extrinsics(frame['c2w'])
        intrinsic = np.asarray(frame['intrinsic'])
        if intrinsic.shape != (3, 3):
            raise ValueError('Camera intrinsic must be 3x3; regenerate converted data')
        width, height = image.size
        cam_infos.append(CameraInfo(uid=int(cam_id), R=R, T=T,
            FovX=focal2fov(intrinsic[0, 0], width), FovY=focal2fov(intrinsic[1, 1], height),
            image=image, image_path=image_path, image_name=Path(image_path).stem,
            width=width, height=height, fid=frame['time'], alpha=mask))
    return cam_infos


def readSpringGausMPMSyntheticInfo(path, config_path, white_background, num_frame, eval_cam_id=0):
    with open(os.path.join(config_path), 'r') as f:
        cfg = json.load(f)
    print("Reading data")
    cam_infos = readCamerasFromFrameAndCamera(path, white_background)
    if num_frame != -1:
        assert num_frame > 0
        fids = list(sorted(set([cam_info.fid for cam_info in cam_infos])))
        num_frame = min(num_frame, len(fids))
        max_fid = fids[num_frame-1]
        cam_infos = [cam_info for cam_info in cam_infos if cam_info.fid <= max_fid]
    eval_cam_infos = [cam_info for cam_info in cam_infos if  cam_info.uid == eval_cam_id]

    nerf_normalization = getNerfppNorm(cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    # We create random points inside the bounds of the synthetic Blender scenes
    xyz_min = np.asarray(cfg['data']['xyz_min']).reshape(1, 3)
    xyz_max = np.asarray(cfg['data']['xyz_max']).reshape(1, 3)
    bounds = xyz_max[0] - xyz_min[0]
    num_pts = np.prod(bounds*50)
    num_pts = int(min(max(num_pts, 1e4), 2e5))
    # Since this data set has no colmap data, we start with random points
    print(f"Generating random point cloud ({num_pts})...")
    xyz = np.random.random((num_pts, 3)) * (xyz_max - xyz_min) + xyz_min
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(
        shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=cam_infos,
                           test_cameras=eval_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromFrameAndCamera(path, white_background):
    path = Path(path)
    cam_infos = []

    with open(path/"camera.json") as json_file:
        cam_list = json.load(json_file)
    with open(path/"frame.json") as json_file:
        fid_list = json.load(json_file)
    
    for cam_path in tqdm(path.glob('camera_*')):
        for d in cam_list:
            if d['camera'] == cam_path.name:
                c2w = d['c2w']
                intrinsic = d['K']
                break
        c2w = np.asarray(c2w)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        
        cam_id = int(cam_path.name.split('_')[-1])
        for image_path in cam_path.glob('*'):
            img_id = int(image_path.name.split('.')[0])
            fid = list(fid_list[img_id].values())[0]
            
            image_name = image_path.stem
            image = Image.open(image_path)
            im_data = np.array(image.convert("RGBA"))
            bg = np.array(
                [1, 1, 1]) if white_background else np.array([0, 0, 0])
            norm_data = im_data / 255.0
            mask = norm_data[..., 3:4]
            arr = norm_data[:, :, :3] * mask + bg * (1 - mask)
            image = Image.fromarray(
                np.array(arr * 255.0, dtype=np.byte), "RGB")
            
            fovy = focal2fov(intrinsic[1][1], image.size[0])
            fovx = focal2fov(intrinsic[0][0], image.size[1])
            FovY = fovx
            FovX = fovy

            cam_infos.append(CameraInfo(uid=cam_id, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[
                                            0],
                                        height=image.size[1], fid=fid, alpha=mask))

    return cam_infos

    
sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,  # colmap dataset reader from official 3D Gaussian [https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/]
    "Blender": readNerfSyntheticInfo,  # D-NeRF dataset [https://drive.google.com/file/d/1uHVyApwqugXTFuIRRlE4abTW8_rrVeIK/view?usp=sharing]
    "PacNeRF": readPACNeRFInfo, 
    "SpringGausMPMSynthetic": readSpringGausMPMSyntheticInfo, 
    "SpringGausRealCapture": readSpringGausCaptureRealInfo, 
}
