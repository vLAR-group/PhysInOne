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
import copy
import json
import torch
import random
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from scene.deform_model import DeformModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


class Scene:
    gaussians: GaussianModel

    def __init__(self, args: ModelParams, gaussians: GaussianModel, load_iteration=None, shuffle=True,
                 resolution_scales=[1.0], pcd=None, load_fix_pcd=False, cam_info=None):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {} if cam_info is None else cam_info.get("train_cams")
        self.test_cameras = {} if cam_info is None else cam_info.get("test_cams")
        read_cam = True if cam_info is None else False
        if not read_cam:
            print('use given cams')
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "all_data.json")):
            print("Found all_data.json file, assuming PacNeRF data set!")
            scene_info = sceneLoadTypeCallbacks["PacNeRF"](args.source_path, args.config_path, args.white_background, load_fix_pcd=load_fix_pcd, read_cam=read_cam, max_frames=args.num_frame)
        elif os.path.exists(os.path.join(args.source_path, "camera.json")) and os.path.exists(os.path.join(args.source_path, "frame.json")):
            print("Found all_data.json file, assuming SpringGaus MPM Synthetic data set!")
            scene_info = sceneLoadTypeCallbacks["SpringGausMPMSynthetic"](args.source_path, args.config_path, args.white_background, args.num_frame)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif 'real_capture' in args.source_path:
            print("Found real_capture, assuming Spring-Gaus Real Capture data set!")
            scene_info = sceneLoadTypeCallbacks["SpringGausRealCapture"](args.source_path, args.white_background, args.eval)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter and read_cam:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply"),
                                                                   'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle and read_cam:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling
        
        self.cameras_extent = scene_info.nerf_normalization["radius"] if read_cam else cam_info.get("cameras_extent")

        for resolution_scale in resolution_scales:
            if not read_cam:
                continue
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale,
                                                                            args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale,
                                                                           args)
        if pcd is not None:
            self.gaussians.create_from_pcd(pcd, self.cameras_extent)
        else:
            if self.loaded_iter:
                if not load_fix_pcd:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"),
                                            og_number_points=len(scene_info.point_cloud.points))
                else:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                        "point_cloud_fix_pcd",
                                                        "iteration_" + str(self.loaded_iter),
                                                        "point_cloud.ply"))
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration, fix_pcd=False):
        name = "point_cloud/iteration_{}".format(iteration) if not fix_pcd else "point_cloud_fix_pcd/iteration_{}".format(iteration)
        point_cloud_path = os.path.join(self.model_path, name)
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def clipTrainCamerasbyframes(self, f):
        new_cameras = {}
        for scale, cam_list in self.train_cameras.items():
            cam_frames = len(torch.unique(torch.stack([view.fid for view in cam_list])))
            if f < cam_frames:
                sorted_times, _ = torch.sort(torch.unique(torch.stack([view.fid for view in cam_list])))
                max_t = sorted_times[:f][-1]
                new_cameras[scale] = [v for v in cam_list if v.fid <= max_t] #<=
            else:
                new_cameras[scale] = cam_list
        self.train_cameras = new_cameras

    def clipTestCamerasbyframes(self, f):
        new_cameras = {}
        for scale, cam_list in self.test_cameras.items():
            cam_frames = len(torch.unique(torch.stack([view.fid for view in cam_list])))
            if f < cam_frames:
                sorted_times, _ = torch.sort(torch.unique(torch.stack([view.fid for view in cam_list])))
                max_t = sorted_times[:f][-1]
                new_cameras[scale] = [v for v in cam_list if v.fid <= max_t] #<=
            else:
                new_cameras[scale] = cam_list
        self.test_cameras = new_cameras

    
    def overwrite_alphas(self, pipeline, dataset: ModelParams, deform: DeformModel):
        from gaussian_renderer import render
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        xyz_canonical = self.gaussians.get_xyz.detach()
        def overwrite(cam_dict):
            for scale, cam_list in cam_dict.items():
                for view in cam_list:
                    fid = view.fid
                    time_input = fid.unsqueeze(0).expand(1, -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz_canonical, time_input)
                    results = render(view, self.gaussians, pipeline, background, d_xyz, d_rotation, d_scaling, False)
                    alpha = results["alpha"]
                    view.gt_alpha_mask = alpha.to(view.data_device)
            return copy.deepcopy(cam_dict)
        train_cams = overwrite(self.train_cameras)
        test_cams = overwrite(self.test_cameras)
        cameras_extent = copy.deepcopy(self.cameras_extent)
        return train_cams, test_cams, cameras_extent