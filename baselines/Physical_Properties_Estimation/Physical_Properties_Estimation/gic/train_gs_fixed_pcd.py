
import os
import torch
import trimesh
import torchvision
import numpy as np
from tqdm import tqdm
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
from scene import Scene, GaussianModel
from utils.system_utils import check_gs_model
from utils.image_utils import psnr
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud


def train(dataset, opt, pipe, testing_iterations, saving_iterations, pcd, d_xyz_list=None, fps=24, cam_info=None, grid_size=0.12):
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, resolution_scales=[dataset.res_scale], pcd=pcd, load_fix_pcd=True, cam_info=cam_info)
    if d_xyz_list:
        scene.clipTrainCamerasbyframes(len(d_xyz_list))
        scene.clipTestCamerasbyframes(len(d_xyz_list))

    gaussians.training_setup(opt, fix_pcd=True)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training fixed pcd progress")
    print(f"reg alpha {opt.reg_alpha}")
    print(f"reg scale {opt.reg_scale}")
    for iteration in range(1, opt.iterations + 1):

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            views = scene.getTrainCameras(scale=dataset.res_scale).copy()
            if iteration < opt.warm_up or d_xyz_list is None:
                # viewpoint_stack = views
                warm_up_fid = torch.unique(torch.stack([view.fid for view in views]))[0]
                viewpoint_stack = [view for view in views if view.fid == warm_up_fid]
            else:
                viewpoint_stack = views

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            if d_xyz_list is None:
                d_xyz = 0.0
            else:
                d_xyz = d_xyz_list[int(torch.round(fid * fps).item())]
            d_rotation, d_scaling = 0.0, 0.0
                
        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii, alpha = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["alpha"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if opt.reg_alpha:
            # gt_op_mask = (torch.var(gt_image, dim=0) > 0.001).to(torch.float32)
            # gt_op_mask = (gt_image.sum(dim=0) > 0.0).to(torch.float32)
            L_alpha = l1_loss(alpha, viewpoint_cam.gt_alpha_mask)
            loss += L_alpha

        if iteration >= opt.warm_up:
            if opt.reg_scale:
                scales = gaussians.get_scaling
                diff_scales = torch.nn.functional.relu(scales-grid_size/16) + torch.nn.functional.relu(grid_size/32-scales)
                L_scale = torch.mean(diff_scales) * 10
                loss = loss + L_scale
            if opt.reg_tgs and iteration > opt.warm_up * 5:
                depth = render_pkg_re["depth"][0]
                xyz_t = gaussians.get_xyz + d_xyz
                gs_w, gs_h, gs_d = viewpoint_cam.pw2pix(xyz_t)
                in_mask = viewpoint_cam.is_in_view(gs_w, gs_h)
                render_pix_d = depth[gs_h[in_mask], gs_w[in_mask]]
                diff_depth = gs_d[in_mask] - render_pix_d
                depth_mask = torch.logical_and(opt.tgs_bound >= diff_depth, render_pix_d > 0)
                if depth_mask.sum() == 0:
                    L_tgs = 0.0
                else:
                    L_tgs = torch.mean(torch.abs(diff_depth[depth_mask])) * 0.1
                loss = loss + L_tgs
                gaussians.add_diff_depth_stats(diff_depth.detach(), in_mask)
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Keep track of max radii in image-space for pruning
            gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                 radii[visibility_filter])

            # Log and save
            torch.cuda.empty_cache()
            cur_psnr = psnr_report(iteration, l1_loss,
                                    testing_iterations, scene, render, (pipe, background), 
                                    dataset.load2gpu_on_the_fly, dataset.res_scale, dataset.is_6dof, d_xyz_list, fps)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, fix_pcd=True)
            
            if iteration % 1000 == 0: 
                vertex_colors = np.concatenate([(gaussians.get_opacity.detach().cpu().numpy()*255).astype(np.uint8)]*3, axis=1)
                file_name = f'gs/fix_gs_{iteration}.ply'
                if iteration < opt.warm_up:
                    xyz0 = gaussians.get_xyz
                else:
                    if d_xyz_list is None:
                        d_xyz = 0.0
                    else:
                        d_xyz = d_xyz_list[int(torch.round(fid/(1/fps)))]
                    d_rotation, d_scaling = 0.0, 0.0
                    xyz0 = gaussians.get_xyz + d_xyz
                    trimesh.Trimesh(
                        xyz0.detach().cpu().numpy(), 
                        vertex_colors=vertex_colors, 
                        ).export(os.path.join(dataset.model_path, file_name))
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def psnr_report(iteration, l1_loss, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, load2gpu_on_the_fly, res_scale=1, is_6dof=False, d_xyz_list=None, fps=24):

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=res_scale)},
                              {'name': 'train',
                               'cameras': scene.getTrainCameras(scale=res_scale)})

        for config in validation_configs:
            min_fid = min([view.fid for view in config['cameras']])
            cams = [view for view in config['cameras'] if view.fid == min_fid] if d_xyz_list is None else config['cameras'] 
            if config['cameras'] and len(cams) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                
                for idx, viewpoint in enumerate(cams):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    if d_xyz_list is None:
                        d_xyz = 0.0
                    else:
                        d_xyz = d_xyz_list[int(torch.round(fid/(1/fps)))]
                    d_rotation, d_scaling = 0.0, 0.0
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test), flush=True)

        torch.cuda.empty_cache()

    return test_psnr

def train_gs_with_fixed_pcd(xyz, dataset, opt, pipe, testing_iterations, saving_iterations, d_xyz_list, fps, force_train=False, cam_info=None, grid_size=0.12):
    xyz = xyz.cpu().detach().numpy()
    num_pts= xyz.shape[0]
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    if (not check_gs_model(dataset.model_path, saving_iterations, fix_pcd=True)) or force_train:
        train(dataset, opt, pipe, testing_iterations, saving_iterations, pcd, d_xyz_list=d_xyz_list, fps=fps, cam_info=cam_info, grid_size=grid_size)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=opt.iterations, shuffle=False, resolution_scales=[1.0], load_fix_pcd=True, cam_info=cam_info)
    return scene


def assign_gs_to_pcd(xyz, xyz_opacity, dataset, opt, pipe, cam_info, grid_size=0.12, scene=None):
    # TODO remove useless params
    xyz = xyz.cpu().detach().numpy()
    num_pts= xyz.shape[0]
    shs = np.random.random((num_pts, 3)) / 255.0
    pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    gaussians = GaussianModel(dataset.sh_degree)
    if scene is None:
        scene = Scene(dataset, gaussians, resolution_scales=[1.0], pcd=pcd, cam_info=cam_info)
    xyz_opacity = xyz_opacity.reshape(-1, 1)
    scene.gaussians._opacity = torch.nn.Parameter(scene.gaussians.inverse_opacity_activation(torch.clamp(xyz_opacity, max=1-1e-4)).requires_grad_(True))
    scales = torch.ones_like(xyz_opacity) * grid_size / 32 * 0.5
    scene.gaussians._scaling = torch.nn.Parameter(scene.gaussians.scaling_inverse_activation(scales).requires_grad_(True))
    return scene
