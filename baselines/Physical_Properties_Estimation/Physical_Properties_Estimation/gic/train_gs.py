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
import time 
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim, kl_divergence
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
from utils.reg_utils import o3d_knn, mini_batch_knn, quat_mult, build_rotation, weighted_l2_loss_v2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
import trimesh

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(dataset)
    deform.train_setting(opt)
    scene = Scene(dataset, gaussians, resolution_scales=[dataset.res_scale])
    gaussians.training_setup(opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")
    smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)
    neighbor_sq_dist, neighbor_indices = None, None
    update_knn = True
    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2,
                                                                                                               0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            views = scene.getTrainCameras(scale=dataset.res_scale).copy()
            if iteration < opt.warm_up:
                # viewpoint_stack = views
                warm_up_fid = torch.unique(torch.stack([view.fid for view in views]))[0]
                viewpoint_stack = [view for view in views if view.fid == warm_up_fid]
            else:
                viewpoint_stack = views

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0)

            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda') * time_interval * smooth_term(iteration)
            d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise)

        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii, alpha = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re["alpha"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        # mask = torch.logical_or(gt_image.sum(0) != 0, 
        #                         image.sum(0) != 0)
        # ids = torch.where(mask)
        # h_min, h_max, w_min, w_max = ids[0].min(), ids[0].max(), ids[1].min(), ids[1].max()
        # h_min, h_max = max(h_min-50, 0), min(h_max+50, image.shape[1])
        # w_min, w_max = max(w_min-50, 0), min(w_max+50, image.shape[2])
        # image = image[:, h_min:h_max, w_min:w_max]
        # gt_image = gt_image[:, h_min:h_max, w_min:w_max]
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        if opt.reg_alpha:
            # l1 loss
            L_alpha = l1_loss(alpha, viewpoint_cam.gt_alpha_mask)
            # cross entropy loss
            # alpha = torch.clamp(alpha, 1e-4, 1-1e-4)
            # L_alpha = torch.mean(-viewpoint_cam.gt_alpha_mask * torch.log(alpha) - (1 - viewpoint_cam.gt_alpha_mask) * torch.log(1 - alpha)) * 0.05
            # weighted loss
            # flag = viewpoint_cam.gt_alpha_mask == 1
            # not_flag = torch.logical_not(flag)
            # weighted l1 loss
            # pos_a = torch.mean(torch.abs(alpha[flag] - viewpoint_cam.gt_alpha_mask[flag]))
            # neg_a = torch.mean(torch.abs(alpha[not_flag] - viewpoint_cam.gt_alpha_mask[not_flag]))
            # L_alpha = 0.5 * (pos_a + neg_a)
            # weighted cross entropy
            # alpha = torch.clamp(alpha, 1e-4, 1-1e-4)
            # pos_a = torch.mean(-torch.log(alpha[flag]))
            # neg_a = torch.mean(-torch.log(1 - alpha[not_flag]))
            # L_alpha = 0.5 * (pos_a + neg_a) * 0.1
            loss += L_alpha
        if iteration >= opt.warm_up:
            
            if opt.reg_rigid:
                xyz_t = gaussians.get_xyz + d_xyz
                rotations_t = gaussians.get_rotation + d_rotation
                inv_rotations_t = rotations_t  # just indicate that it's the inverse of rotation
                inv_rotations_t[:, 1:] = -1 * inv_rotations_t[:, 1:]
                d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + 0.01)
                xyz_t_1 = gaussians.get_xyz + d_xyz
                rotations_t_1 = gaussians.get_rotation + d_rotation
                rel_rot = quat_mult(rotations_t_1, inv_rotations_t)
                rot = build_rotation(rel_rot)
                if neighbor_sq_dist is None or update_knn:
                    neighbor_sq_dist, neighbor_indices = mini_batch_knn(xyz_t.detach(), xyz_t.detach(), opt.num_knn)
                    # neighbor_sq_dist, neighbor_indices = o3d_knn(xyz_t.detach().cpu().numpy(), opt.num_knn)
                    # neighbor_sq_dist = torch.from_numpy(neighbor_sq_dist).to(xyz_t)
                    # neighbor_indices = torch.from_numpy(neighbor_indices).to('cuda').to(torch.int64)
                    neighbor_weight = torch.exp(-2000 * neighbor_sq_dist)
                    update_knn = False
                curr_neighbor_pts = xyz_t[neighbor_indices]
                curr_offset = curr_neighbor_pts - xyz_t[:, None]
                next_neighbor_pts = xyz_t_1[neighbor_indices]
                next_offset = next_neighbor_pts - xyz_t_1[:, None]
                next_offset_in_curr_coord = (rot.transpose(2, 1)[:, None] @ next_offset[:, :, :, None]).squeeze(-1)
                L_rigid = weighted_l2_loss_v2(next_offset_in_curr_coord, curr_offset, neighbor_weight)
                loss = loss + L_rigid
            if opt.reg_scale:
                scales = gaussians.get_scaling + d_scaling
                L_scale = torch.mean(torch.abs(scales)) * 10
                # if iteration < opt.densify_until_iter:
                #     diff_scales = torch.nn.functional.relu(scales-1e-4) + torch.nn.functional.relu(0-scales)
                # else:
                #     diff_scales = torch.nn.functional.relu(0-scales) # torch.nn.functional.relu(0-scales)
                # diff_scales = torch.nn.functional.relu(scales-1e-4) + torch.nn.functional.relu(0-scales)
                # L_scale = torch.mean(diff_scales) * 10
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
            cur_psnr = training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end),
                                       testing_iterations, scene, render, (pipe, background), deform,
                                       dataset.load2gpu_on_the_fly, dataset.res_scale, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(dataset.model_path, iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.001, scene.cameras_extent*0.5, size_threshold) # origin 0.005
                    update_knn = True
                    # print(gaussians.get_xyz.shape[0])

                if iteration > opt.warm_up and iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity_v2(val=0.05)
            
            if opt.reg_tgs and opt.warm_up * 5 < iteration <= opt.tgs_densify_until_iter and iteration % (opt.densification_interval*5) == 0:
                gaussians.truncate_gs(opt.tgs_bound)
                gaussians.densify_surface(opt.tgs_bound, max_screen_size=opt.tgs_max_screen_size)
                print(gaussians.get_xyz.shape[0])
            
            if iteration % 1000 == 0: 
                vertex_colors = np.concatenate([(gaussians.get_opacity.detach().cpu().numpy()*255).astype(np.uint8)]*3, axis=1)
                if iteration < opt.warm_up:
                    xyz0 = gaussians.get_xyz
                else:
                    xyz0 = gaussians.get_xyz+deform.step(gaussians.get_xyz, torch.tensor([[0.0]]).to('cuda'))[0]  # 0.5416666666666666
                trimesh.Trimesh(
                    xyz0.detach().cpu().numpy(), 
                    vertex_colors=vertex_colors, 
                    ).export(os.path.join(dataset.model_path, f'gs/gs_{iteration}.ply'))
                cat_img = torch.cat([image, gt_image], dim=2)
                torchvision.utils.save_image(cat_img, os.path.join(dataset.model_path, f'img/ren_gt_{iteration}.png'))
                cat_mask = torch.cat([alpha, viewpoint_cam.gt_alpha_mask], dim=2)
                torchvision.utils.save_image(cat_mask, os.path.join(dataset.model_path, f'img/mask_ren_gt_{iteration}.png'))
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                deform.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(os.path.join(args.model_path, f'gs'), exist_ok=True)
    os.makedirs(os.path.join(args.model_path, f'img'), exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, res_scale=1, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras(scale=res_scale)},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras(scale=res_scale)[idx % len(scene.getTrainCameras(scale=res_scale))] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = torch.tensor([], device="cuda")
                gts = torch.tensor([], device="cuda")
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    # time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    time_input = fid.unsqueeze(0)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test), flush=True)
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            # tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int,
                        default=[5000, 6000, 7_000] + list(range(10000, 40001, 1000)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000, 20_000, 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    b = time.perf_counter()
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations)
    e = time.perf_counter()
    # All done
    print(f"\nTraining complete, time elapsed: {(e-b)/60} mins.")
