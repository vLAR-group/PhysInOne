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
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel, DeformModel
from utils.general_utils import safe_state, get_linear_noise_func
import uuid
import einops
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
# from fast_pytorch_kmeans import KMeans
import json

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(dataset, opt, pipe, testing_iterations, saving_iterations, vel_start_time=0.0):
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(max_time=dataset.max_time, vel_start_time=vel_start_time, light=dataset.light, physics_code=dataset.physics_code)
    deform.train_setting(opt)

    # Derive the simulation step from the public training metadata.
    train_json_path = os.path.join(dataset.source_path, 'transforms_train.json')
    with open(train_json_path, 'r') as f:
        data = json.load(f)
    total_frames = data['total_frames']
    fps = data['fps']
    if fps == 30:
        dt = 3 / total_frames
    else:
        dt = 6 / total_frames

    # Public training must never load private validation or future test RGB frames.
    scene = Scene(dataset, gaussians, skip_val=True, skip_test=True)
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
        if not viewpoint_stack or iteration == opt.warm_up:
            if iteration < opt.warm_up:
                viewpoint_stack = scene.getInitCameras().copy()
            else:
                viewpoint_stack = scene.getTrainCameras().copy()
                if opt.gradual:
                    cur_max_fid = ((iteration - opt.warm_up) // 1000 + 1.) / ((10000 - opt.warm_up) // 1000 + 1.) * 0.75
                    cur_max_fid = min(cur_max_fid, 0.75)
                    viewpoint_stack = [viewpoint for viewpoint in viewpoint_stack if viewpoint.fid <= cur_max_fid]

        total_frame = len(viewpoint_stack)
        time_interval = 1 / total_frame

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        # elif iteration < 2 * opt.warm_up:
        #     N = gaussians.get_xyz.shape[0]
        #     time_input = fid.unsqueeze(0).expand(N, -1)
        #     ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, 1) * time_interval * smooth_term(iteration)
        #     d_xyz, d_rotation, d_scaling = deform.step(gaussians.get_xyz.detach(), time_input + ast_noise, gaussians.get_deform_code)
        # elif iteration > opt.densify_until_iter:
        #     d_xyz = torch.zeros_like(gaussians.get_xyz)
        #     d_rotation = torch.zeros_like(gaussians.get_xyz)
        #     d_scaling = torch.zeros_like(gaussians.get_xyz)
        #
        #     N = gaussians.get_xyz.shape[0]
        #     time_input = fid.unsqueeze(0).expand(N, -1)
        #     d_xyz[motion_mask], d_rotation[motion_mask], d_scaling[motion_mask] = deform.step_no_rot(
        #         gaussians.get_xyz.detach()[motion_mask], time_input[motion_mask], gaussians.get_deform_code[motion_mask]
        #     )
        #     d_xyz[static_mask], d_rotation[static_mask], d_scaling[static_mask] = deform.step(
        #         gaussians.get_xyz.detach()[static_mask], time_input[static_mask] * 0., gaussians.get_deform_code[static_mask]
        #     )
        else:
            N = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(N, -1)
            ast_noise = 0 if dataset.is_blender else torch.randn(1, 1, device='cuda').expand(N, -1) * time_interval * smooth_term(iteration)
            deform_code = deform.code_field(gaussians.get_xyz.detach())
            d_xyz, d_rotation, d_scaling = deform.step_no_rot(gaussians.get_xyz.detach(), time_input + ast_noise,
                                                              deform_code, dt=dt)


        # Render
        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling, dataset.is_6dof)
        image, viewspace_point_tensor, visibility_filter, radii, depth_filter = render_pkg_re["render"], render_pkg_re[
            "viewspace_points"], render_pkg_re["visibility_filter"], render_pkg_re["radii"], render_pkg_re[
            "depth_filter"]
        # depth = render_pkg_re["depth"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if dataset.max_depth > 0:
            depth = viewpoint_cam.depth
            depth_mask = depth < dataset.max_depth
            depth_mask = depth_mask.unsqueeze(0)
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image*depth_mask, gt_image*depth_mask)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        elif dataset.seg_mask:
            seg = viewpoint_cam.seg
            seg_mask = (seg != 0) & (seg != 255)
            seg_mask = seg_mask.unsqueeze(0)
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image*seg_mask, gt_image*seg_mask)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if iteration >= opt.densify_until_iter and iteration % 3000 == 0:
            with torch.no_grad():
                xyz = gaussians.get_xyz
                deform_code = deform.code_field(xyz)
                deform_seg = deform.code_field.seg(deform_code)

                dxyz = deform.deform.get_translation(xyz, torch.zeros((len(deform_code), 1), device='cuda'), deform_code)
                sampled_time = torch.arange(0, 75, 1, device='cuda') / 100
                sampled_time = einops.repeat(sampled_time, 't -> n_points t', n_points=xyz.shape[0])
                static_mask = torch.ones_like(sampled_time[:, 0], dtype=torch.bool)
                for i in range(75):
                    dxyz_t = deform.deform.get_translation(xyz, sampled_time[:, i:i + 1], deform_code)
                    static_mask &= ((dxyz_t - dxyz).norm(dim=1) < 0.1)
                motion_mask = ~static_mask
                # feature = torch.cat([deform_code * motion_mask.unsqueeze(-1)], dim=-1)
                feature = torch.cat([deform_seg], dim=-1) * motion_mask.unsqueeze(-1)
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
                                       dataset.load2gpu_on_the_fly, dataset.is_6dof)
            if iteration in testing_iterations:
                if cur_psnr.item() > best_psnr:
                    best_psnr = cur_psnr.item()
                    best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(dataset.model_path, iteration)

            # Densification
            # if iteration < opt.densify_until_iter:
            #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            #
            #     if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            #         size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            #         gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
            #
            #     if iteration % opt.opacity_reset_interval == 0 or (
            #             dataset.white_background and iteration == opt.densify_from_iter):
            #         gaussians.reset_opacity()

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter],
                                                                     radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and scene.gaussians.get_xyz.shape[0]<400000:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.08, scene.cameras_extent,
                                                size_threshold)
                    # gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())

                if iteration % opt.opacity_reset_interval == 0 or (
                        dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # if iteration % 100 == 0 and iteration > opt.densify_until_iter:
            #     if iteration < opt.iterations - 100:
            #         # don't update in the end of training
            #         gaussians.compute_3D_filter(cameras=scene.getTrainCameras().copy())

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
    with open(os.path.join(args.model_path, "config.json"), "w", encoding="utf-8") as cfg_log_f:
        json.dump(vars(args), cfg_log_f, indent=2)
        cfg_log_f.write("\n")

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene: Scene, renderFunc,
                    renderArgs, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    report_psnr = torch.tensor(0.0)
    # Report only public training samples. Future test RGB is intentionally unavailable.
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())]
                                           for idx in range(5, 30, 5)]},)

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                # images = torch.tensor([], device="cuda")
                # gts = torch.tensor([], device="cuda")
                images = []
                gts = []
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()
                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    # deform_code = scene.gaussians.get_deform_code
                    deform_code = deform.code_field(xyz)
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input, deform_code)
                    image = torch.clamp(
                        renderFunc(viewpoint, scene.gaussians, *renderArgs, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0).cpu()
                    gt_image = torch.clamp(viewpoint.original_image.cpu(), 0.0, 1.0)
                    # images = torch.cat((images, image.unsqueeze(0)), dim=0)
                    # gts = torch.cat((gts, gt_image.unsqueeze(0)), dim=0)
                    images.append(image)
                    gts.append(gt_image)

                    if load2gpu_on_the_fly:
                        viewpoint.load2device('cpu')
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name),
                                             image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name),
                                                 gt_image[None], global_step=iteration)

                images = torch.stack(images, dim=0)
                gts = torch.stack(gts, dim=0)
                l1_test = l1_loss(images, gts)
                psnr_test = psnr(images, gts).mean()
                report_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return report_psnr


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[40001])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[4_000, 10_000, 20_000,  25_000 , 30_000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--vel_start_time', type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations,
             args.save_iterations, args.vel_start_time)

    # All done
    print("\nTraining complete.")
