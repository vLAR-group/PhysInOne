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
import uuid
import torch
import torchvision
from random import randint
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from utils.loss_utils import l1_loss, ssim
from utils.image_utils import psnr
from utils.general_utils import safe_state
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, DeformModel
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training_report(tb_writer, iteration, Ll1, loss, l1_loss_fn, elapsed, testing_iterations, scene: Scene, render_func,
                    render_args, deform, load2gpu_on_the_fly, is_6dof=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    test_psnr = 0.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()},
                              {'name': 'train',
                               'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in
                                           range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                images = []
                gts = []
                for idx, viewpoint in enumerate(config['cameras']):
                    if load2gpu_on_the_fly:
                        viewpoint.load2device()

                    fid = viewpoint.fid
                    xyz = scene.gaussians.get_xyz
                    time_input = fid.unsqueeze(0).expand(xyz.shape[0], -1)
                    d_xyz, d_rotation, d_scaling = deform.step(xyz.detach(), time_input)

                    image = torch.clamp(
                        render_func(viewpoint, scene.gaussians, *render_args, d_xyz, d_rotation, d_scaling, is_6dof)["render"],
                        0.0, 1.0
                    ).cpu()
                    gt_image = torch.clamp(viewpoint.original_image.cpu(), 0.0, 1.0)
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
                l1_test = l1_loss_fn(images, gts)
                psnr_test = psnr(images, gts).mean()
                if config['name'] == 'test' or len(validation_configs[0]['cameras']) == 0:
                    test_psnr = psnr_test
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

    return test_psnr


def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str = os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def dump_t0_view_comparisons(scene, gaussians, deform, pipe, background, dataset, iteration):
    out_root = os.path.join(dataset.model_path, f"t0_compare_iter_{iteration}")
    render_dir = os.path.join(out_root, "render")
    gt_dir = os.path.join(out_root, "gt")
    compare_dir = os.path.join(out_root, "compare")
    os.makedirs(render_dir, exist_ok=True)
    os.makedirs(gt_dir, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)

    cameras = scene.getTrainCameras()
    for idx, viewpoint in enumerate(cameras):
        if dataset.load2gpu_on_the_fly:
            viewpoint.load2device()

        xyz = gaussians.get_xyz
        t0_input = torch.zeros((xyz.shape[0], 1), device=xyz.device, dtype=xyz.dtype)
        d_xyz, d_rotation, d_scaling = deform.step_full_rot(xyz.detach(), t0_input, 1 / dataset.fps)

        rendering = torch.clamp(
            render(viewpoint, gaussians, pipe, background, d_xyz, d_rotation, d_scaling)["render"],
            0.0,
            1.0,
        )
        gt = torch.clamp(viewpoint.original_image.cuda(), 0.0, 1.0)
        compare = torch.cat([rendering, gt], dim=2)

        filename = f"{idx:05d}_{viewpoint.image_name}.png"
        torchvision.utils.save_image(rendering, os.path.join(render_dir, filename))
        torchvision.utils.save_image(gt, os.path.join(gt_dir, filename))
        torchvision.utils.save_image(compare, os.path.join(compare_dir, filename))

        if dataset.load2gpu_on_the_fly:
            viewpoint.load2device('cpu')

    print(f"[ITER {iteration}] Saved t=0 view comparisons to: {out_root}")
    torch.cuda.empty_cache()


def training(dataset, opt, pipe, testing_iterations, saving_iterations, vel_start_time=0.0, max_points=300000,
             use_screen_cull=True, screen_cull_margin=0.05):
    tb_writer = prepare_output_and_logger(dataset)

    gaussians = GaussianModel(dataset.sh_degree)
    deform = DeformModel(max_time=dataset.max_time, vel_start_time=vel_start_time, light=dataset.light,
                         physics_code=dataset.physics_code, freegave=dataset.freegave)
    deform.train_setting(opt)

    scene = Scene(dataset, gaussians, skip_val=True, skip_test=True)
    gaussians.training_setup(opt)
    gaussians.use_screen_cull = use_screen_cull
    gaussians.screen_cull_margin = screen_cull_margin

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    best_psnr = 0.0
    best_iteration = 0
    progress_bar = tqdm(range(opt.iterations), desc="Training progress")

    for iteration in range(1, opt.iterations + 1):
        if network_gui.conn is None:
            network_gui.try_connect()
        while network_gui.conn is not None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.do_shs_python, pipe.do_cov_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam is not None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview(
                        (torch.clamp(net_image, min=0, max=1.0) * 255)
                        .byte()
                        .permute(1, 2, 0)
                        .contiguous()
                        .cpu()
                        .numpy()
                    )
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception:
                network_gui.conn = None

        iter_start.record()

        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        if not viewpoint_stack or iteration == opt.warm_up:
            if iteration < opt.warm_up:
                viewpoint_stack = scene.getInitCameras().copy()
            else:
                viewpoint_stack = scene.getTrainCameras().copy()
                dataset_max_time = max([viewpoint.fid for viewpoint in viewpoint_stack])
                if opt.gradual:
                    cur_max_fid = ((iteration - opt.warm_up) // 1000 + 1.) / ((10000 - opt.warm_up) // 1000 + 1.) * dataset_max_time
                    cur_max_fid = min(cur_max_fid, dataset_max_time)
                    viewpoint_stack = [viewpoint for viewpoint in viewpoint_stack if viewpoint.fid <= cur_max_fid]

        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))
        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device()
        fid = viewpoint_cam.fid

        if iteration < opt.warm_up:
            d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        else:
            num_points = gaussians.get_xyz.shape[0]
            time_input = fid.unsqueeze(0).expand(num_points, -1)
            d_xyz, d_rotation, d_scaling = deform.step_full_rot(gaussians.get_xyz.detach(), time_input, 1 / dataset.fps)

        render_pkg_re = render(viewpoint_cam, gaussians, pipe, background, d_xyz, d_rotation, d_scaling)
        image = render_pkg_re["render"]
        viewspace_point_tensor = render_pkg_re["viewspace_points"]
        visibility_filter = render_pkg_re["visibility_filter"]
        radii = render_pkg_re["radii"]

        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        if dataset.load2gpu_on_the_fly:
            viewpoint_cam.load2device('cpu')

        with torch.no_grad():
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            gaussians.max_radii2D[visibility_filter] = torch.max(
                gaussians.max_radii2D[visibility_filter],
                radii[visibility_filter]
            )

            # cur_psnr = training_report(
            #     tb_writer,
            #     iteration,
            #     Ll1,
            #     loss,
            #     l1_loss,
            #     iter_start.elapsed_time(iter_end),
            #     testing_iterations,
            #     scene,
            #     render,
            #     (pipe, background),
            #     deform,
            #     dataset.load2gpu_on_the_fly,
            # )
            cur_psnr = 0.0

            if iteration in testing_iterations and cur_psnr.item() > best_psnr:
                best_psnr = cur_psnr.item()
                best_iteration = iteration

            if iteration in saving_iterations:
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                deform.save_weights(dataset.model_path, iteration)

            # if iteration == 3000:
            #     dump_t0_view_comparisons(scene, gaussians, deform, pipe, background, dataset, iteration)

            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0 and iteration != opt.warm_up:
                    cur_max_points = max_points
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.prune_min_opacity,
                        scene.cameras_extent,
                        size_threshold,
                        max_points=cur_max_points
                    )

                if iteration % opt.opacity_reset_interval == 0 or (
                    dataset.white_background and iteration == opt.densify_from_iter
                ):
                    if  iteration != opt.warm_up:
                        gaussians.reset_opacity()

            if iteration % 500 == 0:
                cur_max_points = max_points
                print(f"[ITER {iteration}] Gaussian points: {gaussians.get_xyz.shape[0]:,} / {cur_max_points:,}")

            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.update_learning_rate(iteration)
                gaussians.optimizer.zero_grad(set_to_none=True)

                deform.optimizer.step()
                deform.optimizer.zero_grad()
                deform.update_learning_rate(iteration)

    print("Best PSNR = {} in Iteration {}".format(best_psnr, best_iteration))


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[40001])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7000, 10000, 20000, 30000, 40000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--vel_start_time', type=float, default=0.0)
    parser.add_argument('--max_points', type=int, default=300000, help="Maximum number of Gaussian points after iter >= 3000 (iter < 3000 is fixed to 400000)")
    parser.add_argument('--disable_screen_cull', action='store_true', default=False,
                        help='Disable screen-space culling in filter_gaussians.')
    parser.add_argument('--screen_cull_margin', type=float, default=0.05,
                        help='Screen-space culling margin ratio. 0.05 means keep [-5%, 105%] image bounds.')

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        vel_start_time=args.vel_start_time,
        max_points=args.max_points,
        use_screen_cull=not args.disable_screen_cull,
        screen_cull_margin=args.screen_cull_margin,
    )

    print("\nTraining complete.")
