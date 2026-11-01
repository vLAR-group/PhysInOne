"""Render property-estimation results from one shared particle trajectory."""
import argparse
import os
import numpy as np
import torch
import mmcv

from lib import PACNeRF
from lib.numerical_checks import assert_finite_tensors
from lib.trajectory_cache import ParticleTrajectory, parse_camera_ids, trajectory_metadata
from model_params import save_estimated_model_params
from property_export import (load_camera_rays, load_render_cameras,
                             prepare_property_output, render_property_views)


device = torch.device("cuda:0")
torch.manual_seed(123)
np.random.seed(123)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("yes", "true", "t", "1", "y"):
        return True
    if value in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_float_list(value: str, expected_len: int, name: str):
    if value is None:
        return None
    values = [float(v.strip()) for v in value.split(",") if v.strip()]
    if len(values) != expected_len:
        raise argparse.ArgumentTypeError(f"{name} expects {expected_len} comma-separated values.")
    return values


def create_model(cfg, stage, pnerf=None):
    base_dir = cfg["base_dir"]
    runtime_keys = (
        "data_dir",
        "base_dir",
        "n_cameras",
        "n_frames",
        "frame_dt",
        "H",
        "W",
        "BC",
        "use_complex_boundary",
        "rotation",
        "translation",
        "center",
    )
    runtime_cfg = {key: cfg[key] for key in runtime_keys if key in cfg}
    checkpoint = torch.load(os.path.join(base_dir, f"model/train_{stage}.pt"))
    assert_finite_tensors(
        checkpoint["model_state_dict"],
        f"Loading {os.path.join(base_dir, f'model/train_{stage}.pt')}",
    )
    cfg.update(checkpoint["cfg"])
    cfg.update(runtime_cfg)
    if pnerf is None:
        pnerf = PACNeRF(**cfg)
        pnerf.to(device)
    pnerf.load_state_dict(checkpoint["model_state_dict"], strict=False)
    return pnerf


def get_reinit_transform(cfg, args):
    rotation = parse_float_list(args.rotation, 3, "rotation")
    translation = parse_float_list(args.translation, 3, "translation")
    center = parse_float_list(args.center, 3, "center")
    rotation = rotation if rotation is not None else cfg.get("rotation", [0.0, 0.0, 0.0])
    translation = translation if translation is not None else cfg.get("translation", [0.0, 0.0, 0.0])
    center = center if center is not None else cfg.get("center", [0.0, 0.0, 0.0])
    return rotation, translation, torch.tensor(center)


def config_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", required=True, help="config file path")
    parser.add_argument("--num-frame", required=True, type=int, help="number of frames")
    parser.add_argument("--cam-id", default="all",
                        help="all or comma-separated indices; reinit uses merged original-camera order")
    parser.add_argument("--render-kind", choices=("novel", "reinit", "normal", "resimulate"), default="novel")
    parser.add_argument("--stage", default="dynamic")
    parser.add_argument("--complex-boundary", type=str2bool, default=None)
    parser.add_argument("--write-video", type=str2bool, default=False)
    parser.add_argument("--fps", type=float, default=None,
                        help="Video playback FPS; defaults to 1/frame_dt from the render config")
    parser.add_argument("--simulation-cache", choices=("auto", "refresh", "off"), default="auto",
                        help="Reuse a disk trajectory, regenerate it, or share only in memory.")
    parser.add_argument("--simulation-cache-dir", default=None)
    parser.add_argument("--save-params", type=str2bool, default=True)
    parser.add_argument("--output-dir", "--params-output", dest="output_dir", default=None,
                        help="Export root; defaults to <base_dir>/property_estimation.")
    parser.add_argument("--camera-mapping", default=None)
    parser.add_argument("--camera-split", choices=("train", "test"), default=None,
                        help="Novel-view mapping override; reinit always uses train + test.")
    parser.add_argument("--alpha", "--alpha-output", dest="alpha_output", action="store_true",
                        help="Export foreground RGB JPEGs on white instead of the scene background.")
    parser.add_argument("--rotation", default=None, help="reinit Euler angles in degrees, e.g. 90,0,0")
    parser.add_argument("--translation", default=None, help="reinit translation, e.g. 0,0,0")
    parser.add_argument("--center", default=None, help="reinit rotation center, e.g. 0,0.4,0")
    return parser


def main(argv=None, defaults=None):
    parser = config_parser()
    if defaults:
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if args.num_frame < 1:
        parser.error('--num-frame must be positive')
    if args.fps is not None and args.fps <= 0:
        parser.error('--fps must be positive')
    mode = 'reinit' if args.render_kind in ('reinit', 'resimulate') else 'novel'
    cfg = mmcv.Config.fromfile(args.config)["cfg"]
    if args.fps is None:
        args.fps = 1.0 / cfg["frame_dt"]
    if args.complex_boundary is not None:
        cfg['use_complex_boundary'] = args.complex_boundary
    camera_metadata = load_render_cameras(cfg, args, mode)
    camera_ids = parse_camera_ids(args.cam_id, cfg['n_cameras'])
    rays = load_camera_rays(cfg['data_dir'], cfg['n_cameras'], cfg['H'], cfg['W'],
                            cameras=camera_metadata[0])
    outputs, backgrounds = prepare_property_output(
        cfg, args, mode, camera_ids, camera_metadata=camera_metadata)
    pnerf = create_model(cfg, args.stage)
    if args.save_params:
        save_estimated_model_params(pnerf, cfg, args, stage=args.stage)
    transform = get_reinit_transform(cfg, args) if mode == 'reinit' else None
    checkpoint = os.path.join(cfg['base_dir'], f'model/train_{args.stage}.pt')
    metadata = trajectory_metadata(cfg, checkpoint, transform)
    cache_root = args.simulation_cache_dir or os.path.join(cfg['base_dir'], 'inference_cache')
    trajectory = ParticleTrajectory(cache_root, metadata, mode=args.simulation_cache)
    trajectory.prepare(args.num_frame,
                       lambda cache: pnerf.simulate_trajectory(args.num_frame, cache, transform=transform))
    return render_property_views(pnerf, cfg, args, camera_ids, rays, trajectory, outputs, backgrounds)


if __name__ == '__main__':
    main()
