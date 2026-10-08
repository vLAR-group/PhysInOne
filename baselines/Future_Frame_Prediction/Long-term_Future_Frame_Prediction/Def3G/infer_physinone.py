#!/usr/bin/env python3
"""Render PhysInOne future frames without opening private RGB ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from utils.graphics_utils import focal2fov, fov2focal, getProjectionMatrix, getWorld2View2

METHOD = "Def3G"


class PredictionCamera:
    def __init__(self, frame: dict, metadata: dict, device: torch.device):
        self.uid = int(frame.get("frame", 0))
        self.view = int(frame.get("view", -1))
        self.image_name = Path(frame["file_path"]).name
        self.relative_path = Path(frame["file_path"])

        c2w = np.asarray(frame["transform_matrix"], dtype=np.float32).copy()
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        self.R = np.transpose(w2c[:3, :3])
        self.T = w2c[:3, 3]
        self.image_width = int(metadata["img_w"])
        self.image_height = int(metadata["img_h"])
        self.FoVx = float(metadata["camera_angle_x"])
        focal = fov2focal(self.FoVx, self.image_width)
        self.FoVy = focal2fov(focal, self.image_height)
        self.fid = torch.tensor([float(frame["time"])], dtype=torch.float32, device=device)

        self.znear = 0.8
        self.zfar = 100.0
        self.world_view_transform = torch.tensor(
            getWorld2View2(self.R, self.T), dtype=torch.float32, device=device
        ).transpose(0, 1)
        if frame.get("proj_transform") is not None:
            self.projection_matrix = torch.tensor(
                frame["proj_transform"], dtype=torch.float32, device=device
            )
        else:
            self.projection_matrix = getProjectionMatrix(
                znear=0.01, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            ).transpose(0, 1).to(device)
        self.full_proj_transform = self.world_view_transform.unsqueeze(0).bmm(
            self.projection_matrix.unsqueeze(0)
        ).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=f"Ground-truth-free {METHOD} inference for PhysInOne."
    )
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--transforms", default="transforms_test.json")
    parser.add_argument("--iteration", type=int, default=25000)
    parser.add_argument("--views", nargs="*", type=int)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--image-extension", choices=("jpg", "png"), default="jpg")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--sh-degree", type=int, default=3)
    parser.add_argument("--max-time", type=float, default=0.45)
    parser.add_argument("--vel-start-time", type=float, default=0.0)
    parser.add_argument("--physics-code", type=int, default=16)
    parser.add_argument("--light", action="store_true")
    parser.add_argument("--is-6dof", action="store_true")
    parser.add_argument("--black-background", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def read_frames(path: Path, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    required = {"camera_angle_x", "img_w", "img_h", "frames"}
    missing = sorted(required.difference(metadata))
    if missing:
        raise ValueError(f"{path} is missing required keys: {', '.join(missing)}")

    selected_views = set(args.views) if args.views else None
    frames = []
    for frame in metadata["frames"]:
        frame_index = int(frame.get("frame", Path(frame["file_path"]).name))
        view_id = int(frame.get("view", -1))
        if selected_views is not None and view_id not in selected_views:
            continue
        if args.frame_start is not None and frame_index < args.frame_start:
            continue
        if args.frame_end is not None and frame_index > args.frame_end:
            continue
        frames.append(frame)
    if not frames:
        raise ValueError("No transform frames matched the requested filters.")
    return metadata, frames


def resolve_iteration(model_path: Path, requested: int) -> int:
    from utils.system_utils import searchForMaxIteration
    if requested != -1:
        return requested
    iteration = searchForMaxIteration(str(model_path / "point_cloud"))
    if iteration is None:
        raise FileNotFoundError(f"No point-cloud checkpoint under {model_path}")
    return iteration


def save_rgb(image: torch.Tensor, path: Path, args: argparse.Namespace) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = image.detach().clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    output = Image.fromarray(array, mode="RGB")
    if args.image_extension == "jpg":
        output.save(path, quality=args.jpeg_quality, subsampling=0)
    else:
        output.save(path)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError(f"{METHOD} inference requires CUDA.")

    from gaussian_renderer import render
    from scene.deform_model import DeformModel
    from scene.gaussian_model import GaussianModel
    from utils.general_utils import safe_state

    safe_state(args.quiet)
    device = torch.device("cuda")
    source_path = Path(args.source_path).resolve()
    model_path = Path(args.model_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    metadata, frames = read_frames(source_path / args.transforms, args)
    iteration = resolve_iteration(model_path, args.iteration)
    point_cloud = model_path / "point_cloud" / f"iteration_{iteration}" / "point_cloud.ply"
    if not point_cloud.is_file():
        raise FileNotFoundError(point_cloud)

    gaussians = GaussianModel(args.sh_degree)
    gaussians.load_ply(str(point_cloud))
    if METHOD == "TRACE":
        deform = DeformModel(
            is_blender=True,
            is_6dof=args.is_6dof,
            max_time=args.max_time,
            vel_start_time=args.vel_start_time,
            light=args.light,
            physics_code=args.physics_code,
            freegave=False,
        )
    else:
        deform = DeformModel(
            is_blender=True,
            is_6dof=args.is_6dof,
            max_time=args.max_time,
            vel_start_time=args.vel_start_time,
        )
    deform.load_weights(str(model_path), iteration)
    deform.deform.eval()
    if METHOD == "TRACE":
        deform.vel.eval()

    pipeline = SimpleNamespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
    value = 0.0 if args.black_background else 1.0
    background = torch.tensor([value, value, value], dtype=torch.float32, device=device)
    fps = float(metadata.get("fps", 30.0))
    dt = 1.0 / fps
    xyz = gaussians.get_xyz
    written = []

    for frame in tqdm(frames, desc=f"Rendering {METHOD} predictions"):
        camera = PredictionCamera(frame, metadata, device)
        time_input = camera.fid.unsqueeze(0).expand(xyz.shape[0], -1)
        if METHOD == "TRACE":
            d_xyz, d_rotation, d_scaling = deform.step(xyz, time_input, dt=dt)
        else:
            d_xyz, d_rotation, d_scaling = deform.step(
                xyz, time_input, gaussians.get_deform_code, dt=dt, max_time=args.max_time
            )
        image = render(
            camera, gaussians, pipeline, background,
            d_xyz, d_rotation, d_scaling, args.is_6dof
        )["render"]
        relative = camera.relative_path.with_suffix(f".{args.image_extension}")
        save_rgb(image, output_dir / relative, args)
        written.append(relative.as_posix())

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": METHOD,
        "source_scene": source_path.name,
        "checkpoint_iteration": iteration,
        "transforms": args.transforms,
        "prediction_count": len(written),
        "contains_ground_truth": False,
        "files": written,
    }
    with (output_dir / "prediction_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print(f"Saved {len(written)} predictions to {output_dir}")


if __name__ == "__main__":
    main()
