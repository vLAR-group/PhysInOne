#!/usr/bin/env python3
"""Generate ExtDM predictions from public observed frames only."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from einops import rearrange
from PIL import Image
from tqdm import tqdm



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground-truth-free ExtDM inference for PhysInOne.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--flowae-checkpoint", required=True)
    parser.add_argument("--dm-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="config/DM/physicsbench.yaml")
    parser.add_argument(
        "--unet3d-arch",
        default="DenoiseNet_STWAtt_w_w_ref_adaptor_cross_multi_traj_ada",
    )
    parser.add_argument("--seed", type=int, default=66)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--max-pred-frames", type=int)
    return parser.parse_args()


def load_rgb_sequence(camera_dir: Path, condition_frames: int) -> torch.Tensor:
    paths = sorted((camera_dir / "rgb").glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No public RGB frames found in {camera_dir / 'rgb'}")
    paths = paths[-condition_frames:]
    frames = [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths]
    while len(frames) < condition_frames:
        frames.insert(0, frames[0].copy())
    return torch.from_numpy(np.stack(frames))


def target_frames(scene_dir: Path) -> dict[int, list[dict]]:
    path = scene_dir / "transforms_test.json"
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    grouped: dict[int, list[dict]] = {}
    for frame in metadata["frames"]:
        view = int(frame.get("view", Path(frame["file_path"]).parts[0].split("_")[-1]))
        grouped.setdefault(view, []).append(frame)
    for values in grouped.values():
        values.sort(key=lambda item: int(item.get("frame", Path(item["file_path"]).name)))
    return grouped


def save_frame(tensor: torch.Tensor, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor.detach().clamp(0, 1)
    image = F.interpolate(image[None], size=(1120, 1120), mode="bilinear", align_corners=False)[0]
    array = image.mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array, mode="RGB").save(path, quality=quality, subsampling=0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    from data.video_dataset import dataset2videos
    from model.BaseDM_adaptor.VideoFlowDiffusion_multi_w_ref_u22 import FlowDiffusion
    from utils.seed import setup_seed
    if not torch.cuda.is_available():
        raise RuntimeError("ExtDM inference requires CUDA.")
    setup_seed(args.seed)

    scene_dir = Path(args.scene_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    model = FlowDiffusion(
        config=config,
        pretrained_pth=args.flowae_checkpoint,
        is_train=False,
        Unet3D_architecture=args.unet3d_arch,
    ).cuda()
    checkpoint = torch.load(args.dm_checkpoint, map_location="cuda")
    model.diffusion.load_state_dict(checkpoint["diffusion"])
    model.eval()

    condition_frames = int(config["dataset_params"]["valid_params"]["cond_frames"])
    pred_chunk = int(config["dataset_params"]["train_params"]["pred_frames"])
    targets = target_frames(scene_dir)
    written = []

    camera_dirs = sorted(
        path for path in scene_dir.iterdir()
        if path.is_dir() and path.name.startswith("CineCamera_") and (path / "rgb").is_dir()
    )
    for camera_dir in tqdm(camera_dirs, desc="ExtDM cameras"):
        view = int(camera_dir.name.rsplit("_", 1)[-1])
        frames = targets.get(view, [])
        if args.max_pred_frames is not None:
            frames = frames[: args.max_pred_frames]
        if not frames:
            continue

        observed = load_rgb_sequence(camera_dir, condition_frames).cuda().float() / 255.0
        observed = observed.permute(0, 3, 1, 2)
        observed = F.interpolate(observed, size=(64, 64), mode="bilinear", align_corners=False)
        real_vid = rearrange(dataset2videos(observed.permute(0, 2, 3, 1)[None]), "b t c h w -> b c t h w")

        chunks = []
        state = real_vid[:, :, -condition_frames:]
        for _ in range(math.ceil(len(frames) / pred_chunk)):
            sample = model.sample_one_video(cond_scale=1.0, real_vid=state)["sample_out_vid"]
            chunks.append(sample[:, :, -pred_chunk:].detach())
            state = sample[:, :, -condition_frames:]
            torch.cuda.empty_cache()
        prediction = torch.cat(chunks, dim=2)[0, :, : len(frames)].permute(1, 0, 2, 3)

        for tensor, frame in zip(prediction, frames):
            relative = Path(frame["file_path"]).with_suffix(".jpg")
            save_frame(tensor, output_dir / relative, args.jpeg_quality)
            written.append(relative.as_posix())

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "ExtDM",
        "source_scene": scene_dir.name,
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
