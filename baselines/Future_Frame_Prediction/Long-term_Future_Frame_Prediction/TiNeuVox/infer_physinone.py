#!/usr/bin/env python3
"""Render TiNeuVox future predictions from transforms_test.json metadata only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground-truth-free TiNeuVox inference.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="configs/nerf-base/PhysicsBench/physinone.py")
    parser.add_argument("--views", nargs="*", type=int)
    parser.add_argument("--frame-start", type=int)
    parser.add_argument("--frame-end", type=int)
    parser.add_argument("--ray-batch-size", type=int, default=1000)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def load_frames(scene_dir: Path, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    with (scene_dir / "transforms_test.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    selected = set(args.views) if args.views else None
    frames = []
    for frame in metadata["frames"]:
        index = int(frame.get("frame", Path(frame["file_path"]).name))
        view = int(frame.get("view", Path(frame["file_path"]).parts[0].split("_")[-1]))
        if selected is not None and view not in selected:
            continue
        if args.frame_start is not None and index < args.frame_start:
            continue
        if args.frame_end is not None and index > args.frame_end:
            continue
        frames.append(frame)
    if not frames:
        raise ValueError("No transform frames matched the requested filters.")
    return metadata, frames


@torch.no_grad()
def main() -> None:
    args = parse_args()
    import imageio.v2 as imageio
    import mmengine
    from lib import tineuvox, utils
    if not torch.cuda.is_available():
        raise RuntimeError("TiNeuVox inference requires CUDA.")

    scene_dir = Path(args.scene_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    metadata, frames = load_frames(scene_dir, args)
    cfg = mmengine.Config.fromfile(args.config)
    model = utils.load_model(tineuvox.TiNeuVox, args.checkpoint).cuda().eval()

    height = int(metadata["img_h"])
    width = int(metadata["img_w"])
    focal = 0.5 * width / np.tan(0.5 * float(metadata["camera_angle_x"]))
    K = torch.tensor(
        [[focal, 0.0, 0.5 * width], [0.0, focal, 0.5 * height], [0.0, 0.0, 1.0]],
        dtype=torch.float32,
        device="cuda",
    )
    near, far = 1.0, 8.0
    render_kwargs = {
        "near": near,
        "far": far,
        "bg": 1 if cfg.data.white_bkgd else 0,
        "stepsize": cfg.model_and_render.stepsize,
    }

    written = []
    for frame in tqdm(frames, desc="Rendering TiNeuVox predictions"):
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32, device="cuda")
        rays_o, rays_d, viewdirs = tineuvox.get_rays_of_a_view(
            height, width, K, c2w, cfg.data.ndc
        )
        rays_o = rays_o.flatten(0, -2)
        rays_d = rays_d.flatten(0, -2)
        viewdirs = viewdirs.flatten(0, -2)
        time = torch.full(
            (rays_o.shape[0], 1), float(frame["time"]), dtype=torch.float32, device="cuda"
        )
        chunks = []
        for ro, rd, vd, ts in zip(
            rays_o.split(args.ray_batch_size),
            rays_d.split(args.ray_batch_size),
            viewdirs.split(args.ray_batch_size),
            time.split(args.ray_batch_size),
        ):
            chunks.append(model(ro, rd, vd, ts, **render_kwargs)["rgb_marched"])
        rgb = torch.cat(chunks).reshape(height, width, 3)
        relative = Path(frame["file_path"]).with_suffix(".jpg")
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(
            destination,
            utils.to8b(rgb.detach().cpu().numpy()),
            quality=args.jpeg_quality,
        )
        written.append(relative.as_posix())

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "TiNeuVox",
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
