#!/usr/bin/env python3
"""Generate MAGI-1 predictions from public conditioning frames only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ground-truth-free MAGI-1 inference for PhysInOne.")
    parser.add_argument("--scene-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--config", default="example/4.5B/4.5B_distill_config.json")
    parser.add_argument("--condition-frames", type=int, default=32)
    parser.add_argument("--max-pred-frames", type=int)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    return parser.parse_args()


def targets_by_view(scene_dir: Path) -> dict[int, list[dict]]:
    with (scene_dir / "transforms_test.json").open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    grouped: dict[int, list[dict]] = {}
    for frame in metadata["frames"]:
        view = int(frame.get("view", Path(frame["file_path"]).parts[0].split("_")[-1]))
        grouped.setdefault(view, []).append(frame)
    for frames in grouped.values():
        frames.sort(key=lambda item: int(item.get("frame", Path(item["file_path"]).name)))
    return grouped


def condition_video(camera_dir: Path, count: int) -> torch.Tensor:
    paths = sorted((camera_dir / "rgb").glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"No public RGB frames in {camera_dir / 'rgb'}")
    paths = paths[-count:]
    frames = [np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8) for path in paths]
    while len(frames) < count:
        frames.insert(0, frames[0].copy())
    tensor = torch.from_numpy(np.stack(frames)).cuda()
    tensor = tensor.permute(0, 3, 1, 2).float()
    tensor = F.interpolate(tensor, size=(480, 480), mode="bilinear", align_corners=False)
    return tensor.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1)


def save_frame(frame: torch.Tensor, path: Path, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = frame.detach().cpu().numpy().transpose(1, 2, 0)
    array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="RGB").resize((1120, 1120), Image.Resampling.LANCZOS)
    image.save(path, quality=quality, subsampling=0)


@torch.no_grad()
def main() -> None:
    args = parse_args()
    import inference.infra.distributed.parallel_state as mpu
    from inference.model.dit import get_dit
    from inference.pipeline import MagiPipeline
    from inference.pipeline.video_process import VaeHelper, encode_prefix_video
    if not torch.cuda.is_available():
        raise RuntimeError("MAGI-1 inference requires CUDA.")

    scene_dir = Path(args.scene_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    targets = targets_by_view(scene_dir)
    if args.max_pred_frames is not None:
        targets = {view: frames[: args.max_pred_frames] for view, frames in targets.items()}
    max_frames = max((len(frames) for frames in targets.values()), default=0)
    if max_frames == 0:
        raise ValueError("transforms_test.json contains no requested prediction frames.")

    pipeline = MagiPipeline(args.config)
    pipeline.config.runtime_config.num_frames = max_frames
    vae_model = VaeHelper.get_vae(pipeline.config.runtime_config.vae_pretrained)
    dit = get_dit(pipeline.config)
    written = []

    camera_dirs = sorted(
        path for path in scene_dir.iterdir()
        if path.is_dir() and path.name.startswith("CineCamera_") and (path / "rgb").is_dir()
    )
    for camera_dir in tqdm(camera_dirs, desc="MAGI-1 cameras"):
        view = int(camera_dir.name.rsplit("_", 1)[-1])
        frames = targets.get(view, [])
        if not frames:
            continue
        prefix = condition_video(camera_dir, args.condition_frames)
        encoded = encode_prefix_video(
            prefix,
            pipeline.config.runtime_config.fps,
            pipeline.config.runtime_config.vae_pretrained,
            pipeline.config.runtime_config.scale_factor,
            parallel_group=mpu.get_tp_group(with_context_parallel=True),
            vae_model=vae_model,
        )
        output = pipeline.v2v(encoded, dit)[-len(frames):]
        for tensor, frame in zip(output, frames):
            relative = Path(frame["file_path"]).with_suffix(".jpg")
            save_frame(tensor, output_dir / relative, args.jpeg_quality)
            written.append(relative.as_posix())

    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": "MAGI-1",
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
