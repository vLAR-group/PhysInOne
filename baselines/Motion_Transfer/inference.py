#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-GPU inference for PhysInOne MotionTransfer baselines."""
from __future__ import annotations

import argparse
import gc
import io
import os
import shutil
import zipfile
from pathlib import Path, PurePosixPath
from typing import Sequence

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from dataset import DataLoader, PhysInOne_Leaderboard_MotionTransfer


DEFAULT_GWF_LORA = "I2V5B_final_i38800_nearest_lora_weights"
REPO_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference-only motion transfer on PhysInOne MotionTransfer data."
    )
    parser.add_argument("--data_root", type=str, default="../../PhysInOne_data", help="Path to the MotionTransfer dataset root.")
    parser.add_argument(
        "--output_path",
        "--out_root",
        dest="output_path",
        default="outputs",
        help="Root directory for generated RGB frame folders or scene ZIP archives.",
    )
    parser.add_argument(
        "--zip",
        dest="zip_output",
        action="store_true",
        help="Save each scene as <scene_name>.zip instead of an uncompressed scene folder.",
    )
    parser.add_argument("--method", choices=["gowiththeflow", "motionpro"], default="motionpro")
    parser.add_argument("--mode", choices=["static", "moving"], default="static")
    parser.add_argument("--resolution", type=int, nargs=2, default=[320, 320], metavar=("H", "W"))
    parser.add_argument("--num_frames", type=int, default=16, help="Reference motion frames sampled per item.")
    parser.add_argument("--frame_stride", type=int, default=1)
    parser.add_argument("--sample_mode", choices=["all", "uniform", "head", "tail"], default="uniform")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Fixed at 1 for this serial single-GPU release.",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--camera", action="append", default=None, help="Optional camera name filter. Can repeat.")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:0.")

    parser.add_argument("--motionpro_ckpt", default="checkpoints/MotionPro_Dense-gs_14k.pt")
    parser.add_argument("--motionpro_config", default="configs/inference_all_flow_from_svd.yaml")
    parser.add_argument("--cotracker_ckpt", default="tools/co-tracker/checkpoints/scaled_offline.pth")
    parser.add_argument("--motionpro_dtype", choices=["float16", "bfloat16", "float32"], default="float16")

    parser.add_argument("--gowiththeflow_lora", default=DEFAULT_GWF_LORA)
    parser.add_argument("--gowiththeflow_steps", type=int, default=15)
    parser.add_argument("--gowiththeflow_guidance_scale", type=float, default=6.0)
    parser.add_argument("--gowiththeflow_low_vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--gowiththeflow_resize_frames", type=float, default=0.5)
    parser.add_argument("--gowiththeflow_resize_flow", type=int, default=8)
    parser.add_argument("--gowiththeflow_downscale_factor", type=int, default=32)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        raise RuntimeError("CPU inference is not supported for these baselines. Please run on a CUDA GPU.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for inference, but no CUDA GPU is visible to PyTorch.")
    return torch.device(requested)


def _torch_dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _require_file(path: str, label: str, *, downloadable: bool = True) -> str:
    requested = Path(path).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(REPO_ROOT / requested)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    guidance = (
        "Run `python scripts/download_ckpt.py --name all` or pass the correct path explicitly."
        if downloadable
        else "Pass the correct path explicitly."
    )
    raise FileNotFoundError(f"{label} not found: {path}\n{guidance}")


def _prepare_gowiththeflow_lora(lora: str) -> str:
    requested = Path(lora).expanduser()
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(REPO_ROOT / requested)

    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
        if candidate.is_dir():
            os.environ["GWF_LORA_DIR"] = str(candidate.resolve())
            return DEFAULT_GWF_LORA
    return lora


class MotionTransferRunner:
    """Load one baseline once and reuse it for every dataset sample."""

    def __init__(self, args: argparse.Namespace, device: torch.device) -> None:
        self.args = args
        self.method = args.method.lower()
        self._close = None

        if self.method == "gowiththeflow":
            from utils.gowiththeflow import clear_pipe_cache, get_pipe, go_with_the_flow_api

            self._gowiththeflow_api = go_with_the_flow_api
            self._model_name = _prepare_gowiththeflow_lora(args.gowiththeflow_lora)
            self._pipe = get_pipe(
                model_name=self._model_name,
                device=args.device,
                low_vram=args.gowiththeflow_low_vram,
            )
            self._close = clear_pipe_cache
            return

        if self.method == "motionpro":
            from utils.motionpro import MotionProDenseRunner

            self._runner = MotionProDenseRunner(
                ckpt_path=_require_file(args.motionpro_ckpt, "MotionPro-Dense checkpoint"),
                config_path=_require_file(
                    args.motionpro_config,
                    "MotionPro inference config",
                    downloadable=False,
                ),
                cotracker_ckpt=_require_file(args.cotracker_ckpt, "CoTracker checkpoint"),
                seed=args.seed,
                fps=args.fps,
                num_frames=args.num_frames,
                dtype=_torch_dtype(args.motionpro_dtype),
                device=device,
            )
            self._close = self._runner.close
            return

        raise ValueError(f"Unsupported method: {args.method}")

    @torch.no_grad()
    def __call__(
        self,
        reference_videos: torch.Tensor,
        first_frames: torch.Tensor,
        captions: Sequence[str],
    ) -> torch.Tensor:
        if self.method == "gowiththeflow":
            return self._gowiththeflow_api(
                refs=reference_videos,
                fr1s=first_frames,
                caps=captions,
                model_name=self._model_name,
                pipe=self._pipe,
                device=self.args.device,
                low_vram=self.args.gowiththeflow_low_vram,
                num_inference_steps=self.args.gowiththeflow_steps,
                guidance_scale=self.args.gowiththeflow_guidance_scale,
                resize_frames=self.args.gowiththeflow_resize_frames,
                resize_flow=self.args.gowiththeflow_resize_flow,
                downscale_factor=self.args.gowiththeflow_downscale_factor,
                save_debug=False,
                save_dirs=None,
            )
        return self._runner(reference_videos, first_frames)

    def close(self) -> None:
        close = self._close
        self._close = None
        if hasattr(self, "_pipe"):
            self._pipe = None
        if close is not None:
            close()
        if hasattr(self, "_runner"):
            self._runner = None


def _match_frame_count(video: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Match a generated video's length to the source video frame count.

    Short videos are linearly interpolated along the time axis. Long videos
    are truncated by keeping their first ``target_frames`` frames.
    """
    if video.ndim != 4 or video.shape[-1] != 3:
        raise ValueError(
            "Expected an output video with shape [frames, height, width, 3], "
            f"but received {tuple(video.shape)}."
        )
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, but received {target_frames}.")

    output_frames = video.shape[0]
    if output_frames == 0:
        raise ValueError("Cannot interpolate an output video containing zero frames.")
    if output_frames == target_frames:
        return video
    if output_frames > target_frames:
        return video[:target_frames]

    # Treat every RGB pixel as a 1-D temporal signal. align_corners=True keeps
    # the generated video's first and last frames at the two endpoints.
    height, width, channels = video.shape[1:]
    temporal_signals = (
        video.to(torch.float32)
        .permute(1, 2, 3, 0)
        .reshape(1, height * width * channels, output_frames)
    )
    interpolated = F.interpolate(
        temporal_signals,
        size=target_frames,
        mode="linear",
        align_corners=True,
    )
    return interpolated.reshape(height, width, channels, target_frames).permute(3, 0, 1, 2).contiguous()

def _video_to_uint8(video: torch.Tensor) -> np.ndarray:
    if video.dtype != torch.float32:
        video = video.to(torch.float32)
    arr = video.detach().cpu().clamp(0, 1).numpy()
    frames = (arr * 255.0 + 0.5).astype(np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(
            "Expected an output video with shape [frames, height, width, 3], "
            f"but received {tuple(frames.shape)}."
        )
    return frames


def _batch_strings(value: object, field_name: str) -> list[str]:
    """Normalize string fields produced by either direct or default-collated batches."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, Sequence):
        values = list(value)
        if all(isinstance(item, str) for item in values):
            return values
    raise TypeError(f"batch[{field_name!r}] must be a string or a sequence of strings.")


def _safe_component(value: str, field_name: str) -> str:
    """Reject values that could escape their intended output directory/archive path."""
    if not value or value in {".", ".."} or "/" in value or "\\" in value or "\x00" in value:
        raise ValueError(f"Invalid {field_name}: {value!r}")
    return value


class FrameOutputWriter:
    """Save generated videos as RGB JPEG sequences, optionally grouped into scene ZIPs."""

    def __init__(self, output_path: str, method: str, zip_output: bool) -> None:
        self.root = Path(output_path) / _safe_component(method, "method")
        self.zip_output = zip_output
        self._initialized_archives: set[Path] = set()
        self._written_cameras: set[tuple[str, str, str]] = set()

    def save(
        self,
        video: torch.Tensor,
        *,
        complexity: str,
        scene_name: str,
        camera: str,
    ) -> Path:
        complexity = _safe_component(complexity, "complexity")
        scene_name = _safe_component(scene_name, "scene_name")
        camera = _safe_component(camera, "camera")

        output_key = (complexity, scene_name, camera)
        if output_key in self._written_cameras:
            raise ValueError(
                "Duplicate output destination for "
                f"complexity={complexity!r}, scene_name={scene_name!r}, camera={camera!r}."
            )
        self._written_cameras.add(output_key)

        frames = _video_to_uint8(video)
        if self.zip_output:
            return self._save_zip(frames, complexity, scene_name, camera)
        return self._save_directory(frames, complexity, scene_name, camera)

    def _save_directory(
        self,
        frames: np.ndarray,
        complexity: str,
        scene_name: str,
        camera: str,
    ) -> Path:
        rgb_dir = self.root / complexity / scene_name / camera / "rgb"
        if rgb_dir.exists():
            shutil.rmtree(rgb_dir)
        rgb_dir.mkdir(parents=True, exist_ok=True)

        for frame_index, frame in enumerate(frames):
            imageio.imwrite(rgb_dir / f"{frame_index:05d}.jpg", frame, quality=95)
        return rgb_dir.parent.parent

    def _save_zip(
        self,
        frames: np.ndarray,
        complexity: str,
        scene_name: str,
        camera: str,
    ) -> Path:
        zip_path = self.root / complexity / f"{scene_name}.zip"
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if zip_path in self._initialized_archives else "w"

        # JPEG data is already compressed, so ZIP_STORED avoids redundant recompression.
        with zipfile.ZipFile(zip_path, mode=mode, compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
            for frame_index, frame in enumerate(frames):
                buffer = io.BytesIO()
                imageio.imwrite(buffer, frame, format="JPEG", quality=95)
                archive_name = PurePosixPath(camera, "rgb", f"{frame_index:05d}.jpg").as_posix()
                archive.writestr(archive_name, buffer.getvalue())

        self._initialized_archives.add(zip_path)
        return zip_path


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    if args.batch_size != 1:
        raise ValueError("This release supports serial single-GPU inference only; use --batch_size 1.")
    device = resolve_device(args.device)
    dataset = PhysInOne_Leaderboard_MotionTransfer(
        args.data_root,
        args.mode,
        resolution=tuple(args.resolution) if args.resolution else None,
    )
    # print(dataset)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    ).loader
    runner = MotionTransferRunner(args, device)
    output_writer = FrameOutputWriter(args.output_path, args.method, args.zip_output)

    try:
        progress = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Evaluating {args.method}",
            unit="sample",
            dynamic_ncols=True,
        )
        for batch_index, batch in enumerate(progress):
            refs = batch["reference_video"]
            firsts = batch["first_frame"]
            captions = _batch_strings(batch["caption"], "caption")
            scene_names = _batch_strings(batch["scene_name"], "scene_name")
            complexities = _batch_strings(batch["complexity"], "complexity")
            cameras = _batch_strings(batch["camera"], "camera")

            if not isinstance(refs, torch.Tensor) or not isinstance(firsts, torch.Tensor):
                raise RuntimeError("All samples in a batch must share the same tensor shape. Use fixed --resize_hw.")

            batch_size = refs.shape[0]
            source_frame_count = refs.shape[1]
            string_fields = {
                "caption": captions,
                "scene_name": scene_names,
                "complexity": complexities,
                "camera": cameras,
            }
            for field_name, values in string_fields.items():
                if len(values) != batch_size:
                    raise RuntimeError(
                        f"batch[{field_name!r}] has {len(values)} values, but the tensor batch size is {batch_size}."
                    )

            if args.method == "motionpro":
                refs = refs.to(device=device, non_blocking=True)
                firsts = firsts.to(device=device, non_blocking=True)

            refs = refs.permute(0, 1, 3, 4, 2).contiguous()  # [B, T, C, H, W] -> [B, T, H, W, C]
            firsts = firsts.permute(0, 2, 3, 1).contiguous()  # [B, C, H, W] -> [B, H, W, C]
            outputs = runner(refs, firsts, captions).detach().cpu().clamp(0, 1)

            if outputs.shape[0] != batch_size:
                raise RuntimeError(
                    f"Model returned {outputs.shape[0]} videos for an input batch of {batch_size}."
                )

            for item_index in range(batch_size):
                output_video = _match_frame_count(outputs[item_index], source_frame_count)
                destination = output_writer.save(
                    output_video,
                    complexity=complexities[item_index],
                    scene_name=scene_names[item_index],
                    camera=cameras[item_index],
                )
                progress.set_postfix(
                    scene=scene_names[item_index],
                    camera=cameras[item_index],
                    refresh=False,
                )
                tqdm.write(f"[saved] {destination}")

            del refs, firsts, outputs
            clear_memory()
    finally:
        runner.close()


if __name__ == "__main__":
    main()
