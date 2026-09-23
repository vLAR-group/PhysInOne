"""Dataset for PhysInOne MotionTransfer inference.

- Origin camera RGB sequence: motion reference video.
- MotionTransfer camera RGB sequence: target appearance first frame.
"""
from __future__ import annotations

import json
import os
import zipfile
from pathlib import Path
from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .utils import decode_rgb_zip_to_tensor


SCENE_LIST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "selected.json",
)

SUBFOLDER = "leaderboard/motion_transfer"


class PhysInOne_Leaderboard_MotionTransfer(Dataset):
    def __init__(
        self,
        data_dir,
        mode: str = "static",
        resolution: Sequence[int] = (320, 320),
    ):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.resolution = self._validate_resolution(resolution)
        self.info = []

        with open(SCENE_LIST, "r", encoding="utf-8") as file:
            scene_list = json.load(file)

        zip_files = list(Path(data_dir).rglob("*.zip"))
        for zip_path in zip_files:
            zip_path = zip_path.resolve()
            complexity = zip_path.parts[-2]
            scene_name = zip_path.stem
            if scene_name not in scene_list:
                continue

            camera_name = scene_list[scene_name][0]
            self.info.append(
                {
                    "scene_name": scene_name,
                    "camera_name": camera_name,
                    "complexity": complexity,
                    "zip_path": zip_path,
                }
            )

    @staticmethod
    def _validate_resolution(resolution: Sequence[int]) -> Tuple[int, int]:
        """Validate and normalize resolution as ``(height, width)``."""
        if len(resolution) != 2:
            raise ValueError(
                "resolution must contain exactly two values (height, width), "
                f"got {resolution!r}"
            )

        height, width = (int(value) for value in resolution)
        if height <= 0 or width <= 0:
            raise ValueError(
                f"resolution values must be positive, got {(height, width)!r}"
            )
        return height, width

    def _resize_inputs(
        self,
        video: torch.Tensor,
        first_frame: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Resize video and first frame to the configured ``(height, width)``.

        The decoder returns the video as ``[T, C, H, W]`` and the first frame
        as ``[C, H, W]``. Both are resized here so every sample has a fixed
        spatial shape before default DataLoader collation and model inference.
        """
        if video.ndim != 4:
            raise ValueError(
                "Expected video with shape [T, C, H, W], "
                f"got {tuple(video.shape)}"
            )
        if first_frame.ndim != 3:
            raise ValueError(
                "Expected first_frame with shape [C, H, W], "
                f"got {tuple(first_frame.shape)}"
            )

        height, width = self.resolution
        video = F.interpolate(
            video.to(torch.float32),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        first_frame = F.interpolate(
            first_frame.unsqueeze(0).to(torch.float32),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        return video, first_frame

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        scene_info = self.info[idx]
        scene_name = scene_info["scene_name"]
        complexity = scene_info["complexity"]
        zip_path = scene_info["zip_path"]

        caption = ""
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            if "caption.txt" in zip_ref.namelist():
                caption = zip_ref.read("caption.txt").decode(
                    "utf-8", errors="ignore"
                ).strip()

            if self.mode == "static":
                camera = scene_info["camera_name"]
            else:
                camera = "CineCamera_Moving"

            video_tensor = decode_rgb_zip_to_tensor(
                zip_ref,
                f"source_video/{camera}/rgb/",
            )
            reference_frame_tensor = decode_rgb_zip_to_tensor(
                zip_ref,
                f"reference_frame/{camera}/rgb/",
            )[0]

        video_tensor, reference_frame_tensor = self._resize_inputs(
            video_tensor,
            reference_frame_tensor,
        )

        return {
            "scene_name": scene_name,
            "complexity": complexity,
            "camera": camera,
            "reference_video": video_tensor,
            "first_frame": reference_frame_tensor,
            "caption": caption,
        }
