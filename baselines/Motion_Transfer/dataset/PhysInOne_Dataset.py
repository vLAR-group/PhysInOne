# -*- coding: utf-8 -*-
"""Dataset for PhysInOne MotionTransfer inference.

Each sample uses one camera pair from ``motion_transfer_mapping.json``:
- Origin camera RGB sequence: motion reference video.
- MotionTransfer camera RGB sequence: target appearance first frame.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple
import zipfile

from torch.utils.data import Dataset

from .utils import (
    decode_rgb_zip_to_tensor,
)

SCENE_LIST = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "selected.json",
)

SUBFOLDER = "leaderboard/motion_transfer"  # Subfolder for leaderboard/test data (if present)
class PhysInOne_Leaderboard_MotionTransfer(Dataset):
    def __init__(self, data_dir, mode='static'):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode

        self.info = []
        with open(SCENE_LIST, "r") as f:
            scene_list = json.load(f)

        zip_files = list(data_dir.rglob('*.zip'))
        for zip in zip_files:
            zip = zip.resolve()
            complexity = zip.parts[-2]  # Assuming the structure is .../complexity/scene.zip
            scene_name = zip.stem
            if scene_name not in scene_list:
                continue
            camera_name = scene_list[scene_name][0]
            self.info.append({
                "scene_name": scene_name,
                "camera_name": camera_name,
                "complexity": complexity,
                "zip_path": zip,
            })

    def __len__(self):
        return len(self.info)

    def __getitem__(self, idx):
        scene_info = self.info[idx]
        scene_name = scene_info["scene_name"]
        complexity = scene_info["complexity"]
        zip_path = scene_info["zip_path"]

        caption = ""
        video_tensor = None
        reference_frame_tensor = None

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # 1. Load Caption (assuming it's at the root of the zip)
            if "caption.txt" in zip_ref.namelist():
                caption = zip_ref.read("caption.txt").decode("utf-8", errors="ignore").strip()

            # 2. Determine Camera Name
            if self.mode == 'static':
                camera = scene_info["camera_name"]
            else:
                camera = "CineCamera_Moving"

            # 3. Load RGB frames directly from memory
            # Note: ZipFile internally uses forward slashes '/' for paths, regardless of OS
            video_tensor = decode_rgb_zip_to_tensor(zip_ref, f"source_video/{camera}/rgb/")
            reference_frame_tensor = decode_rgb_zip_to_tensor(zip_ref, f"reference_frame/{camera}/rgb/")[0]
        
        return {
            "scene_name": scene_name,
            "complexity": complexity,
            "camera": camera,
            "reference_video": video_tensor,
            "first_frame": reference_frame_tensor,
            "caption": caption
        }
