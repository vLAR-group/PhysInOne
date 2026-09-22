# -*- coding: utf-8 -*-
"""Small image-sequence helpers for PhysInOne motion-transfer inference."""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

import zipfile
import io
from PIL import Image
import torch
import torchvision.transforms as transforms  # Adjust if you use a different library


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normpath(path: str) -> str:
    return os.path.normpath(os.path.abspath(path))


def natural_key(text: str) -> List[Any]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def load_mapping(mapping_path: str) -> List[Dict[str, Any]]:
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected mapping JSON to be a list, got {type(data).__name__}: {mapping_path}")
    return data


def list_images(rgb_dir: str) -> List[str]:
    if not os.path.isdir(rgb_dir):
        return []
    files = [
        os.path.join(rgb_dir, name)
        for name in os.listdir(rgb_dir)
        if os.path.splitext(name)[1].lower() in IMG_EXTS
    ]
    files.sort(key=natural_key)
    return files


def _resize_if_needed(img: np.ndarray, size_hw: Optional[Tuple[int, int]]) -> np.ndarray:
    if size_hw is None:
        return img
    height, width = size_hw
    if img.shape[0] == height and img.shape[1] == width:
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _read_rgb(path: str, resize_hw: Optional[Tuple[int, int]]) -> torch.Tensor:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = _resize_if_needed(img, resize_hw)
    arr = img.astype(np.float32) / 255.0
    return torch.from_numpy(arr).contiguous()


def _uniform_indices(n: int, k: int) -> np.ndarray:
    if k <= 0 or k >= n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, k).round().astype(int)


def sample_image_paths(
    rgb_dir: str,
    *,
    frame_stride: int = 1,
    num_frames: Optional[int] = None,
    sample_mode: str = "uniform",
) -> Tuple[List[str], Dict[str, Any]]:
    files_all = list_images(rgb_dir)
    if not files_all:
        raise RuntimeError(f"No RGB frames found in: {rgb_dir}")

    step = max(1, int(frame_stride))
    files = files_all[::step]
    mode = sample_mode.lower()

    if mode == "all" or not (num_frames and num_frames > 0):
        indices = np.arange(len(files), dtype=int)
    elif mode == "uniform":
        indices = _uniform_indices(len(files), int(num_frames))
    elif mode == "head":
        indices = np.arange(min(int(num_frames), len(files)), dtype=int)
    elif mode == "tail":
        start = max(0, len(files) - int(num_frames))
        indices = np.arange(start, len(files), dtype=int)
    else:
        raise ValueError(f"Unknown sample_mode: {sample_mode}")

    picked = [files[i] for i in indices.tolist()]
    info = {
        "total_files": len(files_all),
        "after_stride": len(files),
        "picked": len(picked),
        "indices": indices.tolist(),
        "first_path": picked[0],
        "last_path": picked[-1],
    }
    return picked, info


def decode_rgb_dir_to_tensor(
    rgb_dir: str,
    *,
    resize_hw: Optional[Tuple[int, int]] = None,
    frame_stride: int = 1,
    num_frames: Optional[int] = None,
    sample_mode: str = "uniform",
    return_info: bool = False,
) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, Any]]:
    paths, info = sample_image_paths(
        rgb_dir,
        frame_stride=frame_stride,
        num_frames=num_frames,
        sample_mode=sample_mode,
    )
    frames = [_read_rgb(path, resize_hw) for path in paths]
    video = torch.stack(frames, dim=0).to(torch.float32).contiguous()
    return (video, info) if return_info else video


def read_first_rgb_frame_to_tensor(
    rgb_dir: str,
    *,
    resize_hw: Optional[Tuple[int, int]] = None,
) -> Tuple[torch.Tensor, str]:
    files = list_images(rgb_dir)
    if not files:
        raise RuntimeError(f"No RGB frames found in: {rgb_dir}")
    return _read_rgb(files[0], resize_hw), files[0]


def read_caption_txt(sequence_dir: str) -> str:
    path = os.path.join(sequence_dir, "caption.txt")
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read().strip()

def decode_rgb_zip_to_tensor(zip_ref, prefix):
    """
    Reads all images under a specific prefix in a zip file and returns a stacked tensor.
    Handles cases where the zip file contains an extra root folder.
    """
    # Ensure prefix ends with a slash for accurate directory matching
    if not prefix.endswith('/'):
        prefix += '/'

    # 1. Dynamically find the actual prefix in the zip file
    # This handles cases where the zip has an extra root folder (e.g., 'root_folder/source_video/...')
    actual_prefix = None
    for name in zip_ref.namelist():
        idx = name.find(prefix)
        # Check if the prefix is found and it's a complete directory match 
        # (either at the start of the string or preceded by a '/')
        if idx != -1 and (idx == 0 or name[idx - 1] == '/'):
            actual_prefix = name[:idx + len(prefix)]
            break
            
    if actual_prefix is None:
        # Provide a helpful error message with sample contents for debugging
        sample_contents = zip_ref.namelist()[:5]
        raise ValueError(
            f"Could not find the expected directory structure in the zip file.\n"
            f"Expected prefix: '{prefix}'\n"
            f"Sample zip contents: {sample_contents}"
        )

    # 2. Find all image files under the resolved actual prefix
    image_names = [
        name for name in zip_ref.namelist()
        if name.startswith(actual_prefix) 
        and not name.endswith('/') 
        and name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ]
    
    # 3. Sort to ensure correct temporal order (e.g., frame_001.png, frame_002.png)
    image_names.sort()
    
    if not image_names:
        raise ValueError(f"No image files found under the resolved prefix: '{actual_prefix}'")

    frames = []
    for name in image_names:
        # Read file bytes directly into memory
        with zip_ref.open(name) as f:
            img_bytes = f.read()
        
        # Decode bytes to image
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Convert to tensor (Replace this with your original transform logic if needed)
        # e.g., img_tensor = your_custom_transform(img)
        img_tensor = transforms.ToTensor()(img)
        
        frames.append(img_tensor)
        
    # Stack frames into a single tensor (e.g., shape: [T, C, H, W])
    video_tensor = torch.stack(frames)
    return video_tensor