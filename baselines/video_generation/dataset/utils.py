"""Utility functions for the PhysInOne / PhysicsBenchmark dataset pipeline.

Helpers are grouped into three sections:
  1. Filesystem helpers    -- listing folders, checking paths, reading text files
  2. Video helpers         -- loading frame folders, FPS downsampling
  3. Tensor / misc helpers -- center-crop-and-resize, colored logging, string parsing
"""

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ======================================================================
# Filesystem helpers
# ======================================================================


def sub_files_path(base: str) -> List[Path]:
    """Recursively collect all file paths under `base`.

    Args:
        base: Root directory to scan.

    Returns:
        List of `Path` objects for every file under `base`
        (empty list if `base` does not exist or is not a directory).
    """
    base = Path(base)
    if not base.is_dir():
        print(f"Given path does not exist or is not a directory: {base}")
        return []
    return [file for file in base.rglob("*") if file.is_file()]


def check_folder_exists(folder_path: str) -> bool:
    """Return True if `folder_path` exists and is a directory."""
    return Path(folder_path).is_dir()


def list_subset_names(path: str) -> List[str]:
    """List the names of all entries directly under `path`.

    Args:
        path: Directory to list.

    Returns:
        List of entry names, or an empty list if the path is invalid or
        inaccessible (an error message is printed in that case).
    """
    if not os.path.isdir(path):
        print(f"Error: '{path}' does not exist or is not a directory.")
        return []
    try:
        return os.listdir(path)
    except PermissionError:
        print(f"Error: Permission denied to access '{path}'.")
    except Exception as e:
        print(f"Error listing '{path}': {e}")
    return []


def get_cinecamera_subfolders(root_path: str) -> List[str]:
    """Return names of all subfolders of `root_path` starting with "CineCamera_".

    This matches both numbered cameras (e.g. "CineCamera_3") and the moving
    camera ("CineCamera_Moving").

    Args:
        root_path: Directory to scan.

    Returns:
        List of matching subfolder names (empty list on any error).
    """
    root_dir = Path(root_path)
    if not root_dir.is_dir():
        print(f"Error: '{root_path}' does not exist or is not a directory.")
        return []

    try:
        return [
            entry.name
            for entry in root_dir.iterdir()
            if entry.is_dir() and entry.name.startswith("CineCamera_")
        ]
    except PermissionError:
        print(f"Error: Permission denied to access '{root_path}'.")
    except Exception as e:
        print(f"Unexpected error scanning '{root_path}': {e}")
    return []


def read_all_from_txt(file_path: str) -> str:
    """Read and return the entire content of a UTF-8 text file.

    Args:
        file_path: Path to the text file.

    Returns:
        File content as a single string.

    Raises:
        IOError: If the file cannot be read (including FileNotFoundError).
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except IOError as e:
        raise IOError(f"Error reading file {file_path}: {e}")


# ======================================================================
# Video helpers
# ======================================================================


def read_video(
    path: str,
    resolution: Tuple[int, int] = (1120, 1120),
    start_point: float = 0.0,
) -> Optional[np.ndarray]:
    """Load a video stored as a folder of image frames.

    Frames are read in sorted filename order, resized to `resolution`,
    and converted from BGR (OpenCV default) to RGB.

    Args:
        path: Directory containing frame images (.png / .jpg / .jpeg).
        resolution: Target (width, height) passed to `cv2.resize`.
        start_point: Fraction of the video to skip from the beginning,
            in [0, 1). E.g. 0.1 skips the first 10% of frames.

    Returns:
        Video as a (F, H, W, C) uint8 array, or None if no images are found.
    """
    images = sorted(
        img for img in os.listdir(path)
        if img.endswith((".png", ".jpg", ".jpeg"))
    )
    if not images:
        print(f"No images found in the specified directory: {path}")
        return None

    first_frame_idx = int(len(images) * start_point)

    frames = []
    for image in images[first_frame_idx:]:
        frame = cv2.imread(os.path.join(path, image))
        if frame is None:
            continue
        frame = cv2.resize(frame, resolution)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    return np.array(frames).astype(np.uint8)


def downsample_video(
    video: np.ndarray,
    original_fps: int = 30,
    target_fps: int = 30,
) -> np.ndarray:
    """Temporally downsample a video from `original_fps` to `target_fps`.

    Frames are selected at evenly spaced indices so the output covers the
    same time span as the input.

    Args:
        video: Input video with shape (F, H, W, C).
        original_fps: Frame rate of the input video.
        target_fps: Desired frame rate (0 < target_fps <= original_fps).

    Returns:
        Downsampled video with shape (F', H, W, C),
        where F' ~= F * target_fps / original_fps.

    Raises:
        ValueError: If `target_fps` is non-positive or exceeds `original_fps`.
    """
    if target_fps <= 0:
        raise ValueError("Target FPS must be a positive integer")
    if target_fps > original_fps:
        raise ValueError("Target FPS cannot exceed original FPS (use upsample for that)")

    total_frames = video.shape[0]
    target_total_frames = int(round(total_frames * (target_fps / original_fps)))

    # Evenly spaced frame indices spanning the whole video.
    indices = np.linspace(0, total_frames - 1, target_total_frames, dtype=int)
    return video[indices]


# ======================================================================
# Tensor / misc helpers
# ======================================================================


def center_crop_and_resize(
    img: torch.Tensor,
    target_height: int,
    target_width: int,
) -> torch.Tensor:
    """Center-crop to the target aspect ratio, then resize to the target size.

    Args:
        img: Image tensor of shape (C, H, W), or a batch of shape (B, C, H, W).
            A video tensor (C, F, H, W) also works, with F acting as the
            batch dimension.
        target_height: Output height.
        target_width: Output width.

    Returns:
        Tensor with the same leading dimensions and spatial size
        (target_height, target_width).
    """
    is_batch = img.dim() == 4
    if not is_batch:
        img = img.unsqueeze(0)  # (C, H, W) -> (1, C, H, W)

    _, _, h, w = img.shape
    target_ratio = target_width / target_height
    original_ratio = w / h

    if original_ratio > target_ratio:
        # Image is wider than target: crop width.
        new_width = int(h * target_ratio)
        start_w = (w - new_width) // 2
        cropped = img[..., :, start_w:start_w + new_width]
    else:
        # Image is taller than target: crop height.
        new_height = int(w / target_ratio)
        start_h = (h - new_height) // 2
        cropped = img[..., start_h:start_h + new_height, :]

    resized = F.interpolate(
        cropped,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    if not is_batch:
        resized = resized.squeeze(0)
    return resized


def print_yellow(text: str) -> None:
    """Print text in yellow (ANSI escape codes), typically used for warnings."""
    print(f"\033[33m{text}\033[0m")


def get_physics_label(s: str) -> list:
    """
    Extract physics labels from a scene name string.
    
    Format: {Physics_1}_{Physics2}_{Physics_3}__bg*__*_trajectory
    Physics labels are separated by '_' and end before '__bg'.
    
    Args:
        s: Scene name string
        
    Returns:
        List of physics label strings
    """
    # Split by '__' to isolate the physics part from the background/trajectory part
    parts = s.split('__')
    physics_part = parts[0]
    
    # Split the physics part by '_' to get individual labels
    physics_labels = physics_part.split('_')
    
    # Filter out any empty strings (in case of leading/trailing underscores)
    physics_labels = [label for label in physics_labels if label]
    
    return physics_labels
