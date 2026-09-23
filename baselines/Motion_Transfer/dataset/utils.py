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
import torch
import torchvision.transforms as transforms  # Adjust if you use a different library
from PIL import Image, UnidentifiedImageError
from pathlib import PurePosixPath


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
    Read all image frames below ``prefix`` from an opened ZIP file.

    Returns:
        torch.Tensor:
            Video tensor with shape [T, C, H, W] and values in [0, 1].

    Raises:
        RuntimeError:
            If a frame cannot be read or decoded.
        ValueError:
            If the prefix is missing, no images are found, or frame sizes
            are inconsistent.
    """
    if not prefix.endswith("/"):
        prefix += "/"

    zip_names = zip_ref.namelist()

    # Resolve the actual prefix, including ZIPs with an additional root folder.
    actual_prefix = None
    for name in zip_names:
        normalized_name = name.replace("\\", "/")
        index = normalized_name.find(prefix)

        if index != -1 and (
            index == 0 or normalized_name[index - 1] == "/"
        ):
            actual_prefix = normalized_name[: index + len(prefix)]
            break

    if actual_prefix is None:
        sample_contents = zip_names[:10]
        raise ValueError(
            "Could not find the expected directory in the ZIP.\n"
            f"Expected prefix: {prefix!r}\n"
            f"ZIP file: {getattr(zip_ref, 'filename', '<unknown>')!r}\n"
            f"Sample ZIP contents: {sample_contents}"
        )

    image_extensions = (".png", ".jpg", ".jpeg", ".bmp")
    image_names = [
        name
        for name in zip_names
        if name.startswith(actual_prefix)
        and not name.endswith("/")
        and PurePosixPath(name).suffix.lower() in image_extensions
    ]
    image_names.sort()

    if not image_names:
        raise ValueError(
            f"No image files found under prefix {actual_prefix!r} "
            f"in ZIP {getattr(zip_ref, 'filename', '<unknown>')!r}"
        )

    frames = []
    expected_size = None

    for frame_index, name in enumerate(image_names):
        try:
            # Read the compressed member bytes directly from the ZIP.
            image_bytes = zip_ref.read(name)

            # First pass: validate the encoded image stream.
            with Image.open(io.BytesIO(image_bytes)) as image:
                image_format = image.format
                if image_format not in {"JPEG", "PNG", "BMP"}:
                    raise ValueError(
                        f"unexpected image format {image_format!r}"
                    )
                image.verify()

            # Second pass: fully decode pixel data.
            # verify() alone does not necessarily decode every pixel.
            with Image.open(io.BytesIO(image_bytes)) as image:
                image = image.convert("RGB")
                image.load()
                current_size = image.size
                frame_tensor = transforms.ToTensor()(image)

        except (
            OSError,
            EOFError,
            RuntimeError,
            ValueError,
            UnidentifiedImageError,
        ) as error:
            zip_filename = getattr(zip_ref, "filename", "<unknown ZIP>")
            raise RuntimeError(
                "Failed to decode an image frame.\n"
                f"ZIP: {zip_filename}\n"
                f"Frame: {name}\n"
                f"Frame index: {frame_index}\n"
                f"Error: {error}"
            ) from error

        if expected_size is None:
            expected_size = current_size
        elif current_size != expected_size:
            raise ValueError(
                "Image size mismatch in one video sequence.\n"
                f"ZIP: {getattr(zip_ref, 'filename', '<unknown ZIP>')}\n"
                f"Frame: {name}\n"
                f"Current size: {current_size}\n"
                f"Expected size: {expected_size}"
            )

        frames.append(frame_tensor)

    return torch.stack(frames, dim=0)
