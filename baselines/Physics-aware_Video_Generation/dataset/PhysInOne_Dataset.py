"""PyTorch datasets for the PhysInOne / PhysicsBenchmark video-text corpus.

Classes:
    PhysBenchDataset                -- base dataset of (video, caption) pairs
    PhysBenchDataset_MotionTransfer -- (reference, target) video pairs for motion transfer
    PhysBenchDataset_Toy            -- subset filtered by physics category (toy experiments)

Expected on-disk layout (relative to `data_dir`):
    disk*/Rendered/Videos/<Split>/<Complexity>/<Scene>__bg*__<code>_trajectory/
        caption.txt
        CineCamera_<N>/rgb/*.png
        CineCamera_Moving/rgb/*.png
"""

import json
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from .utils import (
    center_crop_and_resize,
    check_folder_exists,
    downsample_video,
    get_cinecamera_subfolders,
    get_physics_label,
    list_subset_names,
    print_yellow,
    read_all_from_txt,
    read_video,
)

# ======================================================================
# Global constants
# ======================================================================

# Default spatial resolution for video frames (height, width).
RESOLUTION = (1120, 1120)

# Fraction of the video to skip from the beginning when loading.
START_POINT = 0.0

# Subdirectory (under each disk) containing rendered video data.
RENDER = "Rendered/Videos"

# Frame rates.
DEFAULT_FPS = 30   # Standard frame rate.
LASER_FPS = 60     # Higher frame rate for laser-related scenes.
LASER_LABELS = ["FixedPlanarRedirect", "FixedArrayRedirect", "FixedConcaveRedirect", "FixedConvexRedirect", "DynMirrorRedirect", "LaserBlock"]  
# Identifier in the scene name indicating laser physics.

# JSON file mapping each scene name to its manually selected cine camera.
CINE_CHOOSE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "selected_cinecamera.json",
)

DEFAULT_NEGATIVE_PROMPT = (
    '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，'
    '最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，'
    '画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，'
    '杂乱的背景，三条腿，背景人很多，倒着走'
)

# ======================================================================
# Main dataset class
# ======================================================================


class PhysInOne(torch.utils.data.Dataset):
    """Physics-based video-text pairs from the PhysicsBenchmark dataset.

    Supports multiple camera views ('main_camera' and cine cameras) and handles:
      - Video loading and preprocessing
      - Caption loading
      - Frame-rate adjustment
      - Padding / truncation to a fixed number of frames
      - Resolution normalization
    """

    def __init__(self, args, data_dir, split="train", only_main=False, only_one_cine=False):
        """
        Args:
            args: Configuration object (e.g., from argparse).
            data_dir: Root directory of the dataset.
            split: One of ['all', 'train', 'test', 'valid'].
            only_main: If True, only load 'main_camera' views (test split only).
            only_one_cine: If True, load a single pre-selected cine camera per
                scene (test split only, mutually exclusive with `only_main`).
        """
        split = split.lower()
        assert split in ["all", "train", "test", "valid"], \
            f"Invalid split {split} for PhysBenchDataset"

        if split in ["all", "train", "valid"] and (only_main or only_one_cine):
            raise ValueError(
                f"Cannot choose [only main] or [only one cine] for split {split}"
            )
        if only_main and only_one_cine:
            raise ValueError("Cannot choose [only main] and [only one cine] at the same time")
        with open(CINE_CHOOSE, "r") as f:
            cine_choose_sheet = json.load(f)

        self.data_dir = data_dir
        self.split = split
        self.args = args

        # Build the list of valid sample paths (relative to data_dir).
        # Example scene path:
        #   disk1/Rendered/Videos/Train/SinglePhysics/ElasticFall__bg801__AfZbCo_trajectory
        self.info = []
        split_dir = split.capitalize()
        for disk in list_subset_names(data_dir):  # e.g., ['disk1', 'disk2']
            if not disk.startswith("disk"):
                continue
            if disk not in ["disk35", "disk36"]:
                continue

            split_root = os.path.join(data_dir, disk, RENDER, split_dir)
            for complexity in list_subset_names(split_root):  # e.g., 'SinglePhysics'
                for scene in list_subset_names(os.path.join(split_root, complexity)):
                    if split == "test" and scene not in cine_choose_sheet:
                        continue

                    scene_root = os.path.join(split_root, complexity, scene)
                    rel_scene_root = os.path.join(disk, RENDER, split_dir, complexity, scene)

                    # Skip scenes without a caption file.
                    caption_path = os.path.join(scene_root, "caption.txt")
                    if not os.path.isfile(caption_path):
                        print(f"Not found {caption_path}")
                        continue

                    if only_main:
                        # Only the main (moving) camera.
                        if check_folder_exists(os.path.join(scene_root, "CineCamera_Moving")):
                            self.info.append(os.path.join(rel_scene_root, "main_camera"))
                    elif only_one_cine:
                        # Single pre-selected cine camera per scene.
                        chosen_cine = cine_choose_sheet[scene]
                        if check_folder_exists(os.path.join(scene_root, chosen_cine)):
                            self.info.append(os.path.join(rel_scene_root, chosen_cine))
                    else:
                        # All cine cameras (CineCamera_N and CineCamera_Moving).
                        for cam in get_cinecamera_subfolders(scene_root):
                            self.info.append(os.path.join(rel_scene_root, cam))

    def __len__(self):
        return len(self.info)

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def fetch_video(self, item):
        """Load and preprocess one video from disk.

        Args:
            item: Sample path relative to `data_dir` (camera folder).

        Returns:
            video: (F, H, W, C) uint8 numpy array.
            fps: Torch scalar tensor (actual FPS after downsampling).
            frame_mask: (F,) binary mask marking real (1) vs padded (0) frames.
        """
        # Load raw frames, skipping the initial ones.
        video = read_video(
            os.path.join(self.data_dir, item, "rgb"),
            resolution=self.resolution,
            start_point=START_POINT,
        )
        # Determine the source FPS based on the scene type (laser vs standard).
        physics_label = get_physics_label(item.split("/")[-2])
        
        if any([label in LASER_LABELS for label in physics_label]): 
            source_fps = LASER_FPS
        else:
            source_fps = DEFAULT_FPS

        if self.split == "train":
            # Downsample to the target FPS, then pad or truncate to `num_frame`.
            video, fps = self.downsample_video(video, source_fps)
            video = video[:self.num_frame]

            num_real_frames = video.shape[0]
            if num_real_frames < self.num_frame:
                # Pad with the last frame if the video is too short.
                last_frame = video[-1]
                padding = np.tile(last_frame, (self.num_frame - num_real_frames, 1, 1, 1))
                video = np.concatenate((video, padding), axis=0)

            # Frame mask: 1 = real frame, 0 = padded frame.
            frame_mask = torch.zeros((self.num_frame,))
            frame_mask[:num_real_frames] = 1.0
        else:
            fps = source_fps
            frame_mask = torch.zeros((video.shape[0],))

        return video, torch.tensor(fps), frame_mask

    def __getitem__(self, idx):
        """Return a single data sample.

        On failure, fall back to the next index (with modulo wrap-around).
        """
        idx = idx % len(self)
        item = self.info[idx]
        try:
            video, fps, frame_mask = self.fetch_video(item)
            main_frame = video[0]  # First frame after preprocessing.

            # Load the text caption from the scene directory (parent of camera folder).
            text = read_all_from_txt(os.path.join(self.data_dir, item, "../caption.txt"))

            # Normalize video to [0, 1], reshape to (C, F, H, W), and crop/resize.
            video = torch.from_numpy(video) / 255.0
            video = rearrange(video, "f h w c -> c f h w")
            video = center_crop_and_resize(video, self.resolution[0], self.resolution[1])

            # Process the main frame separately (for image-conditioned tasks).
            main_frame = torch.from_numpy(main_frame) / 255.0
            main_frame = rearrange(main_frame, "h w c -> c h w")
            img_res = self.image_resolution
            main_frame = center_crop_and_resize(main_frame, img_res[0], img_res[1])

            # Reshape the frame mask for broadcasting: (F,) -> (1, F, 1, 1).
            frame_mask = frame_mask.reshape(1, frame_mask.size(0), 1, 1)

        except Exception as e:
            # Log the error and retry with the next index (simple fault tolerance).
            print_yellow(f"Fail to fetch {idx}:{item} of the dataset")
            print_yellow(str(e))
            return self.__getitem__(idx + 1)

        return {
            "image": main_frame,                      # (C, H, W) conditioning frame
            "video": video,                           # (C, F, H, W) full video
            "prompt": text,                           # Text caption
            "negative_prompt": self.negative_prompt,  # Negative prompt
            "frame_mask": frame_mask,                 # (1, F, 1, 1) real vs padded
            "fps": fps,                               # Actual FPS after downsampling
            "name": item,                             # Relative path for debugging
        }

    # ------------------------------------------------------------------
    # Configuration properties (read from `args` with sensible defaults)
    # ------------------------------------------------------------------

    @property
    def image_resolution(self):
        """Resolution for the main image (may differ from video resolution)."""
        resolution = getattr(self.args, "image_resolution", None)
        return resolution if resolution is not None else self.resolution

    @property
    def resolution(self):
        """Target (height, width) for video frames."""
        height = getattr(self.args, "height", None)
        width = getattr(self.args, "width", None)
        if height is not None and width is not None:
            return (height, width)
        return RESOLUTION

    @property
    def fps(self):
        """Target FPS ('auto' triggers random FPS selection during training)."""
        return getattr(self.args, "fps", DEFAULT_FPS)

    @property
    def num_frame(self):
        """Number of frames to return per video (-1 means no truncation)."""
        nf = getattr(self.args, "num_frame", -1)
        return nf if isinstance(nf, int) and nf > 0 else -1
    
    @property
    def negative_prompt(self):
        """Negative prompt for diffusion models (avoids undesirable artifacts)."""
        return getattr(self.args, "negative_prompt", DEFAULT_NEGATIVE_PROMPT)

    def downsample_video(self, video, original_fps):
        """Downsample a video to the target FPS.

        If `self.fps` is 'auto', randomly choose an FPS in the range that
        still guarantees at least `num_frame` frames.

        Returns:
            (downsampled_video, fps): The downsampled array and the FPS used.
        """
        if self.fps == "auto":
            min_fps = math.ceil(self.num_frame / video.shape[0] * original_fps)
            max_fps = original_fps
            fps = max_fps if min_fps > max_fps else random.randint(min_fps, max_fps)
        else:
            fps = self.fps if self.fps is not None else DEFAULT_FPS
        return downsample_video(video, original_fps, fps), fps

