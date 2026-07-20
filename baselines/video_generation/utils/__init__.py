"""Shared utilities for train.py and inference.py.

This module centralizes the logic duplicated between the training and
inference entry points:

  - Warning filters and common environment setup (`setup_environment`)
  - Diffusers pipeline loading for all supported model families (`load_pipeline`)
  - Model freezing and trainable-parameter configuration
    (`freeze_models`, `apply_training_mode`)
  - Accelerator construction and logging verbosity (`build_accelerator`,
    `configure_logging_verbosity`)
  - Mixed-precision dtype resolution (`get_weight_dtype`)
  - Checkpoint discovery (`find_latest_checkpoint`)
  - Model summaries and misc helpers (`print_model_info`, `get_video_time`,
    `create_log_video`)
"""

import logging
import os
import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
    ProjectConfiguration,
)
import math

logger = get_logger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# Environment / logging setup
# ======================================================================


def setup_environment():
    """Set common environment variables and silence known noisy warnings."""
    os.environ.setdefault("HF_HOME", "./hf_cache")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Silence a tokenizers warning.

    warnings.filterwarnings(
        "once",
        category=DeprecationWarning,
        message="__array__ implementation doesn't accept a copy keyword, "
                "so passing copy=False failed.",
    )
    warnings.filterwarnings(
        "ignore", category=UserWarning, message="TypedStorage is deprecated"
    )
    warnings.filterwarnings(
        "ignore", message="cc_projection/diffusion_pytorch_model.safetensors not found"
    )
    warnings.filterwarnings(
        "ignore",
        message="The config attributes {'cc_projection': ['pipeline_zero1to3', "
                "'CCProjection']} were passed to Neural_Gaffer_StableDiffusionPipeline, "
                "but are not expected and will be ignored. Please verify your "
                "model_index.json configuration file.",
    )
    warnings.simplefilter(action="ignore", category=FutureWarning)


def build_accelerator(project_dir, logging_dir, nccl_timeout,
                      mixed_precision, log_with=None,
                      gradient_accumulation_steps=1):
    """Create an `Accelerator` with the DDP/NCCL settings shared by both scripts."""
    project_config = ProjectConfiguration(project_dir=project_dir, logging_dir=logging_dir)
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(
        backend="nccl", timeout=timedelta(seconds=nccl_timeout)
    )
    return Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=log_with,
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs, init_kwargs],
    )


def configure_logging_verbosity(accelerator):
    """Configure basic logging and set transformers/diffusers verbosity per process."""
    import diffusers
    import transformers

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
    else:
        transformers.utils.logging.set_verbosity_error()


def get_weight_dtype(accelerator):
    """Resolve the weight dtype from the accelerator's mixed-precision setting."""
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


# ======================================================================
# Pipeline loading
# ======================================================================


def load_pipeline(model_key, device=DEVICE):
    """Load a diffusers image-to-video pipeline and expose its main components.

    Args:
        model_key: HuggingFace model identifier. Supported families:
            'stabilityai/stable-video-diffusion*', 'THUDM/CogVideoX1.5*',
            'Wan-AI/Wan*', and 'nvidia/Cosmos*'.
        device: Device to move the pipeline to.

    Returns:
        Dict with keys: 'pipe', 'vae', 'denoiser', 'text_encoder',
        'image_encoder', 'scheduler'. Components not used by a given
        model family are None.
    """
    vae = denoiser = text_encoder = image_encoder = scheduler = None

    if model_key.startswith("stabilityai/stable-video-diffusion"):
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=torch.float32, variant="fp16"
        )
        pipe.to(device)
        vae = pipe.vae
        scheduler = pipe.scheduler
        denoiser = pipe.unet
        image_encoder = pipe.image_encoder

    elif model_key.startswith("THUDM/CogVideoX1.5"):
        from diffusers import CogVideoXDPMScheduler, CogVideoXImageToVideoPipeline
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_key, torch_dtype=torch.float32
        )
        pipe.to(device)
        vae = pipe.vae
        pipe.scheduler = CogVideoXDPMScheduler.from_pretrained(
            model_key, subfolder="scheduler"
        )
        scheduler = pipe.scheduler
        denoiser = pipe.transformer
        text_encoder = pipe.text_encoder
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    elif model_key.startswith("Wan-AI/Wan"):
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        vae = AutoencoderKLWan.from_pretrained(
            model_key, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_key, vae=vae, torch_dtype=torch.float32
        )
        pipe.to(device)
        vae = pipe.vae
        scheduler = pipe.scheduler
        denoiser = pipe.transformer
        text_encoder = pipe.text_encoder
        image_encoder = getattr(pipe, "image_encoder", None)

    elif model_key.startswith("nvidia/Cosmos"):
        from diffusers import Cosmos2_5_PredictBasePipeline
        pipe = Cosmos2_5_PredictBasePipeline.from_pretrained(
            model_key, revision="diffusers/base/post-trained", torch_dtype=torch.bfloat16
        )
        pipe.to(device)
        scheduler = pipe.scheduler
        denoiser = pipe.transformer
        text_encoder = pipe.text_encoder

    else:
        raise NotImplementedError(f"Unsupported model_key: {model_key}")

    pipe.set_progress_bar_config(disable=True)
    return {
        "pipe": pipe,
        "vae": vae,
        "denoiser": denoiser,
        "text_encoder": text_encoder,
        "image_encoder": image_encoder,
        "scheduler": scheduler,
    }


def build_validation_pipeline(model_key, accelerator, vae, denoiser,
                              image_encoder, text_encoder, weight_dtype):
    """Rebuild a pipeline for validation using the (possibly wrapped) trained modules.

    Used by `log_validation` in train.py to run inference with the current
    training weights.
    """
    if model_key.startswith("stabilityai/stable-video-diffusion"):
        from diffusers import StableVideoDiffusionPipeline
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=torch.float32
        )
        pipe.unet = accelerator.unwrap_model(denoiser).eval()
        pipe.image_encoder = accelerator.unwrap_model(image_encoder).eval()
        pipe.vae = accelerator.unwrap_model(vae).eval()

    elif model_key.startswith("THUDM/CogVideoX1.5"):
        from diffusers import CogVideoXImageToVideoPipeline
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(
            model_key, torch_dtype=torch.float32
        )
        pipe.transformer = accelerator.unwrap_model(denoiser).eval()
        pipe.text_encoder = accelerator.unwrap_model(text_encoder).eval()
        pipe.vae = accelerator.unwrap_model(vae).eval()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

    elif model_key.startswith("Wan-AI/Wan"):
        from diffusers import AutoencoderKLWan, WanImageToVideoPipeline
        fresh_vae = AutoencoderKLWan.from_pretrained(
            model_key, subfolder="vae", torch_dtype=torch.float32
        )
        pipe = WanImageToVideoPipeline.from_pretrained(
            model_key, torch_dtype=torch.float32
        )
        pipe.transformer = accelerator.unwrap_model(denoiser).eval()
        pipe.text_encoder = accelerator.unwrap_model(text_encoder).eval()
        pipe.vae = accelerator.unwrap_model(fresh_vae).eval()

    else:
        raise NotImplementedError(f"Unsupported model_key: {model_key}")

    pipe.set_progress_bar_config(disable=True)
    pipe.to(accelerator.device, dtype=weight_dtype)
    pipe.vae.to(accelerator.device, dtype=weight_dtype)
    return pipe


# ======================================================================
# Model freezing / training-mode configuration
# ======================================================================


def freeze_models(models):
    """Set every non-None model to eval mode and disable gradients."""
    for model in models:
        if model is not None:
            model.eval()
            model.requires_grad_(False)


def _unfreeze_final_layers(denoiser, num_to_tune):
    """Unfreeze the last `num_to_tune` blocks of a denoiser (FLT mode).

    Supports Diffusers UNet architectures (down/mid/up blocks) and
    transformer architectures (`blocks` or `transformer_blocks`).
    """
    # Case 1: UNet-style model (down_blocks, mid_block, up_blocks).
    if all(hasattr(denoiser, a) for a in ("up_blocks", "mid_block", "down_blocks")):
        logger.info("Detected Diffusers UNet architecture", main_process_only=True)
        total_layers = len(denoiser.down_blocks) + 1 + len(denoiser.up_blocks)
        start_idx = max(0, total_layers - num_to_tune)
        for i in range(start_idx, total_layers):
            if i < len(denoiser.down_blocks):
                logger.info(f"Unfreezing down_blocks[{i}]", main_process_only=True)
                block = denoiser.down_blocks[i]
            elif i == len(denoiser.down_blocks):
                logger.info("Unfreezing mid_block", main_process_only=True)
                block = denoiser.mid_block
            else:
                up_idx = i - len(denoiser.down_blocks) - 1
                logger.info(f"Unfreezing up_blocks[{up_idx}]", main_process_only=True)
                block = denoiser.up_blocks[up_idx]
            for param in block.parameters():
                param.requires_grad = True

    # Case 2: generic transformer with `blocks`.
    elif hasattr(denoiser, "blocks"):
        logger.info("Detected encoder-based Transformer", main_process_only=True)
        blocks = denoiser.blocks
        for i in range(max(0, len(blocks) - num_to_tune), len(blocks)):
            logger.info(f"Unfreezing blocks[{i}]", main_process_only=True)
            for param in blocks[i].parameters():
                param.requires_grad = True

    # Case 3: CogVideoX-style transformer with `transformer_blocks`.
    elif hasattr(denoiser, "transformer_blocks"):
        logger.info("Detected CogVideoX Transformer", main_process_only=True)
        blocks = denoiser.transformer_blocks
        for i in range(max(0, len(blocks) - num_to_tune), len(blocks)):
            logger.info(f"Unfreezing transformer_blocks[{i}]", main_process_only=True)
            for param in blocks[i].parameters():
                param.requires_grad = True

    else:
        raise NotImplementedError(f"FLT not implemented for {type(denoiser)}")


def apply_training_mode(denoiser, training_mode, cfg):
    """Configure the denoiser for the given training mode.

    Args:
        denoiser: The denoising model (UNet or transformer).
        training_mode: One of 'sft', 'lora', or 'flt' (case-insensitive).
        cfg: Config namespace providing LoRA/FLT hyperparameters
            (rank, alpha, init_lora_weights, target_module, num_to_tune).

    Returns:
        The (possibly wrapped) denoiser. For LoRA the model is wrapped in a
        `LoraModel`; for SFT/FLT the model is modified in place.
    """
    from peft import LoraConfig, LoraModel

    mode = training_mode.lower()
    if mode == "sft":
        logger.info("Training mode: SFT", main_process_only=True)
        denoiser.train()
        denoiser.requires_grad_(True)
    elif mode == "lora":
        logger.info("Training mode: LoRA", main_process_only=True)
        lora_config = LoraConfig(
            r=cfg.rank,
            lora_alpha=cfg.alpha,
            init_lora_weights=cfg.init_lora_weights,
            target_modules=cfg.target_module,
        )
        denoiser = LoraModel(denoiser, lora_config, "default")
    elif mode == "flt":
        logger.info("Training mode: FLT (final-layer tuning)", main_process_only=True)
        _unfreeze_final_layers(denoiser, getattr(cfg, "num_to_tune", 2))
    else:
        raise NotImplementedError("Only SFT, LoRA and FLT are implemented")
    return denoiser


# ======================================================================
# Checkpoints
# ======================================================================


def find_latest_checkpoint(output_dir):
    """Return the name of the latest 'checkpoint-*' folder, or None if absent."""
    checkpoints = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
    return checkpoints[-1] if checkpoints else None


# ======================================================================
# Misc helpers
# ======================================================================


def print_model_info(model, accelerator):
    """Print a parameter/size summary of a model (main process only)."""
    if not accelerator.is_main_process:
        return
    params = list(model.parameters())
    learnable = sum(p.numel() for p in params if p.requires_grad)
    frozen = sum(p.numel() for p in params if not p.requires_grad)
    total = sum(p.numel() for p in params)
    size_mb = sum(p.numel() * p.element_size() for p in params) / 1024 / 1024
    print("=" * 20)
    print("model name: ", type(model).__name__)
    print("learnable parameters(M): ", learnable / 1e6)
    print("non-learnable parameters(M): ", frozen / 1e6)
    print("total parameters(M): ", total / 1e6)
    print("model size(MB): ", size_mb)


def get_video_time(video, fps):
    """Return the duration (seconds) of a (..., F, H, W) video tensor.

    Returns 0 if `video` is None (useful for autoregressive generation loops).
    """
    if video is None:
        return 0
    num_frames = video.size(-3)
    return float(num_frames / fps)


def create_log_video(image, output, video, channel_first=True):
    """Combine conditioning images, generated videos, and ground truth side by side.

    Args:
        image: Batch of conditioning images, shape (B, C, H, W).
        output: Batch of generated videos, shape (B, C, F, H, W).
        video: Batch of ground-truth videos, shape (B, C, F, H, W).
        channel_first: If False, return frames as (H, W, C) instead of (C, H, W).

    Returns:
        uint8 numpy array of shape (F, C, B*H, 3*W) — for each frame, the
        conditioning image, generated video, and ground truth are stacked
        horizontally, with batch items stacked vertically.
    """
    output = output.transpose(1, 2)  # (B, F, C, H, W)
    video = video.transpose(1, 2)    # (B, F, C, H, W)

    B, frame_size, C, H, W = output.shape

    # Resize static images to the video frame size and repeat over frames.
    image = torch.nn.functional.interpolate(image, size=output.shape[3:])
    image_expanded = image.unsqueeze(1).repeat(1, frame_size, 1, 1, 1)
    assert image_expanded.shape == (B, frame_size, C, H, W), "Image expansion failed"
    assert video.shape == (B, frame_size, C, H, W), "Ground truth video shape mismatch"

    # Stack [image | output | ground truth] horizontally, then merge the batch
    # dimension into height: (B, F, C, H, 3W) -> (F, C, B*H, 3W).
    combined = torch.cat([image_expanded, output, video], dim=4)
    combined = combined.permute(1, 2, 0, 3, 4).reshape(frame_size, C, -1, 3 * W)
    combined = torch.clamp(combined * 255.0, 0.0, 255.0).to(torch.uint8).cpu().numpy()

    if not channel_first:
        combined = np.transpose(combined, (0, 2, 3, 1))
    return combined

def frame_interpolate(
    video: torch.Tensor,
    source_fps: int,
    target_fps: int,
    method: str = "linear",  # "linear" or "rife"
    device: torch.device = "cuda"
) -> torch.Tensor:
    """
    Unified frame interpolation function supporting both fast linear interpolation and high-quality RIFE.

    Args:
        video: Tensor of shape (B, C, F, H, W), dtype float32, values in [0, 1]
        source_fps: original frame rate (e.g., 8)
        target_fps: desired frame rate (e.g., 24)
        method: "linear" (fast) or "rife" (high-quality, slow)
        device: device for RIFE inference

    Returns:
        Interpolated video tensor of shape (B, C, F_out, H, W)
    """
    if source_fps == target_fps:
        return video

    B, C, F_in, H, W = video.shape
    F_target = math.ceil(F_in * target_fps / source_fps)

    if F_target == F_in:
        return video

    if method == "linear":
        # Reshape to (B*C*H*W, F_in) — flatten all non-temporal dims
        video_flat = video.permute(0, 2, 1, 3, 4)  # (B, F, C, H, W)
        video_flat = video_flat.reshape(B * F_in, C * H * W)  # (B*F, C*H*W)
        video_flat = video_flat.t()  # (C*H*W*B, F_in) → but we want (N, F)
        # Better: go to (N, F) where N = B*C*H*W
        video_flat = video.permute(0, 1, 3, 4, 2)  # (B, C, H, W, F_in)
        N = B * C * H * W
        video_flat = video_flat.reshape(N, F_in)  # (N, F_in)

        # Add channel dim for interpolate: (N, 1, F_in)
        video_3d = video_flat.unsqueeze(1)  # (N, 1, F_in)

        # Interpolate along time (last dim)
        video_interp_3d = torch.nn.functional.interpolate(
            video_3d,
            size=F_target,
            mode='linear',
            align_corners=False
        )  # (N, 1, F_target)

        # Remove channel dim and reshape back
        video_interp_flat = video_interp_3d.squeeze(1)  # (N, F_target)
        video_interp = video_interp_flat.reshape(B, C, H, W, F_target)
        video_interp = video_interp.permute(0, 1, 4, 2, 3)  # (B, C, F_target, H, W)

        return video_interp

    elif method == "rife":
        if C != 3:
            raise ValueError("RIFE only supports 3-channel RGB video (C=3).")
        if not (video.dtype == torch.float32 and video.min() >= 0.0 and video.max() <= 1.0):
            raise ValueError("RIFE expects video in [0, 1] float32.")

        # RIFE only inserts frames → cannot downsample
        if target_fps <= source_fps:
            # Fallback to linear for downsampling
            return frame_interpolate(video, source_fps, target_fps, method="linear", device=device)

        model = get_rife_model(device)

        # How many total frames we need
        ratio = target_fps / source_fps
        F_out_exact = F_in * ratio

        # We'll iteratively interpolate until we reach or exceed F_target
        current_video = video  # (B, 3, F_curr, H, W)
        current_fps = source_fps

        # RIFE doubles frame count each full pass (inserts 1 frame between each pair)
        # To reach arbitrary ratio, we use multi-stage interpolation
        while current_fps < target_fps:
            B, C, F_curr, H, W = current_video.shape
            new_list = []
            for b in range(B):
                frames = current_video[b]  # (3, F, H, W)
                interp_frames = []
                for i in range(F_curr - 1):
                    I0 = frames[:, i, :, :]     # (3, H, W)
                    I1 = frames[:, i + 1, :, :] # (3, H, W)
                    mid = model.inference(I0, I1)  # (3, H, W)
                    interp_frames.append(I0)
                    interp_frames.append(mid)
                interp_frames.append(frames[:, -1, :, :])  # last frame
                interp_frames = torch.stack(interp_frames, dim=1)  # (3, 2*F-1, H, W)
                new_list.append(interp_frames)
            current_video = torch.stack(new_list, dim=0)  # (B, 3, 2*F-1, H, W)
            current_fps *= 2

        # Now we may have overshot (e.g., 8 → 15 → 29 for target 24)
        # So we finally resample to exact F_target using linear interpolation
        B, C, F_curr, H, W = current_video.shape
        if F_curr != F_target:
            current_video = current_video.view(B * C, F_curr, H, W)
            current_video = torch.nn.functional.interpolate(
                current_video,
                size=(F_target, H),
                mode="linear",
                align_corners=False
            )
            current_video = current_video.view(B, C, F_target, H, W)

        return current_video

    else:
        raise ValueError(f"Unsupported interpolation method: {method}. Choose 'linear' or 'rife'.")
    

def center_crop_and_resize(img, target_height, target_width):
    """
    First crop the image from the center to match the target aspect ratio, 
    then resize to the target resolution.
    
    Parameters:
        img: Input tensor with shape (C, H, W) for single image or (B, C, H, W) for batch
        target_height: Desired output height
        target_width: Desired output width
    
    Returns:
        Processed tensor with shape (C, target_height, target_width) or 
        (B, C, target_height, target_width)
    """
    # Check input dimensions
    is_batch = img.dim() == 4
    if not is_batch:
        # Add batch dimension if single image
        img = img.unsqueeze(0)  # Becomes (1, C, H, W)
    
    _, _, h, w = img.shape
    
    # Calculate target aspect ratio
    target_ratio = target_width / target_height
    # Calculate original image aspect ratio
    original_ratio = w / h
    
    # Determine cropping strategy based on aspect ratios
    if original_ratio > target_ratio:
        # Original image is wider, crop width to match target ratio
        new_width = int(h * target_ratio)
        new_height = h
        # Calculate crop start position for width
        start_w = (w - new_width) // 2
        end_w = start_w + new_width
        # Perform cropping
        cropped = img[..., :, start_w:end_w]
    else:
        # Original image is taller, crop height to match target ratio
        new_height = int(w / target_ratio)
        new_width = w
        # Calculate crop start position for height
        start_h = (h - new_height) // 2
        end_h = start_h + new_height
        # Perform cropping
        cropped = img[..., start_h:end_h, :]
    
    # Resize cropped region to target resolution
    resized = torch.nn.functional.interpolate(
        cropped, 
        size=(target_height, target_width), 
        mode='bilinear',  # Bilinear interpolation, suitable for images
        align_corners=False
    )
    
    # Remove batch dimension if input was single image
    if not is_batch:
        resized = resized.squeeze(0)
    
    return resized