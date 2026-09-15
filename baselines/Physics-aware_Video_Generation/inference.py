"""Inference entry point for PhysInOne video diffusion models.

Loads a (optionally fine-tuned) image-to-video pipeline and generates videos
for the test split, autoregressively extending each clip until it matches the
ground-truth duration. Results (output.mp4 + caption.txt) are written per
sample under the output directory.

Usage:
    # Vanilla validation (no checkpoint, config.yaml specified directly):
    python inference.py --config ./config.yaml --data_path /path/to/dataset \
        [--output_path ./output/] [--only_main] [--skip_exist]

    # Fine-tuned validation (config.yaml is read from the model directory):
    python inference.py --from_pretrained ./models/lora/lora_Wan-AI/... \
        --data_path /path/to/dataset \
        [--checkpoint latest] [--output_path ./output/] [--only_main] [--skip_exist]

    # Vanilla testing on leaderboard (no checkpoint, config.yaml specified directly):
        python inference.py --config ./config.yaml --data_path /path/to/leaderboard_dataset \
            --leaderboard True --leaderboard_branch cine (or moving) [--output_path ./output/] [--skip_exist]
"""

import argparse
import os
os.environ['HF_HOME'] = './hf_cache'
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
import torchvision
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from utils.args import read_yaml_to_namespce
from utils import (
    apply_training_mode,
    build_accelerator,
    configure_logging_verbosity,
    find_latest_checkpoint,
    freeze_models,
    get_video_time,
    get_weight_dtype,
    load_pipeline,
    print_model_info,
    setup_environment,
    center_crop_and_resize,
    frame_interpolate
)

setup_environment()

NCCL_TIMEOUT = 36000

logger = get_logger(__name__)


@torch.no_grad()
def log_validation(validation_dataloader, pipe, args, accelerator, weight_dtype,
                   split="val", cur_step=0):
    """Generate and save one video per test sample.

    For each sample, the pipeline is called autoregressively (feeding the last
    generated frame back as the conditioning image) until the generated clip
    is at least as long as the ground truth, then truncated and frame-rate
    interpolated to match.
    """
    cfg = args.config
    logger.info(f"Running {split} validation... ")

    if cfg.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(cfg.seed)

    bar = tqdm(enumerate(validation_dataloader), desc=split, total=len(validation_dataloader))
    for valid_step, batch in bar:
        # assert len(batch["image"]) == 1, "This log_validation only supports batch_size=1"

        if "video" in batch:
            video = batch["video"].to(accelerator.device, dtype=weight_dtype)  # (1, C, F, H, W)
        else:
            video = None
        image = batch["image"].to(accelerator.device, dtype=weight_dtype)  # (1, C, H, W)
        prompt = batch["prompt"][0]
        fps = batch["fps"]  # scalar tensor or int
        name = batch["name"][0]
        camera_angle_x = batch["camera_angle_x"][0] if "camera_angle_x" in batch else None
        total_frames = batch["total_frames"][0] if "total_frames" in batch else None
        transforms = batch["transforms"][0] if "transforms" in batch else None

        sample_output_dir = os.path.join(args.output_path, name)
        if args.skip_exist and os.path.exists(sample_output_dir):
            print(f"Skip {sample_output_dir} that is already existing")
            continue

        # Target duration (seconds) from the ground truth.

        if video is not None:
            target_duration = get_video_time(video, fps)
        else: # video is None, e.g. for leaderboard submission, we provide the ground-truth duration
            if total_frames is not None and fps is not None:
                target_duration = total_frames.item() / fps.item()
            else:
                raise ValueError(f"Cannot determine target duration for {name}: "
                                    "video, total_frames, and/or fps are missing")

        # ---- Autoregressive generation until the clip is long enough ----
        output = None
        current_image = image.squeeze(0)  # (C, H, W)
        image_resolution = image.size()[-2:]
        while get_video_time(output, cfg.fps) < target_duration:
            _output = pipe(
                image=current_image.unsqueeze(0),  # (1, C, H, W)
                prompt=prompt,
                fps=cfg.fps,
                output_type="pt",
                generator=generator,
                height=cfg.height,
                width=cfg.width,
            ).frames  # (1, F_new, C, H, W)
            _output = _output.permute(0, 2, 1, 3, 4).squeeze(0)  # (C, F_new, H, W)

            if output is None:
                output = _output
            else:
                # Skip the first frame to avoid duplicating the seam frame.
                output = torch.cat([output, _output[:, 1:, :, :]], dim=1)

            # Condition the next chunk on the last generated frame.
            current_image = center_crop_and_resize(
                output[:, -1, :, :], image_resolution[0], image_resolution[1]
            )

        # ---- Match ground-truth length and frame rate ----
        num_gt_frames = total_frames.item() if total_frames is not None else video.size(2) 
        output = output[:, :num_gt_frames, :, :]  # (C, F_gt, H, W)

        fps_val = fps.item() if torch.is_tensor(fps) else fps
        output = frame_interpolate(
            output.unsqueeze(0),  # (1, C, F_gt, H, W)
            source_fps=cfg.fps,
            target_fps=fps_val,
            method="linear",
            device=accelerator.device,
        )
        output = output.squeeze(0)[:, :num_gt_frames, :, :]  # (C, F_gt, H, W)

        # ---- Save results ----
        # Convert to uint8 [0, 255] with shape (F, H, W, C).
        out_frames = (output * 255.0).to(torch.uint8).permute(1, 2, 3, 0)
        if video is not None:
            gt_frames = (video.squeeze(0) * 255.0).to(torch.uint8).permute(1, 2, 3, 0)
        else:
            gt_frames = torch.empty(0)  # Empty tensor for leaderboard submission
        
        if args.output_format == 'mp4':
            os.makedirs(sample_output_dir, exist_ok=True)
            torchvision.io.video.write_video(
                os.path.join(sample_output_dir, "output.mp4"), out_frames, fps=fps_val
            )
        elif args.output_format in ['jpg', 'images']:
            rgb_dir = os.path.join(sample_output_dir, "rgb")
            os.makedirs(rgb_dir, exist_ok=True)
            
            for n, frame in enumerate(out_frames):
                # torchvision.io.write_jpeg requires the tensor to be on CPU and in (C, H, W) format
                frame_cwh = frame.cpu().permute(2, 0, 1)
                torchvision.io.write_jpeg(frame_cwh, os.path.join(rgb_dir, f"{n:04d}.jpg"))
        else:
            raise ValueError(f"Invalid output format: {args.output_format}")
        with open(os.path.join(sample_output_dir, "caption.txt"), "w", encoding="utf-8") as f:
            f.write(prompt)


def main(args):
    cfg = args.config
    accelerator = build_accelerator(
        project_dir=args.output_path,
        logging_dir=Path(args.output_path),
        nccl_timeout=NCCL_TIMEOUT,
        mixed_precision=cfg.mixed_precision,
        log_with=None,
    )
    configure_logging_verbosity(accelerator)

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    components = load_pipeline(cfg.model_key)
    pipe = components["pipe"]
    vae = components["vae"]
    denoiser = components["denoiser"]
    text_encoder = components["text_encoder"]
    image_encoder = components["image_encoder"]

    if cfg.model_key.startswith("stabilityai/stable-video-diffusion") and args.batch_size > 4:
        logger.info("It is stabilityai, we cannot use batch size > 4")
        args.batch_size = 4

    freeze_models([denoiser, vae, text_encoder, image_encoder])

    # When loading a fine-tuned checkpoint, the denoiser must be wrapped or
    # modified the same way it was during training so the state dict matches.
    if args.checkpoint is not None:
        denoiser = apply_training_mode(denoiser, args.training_mode, cfg)

    # Freeze again: apply_training_mode may have re-enabled gradients.
    freeze_models([denoiser, vae, text_encoder, image_encoder])

    # Enable TF32 for faster inference on Ampere GPUs.
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    for model in [denoiser, vae, text_encoder, image_encoder]:
        if model is not None:
            print_model_info(model, accelerator)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    if args.leaderboard:
        logger.info("Running inference for leaderboard submission...")
        from dataset.PhysInOne_Dataset import PhysInOne_Leaderboard_VideoGeneration as PhysInOne

        validation_dataset = PhysInOne(
            cfg.dataset_path,
            args.leaderboard_branch,
            cfg.resolution
        )
    else:
        logger.info("Running inference for validation...")
        from dataset.PhysInOne_Dataset import PhysInOne
        validation_dataset = PhysInOne(
            cfg,
            cfg.dataset_path,
            split="valid",
            only_moving=args.only_moving,
            only_one_cine=not args.only_moving,
        )

    validation_dataloader = torch.utils.data.DataLoader(
        validation_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=cfg.dataloader_num_workers,
        pin_memory=True,
    )

    denoiser, validation_dataloader = accelerator.prepare(denoiser, validation_dataloader)

    # For mixed precision, cast frozen inference-only models to half precision.
    weight_dtype = get_weight_dtype(accelerator)
    pipe.to(accelerator.device, dtype=weight_dtype)
    pipe.vae.to(accelerator.device, dtype=weight_dtype)

    do_classifier_free_guidance = cfg.guidance_scale > 1.0
    logger.info("***** Running inference *****")
    logger.info(f"  Num examples = {len(validation_dataset)}")
    logger.info(f"  Num batches = {len(validation_dataloader)}")
    logger.info(f"  Batch Size = {args.batch_size}")
    logger.info(f"  do_classifier_free_guidance = {do_classifier_free_guidance}")

    # ------------------------------------------------------------------
    # Load fine-tuned weights (if requested) and attach to the pipeline
    # ------------------------------------------------------------------
    if args.checkpoint:
        if args.checkpoint != "latest":
            path = f"checkpoint-{args.checkpoint}"
        else:
            path = find_latest_checkpoint(args.from_pretrained)

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.checkpoint}' does not exist. "
                "Using the original model for inference."
            )
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.from_pretrained, path))

        if cfg.model_key.startswith("stabilityai/stable-video-diffusion"):
            pipe.unet = denoiser
        elif cfg.model_key.startswith(("THUDM/CogVideoX", "Wan-AI/Wan")):
            pipe.transformer = denoiser

    if validation_dataloader is not None:
        log_validation(
            validation_dataloader, pipe, args, accelerator, weight_dtype,
            split="test", cur_step=0,
        )


def resolve_output_path(args):
    """Derive `args.output_path` and `args.training_mode` from the CLI arguments.

    Vanilla mode (--config, no --from_pretrained): the output path is
    'vanilla_<model_key>' under --output_path.

    Pretrained mode (--from_pretrained): the output path mirrors the last two
    components of the model directory (e.g. 'lora/lora_Wan-AI'). Without a
    checkpoint the run is 'vanilla' and the mode prefix in the folder name is
    replaced accordingly; with a checkpoint the mode is parsed from the folder
    prefix and a '_ckpt_<id>' suffix is appended.
    """
    if args.from_pretrained is None:
        # Vanilla mode: run the base pipeline defined by the config file.
        args.training_mode = "vanilla"
        args.output_path = os.path.join(
            args.output_path, f"vanilla_{args.config.model_key}"
        )
        return

    args.output_path = os.path.join(
        args.output_path, "/".join(Path(args.from_pretrained).resolve().parts[-2:])
    )

    parts = os.path.normpath(args.output_path).split(os.sep)
    if args.checkpoint is None:
        # Base model inference: mark the run folder as 'vanilla_*'.
        args.training_mode = "vanilla"
        if len(parts) >= 2:
            orig = parts[-2]
            suffix = orig.split("_", 1)[1] if "_" in orig else orig
            parts[-2] = f"vanilla_{suffix}"
        args.output_path = "/".join(parts)
    else:
        # Fine-tuned inference: parse the mode from the folder prefix,
        # e.g. 'lora' from 'lora_v1'.
        args.training_mode = parts[-2].split("_")[0] if len(parts) >= 2 else "unknown"
        args.output_path = f"{args.output_path}_ckpt_{args.checkpoint}"


if __name__ == "__main__":
    torch.manual_seed(6)
    np.random.seed(66)

    parser = argparse.ArgumentParser(description="Inference configuration parser")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a config.yaml for vanilla inference "
                             "(no fine-tuned weights); mutually exclusive with "
                             "--from_pretrained")
    parser.add_argument("--from_pretrained", type=str, default=None,
                        help="Model directory containing config.yaml and checkpoints")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root directory of the test dataset "
                             "(overrides any dataset path in the config file)")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./outputs/")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--only_moving", action="store_true")
    parser.add_argument("--skip_exist", action="store_true")
    parser.add_argument("--output_format", type=str, default="jpg")
    parser.add_argument("--leaderboard", type=bool, default=False)
    parser.add_argument("--leaderboard_branch", type=str, choices=["static", 'moving'], default="static",
                        help="Specify the branch for leaderboard submission")

    args = parser.parse_args()

    # Exactly one of --config (vanilla) or --from_pretrained (fine-tuned)
    # must be provided.
    if (args.config is None) == (args.from_pretrained is None):
        parser.error("Specify exactly one of --config (vanilla inference) "
                     "or --from_pretrained (fine-tuned inference)")
    if args.from_pretrained is None and args.checkpoint is not None:
        parser.error("--checkpoint requires --from_pretrained")

    if args.from_pretrained is not None:
        config_path = os.path.join(args.from_pretrained, "config.yaml")
    else:
        config_path = args.config
    args.config = read_yaml_to_namespce(config_path)
    # The dataset path is specified externally via CLI, not read from the config.
    args.config.dataset_path = args.data_path
    args.config.resolution = args.config.resolution if hasattr(args.config, "resolution") else (args.config.height, args.config.width)

    resolve_output_path(args)
    os.makedirs(args.output_path, exist_ok=True)

    # Keep a copy of the config alongside the outputs for reproducibility.
    shutil.copy(config_path, os.path.join(args.output_path, "config.yaml"))

    main(args)
