"""Training entry point for PhysInOne video diffusion fine-tuning.

Supports SFT, LoRA, and final-layer tuning (FLT) of image-to-video diffusion
models (Stable Video Diffusion, CogVideoX 1.5, Wan) using HuggingFace
Accelerate for distributed training.

Usage:
    accelerate launch train.py --config ./config.yaml
"""

import argparse
import math
import os
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.utils.checkpoint
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers.optimization import get_scheduler
from diffusers.utils import is_wandb_available
from huggingface_hub import create_repo
from torch.utils.data import Subset
from tqdm.auto import tqdm

from dataset.PhysInOne_Dataset import PhysInOne
from utils.args import read_yaml_to_namespce
from utils import (
    DEVICE,
    apply_training_mode,
    build_accelerator,
    build_validation_pipeline,
    configure_logging_verbosity,
    create_log_video,
    find_latest_checkpoint,
    freeze_models,
    get_weight_dtype,
    load_pipeline,
    print_model_info,
    setup_environment,
)

setup_environment()

if is_wandb_available():
    import wandb
os.environ["WANDB_CONFIG_DIR"] = "/tmp/.config-" + os.environ["USER"]

NCCL_TIMEOUT = 360000

logger = get_logger(__name__)


@torch.no_grad()
def log_validation(validation_dataloader, vae, denoiser, image_encoder, text_encoder,
                   args, accelerator, weight_dtype, split="val", cur_step=0):
    """Run validation inference and log side-by-side videos to trackers."""
    logger.info(f"Running {split} validation... ")

    pipe = build_validation_pipeline(
        args.model_key, accelerator, vae, denoiser,
        image_encoder, text_encoder, weight_dtype,
    )

    log_image_num = 16
    log_case_rate = max(1, len(validation_dataloader) // log_image_num)

    if args.seed is None:
        generator = None
    else:
        generator = torch.Generator(device=accelerator.device).manual_seed(
            args.seed + accelerator.process_index
        )

    image_logs = []
    bar = tqdm(enumerate(validation_dataloader), desc=split, total=len(validation_dataloader))
    for valid_step, batch in bar:
        video = batch["video"].to(DEVICE, dtype=weight_dtype)
        image = batch["image"].to(DEVICE, dtype=weight_dtype)
        prompt = batch["prompt"]
        negative_prompt = batch["negative_prompt"]
        frame_mask = batch["frame_mask"].to(DEVICE, dtype=weight_dtype)

        output = pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            fps=args.fps,
            output_type="pt",
            generator=generator,
            height=args.height,
            width=args.width,
        ).frames  # (B, F, C, H, W)
        output = output.permute(0, 2, 1, 3, 4)  # (B, C, F, H, W)

        # Pad the generated video to the ground-truth frame count, then mask
        # out padded frames in both videos.
        pad_amount = frame_mask.size(2) - output.size(2)
        output = torch.nn.functional.pad(
            output, pad=(0, 0, 0, 0, 0, pad_amount), mode="constant", value=0
        )
        video = torch.clamp(video * frame_mask, min=0.0, max=1.0)
        output = torch.clamp(output * frame_mask, min=0.0, max=1.0)

        if accelerator.is_main_process and valid_step % log_case_rate == 0:
            caption = "".join(
                f"{j}:{prompt[j]}\n" for j in range(output.size(0))
            )
            image_logs.append({
                "video": create_log_video(image, output, video),
                "caption": caption,
            })

    val_metrics = {}
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            formatted = [
                wandb.Video(log["video"], caption=log["caption"], format="mp4", fps=4)
                for log in image_logs
            ]
            tracker.log({split: formatted}, step=cur_step)
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")
    return image_logs, val_metrics


def main(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator = build_accelerator(
        project_dir=args.output_dir,
        logging_dir=logging_dir,
        nccl_timeout=NCCL_TIMEOUT,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    configure_logging_verbosity(accelerator)

    if args.seed is not None:
        set_seed(args.seed)

    # Handle output directory and hub repository creation.
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        if args.push_to_hub:
            create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name,
                exist_ok=True,
                token=args.hub_token,
            )

    # ------------------------------------------------------------------
    # Models
    # ------------------------------------------------------------------
    components = load_pipeline(args.model_key, device=DEVICE)
    pipe = components["pipe"]
    vae = components["vae"]
    denoiser = components["denoiser"]
    text_encoder = components["text_encoder"]
    image_encoder = components["image_encoder"]

    # Freeze everything, then re-enable gradients per training mode.
    freeze_models([denoiser, vae, text_encoder, image_encoder])
    denoiser = apply_training_mode(denoiser, args.training_mode, args)

    training_layers = [p for p in denoiser.parameters() if p.requires_grad]

    # Enable TF32 for faster training on Ampere GPUs.
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps
            * args.training_batch_size * accelerator.num_processes
        )

    if args.gradient_checkpointing:
        denoiser.enable_gradient_checkpointing()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    # Use 8-bit Adam for lower memory usage (fits fine-tuning on 16GB GPUs).
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install bitsandbytes: `pip install bitsandbytes`."
            )
        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW

    optimizer = optimizer_class(
        [{"params": training_layers, "lr": args.learning_rate}],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=float(args.adam_epsilon),
    )

    for model in [denoiser, vae, text_encoder, image_encoder]:
        if model is not None:
            print_model_info(model, accelerator)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    train_dataset = PhysInOne(args, args.dataset_path)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
        prefetch_factor=1,
        persistent_workers=True,
    )

    # A small, evenly spaced subset of the training set for periodic logging
    # (main process only).
    if accelerator.is_main_process:
        num_log_samples = min(16, len(train_dataset))
        if num_log_samples > 0:
            stride = len(train_dataset) // num_log_samples
            log_indices = list(range(0, len(train_dataset), stride))[:num_log_samples]
        else:
            log_indices = []
        train_log_dataloader = torch.utils.data.DataLoader(
            Subset(train_dataset, log_indices),
            shuffle=False,
            batch_size=1,  # batch_size=4 raises a CUDA invalid-config error for SVD
            num_workers=args.dataloader_num_workers,
            pin_memory=False,
        )
    else:
        train_log_dataloader = None
    valid_dataloader = None

    # ------------------------------------------------------------------
    # LR scheduler and accelerator preparation
    # ------------------------------------------------------------------
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    denoiser, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        denoiser, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision, cast frozen inference-only models to half precision.
    weight_dtype = get_weight_dtype(accelerator)
    pipe.to(accelerator.device, dtype=weight_dtype)
    pipe.vae.to(accelerator.device, dtype=weight_dtype)

    # Recalculate steps/epochs (the dataloader size may have changed after prepare).
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    total_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    do_classifier_free_guidance = args.guidance_scale > 1.0
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    logger.info(f"  Instantaneous batch size per device = {args.batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  do_classifier_free_guidance = {do_classifier_free_guidance}")
    logger.info(f"  conditioning_dropout_prob = {args.conditioning_dropout_prob}")

    global_step = 0
    first_epoch = 0

    # Potentially resume from a previous checkpoint.
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            path = find_latest_checkpoint(args.output_dir)

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. "
                "Starting a new training run."
            )
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,  # Show once per machine.
    )

    for epoch in range(first_epoch, args.num_train_epochs):
        loss_epoch = 0.0
        num_train_elems = 0
        if global_step >= args.max_train_steps:
            break

        for step, batch in enumerate(train_dataloader):
            # ---- Forward / backward ----
            with accelerator.accumulate(denoiser):
                video = batch["video"].to(DEVICE, dtype=weight_dtype)
                image = batch["image"].to(DEVICE, dtype=weight_dtype)
                prompt = batch["prompt"]
                negative_prompt = batch["negative_prompt"]
                frame_mask = batch["frame_mask"].to(DEVICE, dtype=weight_dtype)

                # TODO: use FPS as a condition.
                loss = pipe.training(
                    video=video, image=image, prompt=prompt,
                    frame_mask=frame_mask, negative_prompt=negative_prompt,
                    fps=args.fps,
                )
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(training_layers, args.max_grad_norm)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=args.set_grads_to_none)

            # ---- Console / tracker logging ----
            loss_epoch += loss.detach().item()
            num_train_elems += 1
            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "loss_epoch": loss_epoch / num_train_elems,
                "epoch": epoch,
            }
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            # ---- Checkpointing and validation (on real optimization steps) ----
            if accelerator.sync_gradients:
                if accelerator.is_main_process:
                    step_log = {}

                    # Checkpointing (respecting the total checkpoint limit).
                    if global_step % args.checkpointing_steps == 0:
                        if args.checkpoints_total_limit is not None:
                            checkpoints = sorted(
                                (d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")),
                                key=lambda x: int(x.split("-")[1]),
                            )
                            if len(checkpoints) >= args.checkpoints_total_limit:
                                num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                                for cp in checkpoints[:num_to_remove]:
                                    shutil.rmtree(os.path.join(args.output_dir, cp))

                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved checkpoint to {save_path}")

                    # Validation.
                    if global_step > 0 and (global_step % args.validation_steps == 0 or global_step == 100):
                        for dataloader, split in [
                            (train_log_dataloader, "train_log"),
                            (valid_dataloader, "valid"),
                        ]:
                            if dataloader is None:
                                continue
                            _, val_metrics = log_validation(
                                dataloader, vae, denoiser, image_encoder, text_encoder,
                                args, accelerator, weight_dtype,
                                split=split, cur_step=global_step,
                            )
                            step_log.update(val_metrics)

                    if step_log:
                        accelerator.log(step_log, step=global_step)

                    global_step += 1
                    progress_bar.update(1)

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    torch.manual_seed(6)
    np.random.seed(66)

    parser = argparse.ArgumentParser(description="Training configuration parser")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the YAML configuration file")
    parser.add_argument("--data_path", type=str, required=True,
                        help="Root directory of the training dataset "
                             "(overrides any dataset path in the config file)")
    cli_args = parser.parse_args()
    config_path = cli_args.config
    args = read_yaml_to_namespce(config_path)
    # The dataset path is specified externally via CLI, not read from the config.
    args.dataset_path = cli_args.data_path

    # Derive the training mode from the config's parent folder name and build
    # a unique output directory, e.g. 'lora_Wan-AI/Wan2.2-TI2V-5B-Diffusers'.
    args.training_mode = config_path.split("/")[-2]
    name = f"{args.training_mode}_{args.model_key}"
    args.output_dir = os.path.join(args.output_dir, name)
    os.makedirs(args.output_dir, exist_ok=True)

    # Keep a copy of the config alongside the checkpoints for reproducibility.
    shutil.copy(config_path, os.path.join(args.output_dir, "config.yaml"))

    main(args)
