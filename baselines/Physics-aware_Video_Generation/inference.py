"""Multi-GPU inference entry point for PhysInOne video diffusion models.

The script uses Accelerate for data-parallel inference. Every process owns one
GPU and one complete pipeline replica. The dataset is partitioned across
processes without padding or duplicate samples, and ``--batch_size`` is the
number of samples processed by each GPU in one pipeline call.

Examples:
    # Select physical GPUs 0, 2, and 5; the script launches Accelerate itself:
    python inference.py --gpu_ids 0,2,5 --config ./config.yaml \
        --data_path /path/to/dataset --batch_size 2

    # One GPU, batch size 2:
    python inference.py --config ./config.yaml \
        --data_path /path/to/dataset --batch_size 2

    # Four GPUs, batch size 2 per GPU (global batch size 8):
    accelerate launch --num_processes 4 inference.py \
        --config ./config.yaml --data_path /path/to/dataset --batch_size 2

    # Fine-tuned checkpoint on four GPUs:
    accelerate launch --num_processes 4 inference.py \
        --from_pretrained ./models/lora/lora_Wan-AI/... \
        --checkpoint latest --data_path /path/to/dataset --batch_size 2

    # Leaderboard inference:
    accelerate launch --num_processes 4 inference.py \
        --config ./config.yaml --data_path /path/to/leaderboard_dataset \
        --leaderboard --leaderboard_branch static --batch_size 2

Notes:
    * Effective global batch size = batch_size * number of processes/GPUs.
    * ``--gpu_ids`` must be supplied to a direct Python invocation. GPU
      visibility has to be configured before Accelerate creates its workers.
      Alternatively, set CUDA_VISIBLE_DEVICES yourself before running
      ``accelerate launch``.
    * Samples in one batch must have compatible tensor shapes so PyTorch's
      default DataLoader collation can stack them.
    * Autoregressive generation continues until the longest sample in the
      local per-GPU batch reaches its target duration. Each result is then
      independently converted to its requested FPS and frame count.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ["HF_HOME"] = "./hf_cache"

import numpy as np
import torch
import torch.utils.checkpoint
import torchvision
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from tqdm.auto import tqdm

from utils.args import read_yaml_to_namespce
from utils import (
    apply_training_mode,
    build_accelerator,
    center_crop_and_resize,
    configure_logging_verbosity,
    find_latest_checkpoint,
    frame_interpolate,
    freeze_models,
    get_weight_dtype,
    load_pipeline,
    print_model_info,
    setup_environment,
)


setup_environment()

NCCL_TIMEOUT = 36000
ACCELERATE_RELAUNCH_ENV = "PHYSINONE_ACCELERATE_RELAUNCHED"
logger = get_logger(__name__)


def inference_collate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate inference samples while preserving variable-length videos.

    Conditioning images must share a shape and are stacked normally. Video
    tensors are intentionally kept in a list because validation videos may
    contain different frame counts. Other fields use PyTorch's default
    collation when possible and otherwise remain Python lists.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample list")

    keys = set(samples[0])
    if any(set(sample) != keys for sample in samples[1:]):
        raise ValueError("Every sample in one inference batch must have the same keys")

    batch: dict[str, Any] = {}
    for key in keys:
        values = [sample[key] for sample in samples]
        if key == "video":
            batch[key] = values
            continue

        try:
            batch[key] = default_collate(values)
        except (RuntimeError, TypeError):
            batch[key] = values

    return batch


def _batch_strings(value: Any, batch_size: int, field_name: str) -> list[str]:
    """Normalize a default-collated string field to one string per sample."""
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, Sequence):
        values = list(value)
    else:
        raise TypeError(
            f"batch[{field_name!r}] must be a string or sequence of strings, "
            f"got {type(value).__name__}"
        )

    if len(values) != batch_size or not all(isinstance(item, str) for item in values):
        raise ValueError(
            f"batch[{field_name!r}] must contain {batch_size} strings, got {values!r}"
        )
    return values


def _batch_numbers(value: Any, batch_size: int, field_name: str) -> list[float]:
    """Normalize scalar/tensor/list metadata to one numeric value per sample."""
    if torch.is_tensor(value):
        if value.ndim == 0:
            values = [value.item()] * batch_size
        elif value.shape[0] == batch_size:
            values = [value[index].item() for index in range(batch_size)]
        else:
            raise ValueError(
                f"batch[{field_name!r}] has shape {tuple(value.shape)}, but batch "
                f"size is {batch_size}"
            )
    elif isinstance(value, np.ndarray):
        flattened = value.reshape(-1).tolist()
        values = flattened if len(flattened) != 1 else flattened * batch_size
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = [item.item() if torch.is_tensor(item) else item for item in value]
    elif isinstance(value, (int, float)):
        values = [value] * batch_size
    else:
        raise TypeError(
            f"batch[{field_name!r}] must contain numeric values, "
            f"got {type(value).__name__}"
        )

    if len(values) != batch_size:
        raise ValueError(
            f"batch[{field_name!r}] has {len(values)} values, but batch size is "
            f"{batch_size}"
        )

    try:
        return [float(item) for item in values]
    except (TypeError, ValueError) as error:
        raise TypeError(f"batch[{field_name!r}] contains a non-numeric value") from error


def _ensure_frame_count(video: torch.Tensor, target_frames: int) -> torch.Tensor:
    """Trim or last-frame-pad a ``[1, C, F, H, W]`` video to an exact length."""
    if video.ndim != 5 or video.shape[0] != 1:
        raise ValueError(
            "Expected one video with shape [1, C, F, H, W], "
            f"got {tuple(video.shape)}"
        )
    if target_frames <= 0:
        raise ValueError(f"target_frames must be positive, got {target_frames}")

    current_frames = video.shape[2]
    if current_frames == 0:
        raise ValueError("The generated video contains no frames")
    if current_frames >= target_frames:
        return video[:, :, :target_frames]

    padding = video[:, :, -1:].expand(
        -1,
        -1,
        target_frames - current_frames,
        -1,
        -1,
    )
    return torch.cat([video, padding], dim=2)


def _save_video(
    output: torch.Tensor,
    prompt: str,
    fps: float,
    sample_output_dir: Path,
    output_format: str,
) -> None:
    """Save one ``[C, F, H, W]`` video and its prompt."""
    sample_output_dir.mkdir(parents=True, exist_ok=True)
    out_frames = (
        output.detach()
        .clamp(0, 1)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .contiguous()
        .cpu()
    )

    if output_format == "mp4":
        torchvision.io.video.write_video(
            str(sample_output_dir / "output.mp4"),
            out_frames,
            fps=fps,
        )
    elif output_format in {"jpg", "images"}:
        rgb_dir = sample_output_dir / "rgb"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, frame in enumerate(out_frames):
            torchvision.io.write_jpeg(
                frame.permute(2, 0, 1).contiguous(),
                str(rgb_dir / f"{frame_index:04d}.jpg"),
                quality=95,
            )
    else:
        raise ValueError(f"Unsupported output format: {output_format}")

    (sample_output_dir / "caption.txt").write_text(prompt, encoding="utf-8")


@torch.inference_mode()
def log_validation(
    validation_dataloader: DataLoader,
    pipe,
    args: argparse.Namespace,
    accelerator,
    weight_dtype: torch.dtype,
    split: str = "test",
) -> None:
    """Generate batched videos on every GPU and save each sample separately."""
    cfg = args.config
    logger.info(
        "Running %s inference on process %d with %d local batch(es)",
        split,
        accelerator.process_index,
        len(validation_dataloader),
        main_process_only=False,
    )

    if cfg.seed is None:
        generator = None
    else:
        # Different process seeds prevent different GPUs from producing the same
        # random samples while remaining reproducible for a fixed world size.
        process_seed = int(cfg.seed) + accelerator.process_index
        generator = torch.Generator(device=accelerator.device).manual_seed(process_seed)

    progress = tqdm(
        validation_dataloader,
        desc=f"{split} (rank {accelerator.process_index})",
        total=len(validation_dataloader),
        disable=not accelerator.is_local_main_process,
        dynamic_ncols=True,
    )

    for batch in progress:
        image = batch["image"].to(
            accelerator.device,
            dtype=weight_dtype,
            non_blocking=True,
        )
        if image.ndim != 4:
            raise ValueError(
                f"Expected batch['image'] with shape [B, C, H, W], got {tuple(image.shape)}"
            )

        batch_size = image.shape[0]
        prompts = _batch_strings(batch["prompt"], batch_size, "prompt")
        names = _batch_strings(batch["name"], batch_size, "name")
        fps_values = _batch_numbers(batch["fps"], batch_size, "fps")

        videos = batch.get("video")

        if "total_frames" in batch:
            target_frame_counts = [
                int(value)
                for value in _batch_numbers(
                    batch["total_frames"], batch_size, "total_frames"
                )
            ]
        elif videos is not None:
            if torch.is_tensor(videos):
                if videos.ndim != 5 or videos.shape[0] != batch_size:
                    raise ValueError(
                        "Expected batch['video'] with shape [B, C, F, H, W], "
                        f"got {tuple(videos.shape)}"
                    )
                target_frame_counts = [int(videos.shape[2])] * batch_size
            elif isinstance(videos, Sequence) and len(videos) == batch_size:
                target_frame_counts = []
                for index, sample_video in enumerate(videos):
                    if not torch.is_tensor(sample_video) or sample_video.ndim != 4:
                        raise ValueError(
                            "Each unstacked video must have shape [C, F, H, W]; "
                            f"sample {index} has {type(sample_video).__name__}"
                        )
                    target_frame_counts.append(int(sample_video.shape[1]))
            else:
                raise ValueError(
                    "batch['video'] must be a [B, C, F, H, W] tensor or a "
                    "sequence of [C, F, H, W] tensors"
                )
        else:
            raise ValueError(
                "Cannot determine target video lengths: the batch contains neither "
                "'total_frames' nor a ground-truth 'video'."
            )

        for name, fps, frame_count in zip(names, fps_values, target_frame_counts):
            if fps <= 0:
                raise ValueError(f"Sample {name!r} has invalid FPS: {fps}")
            if frame_count <= 0:
                raise ValueError(
                    f"Sample {name!r} has invalid target frame count: {frame_count}"
                )

        # Skip completed samples individually instead of discarding an entire
        # per-GPU batch when only some outputs already exist.
        active_indices = [
            index
            for index, name in enumerate(names)
            if not (
                args.skip_exist
                and (Path(args.output_path) / name).exists()
            )
        ]
        if not active_indices:
            continue

        if len(active_indices) != batch_size:
            index_tensor = torch.tensor(
                active_indices,
                dtype=torch.long,
                device=image.device,
            )
            image = image.index_select(0, index_tensor)
            prompts = [prompts[index] for index in active_indices]
            names = [names[index] for index in active_indices]
            fps_values = [fps_values[index] for index in active_indices]
            target_frame_counts = [
                target_frame_counts[index] for index in active_indices
            ]
            batch_size = len(active_indices)

        target_durations = [
            frame_count / fps
            for frame_count, fps in zip(target_frame_counts, fps_values)
        ]
        longest_target_duration = max(target_durations)

        # Generate complete batches autoregressively. Shorter samples may be
        # over-generated while waiting for the longest sample in this local
        # batch; each result is independently trimmed below.
        output = None
        current_image = image
        image_height, image_width = image.shape[-2:]

        while (
            output is None
            or output.shape[2] / float(cfg.fps) < longest_target_duration
        ):
            chunk = pipe(
                image=current_image,
                prompt=prompts,
                fps=cfg.fps,
                output_type="pt",
                generator=generator,
                height=cfg.height,
                width=cfg.width,
            ).frames

            if not torch.is_tensor(chunk) or chunk.ndim != 5:
                raise RuntimeError(
                    "The pipeline must return a tensor in .frames with shape "
                    f"[B, F, C, H, W], got {type(chunk).__name__}"
                )
            if chunk.shape[0] != batch_size:
                raise RuntimeError(
                    f"The pipeline returned {chunk.shape[0]} videos for a local "
                    f"batch of {batch_size} samples"
                )

            chunk = chunk.permute(0, 2, 1, 3, 4).contiguous()
            if output is None:
                output = chunk
            else:
                # The new chunk begins with its conditioning frame, which is
                # already the final frame of the preceding chunk.
                output = torch.cat([output, chunk[:, :, 1:]], dim=2)

            current_image = center_crop_and_resize(
                output[:, :, -1],
                image_height,
                image_width,
            )

        # FPS conversion and output writing are per sample because samples in a
        # local batch may have different requested FPS values and frame counts.
        for index, (name, prompt, target_fps, target_frames, target_duration) in enumerate(
            zip(
                names,
                prompts,
                fps_values,
                target_frame_counts,
                target_durations,
            )
        ):
            required_source_frames = min(
                output.shape[2],
                max(1, math.ceil(target_duration * float(cfg.fps))),
            )
            sample_output = output[
                index : index + 1,
                :,
                :required_source_frames,
            ]
            sample_output = frame_interpolate(
                sample_output,
                source_fps=cfg.fps,
                target_fps=target_fps,
                method="linear",
                device=accelerator.device,
            )
            sample_output = _ensure_frame_count(sample_output, target_frames)

            _save_video(
                sample_output.squeeze(0),
                prompt,
                target_fps,
                Path(args.output_path) / name,
                args.output_format,
            )

        del image, videos, output, current_image


def _attach_denoiser(pipe, denoiser, model_key: str) -> None:
    """Attach a loaded/fine-tuned denoiser to the correct pipeline attribute."""
    if model_key.startswith("stabilityai/stable-video-diffusion"):
        pipe.unet = denoiser
    elif model_key.startswith(("THUDM/CogVideoX", "Wan-AI/Wan", "nvidia/Cosmos")):
        pipe.transformer = denoiser
    elif getattr(pipe, "unet", None) is denoiser:
        pipe.unet = denoiser
    elif getattr(pipe, "transformer", None) is denoiser:
        pipe.transformer = denoiser
    else:
        raise NotImplementedError(
            f"Do not know where to attach the denoiser for model {model_key!r}"
        )


def _prepare_output_directory(
    args: argparse.Namespace,
    accelerator,
) -> None:
    """Create shared output metadata once, then synchronize all processes."""
    if accelerator.is_main_process:
        output_path = Path(args.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        source_config = Path(args.config_path).resolve()
        destination_config = (output_path / "config.yaml").resolve()
        if source_config != destination_config:
            shutil.copy2(source_config, destination_config)

    accelerator.wait_for_everyone()


def _build_local_dataloader(
    dataset,
    args: argparse.Namespace,
    accelerator,
) -> DataLoader:
    """Partition dataset indices across Accelerate processes without duplicates."""
    # A manual no-padding partition is used instead of a padded DistributedSampler.
    # This prevents two ranks from writing the same final sample when the dataset
    # size is not divisible by the number of GPUs.
    local_indices = list(
        range(
            accelerator.process_index,
            len(dataset),
            accelerator.num_processes,
        )
    )
    local_dataset = Subset(dataset, local_indices)
    workers = int(args.config.dataloader_num_workers)

    return DataLoader(
        local_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        drop_last=False,
        collate_fn=inference_collate,
    )


def main(args: argparse.Namespace) -> None:
    cfg = args.config
    accelerator = build_accelerator(
        project_dir=args.output_path,
        logging_dir=Path(args.output_path),
        nccl_timeout=NCCL_TIMEOUT,
        mixed_precision=cfg.mixed_precision,
        log_with=None,
    )
    configure_logging_verbosity(accelerator)
    _prepare_output_directory(args, accelerator)

    if cfg.seed is not None:
        set_seed(int(cfg.seed), device_specific=True)

    components = load_pipeline(cfg.model_key)
    pipe = components["pipe"]
    vae = components["vae"]
    denoiser = components["denoiser"]
    text_encoder = components["text_encoder"]
    image_encoder = components["image_encoder"]

    if (
        cfg.model_key.startswith("stabilityai/stable-video-diffusion")
        and args.batch_size > 4
    ):
        accelerator.print(
            "Stable Video Diffusion supports at most batch size 4 per GPU; "
            f"reducing --batch_size from {args.batch_size} to 4."
        )
        args.batch_size = 4

    freeze_models([denoiser, vae, text_encoder, image_encoder])
    if args.checkpoint is not None:
        denoiser = apply_training_mode(denoiser, args.training_mode, cfg)
    freeze_models([denoiser, vae, text_encoder, image_encoder])

    if cfg.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    for model in [denoiser, vae, text_encoder, image_encoder]:
        if model is not None:
            print_model_info(model, accelerator)

    # evaluation_mode=True registers the denoiser for Accelerator checkpoint
    # loading and places it on this process's GPU without wrapping it in DDP.
    # DDP synchronization is unnecessary because each rank performs independent
    # inference on a disjoint dataset shard.
    denoiser = accelerator.prepare_model(denoiser, evaluation_mode=True)

    checkpoint_path = None
    if args.checkpoint is not None:
        if args.checkpoint == "latest":
            checkpoint_path = find_latest_checkpoint(args.from_pretrained)
        else:
            checkpoint_path = f"checkpoint-{args.checkpoint}"

        if checkpoint_path is None:
            accelerator.print(
                f"Checkpoint {args.checkpoint!r} does not exist; using the "
                "currently initialized model."
            )
        else:
            full_checkpoint_path = os.path.join(
                args.from_pretrained,
                checkpoint_path,
            )
            accelerator.print(f"Loading checkpoint {full_checkpoint_path}")
            accelerator.load_state(full_checkpoint_path)

    denoiser = accelerator.unwrap_model(denoiser)
    _attach_denoiser(pipe, denoiser, cfg.model_key)

    weight_dtype = get_weight_dtype(accelerator)
    pipe.to(accelerator.device, dtype=weight_dtype)
    if getattr(pipe, "vae", None) is not None:
        pipe.vae.to(accelerator.device, dtype=weight_dtype)
    if hasattr(pipe, "set_progress_bar_config"):
        pipe.set_progress_bar_config(disable=True)

    if args.leaderboard:
        logger.info("Running leaderboard inference...")
        from dataset.PhysInOne_Dataset import (
            PhysInOne_Leaderboard_VideoGeneration as PhysInOne,
        )

        validation_dataset = PhysInOne(
            cfg.dataset_path,
            args.leaderboard_branch,
            cfg.resolution,
        )
    else:
        logger.info("Running validation inference...")
        from dataset.PhysInOne_Dataset import PhysInOne

        validation_dataset = PhysInOne(
            cfg,
            cfg.dataset_path,
            split="valid",
            only_moving=args.only_moving,
            only_one_cine=not args.only_moving,
        )

    validation_dataloader = _build_local_dataloader(
        validation_dataset,
        args,
        accelerator,
    )

    accelerator.print("***** Running distributed inference *****")
    accelerator.print(f"  Number of GPUs/processes = {accelerator.num_processes}")
    accelerator.print(f"  Number of examples = {len(validation_dataset)}")
    accelerator.print(f"  Per-GPU batch size = {args.batch_size}")
    accelerator.print(
        f"  Maximum global batch size = {args.batch_size * accelerator.num_processes}"
    )
    logger.info(
        "Process %d handles %d samples in %d batches",
        accelerator.process_index,
        len(validation_dataloader.dataset),
        len(validation_dataloader),
        main_process_only=False,
    )

    log_validation(
        validation_dataloader,
        pipe,
        args,
        accelerator,
        weight_dtype,
        split="test",
    )
    accelerator.wait_for_everyone()
    accelerator.print(f"Inference complete. Results: {args.output_path}")


def resolve_output_path(args: argparse.Namespace) -> None:
    """Derive output path and training mode from command-line arguments."""
    if args.from_pretrained is None:
        args.training_mode = "vanilla"
        args.output_path = str(
            Path(args.output_path) / f"vanilla_{args.config.model_key}"
        )
        return

    pretrained_parts = Path(args.from_pretrained).resolve().parts
    model_tail = Path(*pretrained_parts[-2:])
    output_path = Path(args.output_path) / model_tail

    if args.checkpoint is None:
        args.training_mode = "vanilla"
        parent_name = output_path.parent.name
        suffix = parent_name.split("_", 1)[1] if "_" in parent_name else parent_name
        output_path = output_path.parent.parent / f"vanilla_{suffix}" / output_path.name
    else:
        args.training_mode = output_path.parent.name.split("_")[0]
        output_path = Path(f"{output_path}_ckpt_{args.checkpoint}")

    args.output_path = str(output_path)


def _parse_gpu_ids(value: str) -> list[int]:
    """Parse and validate a comma-separated list of physical CUDA GPU IDs."""
    fields = [field.strip() for field in value.split(",")]
    if not fields or any(not field for field in fields):
        raise ValueError(
            f"Invalid --gpu_ids value {value!r}; expected a list such as 0,2,5"
        )

    try:
        gpu_ids = [int(field) for field in fields]
    except ValueError as error:
        raise ValueError(
            f"Invalid --gpu_ids value {value!r}; every GPU ID must be an integer"
        ) from error

    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"GPU IDs must be non-negative, got {gpu_ids}")
    if len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError(f"GPU IDs must be unique, got {gpu_ids}")
    return gpu_ids


def maybe_relaunch_with_selected_gpus(
    args: argparse.Namespace,
    script_arguments: Sequence[str],
) -> None:
    """Relaunch this script through Accelerate on the requested physical GPUs.

    CUDA visibility cannot be changed reliably after distributed workers have
    started. When ``--gpu_ids`` is supplied to a direct Python invocation, this
    function replaces the current process with ``accelerate launch`` and sets
    CUDA_VISIBLE_DEVICES before any worker initializes CUDA.
    """
    if args.gpu_ids is None:
        return

    gpu_ids = _parse_gpu_ids(args.gpu_ids)
    if os.environ.get(ACCELERATE_RELAUNCH_ENV) == "1":
        return

    if "LOCAL_RANK" in os.environ or int(os.environ.get("WORLD_SIZE", "1")) > 1:
        raise RuntimeError(
            "--gpu_ids was received after Accelerate had already started. Run "
            "this script directly with `python inference.py "
            "--gpu_ids 0,2 ...`, or set `CUDA_VISIBLE_DEVICES=0,2` before "
            "calling `accelerate launch`."
        )

    accelerate_executable = shutil.which("accelerate")
    if accelerate_executable is None:
        raise RuntimeError(
            "The `accelerate` executable is required for --gpu_ids. Install "
            "Hugging Face Accelerate or run with an activated environment that "
            "provides the `accelerate` command."
        )

    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(gpu_id) for gpu_id in gpu_ids)
    environment[ACCELERATE_RELAUNCH_ENV] = "1"

    command = [accelerate_executable, "launch"]
    if len(gpu_ids) > 1:
        command.append("--multi_gpu")
    command.extend(
        [
            "--num_processes",
            str(len(gpu_ids)),
            str(Path(__file__).resolve()),
            *script_arguments,
        ]
    )

    print(
        "Launching Accelerate on physical GPU(s): "
        + ", ".join(str(gpu_id) for gpu_id in gpu_ids),
        flush=True,
    )
    os.execvpe(accelerate_executable, command, environment)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distributed inference for PhysInOne video diffusion models"
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--config",
        type=str,
        help="Path to config.yaml for vanilla inference",
    )
    source_group.add_argument(
        "--from_pretrained",
        type=str,
        help="Model directory containing config.yaml and checkpoints",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Dataset root; overrides the path in config.yaml",
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--output_path", type=str, default="./outputs/")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        metavar="ID[,ID...]",
        help=(
            "Physical CUDA GPU IDs to use, for example 0,2,5. When supplied "
            "during a direct Python invocation, the script automatically "
            "relaunches itself through Accelerate with one process per GPU."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of samples processed concurrently on each GPU",
    )
    parser.add_argument("--only_moving", action="store_true")
    parser.add_argument("--skip_exist", action="store_true")
    parser.add_argument(
        "--output_format",
        choices=("mp4", "jpg", "images"),
        default="jpg",
    )
    parser.add_argument(
        "--leaderboard",
        action="store_true",
        help="Use the leaderboard dataset instead of the validation split",
    )
    parser.add_argument(
        "--leaderboard_branch",
        choices=("static", "moving"),
        default="static",
        help="Leaderboard submission branch",
    )

    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch_size must be a positive integer")
    if args.from_pretrained is None and args.checkpoint is not None:
        parser.error("--checkpoint requires --from_pretrained")

    if args.from_pretrained is not None:
        config_path = os.path.join(args.from_pretrained, "config.yaml")
    else:
        config_path = args.config

    args.config_path = config_path
    args.config = read_yaml_to_namespce(config_path)
    args.config.dataset_path = args.data_path
    if not hasattr(args.config, "resolution"):
        args.config.resolution = (args.config.height, args.config.width)

    resolve_output_path(args)
    return args


if __name__ == "__main__":
    parsed_args = parse_args()
    maybe_relaunch_with_selected_gpus(parsed_args, sys.argv[1:])
    main(parsed_args)
