#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tensor-based Go-with-the-Flow inference adapter.

This module exposes an in-memory batched API for applying Go-with-the-Flow's
CogVideoX I2V LoRA to PhysInOne motion-transfer samples.

    refs = batch["reference_video"]   # [B, F, H, W, 3] in [0,1] or uint8
    fr1s = batch["first_frame"]        # [B, H, W, 3]   in [0,1] or uint8
    caps = batch.get("caption", [])     # list[str], len B

Key properties:
- **No input.mp4** is written. The noise-warp API is called with a **THWC numpy array**.
- **Optional debug**: only `noise.npy` can be saved to a given directory per sample.
- **Batch inference**: builds latents `[B,13,16,60,90]` then calls I2V pipeline once.

Dependencies: numpy, torch, einops, pillow, diffusers, transformers, accelerate, fire
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import einops
import numpy as np
import torch
from diffusers import (
    AutoencoderKLCogVideoX,
    CogVideoXImageToVideoPipeline,
    CogVideoXTransformer3DModel,
)
from PIL import Image
from transformers import T5EncoderModel

REPO_ROOT = Path(__file__).resolve().parents[1]
_PIPE_CACHE = {}  # key: (pipe_key, lora_name, low_vram, str(device))

# ================== Config ==================
DTYPE = torch.bfloat16
TARGET_T = 49
TARGET_H, TARGET_W = 480, 720
F_LAT = 13
H_LAT, W_LAT = 60, 90   # 480/8, 720/8
NOISE_C = 16

# HF models & LoRA registry
PIPE_IDS = dict(
    I2V5B="THUDM/CogVideoX-5b-I2V",
)
COGVIDEOX_I2V_REVISION = "a6f0f4858a8395e7429d82493864ce92bf73af11"
COMMON_SOURCE_DIR = REPO_ROOT / "third_party" / "CommonSource"

# ================== Utilities ==================

def _import_noise_warp():
    """Import the pinned CommonSource noise-warp implementation."""
    override = os.environ.get("NOISE_WARP_PY", "").strip()
    source_path = (
        Path(override).expanduser()
        if override
        else COMMON_SOURCE_DIR / "noise_warp.py"
    )
    if not source_path.is_file():
        raise FileNotFoundError(
            f"CommonSource noise_warp.py not found: {source_path}\n"
            "Run `python scripts/download_ckpt.py --name gowiththeflow`."
        )

    source_dir = str(source_path.resolve().parent)
    if source_dir not in sys.path:
        sys.path.insert(0, source_dir)

    module_name = "physinone_common_source_noise_warp"
    if module_name in sys.modules:
        return sys.modules[module_name]

    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load CommonSource module from {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


def _to_float01(img: np.ndarray) -> np.ndarray:
    if img.dtype in (np.float32, np.float64):
        x = img
    else:
        x = img.astype(np.float32) / 255.0
    return np.clip(x, 0.0, 1.0)


def _resize_cover_center_crop(img: np.ndarray, th: int, tw: int) -> np.ndarray:
    """Resize with cover strategy then center-crop to (th, tw). img: HWC, float 0..1 or uint8."""
    x = _to_float01(img)
    h, w = x.shape[:2]
    if h == th and w == tw:
        y = x
    else:
        scale = max(tw / w, th / h)
        nh, nw = int(round(h * scale)), int(round(w * scale))
        pil = Image.fromarray((x * 255.0 + 0.5).astype(np.uint8))
        pil = pil.resize((nw, nh), Image.BICUBIC)
        y = np.asarray(pil).astype(np.float32) / 255.0
        # center crop
        top = max(0, (nh - th) // 2)
        left = max(0, (nw - tw) // 2)
        y = y[top:top+th, left:left+tw]
        if y.shape[0] != th or y.shape[1] != tw:
            # pad if rounding caused off-by-one
            canvas = np.zeros((th, tw, y.shape[2]), dtype=y.dtype)
            canvas[:y.shape[0], :y.shape[1]] = y
            y = canvas
    return (y * 255.0 + 0.5).astype(np.uint8)


def _resample_to_T(frames: Union[np.ndarray, List[np.ndarray]], T: int) -> List[np.ndarray]:
    if isinstance(frames, np.ndarray):
        assert frames.ndim == 4, f"expected [T,H,W,3], got {frames.shape}"
        src = [frames[i] for i in range(frames.shape[0])]
    else:
        src = list(frames)
    if len(src) == T:
        return src
    idx = np.linspace(0, len(src) - 1, num=T)
    idx = np.clip(np.round(idx).astype(int), 0, len(src) - 1)
    return [src[i] for i in idx]


def _prep_frames_to_49_480x720(video_thwc: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray:
    """Return uint8 THWC [49,480,720,3]. Accepts float[0,1] or uint8 input."""
    # ensure list of frames
    if isinstance(video_thwc, np.ndarray):
        frames = [video_thwc[i] for i in range(video_thwc.shape[0])]
    else:
        frames = list(video_thwc)
    frames = _resample_to_T(frames, TARGET_T)
    frames = [_resize_cover_center_crop(f, TARGET_H, TARGET_W) for f in frames]
    return np.asarray(frames, dtype=np.uint8)


def _first_frame_to_pil(ff: np.ndarray) -> Image.Image:
    x = _resize_cover_center_crop(ff, TARGET_H, TARGET_W)
    return Image.fromarray(x)

# ---- temporal downsampling for noises (49 -> 13) ----

def _downsamp_mean_13(noise_t: torch.Tensor, out_T: int = F_LAT) -> torch.Tensor:
    # noise_t: [T,C,H,W]
    T = noise_t.shape[0]
    idx = torch.linspace(0, T - 1, steps=out_T).round().long()
    idx = idx.clamp_(0, T - 1)
    return noise_t.index_select(0, idx)


def _normalized_noises(noise_t: torch.Tensor) -> torch.Tensor:
    # per-frame / per-channel std normalization
    return (noise_t / (noise_t.std(dim=(1,2,3), keepdim=True) + 1e-8)).to(noise_t.dtype)


def get_downtemp_noise(noise_t: torch.Tensor, mode: str = 'nearest') -> torch.Tensor:
    assert mode in {'nearest', 'blend', 'blend_norm', 'randn'}
    if mode == 'nearest':
        return _downsamp_mean_13(noise_t, F_LAT)
    elif mode == 'blend':
        return _downsamp_mean_13(noise_t, F_LAT)  # simple approximation; keep nearest for stability
    elif mode == 'blend_norm':
        x = _downsamp_mean_13(noise_t, F_LAT)
        return _normalized_noises(x)
    elif mode == 'randn':
        return torch.randn(F_LAT, noise_t.shape[1], noise_t.shape[2], noise_t.shape[3], dtype=noise_t.dtype, device=noise_t.device)
    raise AssertionError('unreachable')

# ================== Noise (pure in-memory) ==================

def compute_warped_noise_from_frames(
    video_thwc: Union[np.ndarray, List[np.ndarray]],
    *,
    save_debug: bool = False,
    save_dir: Optional[Union[str, Path]] = None,
    noise_channels: int = NOISE_C,
    resize_frames: Union[float, Tuple[float, float], None] = 0.5,   # before flow
    resize_flow: int = 8,                                           # upsample flows
    downscale_factor: int = 32,                                      # final noise H,W := 480/8,720/8
    noise_downtemp_interp: str = 'nearest',
) -> dict:
    """
    Compute warped noise (latent resolution) from **in-memory** video.

    Returns dict:
      - noise_latents: torch.Tensor [13,16,60,90] (bfloat16)
      - input_49: np.ndarray uint8 [49,480,720,3]
      - saved: dict with optional {"noise": path}
    """
    nw = _import_noise_warp()

    # Preprocess: make THWC uint8 [49,480,720,3]
    video49 = _prep_frames_to_49_480x720(video_thwc)

    # Coerce parameter types for noise_warp
    _rf = resize_frames
    if isinstance(_rf, (int, float)) and _rf is not None:
        _rf = float(_rf)  # API allows float scalar
    elif isinstance(_rf, tuple):
        _rf = tuple(_rf)
    _rflow = int(resize_flow)
    _down = int(downscale_factor)

    # Call noise_warp with THWC numpy array directly
    out = nw.get_noise_from_video(
        video_path=video49,              # THWC numpy array
        noise_channels=int(noise_channels),
        output_folder=None,
        visualize=False,
        resize_frames=_rf,
        resize_flow=_rflow,
        downscale_factor=_down,
        save_files=False,
        remove_background=False,
    )

    noises_np = np.asarray(out.numpy_noises)  # [49,60,90,16]
    noise_t = torch.tensor(noises_np, dtype=DTYPE)            # [49,60,90,16]
    noise_t = einops.rearrange(noise_t, 't h w c -> t c h w') # [49,16,60,90]
    noise_t13 = get_downtemp_noise(noise_t, noise_downtemp_interp)

    saved = {}
    if save_debug and save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(str(save_dir / 'noise.npy'), noises_np)
        saved = {"noise": str(save_dir / 'noise.npy')}

    return dict(noise_latents=noise_t13, input_49=video49, saved=saved)


def _pick_local_base_dir(hub_id: str) -> str:
    """Return a local snapshot containing transformer, text_encoder, and vae."""
    from huggingface_hub import scan_cache_dir

    # Prefer a complete Hugging Face cache snapshot.
    info = scan_cache_dir()
    for r in info.repos:
        if getattr(r, "repo_type", "model") == "model" and r.repo_id == hub_id:
            for rev in r.revisions:
                if getattr(rev, "commit_hash", None) != COGVIDEOX_I2V_REVISION:
                    continue
                sp = getattr(rev, "snapshot_path", None)
                if (
                    sp
                    and (sp / "transformer").is_dir()
                    and (sp / "text_encoder").is_dir()
                    and (sp / "vae").is_dir()
                ):
                    return str(sp)

    raise FileNotFoundError(
        f"Cannot find the pinned local base model snapshot for {hub_id} "
        f"at revision {COGVIDEOX_I2V_REVISION} "
        "(missing transformer/text_encoder/vae). Run "
        "`python scripts/download_ckpt.py --name gowiththeflow`."
    )


def resolve_lora_path(name: str) -> Optional[str]:
    """Resolve a Go-with-the-Flow LoRA name/file against local release paths."""
    direct = Path(name).expanduser()
    if direct.is_file():
        return str(direct.resolve())
    if not direct.is_absolute():
        repo_relative = REPO_ROOT / direct
        if repo_relative.is_file():
            return str(repo_relative.resolve())
    filename = name if name.endswith('.safetensors') else f"{name}.safetensors"

    cand_dirs = []
    for env_key in ["GWF_LORA_DIR", "LORA_DIR"]:
        d = os.environ.get(env_key)
        if d:
            cand_dirs.append(d)

    cand_dirs.append(REPO_ROOT / "lora_models")

    for d in cand_dirs:
        p = Path(d) / filename
        if p.is_file():
            return str(p)

    return None

# ================== Pipeline Loader ==================

def get_pipe(
    model_name: str,
    *,
    device: Optional[Union[str, int]] = None,
    low_vram: bool = True,
):
    """Load CogVideoX I2V + optional Go-with-the-Flow LoRA (local-first).

    - If `model_name` is one of PIPE_IDS keys -> base I2V model only.
    - Else it's treated as a **LoRA key** (e.g. "I2V5B_final_i38800_nearest_lora_weights").
      We'll try to load the base I2V pipe and then attach the LoRA weights from disk.
    """
    # --------- decide base pipe vs lora ---------
    if model_name in PIPE_IDS:
        lora_name = None
        lora_path = None
        pipe_key = model_name
    else:
        lora_name = model_name
        pipe_key = Path(lora_name).name.split('_')[0]
        lora_path = resolve_lora_path(lora_name)
        if lora_path is None:
            raise FileNotFoundError(
                f"Cannot find Go-with-the-Flow LoRA weights for {lora_name}. "
                "Run `python scripts/download_ckpt.py --name gowiththeflow` "
                "or set GWF_LORA_DIR / pass --gowiththeflow_lora."
            )

    is_i2v = "I2V" in pipe_key
    hub_id = PIPE_IDS[pipe_key]
    dev_str = str(device) if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    cache_key = (pipe_key, lora_path, bool(low_vram), dev_str)
    #print("model name:", model_name)

    # Reuse an in-process pipeline if available.
    if cache_key in _PIPE_CACHE:
        return _PIPE_CACHE[cache_key]

    # --------- create base pipe ---------
    local_dir = _pick_local_base_dir(hub_id)
    transformer = CogVideoXTransformer3DModel.from_pretrained(local_dir, subfolder="transformer", torch_dtype=DTYPE)
    text_encoder = T5EncoderModel.from_pretrained(local_dir, subfolder="text_encoder", torch_dtype=DTYPE)
    vae = AutoencoderKLCogVideoX.from_pretrained(local_dir, subfolder="vae", torch_dtype=DTYPE)

    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
        local_dir,
        torch_dtype=DTYPE,
        vae=vae,
        transformer=transformer,
        text_encoder=text_encoder,
    )

    # --------- attach LoRA if requested ---------
    if lora_path is not None:
        print("LOADING LORA WEIGHTS...", lora_path, flush=True)
        pipe.load_lora_weights(lora_path)
        pipe._active_lora = lora_name
        print("DONE!")

    # --------- device / offload ---------
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if low_vram:
        pipe = pipe.to('cpu')
        pipe.enable_sequential_cpu_offload(device=device)
    else:
        pipe = pipe.to(device)

    pipe.is_i2v = is_i2v
    _PIPE_CACHE[cache_key] = pipe
    return pipe


def _coerce_video_item_to_numpy_thwc(item):
    """Convert one pipeline output item to a uint8 THWC ndarray."""
    import numpy as np

    if isinstance(item, np.ndarray):
        arr = item
        if arr.ndim == 3:
            arr = arr[None, ...]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return arr

    frames = []
    for f in item:
        a = np.asarray(f)
        if a.ndim == 2:
            a = np.repeat(a[..., None], 3, axis=-1)
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        frames.append(a)
    return np.stack(frames, axis=0)

# ================== Public Batched API ==================
def clear_pipe_cache():
    import gc
    _PIPE_CACHE.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def go_with_the_flow_api(
    refs: Union[torch.Tensor, np.ndarray],          # [B,F,H,W,3]  (0..1 float or uint8)
    fr1s: Union[torch.Tensor, np.ndarray],          # [B,H,W,3]
    caps: Optional[Sequence[str]] = None,           # len B
    model_name: str = 'I2V5B_final_i38800_nearest_lora_weights',
    pipe=None,
    device: Optional[Union[str, int]] = None,
    low_vram: bool = True,
    num_inference_steps: int = 3,
    guidance_scale: float = 6.0,
    noise_downtemp_interp: str = 'nearest',
    resize_frames: Union[float, Tuple[float, float], None] = 0.5,
    resize_flow: int = 8,
    downscale_factor: int = 32,
    save_debug: bool = False,
    save_dirs: Optional[Sequence[Optional[Union[str, Path]]]] = None,  # per-item folder for noise.npy if desired
) -> torch.Tensor:
    # Convert to numpy
    refs_np = refs.detach().cpu().numpy() if isinstance(refs, torch.Tensor) else np.asarray(refs)
    fr1s_np = fr1s.detach().cpu().numpy() if isinstance(fr1s, torch.Tensor) else np.asarray(fr1s)

    assert refs_np.ndim == 5 and refs_np.shape[-1] == 3, f"refs must be [B,F,H,W,3], got {refs_np.shape}"
    assert fr1s_np.ndim == 4 and fr1s_np.shape[-1] == 3, f"fr1s must be [B,H,W,3], got {fr1s_np.shape}"

    B = refs_np.shape[0]
    caps = list(caps) if caps is not None else [""] * B
    assert len(caps) == B, f"len(caps)={len(caps)} must equal B={B}"

    if save_dirs is None:
        save_dirs = [None] * B
    else:
        assert len(save_dirs) == B

    noise_list: List[torch.Tensor] = []
    first_images_pil: List[Image.Image] = []

    for b in range(B):
        frames = refs_np[b]  # THWC*F -> [F,H,W,3]
        # compute warped noises per sample (in-memory)
        out = compute_warped_noise_from_frames(
            frames,
            save_debug=save_debug,
            save_dir=save_dirs[b],
            noise_channels=NOISE_C,
            resize_frames=resize_frames,
            resize_flow=resize_flow,
            downscale_factor=downscale_factor,
            noise_downtemp_interp=noise_downtemp_interp,
        )
        noise_list.append(out['noise_latents'])

        # first frame PIL
        first_images_pil.append(_first_frame_to_pil(fr1s_np[b]))

    latents = torch.stack(noise_list, dim=0).to(DTYPE)  # [B,13,16,60,90]

    if pipe is None:
        pipe = get_pipe(model_name=model_name, device=device, low_vram=low_vram)
    assert pipe.is_i2v, "Use an I2V model_name (contains 'I2V')."

    result = pipe(
        prompt=caps,
        image=first_images_pil,
        latents=latents,
        num_inference_steps=int(num_inference_steps),
        guidance_scale=float(guidance_scale),
    )

    # Convert pipeline frames to [B,T,H,W,3] torch float in [0,1].
    frames_list = result.frames
    videos_np_list = [_coerce_video_item_to_numpy_thwc(v) for v in frames_list]  # List[np.ndarray[T,H,W,3]]
    videos = torch.from_numpy(np.stack(videos_np_list, axis=0)).float().div(255.0)  # [B,T,H,W,3] in [0,1]

    return videos
