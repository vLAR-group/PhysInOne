import gc
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

sys.dont_write_bytecode = True  # Avoid __pycache__

from sgm.util import instantiate_from_config

REPO_ROOT = Path(__file__).resolve().parents[1]
COTRACKER_ROOT = REPO_ROOT / "tools" / "co-tracker"
if str(COTRACKER_ROOT) not in sys.path:
    sys.path.insert(0, str(COTRACKER_ROOT))

from cotracker.predictor import CoTrackerPredictor


class MotionProDenseRunner:
    """Persistent MotionPro-Dense and CoTracker inference runner."""

    def __init__(
        self,
        *,
        ckpt_path: str,
        config_path: str,
        cotracker_ckpt: str,
        seed: int = 2025,
        fps: int = 8,
        num_frames: Optional[int] = None,
        dtype: torch.dtype = torch.float16,
        device: Optional[torch.device] = None,
    ) -> None:
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dtype = dtype
        self.fps = fps
        self.num_frames = num_frames

        seed_everything(seed)
        self.cotracker = CoTrackerPredictor(checkpoint=cotracker_ckpt).to(self.device)
        self.cotracker.eval()

        config = OmegaConf.load(config_path)
        model_cfg = config.get("model")
        if model_cfg is None:
            raise ValueError(f"Missing `model` section in config: {config_path}")
        model_cfg.params.ckpt_path = ckpt_path

        inference_cfg = config.get("inference", OmegaConf.create())
        log_images_cfg = inference_cfg.get("log_images_kwargs", OmegaConf.create())
        self.log_images_kwargs = dict(
            OmegaConf.to_container(log_images_cfg, resolve=True) or {}
        )

        self.model = instantiate_from_config(model_cfg).to(
            dtype=self.dtype, device=self.device
        )
        self.model.eval()

    @torch.no_grad()
    def __call__(
        self,
        reference_video: torch.Tensor,
        first_frame: torch.Tensor,
    ) -> torch.Tensor:
        if reference_video.ndim != 5 or reference_video.shape[-1] != 3:
            raise ValueError(
                "reference_video must be [B,F,H,W,3], got "
                f"{tuple(reference_video.shape)}"
            )
        if first_frame.ndim != 4 or first_frame.shape[-1] != 3:
            raise ValueError(
                f"first_frame must be [B,H,W,3], got {tuple(first_frame.shape)}"
            )

        batch_size, input_frames, height, width, _ = reference_video.shape
        if first_frame.shape[:3] != (batch_size, height, width):
            raise ValueError("first_frame must share (B,H,W) with reference_video")

        ref = reference_video.to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )
        first = first_frame.to(
            device=self.device, dtype=torch.float32, non_blocking=True
        )

        if (
            self.num_frames is not None
            and self.num_frames > 0
            and self.num_frames != input_frames
        ):
            indices = _uniform_indices(
                input_frames, self.num_frames, device=ref.device
            )
            ref = ref.index_select(dim=1, index=indices)

        frame_count = ref.shape[1]
        ref_cf = ref.permute(0, 1, 4, 2, 3).contiguous()
        obj_flow, vis_mask = _build_dense_flow_with_cotracker(
            ref_cf, cotracker=self.cotracker
        )

        frames = first[:, None, ...].expand(
            batch_size, frame_count, height, width, 3
        )
        video_chw = frames.mul(2.0).sub(1.0).permute(0, 4, 1, 2, 3)
        motion_bucket = torch.mean(
            torch.linalg.norm(obj_flow, dim=-1), dim=(1, 2, 3)
        )

        batch = {
            "video": video_chw.to(self.dtype),
            "flow_ori": obj_flow.to(self.dtype),
            "vis_mask_sq": vis_mask.to(self.dtype),
            "caption": None,
            "fps_id": torch.full(
                (batch_size,),
                float(self.fps),
                device=self.device,
                dtype=self.dtype,
            ),
            "motion_bucket_id": motion_bucket.to(
                device=self.device, dtype=self.dtype
            ),
            "video_name": "final",
        }
        log_images_kwargs = dict(self.log_images_kwargs)
        log_images_kwargs["N"] = batch_size
        sample_results: Dict[str, Any] = self.model.log_images(
            batch, split="test", use_optimization=True, **log_images_kwargs
        )
        return _extract_video_from_log_images(sample_results, expect_b=batch_size)

    def close(self) -> None:
        self.model = None
        self.cotracker = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def convert_pred_points_into_flow(pred_tracks, video_interp, pred_visibility):
    """
    Map sparse tracks to a per-pixel flow field on query grid positions
    pred_tracks: [T, N, 2] where N queried points, T frames
    video_interp: [B, T, C, H, W] (here B=1)
    pred_visibility: [T, N]
    returns:
        opt_flow: [T, H, W, 2]
        vis_mask_final: [T, H, W, 1] (bool)
    """
    opt_flow = (pred_tracks - pred_tracks[0][None])
    opt_flow = opt_flow.permute(1, 0, 2)  # [N, T, 2]
    vis_mask_sq = pred_visibility.unsqueeze(-1).permute(1, 0, 2)  # [N, T, 1]
    n, t, _ = opt_flow.shape
    _, _, h, w = video_interp.shape

    device = pred_tracks.device if torch.is_tensor(pred_tracks) else "cpu"
    opt_final      = torch.zeros((t, h, w, 2), device=device, dtype=torch.float32)  # [T,H,W,2]
    vis_mask_final = torch.zeros((t, h, w, 1), device=device, dtype=torch.bool)     # [T,H,W,1]

    for i in range(n):
        x = int(pred_tracks[0, i, 0])
        y = int(pred_tracks[0, i, 1])
        if 0 <= x < w and 0 <= y < h:
            opt_final[:, y, x, :] = opt_flow[i]
            vis_mask_final[:, y, x, :] = vis_mask_sq[i]

    return opt_final, vis_mask_final # [T,H,W,2], [T,H,W,1]

def _uniform_indices(n: int, k: int, device=None) -> torch.Tensor:
    if k <= 0:
        return torch.arange(n, device=device)
    if k == n:
        return torch.arange(n, device=device)
    idx = torch.linspace(0, n-1, steps=k, device=device).round().to(torch.long)
    return idx


def _build_dense_flow_with_cotracker(
    ref_cf: torch.Tensor,                          # [B,F,3,H,W] 0~1, fp32
    *, cotracker: CoTrackerPredictor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        flow: [B,F,H,W,2] xy displacement in pixels.
        mask: [B,F,H,W,1] visibility mask.

    The implementation is fully in-memory. Tracking or shape errors are raised
    because zero-flow substitution would silently change the requested method.
    """
    assert ref_cf.ndim == 5 and ref_cf.shape[2] == 3, f"ref_cf must be [B,F,3,H,W], got {tuple(ref_cf.shape)}"
    B, F, _, H, W = ref_cf.shape
    device = ref_cf.device

    pred_tracks, pred_visibility = cotracker(
        ref_cf, grid_query_frame=0, backward_tracking=False
    )

    flow_list: List[torch.Tensor] = []
    mask_list: List[torch.Tensor] = []

    for b in range(B):
        tracks_b = pred_tracks[b]
        vis_b = pred_visibility[b]
        flow_b, mask_b = convert_pred_points_into_flow(
            tracks_b, ref_cf[b], vis_b
        )
        flow_b = flow_b.to(device=device, dtype=torch.float32)
        mask_b = mask_b.to(device=device, dtype=torch.float32)
        if mask_b.ndim == 3:
            mask_b = mask_b.unsqueeze(-1)

        if flow_b.shape != (F, H, W, 2):
            raise RuntimeError(
                "Optical flow shape mismatch: "
                f"got {tuple(flow_b.shape)}, expected {(F, H, W, 2)}"
            )
        if mask_b.shape != (F, H, W, 1):
            raise RuntimeError(
                "Visibility mask shape mismatch: "
                f"got {tuple(mask_b.shape)}, expected {(F, H, W, 1)}"
            )

        flow_list.append(flow_b)
        mask_list.append(mask_b)

    flow = torch.stack(flow_list, dim=0)
    mask = torch.stack(mask_list, dim=0)
    return flow.contiguous(), mask.contiguous()

def _extract_video_from_log_images(sample_results, expect_b: int | None = None,
                                   prefer_key: str | None = "samples-video") -> torch.Tensor:
    """
    Extract a video tensor from model.log_images and normalize it to
    [B,F,H,W,3] in [0,1].
    """
    candidates = []

    if prefer_key and prefer_key in sample_results:
        v = sample_results[prefer_key]
        if isinstance(v, torch.Tensor):
            candidates.append(v)

    for k in ("video", "samples", "final", "output", "out", "reconstructions", "reconstruction"):
        v = sample_results.get(k)
        if isinstance(v, torch.Tensor):
            candidates.append(v)
        elif isinstance(v, (list, tuple)) and v and isinstance(v[0], torch.Tensor):
            candidates.append(v[0])
        elif isinstance(v, dict):
            vv = v.get("video")
            if isinstance(vv, torch.Tensor):
                candidates.append(vv)

    if not candidates:
        for v in sample_results.values():
            if isinstance(v, torch.Tensor) and v.ndim == 5:
                candidates.append(v)
                break

    if not candidates:
        raise RuntimeError("Cannot find 5D video tensor in sample_results; "
                           "try passing prefer_key='reconstructions-video' explicitly.")

    vid = candidates[0]

    if vid.ndim == 5 and vid.shape[1] in (1, 3):
        out = vid.permute(0, 2, 3, 4, 1).contiguous()
    elif vid.ndim == 5 and vid.shape[-1] == 3:
        out = vid.contiguous()
    elif vid.ndim == 4 and expect_b == 1:
        if vid.shape[0] in (1, 3):
            out = vid.permute(1, 2, 3, 0).unsqueeze(0).contiguous()
        elif vid.shape[-1] == 3:
            out = vid.unsqueeze(0).contiguous()
        else:
            raise RuntimeError(f"Unsupported 4D shape: {tuple(vid.shape)}")
    else:
        raise RuntimeError(f"Unsupported video tensor shape: {tuple(vid.shape)}")

    out = out.to(torch.float32)
    if out.min() < 0.0:
        out = (out + 1.0) * 0.5
    out.clamp_(0.0, 1.0)
    return out
