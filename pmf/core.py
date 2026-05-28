import os
import torch
from typing import Optional
import torch

EPS = 1e-8

def _ensure_5d_b_t_c_h_w(video: torch.Tensor) -> torch.Tensor:
    """
    Ensure input tensor is in (B, T, C, H, W) format.
    Automatically detects and converts (B, C, T, H, W) if C is 1 or 3.
    """
    if video.ndim == 5:
        # Heuristic: if dim1 looks like channels (1 or 3) and dim2 is clearly temporal
        if video.shape[1] in (1, 3) and video.shape[2] > video.shape[1]:
            return video.permute(0, 2, 1, 3, 4).contiguous()
        return video
    elif video.ndim == 4:
        return video.unsqueeze(0)  # (T, C, H, W) -> (1, T, C, H, W)
    elif video.ndim == 3:
        return video.unsqueeze(0).unsqueeze(0)  # (T, H, W) -> (1, 1, T, H, W)
    else:
        raise ValueError(f"Expected 3D, 4D or 5D tensor, got {video.ndim}D with shape {video.shape}")


def align_pred_to_gt(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    Align prediction video to ground truth in time and space using interpolation.
    Expects both tensors in (B, T, C, H, W) format.
    """
    B, T_p, C, H_p, W_p = pred.shape
    _, T_g, _, H_g, W_g = gt.shape

    # --- Temporal alignment ---
    if T_p != T_g:
        # Reshape to (N, 1, T) for 1D interpolation
        pred = pred.permute(0, 2, 3, 4, 1).contiguous()  # (B, C, H_p, W_p, T_p)
        pred = pred.view(B * C * H_p * W_p, 1, T_p)
        pred = torch.nn.functional.interpolate(
            pred, size=T_g, mode='linear', align_corners=False
        )
        pred = pred.view(B, C, H_p, W_p, T_g).permute(0, 4, 1, 2, 3).contiguous()  # (B, T_g, C, H_p, W_p)

    # --- Spatial alignment ---
    if H_p != H_g or W_p != W_g:
        B, T, C, _, _ = pred.shape
        pred = pred.view(B * T, C, H_p, W_p)
        pred = torch.nn.functional.interpolate(
            pred, size=(H_g, W_g), mode='bilinear', align_corners=False
        )
        pred = pred.view(B, T, C, H_g, W_g)

    return pred


def _compute_fft(video: torch.Tensor, device: Optional[str] = None) -> torch.Tensor:
    """
    Compute 3D FFT over the temporal (T) and spatial (H, W) dimensions.
    Returns complex tensor of shape (B, T, C, H, W).
    """
    if device is not None:
        video = video.to(device)
    video = video.to(torch.float32)
    return torch.fft.fftn(video, dim=(1, 3, 4))


def _compute_3d_energy_difference(
    fft_gt: torch.Tensor, 
    fft_out: torch.Tensor,
    eps: float = EPS
) -> torch.Tensor:
    """
    Compute transformed Total Variation distance between 3D energy distributions.
    Higher score = higher similarity.
    """
    B, T, C, H, W = fft_gt.shape
    N = T * H * W

    # Power spectral density (averaged over channels)
    power_gt = 0.5 * torch.abs(fft_gt).pow(2).mean(dim=2)  # (B, T, H, W)
    power_out = 0.5 * torch.abs(fft_out).pow(2).mean(dim=2)

    # Center zero-frequency components
    power_gt = torch.fft.fftshift(power_gt, dim=(1, 2, 3))
    power_out = torch.fft.fftshift(power_out, dim=(1, 2, 3))

    # Flatten and normalize to valid probability distributions
    P_gt = power_gt.view(B, N)
    P_out = power_out.view(B, N)
    
    P_gt = P_gt / (P_gt.sum(dim=1, keepdim=True) + eps)
    P_out = P_out / (P_out.sum(dim=1, keepdim=True) + eps)

    # Total Variation distance: 0.5 * L1 norm (bounded in [0, 1])
    tv_distance = 0.5 * torch.abs(P_gt - P_out).sum(dim=1)  # (B,)

    # -log transform to amplify sensitivity to small differences
    transformed_tv_distance = -torch.log(torch.clamp(tv_distance, min=eps))

    return transformed_tv_distance


def compute_pmf(ground_truth: torch.Tensor, output: torch.Tensor, device: str = 'cpu') -> torch.Tensor:
    """
    Compute the PMF similarity metric between two videos.
    
    Args:
        ground_truth: (B, T, C, H, W) or (B, C, T, H, W)
        output: same shape convention as ground_truth
        device: 'cpu' or 'cuda'
        
    Returns:
        similarity_scores: (B,) tensor, higher = more similar
    """
    gt = _ensure_5d_b_t_c_h_w(ground_truth).to(device)
    pred = _ensure_5d_b_t_c_h_w(output).to(device)

    # Align prediction to ground truth dimensions
    pred = align_pred_to_gt(pred=pred, gt=gt)
    
    # Compute FFTs
    fft_gt = _compute_fft(gt, device=device)
    fft_out = _compute_fft(pred, device=device)

    return _compute_3d_energy_difference(fft_gt, fft_out)