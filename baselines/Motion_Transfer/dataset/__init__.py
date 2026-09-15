# -*- coding: utf-8 -*-
"""DataLoader wrapper for single-GPU PhysInOne MotionTransfer inference."""
from __future__ import annotations

from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from .PhysInOne_Dataset import PhysInOne_Leaderboard_MotionTransfer


def _can_stack(tensors: List[torch.Tensor]) -> bool:
    return bool(tensors) and len({tuple(x.shape) for x in tensors}) == 1


class DataLoader:
    def __init__(
        self,
        dataset: PhysInOne_Leaderboard_MotionTransfer,
        batch_size: int = 1,
        num_workers: int = 0,
        shuffle: bool = False,
        drop_last: bool = False,
        pin_memory: bool = True,
        persistent_workers: bool = False,
    ) -> None:
        self.loader = TorchDataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            drop_last=drop_last,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers if num_workers > 0 else False,
            collate_fn=self._collate_channel_last,
        )

    @staticmethod
    def _collate_channel_last(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        refs = [b["reference_video"] for b in batch]
        firsts = [b["first_frame"] for b in batch]

        refs_b = torch.stack(refs, dim=0) if _can_stack(refs) else refs
        firsts_b = torch.stack(firsts, dim=0) if _can_stack(firsts) else firsts

        return {
            "reference_video": refs_b,
            "first_frame": firsts_b,
            "complexity": [b["complexity"] for b in batch],
            "camera": [b["camera"] for b in batch],
            "scene_name": [b["scene_name"] for b in batch],
            "caption": [b.get("caption", "") for b in batch],
        }
