<p align="center">
  <h1 align="center">PhysInOne: Visual Physics Learning and Reasoning in One Suite</h1>
  <p align="center">
    <strong>CVPR 2026</strong>
  </p>
  <p align="center">
    <a href="https://arxiv.org/pdf/2604.09415"><img src="https://img.shields.io/badge/arXiv-2604.09415-b31b1b.svg" alt="arXiv"></a>
    <a href="https://vlar-group.github.io/PhysInOne.html"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
    <a href="https://huggingface.co/datasets/vLAR/PhysInOne"><img src="https://img.shields.io/badge/🤗-Dataset-yellow" alt="Dataset"></a>
    <a href="https://huggingface.co/datasets/vLAR/PhysInOne"><img src="https://img.shields.io/badge/All%20repo%20downloads-728%2C039-FFD21E?logo=huggingface" alt="All PhysInOne repositories: 728,039 historical downloads"></a>
    <a href="#license"><img src="https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg" alt="License"></a>
  </p>
</p>

<p align="center">
  <img src="teaser.png" alt="PhysInOne Teaser" width="100%">
</p>

## Overview

We present **PhysInOne**, the largest dataset addressing the critical scarcity of physically-grounded training data for AI systems.

### Scale and Diversity

- **2 million videos** generated from **153,810 dynamic 3D scenes**
- Covers **71 fundamental physical phenomena** in everyday environments, spanning four major domains: **Mechanics**, **Optics**, **Fluid Dynamics**, **Magnetism**
- Includes **2,231 common objects** tailored to daily physical interactions
- Enriched with **623 materials** across five categories: plastic, metal, wood, stone, and fabric
- Features **528 diverse 3D backgrounds** to ensure realism and environmental variety

### Scene Characteristics

- Each scene involves **1–3 physical phenomena**, reflecting real-world activities
- Supports **complex multi-object interactions**, with increasing scene complexity
- Average number of objects per scene: **3.9** (single-physics), **6.3** (double-physics), **7.8** (triple-physics)
- Each scene is captured from **13 viewpoints**: 12 static cameras and 1 moving camera

### Rich Annotations

- 3D geometry
- Semantic labels
- Object motion and dynamics
- Physical properties
- Natural-language scene descriptions

### Supported Applications

- Physics-aware video generation
- Short- and long-term future frame prediction
- Physical property estimation
- Motion transfer
- And more...

## 🚀 Release Timetable

| Component             | Progress                      | Status       | Notes                             |
| --------------------- | ----------------------------- | ------------ | --------------------------------- |
| SubSet                | `██████████`100%              | Released     |                                   |
| Rendered Data - Train | `██████████` 100%(122988/122988) | Released     | Last updated: Aug 21              |
| Rendered Data - Test  | `░░░░░░░░░░` 0%(0/15411)      | In progress  |                                   |
| Rendered Data - Val   | `░░░░░░░░░░` 1%(103/15411)    | In progress  |                                   |
| 3D Assets             | `░░░░░░░░░░` 0%               | Not released | Expected around Aug              |
| Leaderboard           | `░░░░░░░░░░` 0%               | Ongoing      | Link will be added when available |
| PMF                   | `██████████`100%              | Released     |                                   |
| Baselines             | `███░░░░░░░`25%               | In progress | Last updated: Jul 23              |
| Data processing       | `░░░░░░░░░░` 0%               | Not released | Expected around Aug              |

## Links

| Resource        | Link                                                                          |
| --------------- | ----------------------------------------------------------------------------- |
| 📄 Paper        | [arXiv](https://arxiv.org/pdf/2604.09415)                                     |
| 🌐 Project Page | [vlar-group.github.io/PhysInOne](https://vlar-group.github.io/PhysInOne.html) |
| 🤗 Dataset      | [Hugging Face](https://huggingface.co/datasets/vLAR/PhysInOne)                |

### Shard Repositories

Due to the large scale of PhysInOne, the rendered data and annotations are split across multiple Hugging Face repositories.

- PhysInOneP01: https://huggingface.co/datasets/PhysInOneP01/PhysInOneP01 4.52 TB ( Complete )
- PhysInOneP02: https://huggingface.co/datasets/PhysInOneP02/PhysInOneP02 7.09 TB ( Complete )
- PhysInOneP03: https://huggingface.co/datasets/PhysInOneP03/PhysInOneP03 7.59 TB ( Complete )
- PhysInOneP04: https://huggingface.co/datasets/PhysInOneP04/PhysInOneP04 7.58 TB ( Complete )
- PhysInOneP05: https://huggingface.co/datasets/PhysInOneP05/PhysInOneP05 7.16 TB ( Complete )
- PhysInOneP06: https://huggingface.co/datasets/PhysInOneP06/PhysInOneP06 7.19 TB ( Complete )
- PhysInOneP07: https://huggingface.co/datasets/PhysInOneP07/PhysInOneP07 7.21 TB ( Complete )
- PhysInOneP08: https://huggingface.co/datasets/PhysInOneP08/PhysInOneP08 7.21 TB ( Complete )
- PhysInOneP09: https://huggingface.co/datasets/PhysInOneP09/PhysInOneP09 7.20 TB ( Complete )
- PhysInOneP10: https://huggingface.co/datasets/PhysInOneP10/PhysInOneP10 7.25 TB ( Complete )
- PhysInOneP11: https://huggingface.co/datasets/PhysInOneP11/PhysInOneP11 7.48 TB ( Complete )
- PhysInOneP12: https://huggingface.co/datasets/PhysInOneP12/PhysInOneP12 6.68 TB ( Complete )
- PhysInOneP13: https://huggingface.co/datasets/PhysInOneP13/PhysInOneP13 6.66 TB ( Complete )
- PhysInOneP14: https://huggingface.co/datasets/PhysInOneP14/PhysInOneP14 6.72 TB ( Complete )
- PhysInOneP15: https://huggingface.co/datasets/PhysInOneP15/PhysInOneP15 7.93 TB ( Complete )
- PhysInOneP16: https://huggingface.co/datasets/PhysInOneP16/PhysInOneP16 1.57 TB ( Complete )

## 📈 Download Statistics

The badges below query the official Hugging Face API and update automatically.

> **Combined snapshot (Aug 21, 2026):** P01-P16 have **714,894** all-time downloads and **450,894** downloads in the last 30 days. Including the main repository, the corresponding per-repository sums are **728,039** and **452,403**.

| Repository | All-time downloads | Last 30 days |
| --- | ---: | ---: |
| [Main](https://huggingface.co/datasets/vLAR/PhysInOne) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FvLAR%2FPhysInOne%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/vLAR/PhysInOne) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FvLAR%2FPhysInOne%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/vLAR/PhysInOne) |
| [P01](https://huggingface.co/datasets/PhysInOneP01/PhysInOneP01) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP01%2FPhysInOneP01%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP01/PhysInOneP01) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP01%2FPhysInOneP01%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP01/PhysInOneP01) |
| [P02](https://huggingface.co/datasets/PhysInOneP02/PhysInOneP02) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP02%2FPhysInOneP02%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP02/PhysInOneP02) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP02%2FPhysInOneP02%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP02/PhysInOneP02) |
| [P03](https://huggingface.co/datasets/PhysInOneP03/PhysInOneP03) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP03%2FPhysInOneP03%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP03/PhysInOneP03) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP03%2FPhysInOneP03%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP03/PhysInOneP03) |
| [P04](https://huggingface.co/datasets/PhysInOneP04/PhysInOneP04) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP04%2FPhysInOneP04%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP04/PhysInOneP04) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP04%2FPhysInOneP04%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP04/PhysInOneP04) |
| [P05](https://huggingface.co/datasets/PhysInOneP05/PhysInOneP05) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP05%2FPhysInOneP05%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP05/PhysInOneP05) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP05%2FPhysInOneP05%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP05/PhysInOneP05) |
| [P06](https://huggingface.co/datasets/PhysInOneP06/PhysInOneP06) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP06%2FPhysInOneP06%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP06/PhysInOneP06) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP06%2FPhysInOneP06%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP06/PhysInOneP06) |
| [P07](https://huggingface.co/datasets/PhysInOneP07/PhysInOneP07) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP07%2FPhysInOneP07%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP07/PhysInOneP07) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP07%2FPhysInOneP07%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP07/PhysInOneP07) |
| [P08](https://huggingface.co/datasets/PhysInOneP08/PhysInOneP08) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP08%2FPhysInOneP08%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP08/PhysInOneP08) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP08%2FPhysInOneP08%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP08/PhysInOneP08) |
| [P09](https://huggingface.co/datasets/PhysInOneP09/PhysInOneP09) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP09%2FPhysInOneP09%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP09/PhysInOneP09) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP09%2FPhysInOneP09%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP09/PhysInOneP09) |
| [P10](https://huggingface.co/datasets/PhysInOneP10/PhysInOneP10) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP10%2FPhysInOneP10%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP10/PhysInOneP10) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP10%2FPhysInOneP10%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP10/PhysInOneP10) |
| [P11](https://huggingface.co/datasets/PhysInOneP11/PhysInOneP11) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP11%2FPhysInOneP11%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP11/PhysInOneP11) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP11%2FPhysInOneP11%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP11/PhysInOneP11) |
| [P12](https://huggingface.co/datasets/PhysInOneP12/PhysInOneP12) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP12%2FPhysInOneP12%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP12/PhysInOneP12) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP12%2FPhysInOneP12%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP12/PhysInOneP12) |
| [P13](https://huggingface.co/datasets/PhysInOneP13/PhysInOneP13) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP13%2FPhysInOneP13%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP13/PhysInOneP13) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP13%2FPhysInOneP13%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP13/PhysInOneP13) |
| [P14](https://huggingface.co/datasets/PhysInOneP14/PhysInOneP14) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP14%2FPhysInOneP14%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP14/PhysInOneP14) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP14%2FPhysInOneP14%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP14/PhysInOneP14) |
| [P15](https://huggingface.co/datasets/PhysInOneP15/PhysInOneP15) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP15%2FPhysInOneP15%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP15/PhysInOneP15) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP15%2FPhysInOneP15%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP15/PhysInOneP15) |
| [P16](https://huggingface.co/datasets/PhysInOneP16/PhysInOneP16) | [![total](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP16%2FPhysInOneP16%3Fexpand%3DdownloadsAllTime&query=%24.downloadsAllTime&label=total&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP16/PhysInOneP16) | [![30d](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP16%2FPhysInOneP16%3Fexpand%3Ddownloads&query=%24.downloads&label=30d&logo=huggingface&color=FFD21E&cacheSeconds=3600)](https://huggingface.co/datasets/PhysInOneP16/PhysInOneP16) |

> Counts are repository-level download events, not deduplicated users. A user accessing multiple shards can be counted once in each shard.



## 💻 Code

### PMF Metric
The **PMF (Physical Motion Fidelity)** evaluates video similarity in the frequency domain using 3D FFT-based energy distributions. 
It is designed for physics-aware video generation, future prediction, and motion transfer tasks in the PhysInOne benchmark.

#### Install via pip (Recommended)
```bash
# Step 1: Install PyTorch first (choose your variant)
# CPU only:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# CUDA 12.6:
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Step 2: Install pmf from this repo
pip install git+https://github.com/vLAR-group/PhysInOne.git#subdirectory=pmf
```

#### Demo
```bash
#!/usr/bin/env python
"""Test PMF metric with random tensors."""

import torch
from pmf import compute_pmf

def main():
    torch.manual_seed(42)
    B, T, C, H, W = 1, 81, 3, 128, 128
    video_pred = torch.randn(B, T, C, H, W)
    video_gt = torch.randn(B, T, C, H, W)

    score = compute_pmf(video_pred, video_gt, device='cpu') 
    # If you want to use gpu, set device='cuda'
    # score = compute_pmf(video_pred, video_gt, device='cuda') 
    if isinstance(score, torch.Tensor):
        score = score.item()
        
    print(f"PMF similarity score: {score:.4f}")

if __name__ == "__main__":
    main()
```

## Baselines

We provide baseline implementations under the `./baselines` directory for your reference. We welcome your feedback, please feel free to contact us if you need anything..

> **📅 Update Schedule:** This section is actively being updated throughout July and August.

## 🚧 **Coming Soon** 🚧

Data processing code will be released soon. Stay tuned!

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhou2026physinone,
         title={PhysInOne: Visual Physics Learning and Reasoning in One Suite}, 
         author={Siyuan Zhou and Hejun Wang and Hu Cheng and Jinxi Li and Dongsheng Wang and Junwei Jiang and Yixiao Jin and Jiayue Huang and Shiwei Mao and Shangjia Liu and Yafei Yang and Hongkang Song and Shenxing Wei and Zihui Zhang and Peng Huang and Shijie Liu and Zhengli Hao and Hao Li and Yitian Li and Wenqi Zhou and Zhihan Zhao and Zongqi He and Hongtao Wen and Shouwang Huang and Peng Yun and Bowen Cheng and Pok Kazaf Fu and Wai Kit Lai and Jiahao Chen and Kaiyuan Wang and Zhixuan Sun and Ziqi Li and Haochen Hu and Di Zhang and Chun Ho Yuen and Bing Wang and Zhihua Wang and Chuhang Zou and Bo Yang},
         year={2026},
         journal={CVPR} 
}
```

## License

This project is licensed under the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license.

## Acknowledgements

We would like to express our sincere gratitude to all contributors who participated in human evaluations and data collection efforts.
