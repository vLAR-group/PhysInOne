<p align="center">
  <h1 align="center">PhysInOne: Visual Physics Learning and Reasoning in One Suite</h1>
  <p align="center">
    <strong>CVPR 2026</strong>
  </p>
  <p align="center">
    <a href="https://arxiv.org/pdf/2604.09415"><img src="https://img.shields.io/badge/arXiv-2604.09415-b31b1b.svg" alt="arXiv"></a>
    <a href="https://vlar-group.github.io/PhysInOne.html"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
    <a href="https://huggingface.co/datasets/vLAR/PhysInOne"><img src="https://img.shields.io/badge/🤗-Dataset-yellow" alt="Dataset"></a>
    <a href="https://huggingface.co/datasets/vLAR/PhysInOne"><img src="https://img.shields.io/badge/All%20repo%20downloads-999%2C927-FFD21E?logo=huggingface" alt="All PhysInOne repositories: 999,927 historical downloads (snapshot Sep 5, 2026)"></a>
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
| Rendered Data - Train | `██████████` 100%(122988/122988) | Released     | Last updated: Aug 21              |
| Rendered Data - Test  | `██████████` 100%               | Released     | All Leaderboard user inputs released; GT excluded |
| Rendered Data - Val   | `░░░░░░░░░░` 1%(103/15411)    | In progress  |                                   |
| 3D Assets             | `█████░░░░░` 50%              | Partially released | Validation scenes and Train resource libraries |
| Leaderboard           | `██████████` 100%              | Released     | Public evaluation inputs for all four tasks; GT excluded |
| PMF                   | `██████████`100%              | Released     |                                   |
| Baselines             | `█████░░░░░`50%               | In progress | Last updated: Sept 18              |
| Data processing       | `░░░░░░░░░░` 0%               | Not released | Expected around Aug              |

## Links

| Resource        | Link                                                                          |
| --------------- | ----------------------------------------------------------------------------- |
| 📄 Paper        | [arXiv](https://arxiv.org/pdf/2604.09415)                                     |
| 🌐 Project Page | [vlar-group.github.io/PhysInOne](https://vlar-group.github.io/PhysInOne.html) |
| 🤗 Dataset      | [Hugging Face](https://huggingface.co/datasets/vLAR/PhysInOne)                |
| 🏆 Leaderboard Data | [Public evaluation inputs](https://huggingface.co/datasets/vLAR/PhysInOne/tree/main/Leaderboard) |
| 🧊 3D Assets | [PhysicBenchmark project assets](https://huggingface.co/datasets/vLAR/PhysInOne/tree/main/Assets/PhysicBenchmark) |

### 🏆 Leaderboard Evaluation Data

All user-facing evaluation inputs required by the public Leaderboard have been released for the four benchmark tasks. Ground-truth outputs remain private and are not included in the public download.

| Task | Public package count |
| ---- | -------------------- |
| Video Generation | 75,865 files |
| Future Prediction | 103 scene ZIP archives |
| Physical Properties Estimation | 72 scene ZIP archives and 2 shared support files |
| Motion Transfer | 217 scene ZIP archives |

Clone this repository once; the downloader uses only the Python standard library and reads its maintained manifests directly from the repository:

```bash
git clone https://github.com/vLAR-group/PhysInOne.git
cd PhysInOne
```

A byte-identical public mirror of `scripts/` is maintained under [`Utils/scripts/`](https://huggingface.co/datasets/vLAR/PhysInOne/tree/main/Utils/scripts).

During downloads, an interactive terminal displays one live progress bar with file count, transferred and estimated total size, current speed, and ETA. Redirected output automatically switches to periodic plain-text progress updates, while complete per-file records remain available in `_download_logs/`.

Each row below is a complete command that can be copied directly. The remote Leaderboard folder names remain unchanged, while local task directories are created without spaces.

| Task | Copy command | Local directory |
| --- | --- | --- |
| Video Generation | `python scripts/download_data.py --task video-generation --output-dir ./PhysInOne_data` | `PhysInOne_data/leaderboard/video_generation/` |
| Future Prediction | `python scripts/download_data.py --task future-prediction --output-dir ./PhysInOne_data` | `PhysInOne_data/leaderboard/future_prediction/` |
| Physical Properties Estimation | `python scripts/download_data.py --task physical-properties-estimation --output-dir ./PhysInOne_data` | `PhysInOne_data/leaderboard/physical_properties_estimation/` |
| Motion Transfer | `python scripts/download_data.py --task motion-transfer --output-dir ./PhysInOne_data` | `PhysInOne_data/leaderboard/motion_transfer/` |
| All four Leaderboard tasks | `python scripts/download_data.py --task video-generation future-prediction physical-properties-estimation motion-transfer --output-dir ./PhysInOne_data` | `PhysInOne_data/leaderboard/` |

### 🧊 3D Assets

The public 3D asset release is now **50% complete**. It contains shared Unreal Engine project files, Scene and Trajectory resources for **1,000 validation scenes**, six validation resource archives, and seven training resource archives. The current release contains **4,306 downloadable files**, including **15 ZIP archives**, totaling approximately **73.87 GiB**.

The seven training archives add approximately **51.61 GiB** of backgrounds, raw backgrounds, breakable objects, interactable objects, solid objects, appearance materials, and physical materials. Training Scene maps and training Sequence/Trajectory assets are **not released yet**.

```bash
python scripts/download_data.py \
  --task 3d_assets \
  --output-dir ./PhysInOne_data
```

The dependency-free downloader currently retrieves and assembles the exact 1,000-scene validation project subset. It resumes incomplete files, validates ZIP archives, writes detailed logs, and creates local directories without spaces. The assembled project is stored at `PhysInOne_data/assets/PhysicBenchmark/PhysInOne.uproject`. The newly released training resource archives are available in their corresponding category folders in the [Hugging Face project browser](https://huggingface.co/datasets/vLAR/PhysInOne/tree/main/Assets/PhysicBenchmark/Content/PhysInOne).

Common options:

| Option | Purpose |
| --- | --- |
| `--task NAME [NAME ...]` | Select one or more public releases. |
| `--output-dir PATH` | Set a whitespace-free local output root. |
| `--scene TEXT` | Download matching scene names or six-character IDs. |
| `--workers N` | Set concurrent downloads. |
| `--extract` / `--no-extract` | Control ZIP extraction. |
| `--delete-zip-after-extract` | Save space after verified extraction. |
| `--dry-run` / `--list-only` | Preview counts or exact public URLs. |

See the [complete download guide](docs/DATA_DOWNLOAD.md), [download script](scripts/download_data.py), and [task manifests](scripts/download_lists/) for all parameters and examples. Full rendered-data subsets can be prepared with [filter_cases.py](scripts/filter_cases.py) and downloaded with [download_selected.py](scripts/download_selected.py).

### 📦 Dataset Repositories & Downloads

Due to the large scale of PhysInOne, the rendered data and annotations are split across 16 Hugging Face repositories. Each entry shows the shard size, release status, live all-time downloads, live downloads in the last 30 days, and its repository link.

> **Combined snapshot (Sep 5, 2026):** P01–P16 have **986,449** all-time downloads and **615,484** downloads in the last 30 days. Including the main repository, the per-repository sums are **999,927** and **616,472**.

<table>
<tr>
<td colspan="2" valign="top">
<strong>Main repository</strong> · <a href="https://huggingface.co/datasets/vLAR/PhysInOne"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FvLAR%2FPhysInOne%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOne total downloads"></a> <a href="https://huggingface.co/datasets/vLAR/PhysInOne"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FvLAR%2FPhysInOne%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOne 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/vLAR/PhysInOne"><code>huggingface.co/datasets/vLAR/PhysInOne</code></a>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<strong>PhysInOneP01</strong> · <strong style="color:#d32f2f;">4.52 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP01/PhysInOneP01"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP01%2FPhysInOneP01%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP01 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP01/PhysInOneP01"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP01%2FPhysInOneP01%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP01 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP01/PhysInOneP01"><code>huggingface.co/datasets/PhysInOneP01/PhysInOneP01</code></a>
</td>
<td width="50%" valign="top">
<strong>PhysInOneP02</strong> · <strong style="color:#d32f2f;">7.09 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP02/PhysInOneP02"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP02%2FPhysInOneP02%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP02 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP02/PhysInOneP02"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP02%2FPhysInOneP02%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP02 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP02/PhysInOneP02"><code>huggingface.co/datasets/PhysInOneP02/PhysInOneP02</code></a>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<strong>PhysInOneP03</strong> · <strong style="color:#d32f2f;">7.59 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP03/PhysInOneP03"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP03%2FPhysInOneP03%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP03 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP03/PhysInOneP03"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP03%2FPhysInOneP03%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP03 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP03/PhysInOneP03"><code>huggingface.co/datasets/PhysInOneP03/PhysInOneP03</code></a>
</td>
<td width="50%" valign="top">
<strong>PhysInOneP04</strong> · <strong style="color:#d32f2f;">7.58 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP04/PhysInOneP04"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP04%2FPhysInOneP04%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP04 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP04/PhysInOneP04"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP04%2FPhysInOneP04%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP04 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP04/PhysInOneP04"><code>huggingface.co/datasets/PhysInOneP04/PhysInOneP04</code></a>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<strong>PhysInOneP05</strong> · <strong style="color:#d32f2f;">7.16 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP05/PhysInOneP05"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP05%2FPhysInOneP05%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP05 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP05/PhysInOneP05"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP05%2FPhysInOneP05%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP05 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP05/PhysInOneP05"><code>huggingface.co/datasets/PhysInOneP05/PhysInOneP05</code></a>
</td>
<td width="50%" valign="top">
<strong>PhysInOneP06</strong> · <strong style="color:#d32f2f;">7.19 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP06/PhysInOneP06"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP06%2FPhysInOneP06%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP06 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP06/PhysInOneP06"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP06%2FPhysInOneP06%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP06 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP06/PhysInOneP06"><code>huggingface.co/datasets/PhysInOneP06/PhysInOneP06</code></a>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<strong>PhysInOneP07</strong> · <strong style="color:#d32f2f;">7.21 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP07/PhysInOneP07"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP07%2FPhysInOneP07%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP07 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP07/PhysInOneP07"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP07%2FPhysInOneP07%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP07 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP07/PhysInOneP07"><code>huggingface.co/datasets/PhysInOneP07/PhysInOneP07</code></a>
</td>
<td width="50%" valign="top">
<strong>PhysInOneP08</strong> · <strong style="color:#d32f2f;">7.21 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP08/PhysInOneP08"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP08%2FPhysInOneP08%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP08 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP08/PhysInOneP08"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP08%2FPhysInOneP08%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP08 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP08/PhysInOneP08"><code>huggingface.co/datasets/PhysInOneP08/PhysInOneP08</code></a>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<strong>PhysInOneP09</strong> · <strong style="color:#d32f2f;">7.20 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP09/PhysInOneP09"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP09%2FPhysInOneP09%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP09 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP09/PhysInOneP09"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP09%2FPhysInOneP09%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP09 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP09/PhysInOneP09"><code>huggingface.co/datasets/PhysInOneP09/PhysInOneP09</code></a>
</td>
<td width="50%" valign="top">
<strong>PhysInOneP10</strong> · <strong style="color:#d32f2f;">7.25 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP10/PhysInOneP10"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP10%2FPhysInOneP10%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP10 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP10/PhysInOneP10"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP10%2FPhysInOneP10%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP10 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP10/PhysInOneP10"><code>huggingface.co/datasets/PhysInOneP10/PhysInOneP10</code></a>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<strong>PhysInOneP11</strong> · <strong style="color:#d32f2f;">7.48 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP11/PhysInOneP11"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP11%2FPhysInOneP11%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP11 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP11/PhysInOneP11"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP11%2FPhysInOneP11%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP11 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP11/PhysInOneP11"><code>huggingface.co/datasets/PhysInOneP11/PhysInOneP11</code></a>
</td>
<td width="50%" valign="top">
<strong>PhysInOneP12</strong> · <strong style="color:#d32f2f;">6.68 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP12/PhysInOneP12"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP12%2FPhysInOneP12%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP12 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP12/PhysInOneP12"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP12%2FPhysInOneP12%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP12 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP12/PhysInOneP12"><code>huggingface.co/datasets/PhysInOneP12/PhysInOneP12</code></a>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<strong>PhysInOneP13</strong> · <strong style="color:#d32f2f;">6.66 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP13/PhysInOneP13"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP13%2FPhysInOneP13%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP13 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP13/PhysInOneP13"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP13%2FPhysInOneP13%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP13 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP13/PhysInOneP13"><code>huggingface.co/datasets/PhysInOneP13/PhysInOneP13</code></a>
</td>
<td width="50%" valign="top">
<strong>PhysInOneP14</strong> · <strong style="color:#d32f2f;">6.72 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP14/PhysInOneP14"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP14%2FPhysInOneP14%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP14 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP14/PhysInOneP14"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP14%2FPhysInOneP14%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP14 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP14/PhysInOneP14"><code>huggingface.co/datasets/PhysInOneP14/PhysInOneP14</code></a>
</td>
</tr>
<tr>
<td width="50%" valign="top">
<strong>PhysInOneP15</strong> · <strong style="color:#d32f2f;">7.93 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP15/PhysInOneP15"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP15%2FPhysInOneP15%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP15 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP15/PhysInOneP15"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP15%2FPhysInOneP15%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP15 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP15/PhysInOneP15"><code>huggingface.co/datasets/PhysInOneP15/PhysInOneP15</code></a>
</td>
<td width="50%" valign="top">
<strong>PhysInOneP16</strong> · <strong style="color:#d32f2f;">1.57 TB</strong> · ✅ Complete · <a href="https://huggingface.co/datasets/PhysInOneP16/PhysInOneP16"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP16%2FPhysInOneP16%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=total&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP16 total downloads"></a> <a href="https://huggingface.co/datasets/PhysInOneP16/PhysInOneP16"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fdatasets%2FPhysInOneP16%2FPhysInOneP16%3Fexpand%3Ddownloads&amp;query=%24.downloads&amp;label=30d&amp;logo=huggingface&amp;color=FFD21E&amp;cacheSeconds=3600" alt="PhysInOneP16 30d downloads"></a><br>
<a href="https://huggingface.co/datasets/PhysInOneP16/PhysInOneP16"><code>huggingface.co/datasets/PhysInOneP16/PhysInOneP16</code></a>
</td>
</tr>
</table>

> Download badges query the official Hugging Face API and update automatically. Counts are repository-level download events, not deduplicated users; accessing multiple shards can produce one event in each shard.

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

> **📅 Update Schedule:** This section is actively being updated throughout Oct.

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
