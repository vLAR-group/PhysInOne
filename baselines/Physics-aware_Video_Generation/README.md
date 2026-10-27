# Physics-Aware Video Generation on PhysInOne

## Table of Contents

- [Physics-Aware Video Generation on PhysInOne](#physics-aware-video-generation-on-physinone)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Quickstart](#quickstart)
    - [1. Installation](#1-installation)
    - [2. Inference](#2-inference)
    - [3. Training](#3-training)
    - [4. Checkpoints](#4-checkpoints)
      - [Available Checkpoints](#available-checkpoints)
      - [Download Checkpoints](#download-checkpoints)
      - [Validate Checkpoints](#validate-checkpoints)
    - [5. Configurations](#5-configurations)
  - [Leaderboard (Coming Soon)](#leaderboard-coming-soon)
    - [Evaluation Tracks and Subsets](#evaluation-tracks-and-subsets)
    - [Input Conditions](#input-conditions)
    - [Inference \& Submission Workflow](#inference--submission-workflow)
      - [Step 1: Download Benchmark Inputs](#step-1-download-benchmark-inputs)
      - [Step 2: Run Inference](#step-2-run-inference)
      - [Step 3: Package and Verify Results](#step-3-package-and-verify-results)
      - [Step 4: Upload and Submit Results](#step-4-upload-and-submit-results)

## Overview

Video generation models have made substantial progress in visual fidelity, but they still struggle to reproduce physically plausible dynamics. **PhysInOne** is a diverse, large-scale dataset of dynamic activities governed by real-world physical phenomena. It supports training and fine-tuning video generation models to better capture physical behavior.

This directory contains the official implementation for fine-tuning video generation models and running inference on the PhysInOne dataset.

**Paper:** *PhysInOne: Visual Physics Learning and Reasoning in One Suite*

[GitHub](https://github.com/vLAR-group/PhysInOne) · [Project Website](https://vlar-group.github.io/PhysInOne.html) · [arXiv](https://arxiv.org/abs/2604.09415) · [Dataset](https://huggingface.co/datasets/vLAR/PhysInOne)

![PhysInOne dataset overview](https://raw.githubusercontent.com/vLAR-group/PhysInOne/main/teaser.png)

## Quickstart

Run the following commands from `baselines/Physics-aware_Video_Generation/` unless otherwise noted. Replace placeholder paths with the locations of your configuration files, dataset, and checkpoints.

### 1. Installation

We recommend using Conda to manage dependencies. Create and activate the environment, then install the local `diffusers` package:

```bash
conda env create -f environment.yaml
conda activate PhysInOne-Application-VideoGeneration
cd diffusers
pip install -e ".[torch]"
cd ..
```

### 2. Inference

Run inference with a baseline model using its configuration file and the path to the PhysInOne dataset:

```bash
python inference.py \
  --config /path/to/baseline/configuration.yaml \
  --data_path /path/to/PhysInOne
```

**Example:** Run the vanilla `Wan2.2-TI2V-5B` model on the dataset stored at `/media/SSD2/PhysInOne`:

```bash
python inference.py \
  --config ./configs/lora/Wan2.2-TI2V-5B.yaml \
  --data_path /media/SSD2/PhysInOne
```

### 3. Training

Use the Hugging Face Accelerate launcher to fine-tune a baseline model on PhysInOne:

```bash
accelerate launch \
  --main_process_port 22525 \
  --config_file /path/to/accelerate/config.yaml \
  train.py \
  --config /path/to/training/config.yaml
```

**Example:** Fine-tune `Wan2.2-TI2V-5B` with LoRA using eight H200 GPUs and bfloat16 precision:

```bash
accelerate launch \
  --main_process_port 22525 \
  --config_file configs/accelerate/8_16bf.yaml \
  train.py \
  --config configs/lora/Wan2.2-TI2V-5B.yaml \
  --data_path /media/SSD2/PhysInOne
```

After training, evaluate the latest saved checkpoint:

```bash
python inference.py \
  --from_pretrained ./models/lora_Wan-AI \
  --data_path /path/to/PhysInOne \
  --checkpoint latest
```

### 4. Checkpoints

We provide fine-tuned checkpoints for several baseline models. Community contributions are welcome—please contact us if you develop an improved model or a new fine-tuning approach.

#### Available Checkpoints

| Base Model | Fine-Tuning Method | Checkpoint Name |
| --- | --- | --- |
| Wan2.2-TI2V-5B | LoRA | `lora_Wan-AI` |
| Wan2.2-TI2V-5B | SFT | `sft_Wan-AI` |
| Wan2.2-TI2V-5B | FLT | `flt_Wan-AI` |
| Stable Video Diffusion XT | LoRA | `lora_Wan-AI` |
| Stable Video Diffusion XT | SFT | `sft_Wan-AI` |
| Stable Video Diffusion XT | FLT | `flt_Wan-AI` |
| CogVideoX-1.5-5B | LoRA | `lora_THUDM` |

> **Note:** The checkpoint names above are retained from the provided checkpoint list. The Stable Video Diffusion XT entries share names with the Wan entries; confirm these identifiers in `scripts/download_ckpt.py` before downloading.

See [Baseline Resources](#baseline-resources) for model papers and official repositories.

#### Download Checkpoints

Download all available checkpoints:

```bash
python scripts/download_ckpt.py
```

To download a specific checkpoint, pass its name with `--name`:

```bash
python scripts/download_ckpt.py --name <checkpoint_name>
```

**Example:** Download the `lora_Wan-AI` checkpoint:

```bash
python scripts/download_ckpt.py --name lora_Wan-AI
```

#### Validate Checkpoints

Run inference on a downloaded checkpoint:

```bash
python inference.py \
  --from_pretrained /path/to/checkpoints \
  --data_path /path/to/PhysInOne \
  --checkpoint latest
```

**Example:** Evaluate the latest checkpoint in `./checkpoints/flt_Wan-AI/Wan2.2-TI2V-5B-Diffusers`:

```bash
python inference.py \
  --from_pretrained ./checkpoints/flt_Wan-AI/Wan2.2-TI2V-5B-Diffusers \
  --data_path /media/SSD2/PhysInOne \
  --checkpoint latest
```

For this example, generated results are saved to:

```
./outputs/flt_Wan-AI/Wan2.2-TI2V-5B-Diffusers_ckpt_latest
```

### 5. Configurations

Example configuration files are available in `./configs`. Customize them to match your hardware, model settings, and experimental requirements.

## Leaderboard (Coming Soon)

We welcome submissions to the **PhysInOne Benchmark**. Benchmark inputs and submission tools are available to help you prepare your evaluation results ahead of the leaderboard launch.

### Evaluation Tracks and Subsets

The video generation benchmark includes two tracks based on camera behavior:

- **Static Camera Track:** Evaluates physical dynamics—including mechanics, optics, fluid dynamics, and magnetism—from a fixed viewpoint.

- **Moving Camera Track:** Evaluates physical dynamics under changing camera positions and perspectives.

For each track, you can evaluate on the **large test dataset** (coming soon) containing 15,411 data pairs or the **mini test dataset** containing 801 data pairs.

### Input Conditions

For each test case across both tracks, you will receive:

* **Initial Frame:** The starting frame of the target sequence ($t = 0$).
* **Textual Prompt:** A natural-language description of the physical scene and activity.
* **Camera Metadata:** Intrinsic parameters and camera extrinsics.

> **Note:** Models may consume any combination of these conditions (e.g., text-only, image + text, or full frame + text + camera trajectory). Please specify your exact conditioning inputs upon submission.


### Inference & Submission Workflow

Follow the steps below to download benchmark inputs, run inference, package and verify your outputs, and submit your results.

#### Step 1: Download Benchmark Inputs

First of all, download the video generation inputs:

```bash
cd ../..
python scripts/download_data.py --task video-generation --output-dir ./data
cd ./baselines/Physics-aware_Video_Generation
```

The downloaded inputs are stored in `../../data/leaderboard/video_generation/`.

#### Step 2: Run Inference

Evaluate your model on the downloaded leaderboard dataset using the following command:

```bash
python inference.py \
  --config <path/to/your/configuration.yaml> \
  --data_path <path/to/leaderboard_dataset> \
  --leaderboard True \
  --leaderboard_track <your track : static or moving> \
  --output_path <path/to/your/output> \
```

**Example: Static Camera Track on Mini Test Dataset**

If you downloaded the leaderboard dataset to `../../data`, you can run inference using the vanilla `Wan2.2-TI2V-5B` model for the **Static Track** on **mini test subset** like this:

```bash
python inference.py \
  --config ./configs/lora/Wan2.2-TI2V-5B.yaml \
  --data_path ../../data \
  --leaderboard \
  --leaderboard_track static \
  --subset mini \
  --output_path ./leaderboard_output
```
#### Step 3: Package and Verify Results

Use the provided utility script to package your outputs in the required submission format:

```bash
bash ./archive_results.sh /path/to/your/output /path/to/your/submission_folder
```

**Example:**

```bash
bash ./archive_results.sh ./leaderboard_output ./submission
```

The script creates individual `.zip` archives in `./submission`.

> **Important:** Each archive must contain the result subfolders, such as `CineCamera_*`, directly at its root. Do not wrap them in an additional parent folder named after the archive.

Before uploading, validate the submission structure using the same track and subset used for inference:

```bash
python check.py /path/to/your/submission_folder <static_or_moving> --subset <mini_or_full>
```

**Example:** Validate a Static Camera Track submission for the mini subset:

```bash
python check.py ./submission static --subset mini
```

Use `moving` for the Moving Camera Track or `full` for the full test dataset.

#### Step 4: Upload and Submit Results

Upload your verified submission folder to your Hugging Face repository:

```bash
python ./upload.py ./submission \
  --repo username/repository \
  --remote_folder submissions/team-name \
  --token <your_hugging_face_token>
```

Replace the repository name, remote folder, and token with your own values. Keep your token private and do not commit it to the repository.

For a reference submission, see [vLAR/PhysInOne-Leaderboard-Run-by-VLAR](https://huggingface.co/datasets/vLAR/PhysInOne-Leaderboard-Run-by-VLAR).

Submit your Hugging Face repository link through the Leaderboard Portal when it becomes available. Your results will be evaluated automatically, and the leaderboard will be updated after evaluation.
