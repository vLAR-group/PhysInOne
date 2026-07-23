# Overview

While video generation models have demonstrated significant advancements in visual fidelity, they frequently struggle to capture physically plausible dynamics. **PhysInOne** is a diverse and extensive dataset comprising dynamic activities governed by a wide variety of real-world physical phenomena. This rich dataset serves as an ideal resource for training and fine-tuning next-generation video models to faithfully emulate real-world physics.

This repository provides the official codebase for fine-tuning and running inference on video generation models using the PhysInOne dataset.

# Quickstart

## 1. Installation

We recommend using Conda to manage dependencies. Create and activate the environment using the provided configuration:

```bash
conda env create -f environment.yaml
conda activate PhysInOne-Application-VideoGeneration
cd diffusers
pip install -e ".[torch]"
cd ..
```

## 2. Inference

To run inference and evaluate baseline video generation models on the PhysInOne dataset, use the following command:

```bash
python inference.py --config /path/to/baseline/configuration --data_path /path/to/PhysInOne
```

**Example:** Evaluating the vanilla `Wan2.2-TI2V-5B` model on the PhysInOne dataset located at `/media/SSD2/PhysInOne`:
```bash
python inference.py --config ./configs/lora/Wan2.2-TI2V-5B.yaml --data_path /media/SSD2/PhysInOne
```

## 3. Training

To fine-tune a baseline video generation model on the PhysInOne dataset, use the Hugging Face `accelerate` launcher:

```bash
accelerate launch --main_process_port 22525 --config_file /path/to/accelerate/config train.py --config /path/to/training/config
```

**Example:** Fine-tuning the `Wan2.2-TI2V-5B` model using LoRA on the PhysInOne dataset (located at `/media/SSD2/PhysInOne`) utilizing 8 H200 GPUs with bfloat16 precision:
```bash
accelerate launch --main_process_port 22525 --config_file configs/accelerate/8_16bf.yaml train.py --config configs/lora/Wan2.2-TI2V-5B.yaml --data_path /media/SSD2/PhysInOne
```

**Validation:** After training, you can evaluate the latest saved checkpoints using the inference script:

```bash
python inference.py --from_pretrained ./models/lora_Wan-AI --data_path /path/to/dataset --checkpoint latest
```

## 4. Checkpoints

We provide several fine-tuned models for you to explore. We highly encourage community contributions; if you have developed an improved model or a novel approach, please feel free to contact us!

### 4.1 Available Checkpoints

| Base Model | Fine-tuning Method | Checkpoint Name |
| :--- | :--- | :--- |
| [**Wan2.2-5B**]([WAN_MODEL_LINK]https://arxiv.org/abs/2503.20314) | [**LoRA**]([LORA_METHOD_LINK]https://arxiv.org/pdf/2106.09685v1/1000) | `lora_Wan-AI` |
| [**Wan2.2-5B**]([WAN_MODEL_LINK]https://arxiv.org/abs/2503.20314) | [**SFT**]([SFT_METHOD_LINK]https://aclanthology.org/P18-1031/) | `sft_Wan-AI` |
| [**Wan2.2-5B**]([WAN_MODEL_LINK]https://arxiv.org/abs/2503.20314) | [**FLT**]([FLT_METHOD_LINK]https://proceedings.neurips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html) | `flt_Wan-AI` |
| [**Stable Video Diffusion XT**]([SVD_MODEL_LINK]https://arxiv.org/abs/2311.15127) | [**LoRA**]([LORA_METHOD_LINK]https://arxiv.org/pdf/2106.09685v1/1000)  | `lora_Wan-AI` |
| [**Stable Video Diffusion XT**]([SVD_MODEL_LINK]https://arxiv.org/abs/2311.15127) | [**SFT**]([SFT_METHOD_LINK]https://aclanthology.org/P18-1031/) | `sft_Wan-AI` |
| [**Stable Video Diffusion XT**]([SVD_MODEL_LINK]https://arxiv.org/abs/2311.15127) | [**FLT**]([FLT_METHOD_LINK]https://proceedings.neurips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html) | `flt_Wan-AI` |
| [**CogVideoX-1.5-5B**]([COGVIDEO_MODEL_LINK]https://proceedings.iclr.cc/paper_files/paper/2025/hash/ce31378e9f41d8907e97dab172b6c559-Abstract-Conference.html) | [**LoRA**]([LORA_METHOD_LINK]https://arxiv.org/pdf/2106.09685v1/1000)  | `lora_THUDM` |

### 4.2 Downloads

To download all available checkpoints at once, run the following command:
```bash
python scripts/download_ckpt.py
```

You can also specify a particular checkpoint name to download:
```bash
python scripts/download_ckpt.py --name <checkpoint_name>
```

**Example:** To download the `lora_Wan-AI` checkpoint:
```bash
python scripts/download_ckpt.py --name lora_Wan-AI
```

### 4.3 Validation

Similarly, use the following command to validate the models:
```bash
python inference.py --from_pretrained /path/to/checkpoints --data_path /path/to/dataset --checkpoint latest
```

**Example:** To validate the **latest** checkpoint located at `./checkpoints/flt_Wan-AI/Wan2.2-TI2V-5B-Diffusers` on the PhysInOne dataset (located at `/media/SSD2/PhysInOne`):
```bash
python inference.py --from_pretrained ./checkpoints/flt_Wan-AI/Wan2.2-TI2V-5B-Diffusers --data_path /media/SSD2/PhysInOne --checkpoint latest
```

The generated inference results will be saved in the `./outputs/flt_Wan-AI/Wan2.2-TI2V-5B-Diffusers_ckpt_latest` directory.

## 5. Configurations

We provide several example configuration files in the `./configs` directory. Feel free to customize these files to suit your specific hardware setups and experimental requirements.