## Overview

While video generation models have demonstrated significant advancements in visual fidelity, they frequently struggle to capture physically plausible dynamics. **PhysInOne** comprises a diverse and extensive collection of dynamic activities governed by a wide variety of real-world physical phenomena. This rich dataset serves as an ideal resource for training and fine-tuning next-generation video models to faithfully emulate real-world physics.

This repository provides the codebase for fine-tuning and running inference with video generation models on the PhysInOne dataset.

## Quickstart

### 1. Installation

We recommend using Conda to manage dependencies. Create and activate the environment using the provided configuration:

```bash
conda env create -f environment.yaml
conda activate PhysInOne-Application-VideoGeneration
cd diffusers
pip install -e ".[torch]"
cd ..
```

### 2. Inference

To run inference and evaluate video generation baselines on this task, use the following command:

```bash
python inference.py --config /path/to/baseline/configuration --data_path /path/to/PhysInOne
```

**Example:** To evaluate the vanilla Wan2.2-TI2V-5B model on the PhysInOne dataset located at `/media/SSD2/PhysInOne`:
```bash
python inference.py --config ./configs/lora/Wan2.2-TI2V-5B.yaml --data_path /media/SSD2/PhysInOne
```

### 3. Training

To fine-tune a video generation baseline on the PhysInOne dataset, use the Hugging Face `accelerate` launcher:

```bash
accelerate launch --main_process_port 22525 --config_file /path/to/accelerate/config train.py --config /path/to/training/config
```

**Example:** To LoRA fine-tune the Wan2.2-TI2V-5B model on the PhysInOne dataset (located at `/media/SSD2/PhysInOne`) using 8 H200 GPUs with bfloat16 precision:
```bash
accelerate launch --main_process_port 22525 --config_file configs/accelerate/8_16bf.yaml train.py --config configs/lora/Wan2.2-TI2V-5B.yaml --data_path /media/SSD2/PhysInOne
```

Then, to validate your fine-tuning results, you can evaluate the latest saved checkpoints using the inference script:

```bash
python inference.py --from_pretrained ./models/lora_Wan-AI --data_path /path/to/dataset --checkpoint latest
```

### 5. Configurations

We provide several example configurations under the `./configs` directory. You are welcome to customize these files to suit your specific hardware and experimental needs.