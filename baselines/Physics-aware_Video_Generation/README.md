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

Below is the polished and structured version of your `README.md` section:

---

# Leaderboard (Coming Soon)

We welcome evaluations on our **PhysInOne Benchmark**! Official evaluation submission pipelines and benchmark datasets are now available.

### Evaluation Tasks & Tracks

For the **Video Generation** task, we divide evaluation into two distinct tracks based on camera behavior:

1. **Static Camera Track:** Focuses on pure dynamic physical phenomena (mechanics, optics, fluid dynamics, magnetism) under a fixed viewpoint.
2. **Moving Camera Track:** Evaluates continuous dynamic visual physics generated under camera trajectory shifts and changing perspectives.

### Input Conditions

For each test case across both tracks, you will receive:

* **Initial Frame:** The starting frame of the target sequence ($t = 0$).
* **Textual Prompt:** A natural-language description of the physical scene and activity.
* **Camera Metadata:** Intrinsic parameters and camera extrinsics (fixed pose or full trajectory).

> **Note:** Models may consume any combination of these conditions (e.g., text-only, image + text, or full frame + text + camera trajectory). Please specify your exact conditioning inputs upon submission.

---

### Submission & Download Workflow

Follow this step-by-step guide to download the benchmark inputs, run inference, package your results, and submit them to the PhysInOne Leaderboard.

---

#### Step 1: Download Benchmark Inputs

First, fetch the leaderboard utility scripts and download the test inputs for your target task.

Download the specific evaluation input cases for your task (e.g., `video-generation`):

```bash
cd ../..
python scripts/download_data.py --task video-generation --output-dir ./PhysInOne_data
cd ./baselines/Physics-aware_Video_Generation
```

The downloaded inputs are stored at `PhysInOne_data/leaderboard/video_generation/`.

#### Step 2: Run Inference

Evaluate your model on the downloaded leaderboard dataset using the following command:

```bash
python inference.py \
  --config <path/to/your/configuration.yaml> \
  --data_path <path/to/leaderboard_dataset> \
  --leaderboard True \
  --leaderboard_branch static \
  --output_path <path/to/your/output> \
  --skip_exist
```

> 💡 **Tip:** Change `--leaderboard_branch static` to `moving` if you are evaluating on the **Moving Camera Track**.

**Example: Static Camera Track**

If you downloaded the leaderboard dataset to `../../PhysInOne_data`, you can run inference using the vanilla `Wan2.2-TI2V-5B` model for the **Static Track** like this:

```bash
python inference.py \
  --config ./configs/lora/Wan2.2-TI2V-5B.yaml \
  --data_path ../../PhysInOne-leaderboard \
  --leaderboard True \
  --leaderboard_branch static \
  --output_path ./leaderboard_output
```

---

#### Step 3: Package Results

Once inference is complete, use the provided utility script to bundle your outputs into the required submission format.

```bash
bash ./archive_results.sh <path/to/your/output> <path/to/your/submission_folder>
```

**Example**
```bash
bash ./archive_results.sh ./leaderboard_output ./submission
```

After the script finishes, you will find individual `.zip` files inside the `./submission` directory.

> ⚠️ **Important Note on Zip Structure:** 
> Each zip archive is structured to contain **only the direct subfolders** (e.g., `CineCamera_*`). It does **not** contain an outer parent folder with the same name as the zip file.

##### Verify Your Submission
Before uploading, strictly verify that your submission directory is properly formatted and legal using our validation script:

```bash
python check.py <path/to/your/submission_folder> [track]
```

**Example: Verify Static Track Submission**
```bash
python check.py ./submission static 
# Use 'moving' instead of 'static' for the moving camera track
```

---

#### Step 4: Submit Your Results

1. **Upload:** Upload your verified `./submission` folder to your own **Hugging Face repository**.
2. **Submit:** Navigate to the **Leaderboard Portal** and submit your Hugging Face repository link.

🎉 **That's it!** We will automatically compute your metrics and update the leaderboard as soon as possible.