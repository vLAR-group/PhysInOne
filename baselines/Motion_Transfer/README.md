# Overview

This repository provides inference pipelines for two open-source motion-transfer methods on PhysInOne.

| Method | Official repository | Checkpoints |
| :--- | :--- | :--- |
| Go-with-the-Flow | [Eyeline-Labs/Go-with-the-Flow](https://github.com/Eyeline-Labs/Go-with-the-Flow) | [Hugging Face](https://huggingface.co/Eyeline-Labs/Go-with-the-Flow) |
| MotionPro-Dense | [HiDream-ai/MotionPro](https://github.com/HiDream-ai/MotionPro) | [Hugging Face](https://huggingface.co/HiDream-ai/MotionPro) |

## Installation

```bash
git clone https://github.com/vLAR-group/PhysInOne.git
cd PhysInOne/baselines/Motion_Transfer

conda env create -f environment.yaml
conda activate PhysInOne-Application-MotionTransfer
```

Download the checkpoints for both methods:

```bash
python scripts/download_ckpt.py --name all
```

To download only one method:

```bash
python scripts/download_ckpt.py --name motionpro
python scripts/download_ckpt.py --name gowiththeflow
```

## Leaderboard (Coming Soon)

We welcome evaluations on our **PhysInOne Benchmark**! Official evaluation submission pipelines and benchmark datasets are now available.

### Evaluation Task
**Motion Transfer** seeks to propagate *motion dynamics* from a **source video** to a **target image**, synthesizing a new video that retains the *target’s visual attributes* while adopting the source’s *motion patterns*.

### Input Conditions

For each test case across both tracks, you will receive:

* **Initial Frame:** The starting frame of the target sequence.
* **Textual Prompt:** A natural-language description of the physical scene and activity.
* **Source Video:** A video in which the motion dynamic would be transferred.


### Submission & Download Workflow

Follow this step-by-step guide to download the benchmark inputs, run inference, package your results, and submit them to the PhysInOne Leaderboard.

#### Step 1: Download Benchmark Inputs

First, fetch the leaderboard utility scripts and download the test inputs for your target task.

From the repository root, run:

```bash
cd ../..
python scripts/download_data.py --task motion-transfer --output-dir ./PhysInOne_data
cd ./baselines/Motion_Transfer
```

#### Step 2: Run Inference

Evaluate your model on the downloaded leaderboard dataset using the following command:

```bash
python inference.py \
  --method <method> # motionpro, gowiththeflow \
  --data_root /path/to/your/output
```

**Example: MotionPro-Dense**
For example, if you would like to inference MotionPro-Dense on GPU-0 and save the output to `./output`

```bash
python inference.py \
  --method motionpro \
  --output ./output
```

Results are saved to `outputs/<method>/` by default. Use `--out_root` to select another output directory.

#### Step 3: Package Results

Once inference is complete, use the provided utility script to bundle your outputs into the required submission format.

```bash
bash ../Physics-aware_Video_Generation/archive_result.sh <path/to/your/output> <path/to/your/submission_folder>
```

**Example: Archive results of MotionPro**
```bash
bash ../Physics-aware_Video_Generation/archive_result.sh ./output/motionpro ./submission
```

After the script finishes, you will find individual `.zip` files inside the `./submission` directory.

> ⚠️ **Important Note on Zip Structure:** 
> Each zip archive is structured to contain **only the direct subfolders** (e.g., `CineCamera_*`). It does **not** contain an outer parent folder with the same name as the zip file.

##### Verify Your Submission
Before uploading, strictly verify that your submission directory is properly formatted and legal using our validation script:

```bash
python ../Physics-aware_Video_Generation/check.py <path/to/your/local_submission_folder>
```

**Example: Verify Submission**
```bash
python ../Physics-aware_Video_Generation/check.py ./submission 
```

---

#### Step 4: Submit Your Results

1. **Upload:** Upload your verified `./submission` folder to your own **Hugging Face repository**.

```bash
python ../Physics-aware_Video_Generation/upload.py ./submission --repo username/repository --remote_folder submissions/team-name --token your-token
```

You can visit our uploading demo in [huggingface repository](https://huggingface.co/datasets/vLAR/PhysInOne-Leaderboard-Run-by-VLAR).

1. **Submit:** Navigate to the **Leaderboard Portal** and submit your Hugging Face repository link.

🎉 **That's it!** We will automatically compute your metrics and update the leaderboard as soon as possible.