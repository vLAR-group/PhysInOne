# MAGI-1 for PhysInOne Future Prediction

This folder contains the cleaned MAGI-1 inference baseline for PhysInOne.

## Training status

MAGI-1 was used as a pretrained video-generation model. We did **not** train or
fine-tune MAGI-1 on PhysInOne, so no PhysInOne training command is provided.
Training the foundation model is outside this baseline protocol. The supported
workflow is checkpoint installation followed by inference and leaderboard
evaluation.

## Installation

The reference setup uses Python 3.10, PyTorch 2.4, and CUDA 12.4. The upstream
Docker image is also suitable.

```bash
conda create -n magi-physinone python=3.10.12 -y
conda activate magi-physinone
conda install pytorch==2.4.0 torchvision==0.19.0   torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
git clone --recursive https://github.com/SandAI-org/MagiAttention.git
pip install --no-build-isolation ./MagiAttention
```

Install the upstream 4.5B distilled MAGI, VAE, and T5 weights as:

```text
checkpoints/
├── magi/4.5B_distill/
├── vae/
└── t5/
```

Then update `runtime_config.load`, `vae_pretrained`, and `t5_pretrained`
in `example/4.5B/4.5B_distill_config.json` to those paths.

## Inference and evaluation

The script conditions on the last 32 available public frames per static camera.
If a clip is shorter, it left-pads with the earliest available frame. Moving
camera folders are not selected by the released future-prediction data.

```bash
export MASTER_ADDR=localhost MASTER_PORT=6009
export GPUS_PER_NODE=1 NNODES=1 WORLD_SIZE=1
export RANK=0 LOCAL_RANK=0
export PAD_HQ=1 PAD_DURATION=1
export OFFLOAD_T5_CACHE=true OFFLOAD_VAE_CACHE=true
export PYTHONPATH="$PWD"

python infer_physinone.py   --scene-dir /path/to/FuturePrediction/SinglePhysics/<scene>_trajectory   --output-dir ./predictions/SinglePhysics/<scene>_trajectory   --config example/4.5B/4.5B_distill_config.json
```

`transforms_test.json` determines the target views, timestamps, frame names,
and prediction count. The script never loads future RGB and writes
`prediction_manifest.json` with `contains_ground_truth: false`. Submit the
result to the official leaderboard for private evaluation.

`manifests/scenes_103.txt` records scene coverage. Model weights, generated
videos, GT exports, metrics, logs, and local progress databases are excluded.
