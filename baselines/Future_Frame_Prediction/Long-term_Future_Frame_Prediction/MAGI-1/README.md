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

MAGI-1 supports **seen-view future prediction only**: predict the second-half
RGB for the 10 seen views (training cameras) identified by
`transforms_train.json`. It does not synthesize the two held-out novel views
identified by `transforms_val.json`.

For each seen camera, the script conditions on the last 32 available frames
of its public first-half clip. If the clip is shorter, it left-pads with the
earliest available public frame.

Use an input directory containing only the public first-half RGB for those
10 seen cameras. The current entry point enumerates RGB camera directories;
it does not independently filter them against the training-camera JSON.
Never point it at a full rendered scene or private GT directory containing
future frames, novel-camera RGB, or moving-camera RGB.

```bash
export MASTER_ADDR=localhost MASTER_PORT=6009
export GPUS_PER_NODE=1 NNODES=1 WORLD_SIZE=1
export RANK=0 LOCAL_RANK=0
export PAD_HQ=1 PAD_DURATION=1
export OFFLOAD_T5_CACHE=true OFFLOAD_VAE_CACHE=true
export PYTHONPATH="$PWD"

python infer_physinone.py   --scene-dir /path/to/FuturePrediction/SinglePhysics/<scene>_trajectory   --output-dir ./predictions/SinglePhysics/<scene>_trajectory   --config example/4.5B/4.5B_distill_config.json
```

For each seen camera, `transforms_test.json` determines the future timestamps,
frame names, and prediction count. That JSON also lists novel-camera targets
for other baselines; those entries are not MAGI-1 submission targets.

Submit only predicted future RGB for all 10 seen cameras. Omit novel-camera
directories entirely, including empty placeholders; do not copy GT or
observed frames into the submission. The leaderboard reports seen-view
metrics and displays novel-view metrics as `-`, not zero. See the
[task submission modes](../../README.md#prediction-layout-and-evaluation).

The script also writes `prediction_manifest.json` for local bookkeeping;
keep this metadata file out of the prediction ZIP.

`manifests/scenes_103.txt` records scene coverage. Model weights, generated
videos, GT exports, metrics, logs, and local progress databases are excluded.
