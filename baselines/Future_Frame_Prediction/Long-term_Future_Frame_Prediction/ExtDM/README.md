# ExtDM for PhysInOne Future Prediction

This folder contains the PhysInOne adaptation of ExtDM. Unlike the
scene-optimization baselines, ExtDM trains one dataset-level model in two
stages: a flow autoencoder (AE), followed by the diffusion model (DM).

## Release status

- AE training: included.
- DM training: included and initialized from the trained AE.
- Inference: included and reads public conditioning RGB only.
- Evaluation: predictions are scored by the private PhysInOne leaderboard.
- Weights: not included yet.

The inspected ExtDM source snapshot did not include a standalone license file.
Read `THIRD_PARTY_NOTICES.md` before publishing this folder.

## Installation

```bash
conda create -n extdm-physinone python=3.9 -y
conda activate extdm-physinone
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116   --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install -e .
```

## Manifests and configuration

Each manifest contains one scene path per line. Relative paths are resolved
against `dataset_params.root_dir`; absolute paths are also accepted.

- `manifests/train_scenes.txt`: create this from the public training split.
- `manifests/scenes_103.txt`: the 103 future-prediction scenes.
- `config/AE/physicsbench.yaml` and `config/DM/physicsbench.yaml`: set
  `root_dir` to the downloaded dataset root.

## Stage 1: train the AE

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. accelerate launch --num_processes=2 scripts/AE/run.py   --config config/AE/physicsbench.yaml   --log_dir ./output/AE   --device_ids 0,1   --postfix physinone
```

The resulting checkpoint is normally
`output/AE/physicsbench_physinone/snapshots/RegionMM.pth`.

## Stage 2: train the DM

```bash
WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. accelerate launch --num_processes=2 scripts/DM/run.py   --config config/DM/physicsbench.yaml   --log_dir ./output/DM   --device_ids 0,1   --postfix physinone   --flowae_checkpoint ./output/AE/physicsbench_physinone/snapshots/RegionMM.pth
```

## Inference and evaluation

```bash
python infer_physinone.py   --scene-dir /path/to/FuturePrediction/SinglePhysics/<scene>_trajectory   --flowae-checkpoint ./checkpoints/RegionMM.pth   --dm-checkpoint ./checkpoints/flowdiff.pth   --output-dir ./predictions/SinglePhysics/<scene>_trajectory
```

The script left-pads short conditioning clips with their earliest available
public frame, autoregressively predicts exactly the frames described by
`transforms_test.json`, and writes no GT or metric artifacts. Submit the
prediction tree to the PhysInOne leaderboard for private evaluation.

Checkpoint files:

```text
checkpoints/
├── RegionMM.pth
└── flowdiff.pth
```
