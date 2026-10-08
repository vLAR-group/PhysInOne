# TiNeuVox for PhysInOne Future Prediction

This folder contains the cleaned TiNeuVox baseline. TiNeuVox is optimized
independently for each PhysInOne scene.

## Release status

- Per-scene training: included as `train_physinone.py`.
- Inference: included and uses test metadata without opening future RGB.
- Evaluation: predictions are submitted to the private PhysInOne leaderboard.
- Weights: not included yet.

## Installation

```bash
conda create -n tineuvox-physinone python=3.9 -y
conda activate tineuvox-physinone
pip install torch torchvision
pip install -r requirements.txt
# Install torch-scatter for the exact PyTorch/CUDA versions in this environment.
```

## Per-scene training

Copy `configs/nerf-base/PhysicsBench/physinone.py` per scene and set
`expname` and `data.datadir`.

```bash
python train_physinone.py   --config configs/nerf-base/PhysicsBench/physinone.py
```

The adapted loader uses RGB from `transforms_train.json` for optimization. If released validation or future RGB is absent, it keeps one observed-image placeholder only to satisfy the legacy split container; those placeholder splits are never used for training. The default run writes `fine_last.tar` under
`<basedir>/<expname>/`.

## Inference and evaluation

```bash
python infer_physinone.py   --scene-dir /path/to/FuturePrediction/SinglePhysics/<scene>_trajectory   --checkpoint ./checkpoints/SinglePhysics/<scene>_trajectory/fine_last.tar   --output-dir ./predictions/SinglePhysics/<scene>_trajectory   --config configs/nerf-base/PhysicsBench/physinone.py
```

The inference entry point loads only `transforms_test.json` camera metadata,
renders the exact target paths, and never reads future RGB. Use
`--views`, `--frame-start`, and `--frame-end` for smoke tests. Submit the
prediction tree to the official leaderboard for private evaluation.

Expected checkpoint layout:

```text
checkpoints/<Physics>/<scene>_trajectory/fine_last.tar
```

The source is subject to `LICENSE`. See `THIRD_PARTY_NOTICES.md` for
attribution and `manifests/scenes_103.txt` for scene coverage.
