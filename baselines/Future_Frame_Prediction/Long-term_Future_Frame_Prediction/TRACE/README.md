# TRACE for PhysInOne Future Prediction

This folder contains the cleaned TRACE baseline used for long-term
future-frame prediction. TRACE performs per-scene optimization from the public
observed multi-view sequence.

## Release status

- Training: included, using `transforms_train.json` and public RGB only.
- Inference: included and ground-truth free.
- Evaluation: performed by submitting predictions to the PhysInOne leaderboard.
- Weights: not included yet; preserve the layout below under `checkpoints/`.

## Installation

```bash
conda env create -f environment.yml
conda activate trace
conda install pytorch==1.13.1 torchvision==0.14.1   torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia
pip install ./submodules/depth-diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

## Per-scene training

```bash
python train_physinone.py   -s /path/to/FuturePrediction/SinglePhysics/<scene>_trajectory   -m ./output/SinglePhysics/<scene>_trajectory   --is_blender --max_time 0.45 --light   --iterations 25000 --save_iterations 25000
```

The released training entry point constructs only training cameras and skips
validation/test RGB, which are not required for optimization.

## Inference and evaluation

```bash
python infer_physinone.py   --source-path /path/to/FuturePrediction/SinglePhysics/<scene>_trajectory   --model-path ./checkpoints/SinglePhysics/<scene>_trajectory   --output-dir ./predictions/SinglePhysics/<scene>_trajectory   --iteration 25000 --light
```

The script reads future camera poses and timestamps from
`transforms_test.json`, never future RGB. It writes the benchmark-relative
camera/frame paths plus `prediction_manifest.json`. Submit those predictions
to the official leaderboard for private evaluation.

Expected checkpoint layout:

```text
checkpoints/<Physics>/<scene>_trajectory/
├── point_cloud/iteration_25000/point_cloud.ply
└── deform/iteration_25000/
    ├── deform.pth
    └── vel.pth
```

See `manifests/scenes_103.txt` for scene coverage. TRACE is distributed under
`LICENSE`; dependency notices are in `THIRD_PARTY_NOTICES.md`.
