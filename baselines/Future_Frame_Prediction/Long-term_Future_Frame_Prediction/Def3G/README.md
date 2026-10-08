# Def3G for PhysInOne Future Prediction

This folder contains the cleaned Def3G baseline used for long-term future-frame
prediction. Def3G is optimized independently for every scene; there is no
single dataset-wide training checkpoint.

## Release status

- Training: included, using public observed RGB frames only.
- Inference: included and ground-truth free.
- Evaluation: submit the generated images to the PhysInOne leaderboard. Private
  future RGB and local metric code are intentionally excluded.
- Weights: not included yet; use `checkpoints/` for the per-scene release.

The inspected Def3G snapshot did not contain a repository-level license. Read
`THIRD_PARTY_NOTICES.md` before publishing this folder.

## Installation

A CUDA-capable NVIDIA GPU is required. The original experiments used a
PyTorch/CUDA stack compatible with the two vendored CUDA extensions.

```bash
conda create -n def3g-physinone python=3.8 -y
conda activate def3g-physinone
pip install torch==1.13.1 torchvision==0.14.1   --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt
pip install ./submodules/depth-diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

## Data layout

```text
<scene>_trajectory/
├── transforms_train.json
├── transforms_val.json
├── transforms_test.json
└── CineCamera_<view>/rgb/0000.jpg ... 0074.jpg
```

Future RGB paths may appear in `transforms_test.json`, but the images
themselves are private and are never opened by `infer_physinone.py`.

## Per-scene training

```bash
python train_physinone.py   -s /path/to/FuturePrediction/SinglePhysics/<scene>_trajectory   -m ./output/SinglePhysics/<scene>_trajectory   --is_blender --max_time 0.45   --iterations 25000 --save_iterations 25000
```

The training entry point explicitly skips validation and test image loading.

## Inference and evaluation

```bash
python infer_physinone.py   --source-path /path/to/FuturePrediction/SinglePhysics/<scene>_trajectory   --model-path ./checkpoints/SinglePhysics/<scene>_trajectory   --output-dir ./predictions/SinglePhysics/<scene>_trajectory   --iteration 25000
```

For a smoke test, add `--views 2 --frame-start 75 --frame-end 75`.
The output mirrors every `file_path` in `transforms_test.json` and includes a
`prediction_manifest.json` declaring `contains_ground_truth: false`.
Package the resulting RGB folders using the official submission instructions;
the leaderboard worker computes the private metrics.

Expected checkpoint layout:

```text
checkpoints/<Physics>/<scene>_trajectory/
├── point_cloud/iteration_25000/point_cloud.ply
└── deform/iteration_25000/deform.pth
```

`manifests/scenes_103.txt` lists the benchmark scenes. Generated outputs,
private GT, experiment logs, and local metric scripts are deliberately absent.
