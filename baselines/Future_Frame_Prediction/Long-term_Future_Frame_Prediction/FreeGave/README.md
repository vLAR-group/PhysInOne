# FreeGave for PhysInOne Future Prediction

This folder contains the cleaned FreeGave baseline code used for the
PhysInOne future-prediction task. It provides:

- a ground-truth-free inference entry point;
- a training entry point that reads only the public training split;
- the CUDA extensions required by Gaussian splatting;
- the expected dataset, checkpoint, and prediction layouts.

The pretrained checkpoints are distributed with this release separately from
the source files, under `checkpoints/`. They are not included in this staging
folder yet.

## Important data boundary

`infer_physinone.py` reads camera poses and timestamps from
`transforms_test.json`, but it never opens the future RGB paths listed in that
file. It writes predictions only and does not compute metrics or copy ground
truth.

`train_physinone.py` loads `transforms_train.json` and its public RGB frames.
Validation and future-test images are skipped during public training.

## Installation

The released checkpoints were produced with the original FreeGave software
stack based on PyTorch 1.13.1 and CUDA 11.6. A CUDA-capable NVIDIA GPU is
required.

```bash
conda env create -f environment.yml
conda activate freegave-physinone

conda install pytorch==1.13.1 torchvision==0.14.1 \
  torchaudio==0.13.1 pytorch-cuda=11.6 -c pytorch -c nvidia

pip install ./submodules/depth-diff-gaussian-rasterization
pip install ./submodules/simple-knn
```

If a newer CUDA/PyTorch combination is used, the two CUDA extensions must be
rebuilt against that exact environment.

## Expected public scene layout

Each PhysInOne future-prediction scene must preserve the released directory
structure:

```text
<scene>_trajectory/
├── transforms_train.json
├── transforms_val.json
├── transforms_test.json
└── CineCamera_<view>/
    └── rgb/
        ├── 0000.jpg
        ├── ...
        └── 0074.jpg
```

Only observed frames are public. Future RGB files referenced by
`transforms_test.json` are intentionally absent.

## Expected checkpoint layout

```text
checkpoints/
└── <Physics>/
    └── <scene>_trajectory/
        ├── point_cloud/
        │   └── iteration_25000/
        │       └── point_cloud.ply
        └── deform/
            └── iteration_25000/
                ├── deform.pth
                ├── code_field.pth
                └── vel.pth
```

The `<Physics>` directory is one of `SinglePhysics`, `DoublePhysics`, or
`TriplePhysics`.

## Inference

Render all future cameras and frames for one scene:

```bash
python infer_physinone.py \
  --source-path /path/to/FuturePrediction/SinglePhysics/<scene>_trajectory \
  --model-path ./checkpoints/SinglePhysics/<scene>_trajectory \
  --output-dir ./predictions/SinglePhysics/<scene>_trajectory \
  --iteration 25000
```

The output mirrors the camera paths in `transforms_test.json`:

```text
predictions/SinglePhysics/<scene>_trajectory/
├── prediction_manifest.json
└── CineCamera_<view>/
    └── rgb/
        ├── 0075.jpg
        ├── ...
        └── 0149.jpg
```

Useful inference options:

| Option | Default | Description |
| --- | ---: | --- |
| `--iteration` | `25000` | Checkpoint iteration; `-1` selects the latest available iteration. |
| `--views` | all | Space-separated camera view IDs to render. |
| `--frame-start` | all | First frame index to render, inclusive. |
| `--frame-end` | all | Last frame index to render, inclusive. |
| `--image-extension` | `jpg` | Output format: `jpg` or `png`. |
| `--jpeg-quality` | `95` | JPEG quality when JPEG output is selected. |
| `--max-time` | `0.45` | Learned deformation-to-velocity transition time. |
| `--static-threshold` | `0.01` | Threshold used to identify static Gaussians. |
| `--black-background` | off | Render against black instead of white. |

For a quick smoke test, render one frame from one view:

```bash
python infer_physinone.py \
  --source-path /path/to/<scene>_trajectory \
  --model-path ./checkpoints/SinglePhysics/<scene>_trajectory \
  --output-dir ./smoke_test \
  --views 2 \
  --frame-start 75 \
  --frame-end 75
```

## Training

The following command reproduces the public-data training path without loading
validation or future-test RGB frames:

```bash
python train_physinone.py \
  --source_path /path/to/FuturePrediction/SinglePhysics/<scene>_trajectory \
  --model_path ./output/SinglePhysics/<scene>_trajectory \
  --iterations 25000 \
  --save_iterations 4000 10000 20000 25000 \
  --max_time 0.45 \
  --physics_code 16
```

The first training run may create `points3d.ply` in the input scene directory.
Use a writable working copy of the public scene when training.

## Scene manifest

`manifests/scenes_103.txt` contains the 103 PhysInOne future-prediction scene
identifiers expected by the checkpoint release. One scene is listed per line as
`<Physics>/<scene>_trajectory`.

## Repository structure

See [`docs/DIRECTORY_STRUCTURE.md`](docs/DIRECTORY_STRUCTURE.md) for a detailed
description of every released directory and the files intentionally excluded
from the original research workspace.

## License and attribution

FreeGave is released under the license in [`LICENSE`](LICENSE). Vendored
components retain their own licenses; see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the license files inside
`submodules/`.

This code is based on FreeGave, Deformable 3D Gaussians, NVIDIA's NVFi, and the
3D Gaussian Splatting rasterization components. Please cite the corresponding
projects and the PhysInOne benchmark when using this release.

### FreeGave citation

```bibtex
@inproceedings{li2025freegave,
  title     = {FreeGave: 3D Physics Learning from Dynamic Videos by Gaussian Velocity},
  author    = {Li, Jinxi and Song, Ziyang and Zhou, Siyuan and Yang, Bo},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year      = {2025}
}
```

Please also follow the citation instructions in the main PhysInOne repository
for the benchmark and dataset.

