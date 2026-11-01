# PAC-NeRF for PhysInOne Physical Property Estimation

This folder contains the PhysInOne adaptation of
[PAC-NeRF](https://github.com/xuan-li/PAC-NeRF), **Physics Augmented Continuum
Neural Radiance Fields for Geometry-Agnostic System Identification**
(ICLR 2023). The model estimates geometry, initial velocity and material
parameters independently for each scene from multi-view video.

## Release status

- Per-scene training: included as `train.py`, with static reconstruction,
  initial-velocity fitting and material-parameter fitting.
- Inference: included as `test.py`, with novel-view and reinitialization modes.
  It reads camera metadata and optional static backgrounds without opening
  test-frame RGB.
- Coverage: 30 single-material cases without SDF boundaries, six each for
  elastic, Newtonian, non-Newtonian, plasticine and granular materials.

## Installation

The core installation recipe is [environment.yml](environment.yml), with
versions pinned from the supplied package metadata.

```bash
conda env create -f environment.yml
conda activate pacnerf
```

The custom CUDA extensions require a CUDA toolkit with `nvcc` and a
compatible C++ compiler. After those are available, install torch-scatter
against this PyTorch environment:

```bash
python -m pip install --no-build-isolation --no-binary=torch-scatter torch-scatter==2.1.2
```

## Data layout

Prepare converted datasets with the shared
[data preparation script](../../data_preprocess/create_dataset.py)
and its [usage guide](../../data_preprocess/README.md):

```text
<data_root>/
└── <case>_trajectory_gapN/
    ├── camera_mapping.json
    ├── train/
    │   ├── all_data.json
    │   └── data/
    │       ├── r_0_0.jpg
    │       ├── m_0_0.png
    │       └── r_0_-1.jpg          # optional static background
    └── test/
        ├── all_data.json
        └── data/
            └── r_0_-1.jpg         # optional static background
```

The current training loader expects JPEG `r_<camera>_<frame>.jpg` metadata
entries and pre-generated `m_<camera>_<frame>.png` images. The `m_` files
contain foreground RGB on white, rather than binary masks; training does
not run background matting.

Each `all_data.json` entry contains `file_path`, `time`, a 3x4 `c2w` matrix
and a 3x3 `intrinsic` matrix. Background entries use frame ID `-1` and do
not count as training frames.

Novel inference reads test calibration; resimulation reads both train and
test calibration (13 + 2 = 15 cameras). Both modes read any declared static
background files. Frame paths can appear in metadata without the images
existing. Original `CineCamera_*` names come from `camera_mapping.json`;
train/test camera indices are independent. Resimulation requires mappings
for both splits to merge them without name collisions.

## Manifests and configuration

The generator selects single-material cases whose supplementary map entry
has an empty `sdf` list. SDF and multi-material cases are excluded. Selected case names, initial transforms, metadata status and run commands
are recorded in `configs/generated/manifest.json`.

All configuration path arguments are required: `--supp-map`, `--data-root`,
`--model-root` and `--output-dir`. The config generator has no default
paths. Run the commands from this method's directory.
Use paths visible on the host that will run training and rendering.

Display every parameter, numerical defaults and an example:

```bash
python generate_configs.py --help
```

Generate configurations with all paths supplied explicitly:

```bash
python generate_configs.py \
  --supp-map /path/to/test_case_supp_map.json \
  --data-root /path/to/converted_datasets \
  --model-root checkpoint \
  --output-dir configs/generated \
  --gap 1
```

| Configuration | Purpose | Data source |
| --- | --- | --- |
| `<case>_gapN.py` | Per-scene training | `train/` |
| `<case>_gapN_novel.py` | Original-state novel-view rendering | `test/` |
| `<case>_gapN_resimulate.py` | Rendering after changing the initial pose | Dataset root: all `train/` + `test/` cameras (15) |

The generated files inherit common defaults, material defaults and the
case's training configuration. Novel and resimulation each inherit directly
from training and share its `base_dir`. Resimulation sets `data_dir` to the
case root and `n_cameras` to the sum of both splits, and adds rotation,
translation and center from the supplementary map. It uses the original
train/test calibration and needs no separate reinit RGB dataset.

Existing metadata supplies valid frame counts, camera counts, resolution
and frame intervals. Missing metadata uses explicit defaults and is marked
in `configs/generated/manifest.json`. Add `--require-data` to reject missing
or incompatible metadata. Use `--overwrite` when updating generated files.

Both generators need only Python's standard library. The configuration
set contains 90 case configurations and six common/material default files.

### Material defaults and frame sampling

| Material directory | MPM model | Initial physical parameters |
| --- | --- | --- |
| elastic | elasticity | E=316228, nu=0.25, rho=1000 |
| newtonian | viscous_fluid | mu=10, kappa=10000, rho=1000 |
| non_newtonian | von_mises | mu=100, kappa=100000, yield_stress=10, plastic_viscosity=1 |
| plasticine | von_mises | E=10000, nu=0.25, yield_stress=1000 |
| sand | drucker_prager | E=1000000, nu=0.3, friction_angle=10 |

Case configs inherit these physical settings. The gap1 pre-contact fitting
length is 32 frames for bg851 and 37 for bg852, following the previous case
settings. This is not an automatic collision-time detector.

When metadata is absent, the defaults are 90 source frames, 150 FPS, 13
training cameras, two test cameras and 1120x1120 images. Valid frame count
is ceil(source_frames/gap), frame_dt is gap/original_fps, and hit_frame is
scaled to preserve the original fitting time. The default maximum physical
dt is 1/6000: training uses 40 substeps per gap1 frame or 200 per gap5
frame. Novel and resimulation always default to 90 frames at 150 FPS,
with frame_dt=1/150 and 40 substeps per output frame. Gap5 training
remains 18 frames at 30 FPS. Test-time frame_dt is preserved when loading
the training checkpoint, including its learned physical parameters. Very short
sequences that cannot fit hit_frame plus warmup_step are rejected.

| Generator option | Default | Purpose |
| --- | --- | --- |
| --supp-map | Required | Supplementary case map JSON file |
| --data-root | Required | Converted datasets root |
| --model-root | Required | Root for per-material, per-case model directories |
| --gap | 1 | One or more sampling intervals |
| --source-frames / --original-fps | 90 / 150 | Render sequence and source fallback before sampling |
| --train-cameras / --test-cameras | 13 / 2 | Fallback camera counts |
| --height / --width | 1120 / 1120 | Fallback image size |
| --substeps-per-source-frame | 40 | Maximum physical dt control |
| --hit-frame | bg851=32, bg852=37 | Override fitting length at original FPS |
| --warmup-step | 3 | Additional dynamic warmup frames |
| --render-frames | Source frame count (90) | Novel/resimulation render length, independent of gap |
| --output-dir | Required | Config and manifest output directory |

In the supplementary map, rotation and transformation are in Blender
coordinates and are converted internally by PAC-NeRF. The generator keeps
them unchanged, but converts center_point to [x,z,-y], because the model
uses the rotation center directly in PAC-NeRF coordinates. This avoids
converting the translation twice.

## Per-scene training

After preparing this case's dataset and generated configuration:

```bash
CASE=ElasticFall__bg851__aZnA3h_trajectory_gap1
python train.py --config "configs/generated/elastic/${CASE}.py"
```

Training reads the configured `train/` split. Each scene has its own model;
there is no single dataset-wide checkpoint. The three stages save:

```text
checkpoint/<material>/<case>_gapN/
└── model/
    ├── train_static.pt
    ├── train_velocity.pt
    └── train_dynamic.pt
```

Inference loads `model/train_dynamic.pt` by default. The checkpoint provides
the learned representation and physical parameters.

## Inference and evaluation

Render novel views from the learned initial state:

```bash
CASE=ElasticFall__bg851__aZnA3h_trajectory_gap1
python test.py \
  --config "configs/generated/elastic/${CASE}_novel.py" \
  --render-kind novel --num-frame 90 --cam-id all
```

Change the initial particle pose and render a new simulation with the same
estimated material parameters, rendering all 15 train + test cameras:

```bash
CASE=ElasticFall__bg851__aZnA3h_trajectory_gap1
python test.py \
  --config "configs/generated/elastic/${CASE}_resimulate.py" \
  --render-kind reinit --num-frame 90 --cam-id all
```

Set `--num-frame` to the desired sequence length; the manifest records the
command for each case/gap. Each physical setup is simulated once and its
trajectory is shared by all cameras. Novel and reinit have different
initial states, so each needs its own trajectory. Reinit merges both splits
by original camera name, sorts names numerically and assigns dense render
indices; `--cam-id all` renders all 15. A numeric `--cam-id` selects from
this merged order. Novel indices remain local to the test split. The default
`--simulation-cache auto` reuses disk trajectories; `refresh` rebuilds them
and `off` shares the trajectory only in memory.

No local metric calculation or `--mode metrics` interface is provided.
Use `test.py` for both rendering modes.

## Batch training and inference

Generate batch scripts from the current configuration manifest:

```bash
python generate_scripts.py
bash scripts/generated/train.sh
bash scripts/generated/novel.sh
bash scripts/generated/resimulate.sh
```

Or run the three stages in order:

```bash
bash scripts/generated/all.sh
```

Each stage covers all 30 case/gap pairs in the current gap1 manifest.
Render commands select all cameras: two test views for novel, and all 15
train + test views for resimulation. Scripts stop on the first failure.
Regenerate scripts after changing the manifest:

```bash
python generate_scripts.py --gpu 0 --overwrite
```

Use `--material`, `--gap` and `--output-dir` for subsets. `--python` selects
the Python executable; the runtime `PYTHON` environment variable takes
priority. Without --gpu, scripts preserve CUDA_VISIBLE_DEVICES; with
--gpu they explicitly set it. Each script changes to the project directory
and uses LF newlines, including when generated on Windows.

For a separate material subset:

```bash
python generate_scripts.py --material elastic sand --output-dir scripts/elastic_sand --gpu 0
```

To generate multiple gaps and a gap5 batch:

```bash
python generate_configs.py \
  --supp-map /path/to/test_case_supp_map.json \
  --data-root /path/to/converted_datasets \
  --model-root checkpoint --output-dir configs/generated \
  --gap 1 5 --overwrite
python generate_scripts.py --gap 5 --output-dir scripts/gap5 --overwrite
```

`generate_scripts.py --gap 5` selects existing gap5 configs; it does not
resample frames. First regenerate configs with `--gap 5 --overwrite`
(or `--gap 1 5 --overwrite` for both). Then regenerate the Bash scripts.
Older manifests without render FPS are rejected with this instruction.


Existing files with different contents require --overwrite. Generators do
not start training or rendering themselves.

## Output layout

The default export root is `<base_dir>/property_estimation/`:

```text
property_estimation/
├── materials_configs.json
├── initvelocity.json
├── novel/
│   └── CineCamera_<original_id>/rgb/0000.jpg
└── reinit/
    └── CineCamera_<original_id>/rgb/0000.jpg
```

Material values are decoded from the model's parameterization. An elastic
case writes `{"ELASTIC": {"E": ..., "nu": ...}}`; velocity is
`{"vx": ..., "vy": ..., "vz": ...}`. Units and velocity coordinates follow
PAC-NeRF's simulation coordinates. Reinitialization changes particle pose
but does not rotate the estimated world-space velocity.

RGB JPEGs are written per frame, composited onto each camera's background
when supplied. `--output-dir` changes the common export root;
`--write-video true` additionally encodes `video.mp4` for each camera,
at the configured render FPS (150 by default); `--fps` changes playback only.
Camera mapping is searched in each split directory, its parent and its
sibling train folder. Use --camera-mapping to select a file. Resimulation
requires complete mappings for both train and test, with distinct original
names; the shared converter writes these in the case-root mapping file.
Missing mappings or a camera count mismatch fail before simulation. For
novel only, --camera-split train/test overrides mapping selection, and
missing mappings fall back to CineCamera_<local_index>.

--params-output is an alias for the export directory, not a JSON filename.
Use --save-params false to skip the two parameter files. --alpha exports
white-background RGB JPEGs; JPEG does not support transparency. Rerunning
replaces numbered JPEGs for selected cameras so shorter runs do not retain
stale frames.

### Material JSON format

Inference writes `property_estimation/materials_configs.json` with exactly
one of the following material objects. These examples use simple illustrative
initial values, not estimated results or reference answers:

```json
{
  "ELASTIC": {
    "E": 100000.0,
    "nu": 0.3
  }
}
```

```json
{
  "SAND": {
    "friction_angle": 30.0
  }
}
```

```json
{
  "NEWTONIAN": {
    "mu": 100.0,
    "kappa": 100000.0
  }
}
```

```json
{
  "NON_NEWTONIAN": {
    "mu": 10000.0,
    "kappa": 1000000.0,
    "yield": 1000.0,
    "visc": 1.0
  }
}
```

```json
{
  "PLASTICINE": {
    "E": 1000000.0,
    "nu": 0.3,
    "yield": 1000.0
  }
}
```

SAND exports only `friction_angle` in degrees. The exporter renames native
`yield_stress` to `yield` and `plastic_viscosity` to `visc` and excludes
`plastic_pow`. Training parameters and checkpoints keep their native names.
`initvelocity.json` retains `{"vx": ..., "vy": ..., "vz": ...}`.
See the shared [submission parameter contract](../../README.md#physical-parameter-jsons)
for accepted fields and numeric ranges. Rerun inference to refresh parameter
JSONs created with the previous export format.

## Attribution

Upstream: [xuan-li/PAC-NeRF](https://github.com/xuan-li/PAC-NeRF).
See the [paper](https://openreview.net/forum?id=tVkrbkz42vc) and
[project page](https://xuan-li.github.io/PAC-NeRF) for the original method.
The source is distributed under [LICENSE](LICENSE) (MIT, Xuan Li, 2023).
