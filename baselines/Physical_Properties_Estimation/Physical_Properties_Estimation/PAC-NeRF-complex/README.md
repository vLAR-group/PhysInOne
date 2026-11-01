# PAC-NeRF-complex for PhysInOne Physical Property Estimation

This folder extends [PAC-NeRF](https://github.com/xuan-li/PAC-NeRF) with
voxel SDF mesh collisions for complex static boundaries. It estimates
geometry, initial velocity and material parameters independently per scene.
The complex-boundary MPM implementation and material defaults are retained.

## Release status

- Training: `train.py`, with static reconstruction, initial-velocity fitting
  and material-parameter fitting.
- Inference: one `test.py` entry for novel views and resimulation, with one
  simulation shared by all selected cameras.
- Coverage: 30 single-material cases with nonempty `sdf` lists, six each for
  elastic, Newtonian, non-Newtonian, plasticine and granular materials.
  The supplementary map also contains three SDF multi-material records;
  these are excluded because this model fits one material per scene.
- Inference reads calibration and optional static backgrounds, without
  opening ground-truth video frames or calculating metrics.

## Installation

Use the same `pacnerf` environment as PAC-NeRF. The supplied
[environment.yml](environment.yml) uses versions from the local conda
package snapshot: Python 3.9, PyTorch 2.5.1 with CUDA 11.8 and Taichi 1.2.0.
Reuse an existing `pacnerf` environment, or create it with:

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

## Data and complex boundaries

Prepare camera data with the shared
[data preparation script](../../data_preprocess/create_dataset.py)
and its [usage guide](../../data_preprocess/README.md):

```text
<data_root>/<case>_gapN/
├── camera_mapping.json
├── train/
│   ├── all_data.json
│   └── data/
│       ├── r_0_0.jpg
│       ├── m_0_0.png
│       └── r_0_-1.jpg
└── test/
    ├── all_data.json
    └── data/r_0_-1.jpg

<sdf_root>/
├── mesh_sdf_elephanttoy.npy
├── mesh_sdf_meta_elephanttoy.json
├── mesh_sdf_kettle.npy
└── mesh_sdf_meta_kettle.json
```

Each camera entry contains `file_path`, `time`, a 3x4 `c2w` and a 3x3
`intrinsic` matrix. Negative frame IDs describe backgrounds or masks and
are excluded from generated training frame counts. The training loader
uses pre-generated `m_<camera>_<frame>.png` foreground RGB on white;
background matting is not run during training.

Novel rendering needs test calibration. Resimulation needs both train and
test calibration: 13 + 2 = 15 cameras. Frame paths may appear in metadata
without the corresponding ground-truth RGB images existing. Declared
static background images must exist when declared in the metadata.

The converter writes split-local indices and original `CineCamera_*`
names to `camera_mapping.json`. Resimulation requires complete mappings
for both splits and distinct original camera names. It merges cameras by
name, sorts numeric camera suffixes and renders all 15 with `--cam-id all`.
A numeric selection addresses this merged order; novel selections remain
local to the test split.

The supplementary SDF name `test_sdf_<name>` resolves to
`mesh_sdf_<name>.npy` and `mesh_sdf_meta_<name>.json`. The generator adds
these file pairs to `cfg.BC` as `mesh`, `mesh_1`, etc. The common ground
collider is inherited alongside the meshes. The existing complex
simulator registers every mesh in training and inference. Resimulation
changes the particles' initial pose while retaining the same boundaries.

SDF metadata supplies `dx_world`; the existing loader uses origin [0,0,0].
The collision transform maps PAC coordinates [x,y,z] to SDF coordinates
[x+0.5,-z+0.5,y]. Existing collision settings remain unchanged:
separating contact, zero friction and the loader's 50-voxel contact band.

## Configuration generation

Both generators use only Python's standard library. All configuration
path arguments are required: `--supp-map`, `--data-root`, `--sdf-root`,
`--model-root` and `--output-dir`. The config generator has no default
paths. Run the commands from this method's directory.
Use paths visible on the host that will run training and rendering.

Display every parameter, numerical defaults and an example:

```bash
python generate_configs.py --help
```

Generate configs and then the batch scripts, replacing the input paths:

```bash
python generate_configs.py \
  --supp-map /path/to/test_case_supp_map.json \
  --data-root /path/to/baseline \
  --sdf-root /path/to/boundary \
  --model-root checkpoint \
  --output-dir configs/generated \
  --gap 1 --overwrite
python generate_scripts.py --overwrite
```

| Configuration | Purpose | Camera source |
| --- | --- | --- |
| `<case>_gapN.py` | Training with SDF mesh collisions | `train/` (13) |
| `<case>_gapN_novel.py` | Original-state novel-view rendering | `test/` (2) |
| `<case>_gapN_resimulate.py` | Changed initial particle pose | Case root: `train/` + `test/` (15) |

Novel and resimulation inherit the training config directly. They share
its model directory and SDF boundaries. The generated set contains 90 case
configs plus six common/material defaults. Case selection, SDF file
availability, camera metadata, transforms and commands are recorded in
`configs/generated/manifest.json`.

Existing metadata supplies frame count, resolution, camera count and frame
interval. Missing metadata uses explicit fallback defaults and is marked
in the manifest. `--require-data` rejects missing or incompatible camera
metadata and missing SDF file pairs. Use `--overwrite` to update files.

| Option | Default | Purpose |
| --- | --- | --- |
| --supp-map | Required | Supplementary case map JSON file |
| --data-root | Required | Converted datasets root |
| --sdf-root | Required | Directory containing SDF arrays and metadata |
| --model-root | Required | Root for per-material, per-case model directories |
| --gap | 1 | One or more frame sampling intervals |
| --source-frames / --original-fps | 90 / 150 | Render sequence and source fallback before sampling |
| --train-cameras / --test-cameras | 13 / 2 | Fallback camera counts |
| --height / --width | 1120 / 1120 | Fallback resolution |
| --substeps-per-source-frame | 40 | Maximum physical dt control |
| --hit-frame | elephanttoy=17, kettle=19 | Pre-contact fitting length at original FPS |
| --warmup-step | 3 | Additional dynamic warmup frames |
| --render-frames | Source frame count (90) | Novel/resimulation render length, independent of gap |
| --output-dir | Required | Config and manifest output directory |

Training frame count is ceil(source_frames/gap); training frame_dt is
gap/original_fps. The physical dt is at most 1/6000 by default: training
uses 40 substeps per gap1 frame or 200 per gap5 frame. Novel and resimulation
always default to 90 frames at 150 FPS, with frame_dt=1/150 and 40 substeps
per output frame. Gap5 training remains 18 frames at 30 FPS. Test-time
frame_dt is preserved when loading the training checkpoint and its learned
physical parameters. The pre-contact fitting length is scaled to preserve elapsed
time, using the previous complex-boundary configuration values of 17/19
frames. This is not an automatic contact detector. Sequences too short for
the fitting length plus warmup are rejected.

| Material directory | MPM model | Initial parameters |
| --- | --- | --- |
| elastic | elasticity | E=316228, nu=0.25, rho=1000 |
| newtonian | viscous_fluid | mu=10, kappa=10000, rho=1000 |
| non_newtonian | von_mises | mu=100, kappa=100000, yield_stress=10, plastic_viscosity=1 |
| plasticine | von_mises | E=10000, nu=0.25, yield_stress=1000 |
| sand | drucker_prager | E=350000, nu=0.3, friction_angle=10 |

The existing elastic ray batch size of 2^17 and granular E=350000 are
retained. `GranularFall__bg853__ZBMM7p_trajectory` retains its existing
dx=0.12 override; other cases inherit their material defaults.

Supplementary rotation and transformation are in Blender coordinates and
are converted internally by PAC-NeRF. Only center_point is converted by
the generator to [x,z,-y], because the model consumes the center directly
in PAC coordinates. Estimated world-space velocity is not rotated.

## Training and inference

```bash
CASE=ElasticFall__bg851__wBxhzP_trajectory_gap1
python train.py --config "configs/generated/elastic/${CASE}.py"
python test.py --config "configs/generated/elastic/${CASE}_novel.py" \
  --render-kind novel --num-frame 90 --cam-id all
python test.py --config "configs/generated/elastic/${CASE}_resimulate.py" \
  --render-kind reinit --num-frame 90 --cam-id all
```

Training saves `model/train_static.pt`, `model/train_velocity.pt` and
`model/train_dynamic.pt` below each case's model directory. Inference
loads the dynamic checkpoint by default; `--stage` selects another stage.
`--render-kind resimulate` is an alias for `reinit`.

Each render invocation initializes the simulation once. For each frame it
advances physics once, builds one feature grid and renders every selected
camera from that grid. Images are streamed to disk. Novel and reinit have
different initial states and each runs its own simulation. This complex
implementation performs one streaming simulation/render pass per command.

CLI `--rotation`, `--translation` and `--center` accept comma-separated
triples to override resimulation transforms. Camera data, boundaries and
transforms in the current config take precedence over checkpoint settings.
No local metric calculation interface remains.

## Batch runs

```bash
bash scripts/generated/train.sh
bash scripts/generated/novel.sh
bash scripts/generated/resimulate.sh
# Or run all three stages in order:
bash scripts/generated/all.sh
```

Each stage contains 30 jobs for the current gap1 manifest. Novel commands
render both test views; resimulation commands render all 15 train + test
views. Scripts stop on the first failure and change to the project root.
The generators only write files; they do not start training or rendering.

```bash
python generate_scripts.py --gpu 0 --overwrite
python generate_scripts.py --material elastic sand --output-dir scripts/elastic_sand
python generate_configs.py \
  --supp-map /path/to/test_case_supp_map.json \
  --data-root /path/to/baseline \
  --sdf-root /path/to/boundary \
  --model-root checkpoint --output-dir configs/generated \
  --gap 1 5 --overwrite
python generate_scripts.py --gap 5 --output-dir scripts/gap5 --overwrite
```

`generate_scripts.py --gap 5` selects existing gap5 configs; it does not
resample frames. First regenerate configs with `--gap 5 --overwrite`
(or `--gap 1 5 --overwrite` for both). Then regenerate the Bash scripts.
Older manifests without render FPS are rejected with this instruction.


`--python` selects the executable, overridden by runtime `PYTHON` when set.
Without `--gpu`, scripts preserve `CUDA_VISIBLE_DEVICES`. With `--gpu`,
they set it explicitly. Generated scripts use LF line endings on Windows
and Linux. Regenerate the scripts after changing the configuration manifest.

## Output layout

```text
<base_dir>/property_estimation/
├── materials_configs.json
├── initvelocity.json
├── novel/CineCamera_<original_id>/rgb/0000.jpg
└── reinit/CineCamera_<original_id>/rgb/0000.jpg
```

An elastic case exports `{"ELASTIC": {"E": ..., "nu": ...}}`; velocity is
`{"vx": ..., "vy": ..., "vz": ...}`. Values follow PAC simulation units and
coordinates. `--output-dir` changes the common root; `--params-output` is
an alias. `--save-params false` skips the two parameter files.

RGB is composited over each camera's static background when declared in
metadata. Cameras without a declared background use white. JPEG has no
transparency.
`--write-video true` additionally writes `video.mp4` per camera at `--fps`
(default: configured render FPS, 150). This option changes playback only.
Rerunning replaces numbered JPEGs for selected cameras.
`--camera-mapping` selects a mapping file explicitly. `--camera-split` is a
novel-only override; resimulation always reads both splits. Novel falls
back to local camera names if a mapping is missing; resimulation rejects
missing, incomplete or overlapping split mappings before simulation.

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
[project page](https://xuan-li.github.io/PAC-NeRF). The complex static SDF
boundary extension is local to this adaptation. See [LICENSE](LICENSE).
