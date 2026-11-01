# GIC for PhysInOne Physical Property Estimation

This folder adapts Gaussian-Informed Continuum (GIC, NeurIPS 2024) for
30 single-material cases without SDF boundaries, six per material.
Complex-boundary and multi-material cases are excluded. The physical
estimator and MPM implementation are retained.

## Environment

[environment.yml](environment.yml) follows the original GIC dependency layout,
Python 3.9.24, PyTorch 2.4.0, CUDA 11.8, NumPy 1.26.4 and Taichi 1.2.0.
The three Git dependencies are pinned to the commits installed in that environment.

On Linux with a CUDA 11.8 toolkit (`nvcc`) and a compatible C++ compiler:

```bash
conda env create -f environment.yml
conda activate gic
```

All dependencies are declared in this single file. Installation and GPU execution
have not been tested on this Windows editing host.

## Data

Prepare data with the shared
[conversion script](../../data_preprocess/create_dataset.py)
and [usage guide](../../data_preprocess/README.md):

```text
<data_root>/<case>_gapN/
├── camera_mapping.json
├── train/
│   ├── all_data.json
│   └── data/m_<camera>_<frame>.png
└── test/
    ├── all_data.json
    └── data/r_<camera>_-1.jpg       # optional static background
```

Training and infer fitting use foreground RGB-on-white `m_*.png` images
from `train/`. Metadata needs 3x3 intrinsics, 3x4 or 4x4 OpenGL camera-to-world
matrices in PAC coordinates and timestamps in seconds. Background/mask
entries with negative frame IDs are excluded. The camera reader preserves
image size and converts the OpenGL camera basis to the rasterizer's
OpenCV convention for both training and rendering.

Novel and resimulate read calibration and optional static backgrounds.
They do not open test-frame RGB or foreground images. Declared background
files must exist; cameras without a background entry use white.
Original `CineCamera_*` names come from `camera_mapping.json`. Resimulate
requires complete, non-overlapping train/test mappings because each split
numbers its cameras independently.

## Four stages

1. **train** runs the original deformable Gaussian training, velocity fitting
   and physical-parameter estimation through `train_dynamic.py`.
2. **infer** runs the learned original-state simulation once, caches particle
   positions, and performs the required fixed-particle Gaussian appearance
   fitting once using the training split. Only the fitting prefix images
   are loaded (five frames by default). It saves a reusable Gaussian model.
3. **novel** loads that model and the original trajectory, rendering all two
   test cameras without fitting again.
4. **resimulate** loads the same fitted model, transforms the initial particles,
   simulates the new state once and renders all 15 train + test cameras.

Both render modes share one trajectory frame across every selected view.
They stream JPEGs to disk and keep only the current rendered sequence frame
on the GPU. No training is invoked inside novel/resimulate. If infer results
are missing or stale, rendering stops with a request to run infer first.

The fixed-Gaussian artifact is checked against the static particle cloud,
physical prediction, training metadata, fitting settings and implementation.
Trajectory keys include physics, particle cloud, transform and simulator
sources. Compatible disk trajectories are reused. Different original and
reinitialized states have separate trajectories. Extending a sequence past
its cached length regenerates that trajectory once.

## Generate configurations

The generator uses only Python's standard library. Every path is explicit:

```bash
python generate_configs.py --help
python generate_configs.py \
  --supp-map /path/to/test_case_supp_map.json \
  --data-root /path/to/converted_datasets \
  --model-root output \
  --output-dir config/generated \
  --gap 5 --overwrite
```

Run commands from this method's directory and supply paths visible to the
training host. There are 120 case configs
(four stages times 30 cases), five retained material templates under
`config/materials/`, and `config/generated/manifest.json` with commands and
metadata status. All four configs for a case share one model directory.

| Path option | Required input |
| --- | --- |
| --supp-map | Supplementary case map JSON |
| --data-root | Root containing converted case train/test folders |
| --model-root | Root for per-material/per-case models |
| --output-dir | Destination for configs and manifest |

| Numerical option | Default |
| --- | --- |
| --gap | 1; accepts multiple values |
| --original-fps / --source-frames | 150 / 90, used when metadata is absent |
| --height / --width | 1120 / 1120, used when metadata is absent |
| --gs-iterations | 40000, initial deformable GS training |
| --infer-iterations | 40000, one-time fixed-particle GS fitting |
| --infer-frames | First 5 training frames |
| --render-frames | Actual training frame count |
| --substeps-per-source-frame | 40, maximum physical dt control |

Valid metadata supplies frame count, interval, resolution and camera count.
Missing metadata is marked as assumed in the manifest. `--require-data`
rejects absent or incompatible metadata, including legacy [FOV,W,H]
intrinsics. For fallback data, gap1 has 90 frames, FPS=150 and 40 MPM steps
per frame; gap5 has 18 frames, FPS=30 and 200 steps. Both use physical
dt=1/6000. Material defaults are preserved. The two existing light granular
cases `MoyIIZ` and `tDtaft` retain voxel_size=0.025 and density_grid_size=0.16.
`reg_scale` and `reg_alpha` remain training regularizers.

Reinitialization uses Blender XYZ Euler angles, `Rz @ Ry @ Rx`, and converts
rotation/translation to PAC coordinates. The generator converts only the
rotation center to [x,z,-y]. Learned velocity remains in world coordinates
and is not rotated. Resimulate reuses fitted appearance and particle order;
it does not fit against a reinitialized video.

## Run one case

```bash
CASE=ElasticFall__bg851__aZnA3h_trajectory_gap1
python train_dynamic.py --config_path "config/generated/elastic/${CASE}_train.json"
python test.py --config "config/generated/elastic/${CASE}_infer.json" --mode infer
python test.py --config "config/generated/elastic/${CASE}_novel.json" --mode novel --cam-id all
python test.py --config "config/generated/elastic/${CASE}_resimulate.json" --mode resimulate --cam-id all
```

`--mode reinit` aliases resimulate. Without `--mode`, `test.py` uses the
stage in the config. `--force-infer` refits the inference Gaussians;
`--refresh-simulation` regenerates the selected trajectory. `--num-frame`
overrides sequence length; infer always includes at least its fitting
prefix. `--write-video` adds per-camera `video.mp4` at `--fps` (default 30).
`python test.py --help` works without importing CUDA/Taichi.

Novel numeric camera IDs are local to the test split. Resimulate IDs follow
the numerically sorted original camera names across both splits. The
standard dataset renders two novel views and 15 resimulated views.

## Batch runs

```bash
python generate_scripts.py --help
python generate_scripts.py --overwrite
bash scripts/generated/train.sh
bash scripts/generated/infer.sh
bash scripts/generated/novel.sh
bash scripts/generated/resimulate.sh
# Or run all four stages in order:
bash scripts/generated/all.sh
```

Each stage contains 30 commands for the current gap1 manifest. Scripts
change to the project directory, use LF line endings and stop on failure.
`--gpu` sets CUDA_VISIBLE_DEVICES; without it the existing selection is
preserved. `--python` sets the executable, overridden by runtime `PYTHON`.
Use `--material`, `--gap`, `--manifest` and `--output-dir` for subsets or a
custom config destination. Generators write files without launching jobs.

## Output

```text
<model_path>/
├── 0-pred.json
├── mpm/static_0.ply
├── point_cloud/iteration_40000/point_cloud.ply
├── point_cloud_fix_pcd/iteration_40000/point_cloud.ply
├── inference/
│   ├── manifest.json
│   └── trajectories/<key>/000000.npy
└── property_estimation/
    ├── materials_configs.json
    ├── initvelocity.json
    ├── novel/CineCamera_*/rgb/0000.jpg
    └── reinit/CineCamera_*/rgb/0000.jpg
```

Iteration directory names follow the configured training lengths. Raw
particle caches cost roughly frames * particles * 3 * 4 bytes per state.
The property JSON files export learned values in PAC simulation units.
GIC's `friction_alpha` prediction stores degrees and is exported as SAND's
`friction_angle`. The native `0-pred.json` keeps its original format.

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

Upstream: [Jukgei/gic](https://github.com/Jukgei/gic),
Gaussian-Informed Continuum for Physical Property Identification and
Simulation (NeurIPS 2024). The original licensing notices are retained;
see [LICENSE](LICENSE).
