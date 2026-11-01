# GIC-complex for PhysInOne Physical Property Estimation

GIC-complex adds SDF mesh collisions to Gaussian-Informed Continuum (GIC,
NeurIPS 2024). This method covers the 30 single-material cases with nonempty
`sdf` lists in the supplementary case map: six cases per material. Cases
without SDFs and multi-material cases are excluded. The existing complex
MPM solver, collision rules and physical estimator are retained.

## Environment

Use the same `gic` conda environment as GIC. [environment.yml](environment.yml)
uses the original compact dependency layout with versions matched to the
Python 3.9.24, PyTorch 2.4.0,
CUDA 11.8, NumPy 1.26.4 and Taichi 1.2.0. The three Git dependencies are
pinned to the installed commits.

On Linux with a CUDA 11.8 toolkit (`nvcc`) and a compatible C++ compiler:

```bash
conda env create -f environment.yml
conda activate gic
```

If that environment already exists for GIC, activate it directly.
GPU training and rendering have not been executed on this Windows editing host.

## Data and SDF boundaries

Prepare training data with the shared
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

<sdf_root>/
├── test_sdf_elephanttoy.npy
├── test_sdf_elephanttoy.json
├── test_sdf_kettle.npy
└── test_sdf_kettle.json
```

The supplementary name `test_sdf_<name>` maps to the `.npy` and `.json` pair
with that same stem. There is no `_meta` suffix. Every referenced boundary
is added to `physics.bc` as `mesh0`, `mesh1`, etc., alongside the ground
collider, for both training and inference. Missing files are listed in the
manifest; `--require-data` rejects them during generation. Inference always
requires the actual boundary files.

The existing SDF loader reads `dx_world`, uses origin [0,0,0] and maps PAC
coordinates [x,y,z] to SDF coordinates [x+0.5,-z+0.5,y]. Resimulation changes
the particles' initial pose while keeping the SDF boundaries fixed.

Training and infer fitting read foreground RGB on white from `train/data/m_*.png`
paths declared through the converter's metadata. Calibration requires 3x3
intrinsics and 3x4 or 4x4 OpenGL camera-to-world matrices in PAC coordinates.
Negative frame IDs are excluded from training. The reader preserves image
size and uses the same OpenGL-to-OpenCV camera conversion for training and
rendering.

Novel and resimulate need camera metadata and any declared static backgrounds;
they do not load ground-truth test-frame images or calculate test metrics.
Resimulate merges both splits by original `CineCamera_*` names from
`camera_mapping.json`. The mappings must be complete and non-overlapping.

## Four stages and time sampling

1. **train** runs deformable Gaussian training, velocity fitting and physical
   estimation through `train_dynamic.py`, including the SDF colliders.
2. **infer** simulates the original state once for 90 frames at 150 FPS,
   caches the particle positions and fits the fixed-particle Gaussian
   appearance once using the first five training frames by default. It
   saves a reusable Gaussian model and manifest.
3. **novel** reuses that Gaussian model and original trajectory, rendering
   all two test cameras for 90 frames at 150 FPS.
4. **resimulate** reuses the fitted Gaussian model, changes the initial
   particle pose, simulates the new state once and renders all 15 train
   and test cameras for 90 frames at 150 FPS.

Both render modes load one particle frame and render every selected view
before advancing. JPEG output is streamed to disk. Novel and resimulate do
not fit Gaussians again; missing or stale inference artifacts require an
infer run first. Compatible repeated infer calls reuse the completed fit.

| Setting | gap1 | gap5 |
| --- | --- | --- |
| Training frames / FPS | 90 / 150 | 18 / 30 |
| Training MPM steps per frame | 40 | 200 |
| Infer cached frames / FPS | 90 / 150 | 90 / 150 |
| Novel and resimulate frames / FPS | 90 / 150 | 90 / 150 |
| Inference MPM steps per frame | 40 | 40 |

Physical dt stays at 1/6000 by default. Gap5 infer fitting reads dense cache
frames 0, 5, 10, ... to match its sparse training timestamps; its fitting
loss still uses the training FPS. Training frame counts come from metadata,
while render length defaults to the unsampled source length (90).

Trajectory and fitted-artifact keys include the SDF file hashes, so changing
boundary content invalidates their reuse. Physics, static particles,
reinitialization transforms and simulator sources also identify trajectories.
Different original/reinitialized states have separate caches. Extending a
sequence beyond the cached length regenerates that trajectory once.

## Generate configurations

Both generators use only Python's standard library. All input/output paths
for configuration generation must be supplied explicitly. Run from this
method's directory using paths visible on the training host:

```bash
python generate_configs.py --help
python generate_configs.py \
  --supp-map /path/to/test_case_supp_map.json \
  --data-root /path/to/baseline \
  --sdf-root /path/to/boundary \
  --model-root output \
  --output-dir config/generated \
  --gap 5 --overwrite
```

Each gap produces 120 configs (four stages times 30 cases) and one manifest.
All stages for a case share the same model directory. Five common material
templates remain under `config/materials/`. The previous case bounds,
material defaults and complex GS pruning threshold are retained;
`reg_scale` and `reg_alpha` remain training regularizers.

| Option | Meaning / default |
| --- | --- |
| --supp-map | Required supplementary case map JSON |
| --data-root | Required converted dataset root |
| --sdf-root | Required SDF file-pair root |
| --model-root | Required model output root |
| --output-dir | Required config and manifest destination |
| --gap | Training sampling gap; default 1, accepts `1 5` |
| --original-fps / --source-frames | Render FPS / frames: 150 / 90; also source fallback |
| --height / --width | Metadata fallback: 1120 / 1120 |
| --gs-iterations / --infer-iterations | 40000 / 40000 |
| --infer-frames | First 5 training frames for appearance fitting |
| --render-frames | Unsampled source frame count (90) |
| --substeps-per-source-frame | 40; maximum physical dt control |
| --require-data | Require current calibration and all SDF assets |
| --overwrite | Update changed generated files |

The manifest records case selection, camera metadata status, SDF availability,
model paths and commands. Missing metadata uses explicit fallback values.

## Run one case

```bash
CASE=ElasticFall__bg851__wBxhzP_trajectory_gap5
python train_dynamic.py --config_path "config/generated/elastic/${CASE}_train.json"
python test.py --config "config/generated/elastic/${CASE}_infer.json" --mode infer
python test.py --config "config/generated/elastic/${CASE}_novel.json" --mode novel --cam-id all
python test.py --config "config/generated/elastic/${CASE}_resimulate.json" --mode resimulate --cam-id all
```

`--mode reinit` aliases resimulate. Without `--mode`, the config selects the
stage. `--force-infer` refits appearance; `--refresh-simulation` regenerates
the selected trajectory. `--num-frame` overrides the sequence length;
infer includes every requested fitting timestamp. `--write-video` adds a
per-camera `video.mp4` at the configured render FPS (150); `--fps` overrides
video playback only. `python test.py --help` does not import CUDA or Taichi.

Reinitialization follows Blender XYZ Euler angles, `Rz @ Ry @ Rx`, converted
to PAC coordinates. The generator converts only the center to [x,z,-y].
Learned velocity remains in world coordinates. Resimulate reuses the canonical
Gaussian appearance and particle order, without a separate reinit RGB dataset.

## Batch runs

```bash
python generate_scripts.py --help
python generate_scripts.py --gap 5 --overwrite
bash scripts/generated/train.sh
bash scripts/generated/infer.sh
bash scripts/generated/novel.sh
bash scripts/generated/resimulate.sh
# Or run all four stages in order:
bash scripts/generated/all.sh
```

The Python script generates Bash files; run the `.py` with `python` and the
resulting `.sh` with `bash`. Each stage contains 30 jobs for the generated
gap5 manifest. To include both gaps, first generate configs with `--gap 1 5`,
then regenerate scripts. Script `--gap` selects existing manifest records;
it does not change training or rendering timestamps.

Scripts use LF line endings, change to the project root and stop on failure.
`--material`, `--manifest` and `--output-dir` select subsets or custom
locations. `--gpu` sets CUDA_VISIBLE_DEVICES; omission preserves the existing
selection. `--python` sets the executable, overridden by runtime `PYTHON`.
Generating configs or scripts does not launch training.

## Output and checks

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

Iteration directories follow the configured training lengths. Particle
caches cost approximately frames * particles * 3 * 4 bytes per state.
Exported material values and velocity retain PAC simulation units.
GIC's `friction_alpha` stores degrees and is exported as SAND's
`friction_angle`. The native `0-pred.json` keeps its original format.

Portable CPU regressions exercise config generation, gap5 sampling,
projection consistency, both SDF registration paths, cache invalidation,
one appearance fit, and dense output for two novel plus 15 reinitialized views:

```bash
python -m unittest discover -s tests -v
```

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

Upstream: [Jukgei/gic](https://github.com/Jukgei/gic), Gaussian-Informed
Continuum for Physical Property Identification and Simulation (NeurIPS 2024).
Original copyright notices and [LICENSE](LICENSE) are retained.
