# Physical Properties Estimation

Estimate a scene's material parameters and initial velocity from its public
training views, then render the learned scene from novel views and simulate it
after changing the initial particle pose. The submission contains physical
parameters and RGB sequences for both settings.

## Task setting

The current baseline and submission-checking scope is **60 SinglePhysics
cases**, with one material per case. DoublePhysics and TriplePhysics cases are
outside this scope.

| Dimension | Setting |
| --- | --- |
| Material categories | ELASTIC, NEWTONIAN, NON_NEWTONIAN, PLASTICINE, SAND |
| Cases per material | 12: six without SDF boundaries and six with SDF boundaries |
| Boundary groups | 30 no-SDF cases and 30 SDF cases |
| Static cameras | 15: 13 training views and 2 novel views |
| Public RGB input | Training cameras only, with segmentation and background images |
| Novel output | Both held-out cameras, using the original initial state |
| Reinitialized output | All 15 static cameras, using the supplied initial-pose change |
| Output sequence | 90 RGB frames, `0000` through `0089`, at 150 FPS |
| Output image size | 1120 x 1120, matching the released camera metadata |
| Moving camera | Excluded |

**Novel-camera membership varies by scene.** Preserve the original
`CineCamera_*` names. The converter writes split-local camera indices and their
original names to `camera_mapping.json`; never treat those split-local indices
as submission camera names.

Material and initial-velocity values must describe the learned scene. Novel
rendering keeps its original initial state. Reinitialization uses the rotation,
translation and center in `test_case_supp_map.json`, while retaining the same
material parameters, estimated world-space velocity and boundary geometry.

## Public download and supplementary resources

Follow the [Leaderboard download guide](https://huggingface.co/datasets/vLAR/PhysInOne/blob/main/Utils/LEADERBOARD_DOWNLOAD.md)
and select **Physical Properties Estimation**. From the PhysInOne repository
root containing `scripts/download_data.py`, download the current single-case
scope and its supplementary resources together:

```bash
python scripts/download_data.py \
  --task physical-properties-estimation \
  --scene SinglePhysics \
  --scene test_supp_info \
  --output-dir ./data \
  --extract
```

This selects **62 files: 60 scene ZIPs, `Test_sdf.zip`, and
`test_case_supp_map.json`**. Public downloads require no access token. The scene
filters are combined with OR; omitting `test_supp_info` would exclude the shared
resources. `--subset` is not a SinglePhysics selector.

The local task root is `data/leaderboard/physical_properties_estimation/`:

```text
physical_properties_estimation/
├── SinglePhysics/
│   ├── <case>.zip
│   └── <case>/
└── test_supp_info/
    ├── test_case_supp_map.json
    ├── Test_sdf.zip
    └── Test_sdf/
        ├── test_sdf_elephanttoy.npy
        ├── test_sdf_elephanttoy.json
        ├── test_sdf_kettle.npy
        └── test_sdf_kettle.json
```

The [supplementary directory](https://huggingface.co/datasets/vLAR/PhysInOne/tree/main/Leaderboard/Physical%20Properties%20Estimation/test_supp_info)
provides both the SDF archive and the mapping JSON. Use the JSON for case
selection, SDF references and reinitialization transforms. It includes cases
outside the current 60-case scope; the baseline generators filter those out.
Point `--sdf-root` at the extracted `Test_sdf` directory containing the actual
`.npy`/`.json` pairs. Their names use `test_sdf_` and have no `_meta` suffix.

Public scene packages provide training images and segmentation, camera JSONs,
backgrounds, `particle_materials.json`, and `recorder_stats.json`. Novel cameras
have backgrounds and calibration, without target RGB. See the download guide
for the complete release description.

## Prepare baseline inputs

Run the following commands from this directory. Set paths visible to your
training host; the paths below are examples:

```bash
TASK_DATA=/absolute/path/to/data/leaderboard/physical_properties_estimation
CONVERTED=/absolute/path/to/converted_baseline_data
SUPP_MAP="$TASK_DATA/test_supp_info/test_case_supp_map.json"
SDF_ROOT="$TASK_DATA/test_supp_info/Test_sdf"

python data_preprocess/create_dataset.py \
  --origin-path "$TASK_DATA/SinglePhysics" \
  --output-path "$CONVERTED" \
  --gap 1 5 \
  --original-fps 150 \
  --test-split-mode train-only
```

The [conversion script](data_preprocess/create_dataset.py) and
[conversion guide](data_preprocess/README.md) describe its dependencies and
options. `train-only` identifies training cameras by public RGB availability
and preserves background-only novel cameras as metadata. It avoids replacing
the released split with a random split.

```text
<converted_root>/<case>_gapN/
├── camera_mapping.json
├── train/
│   ├── all_data.json
│   └── data/m_<local_camera>_<frame>.png
└── test/
    ├── all_data.json
    └── data/r_<local_camera>_-1.jpg
```

Foreground `m_*.png` images contain object RGB on white, not binary masks.
Background records have negative frame IDs and do not count as training frames.
Novel metadata can reference target images that are absent from public inputs.
The converter supplies the 3x3 intrinsics and PAC camera coordinates required
by the baselines.

Sampling gap controls training data only:

| Stage | gap1 | gap5 |
| --- | --- | --- |
| Per-scene training | 90 frames at 150 FPS | 18 frames at 30 FPS |
| GIC infer simulation cache | 90 frames at 150 FPS | 90 frames at 150 FPS |
| Novel rendering | 90 frames at 150 FPS | 90 frames at 150 FPS |
| Reinitialized rendering | 90 frames at 150 FPS | 90 frames at 150 FPS |

Use `--original-fps 150` explicitly when converting this task. Legacy camera
JSON FPS headers do not override the current baseline output setting. Do not
submit only 18 frames for a gap5-trained model.

## Baseline methods

| Method | Cases | Environment | Usage guide |
| --- | --- | --- | --- |
| PAC-NeRF | 30 without SDF | pacnerf | [README](Physical_Properties_Estimation/PAC-NeRF/README.md) |
| PAC-NeRF-complex | 30 with SDF | pacnerf | [README](Physical_Properties_Estimation/PAC-NeRF-complex/README.md) |
| GIC | 30 without SDF | gic | [README](Physical_Properties_Estimation/gic/README.md) |
| GIC-complex | 30 with SDF | gic | [README](Physical_Properties_Estimation/gic-complex/README.md) |

A complete submission combines one family's standard and complex variants:
**PAC-NeRF + PAC-NeRF-complex**, or **GIC + GIC-complex**. Keep different
families in separate submission directories; including all four methods in
one directory would duplicate scenes. Each method fits a separate model per
case rather than one model for the whole dataset.

Each method includes `generate_configs.py` and `generate_scripts.py`.
Configuration paths are required explicitly: `--supp-map`, `--data-root`,
`--model-root`, and `--output-dir`; complex methods additionally require
`--sdf-root`. Supply the downloaded mapping JSON and converted data root.
Use `--gap 5 --overwrite` for the current sparse-training setup.

Run the Python generators with `python`, then run their generated `.sh` files
with `bash`. PAC-NeRF variants generate train, novel, resimulate and all-stage
scripts. GIC variants also generate an independent infer script: run it after
training to fit the fixed-particle Gaussian appearance once. Novel reuses that
fit and its original trajectory. Resimulate reuses the fit and simulates the
changed initial state once. All views share their corresponding simulation.

`generate_scripts.py --gap 5` filters gap5 records already in the manifest; it
does not resample frames or repair old configs. Regenerate configs first when
updating sampling settings. Training must use public training-view data only;
novel and reinitialized target RGB are reserved for private evaluation.

## Package predictions

All four methods write the same payload under
`<model_path>/property_estimation/`:

```text
property_estimation/
├── materials_configs.json
├── initvelocity.json
├── novel/
│   ├── CineCamera_<novel_a>/rgb/0000.jpg
│   ├── ...
│   └── CineCamera_<novel_b>/rgb/0089.jpg
└── reinit/
    ├── CineCamera_0/rgb/0000.jpg
    ├── ...
    └── CineCamera_14/rgb/0089.jpg
```

Each camera directory contains every frame `0000` through `0089`. Both novel
views and all 15 reinit views are required. These are two different initial
states; novel images cannot be copied into reinit as a replacement for the
requested simulation.

Create **one ZIP per official case identifier**, including `_trajectory` and
excluding the training suffix `_gap1` or `_gap5`. The ZIP root must contain
one `property_estimation/` directory, with the two JSON files and the `novel/`
and `reinit/` directories directly inside it. The online Worker does not
remove outer directories. Bare payloads, `<scene>/` wrappers and
`<scene>/property_estimation/` wrappers are not supported by the checker or
the online Worker.
For example, from a directory where `submissions/` should be created:

```bash
CASE=ElasticFall__bg851__aZnA3h_trajectory
MODEL_DIR=/absolute/path/to/output/elastic/${CASE}_gap5
mkdir -p submissions
python -m zipfile -c "submissions/${CASE}.zip" "$MODEL_DIR/property_estimation"
```

Repeat for all 60 cases, combining the two boundary groups of the selected
family. Physics/material category directories outside the ZIPs are optional.
A full submission has **60 ZIPs and 91,800 RGB images**: 90 frames times
17 output views per case. The checker recursively searches the submission
folder for ZIPs and rejects missing, duplicate or extra scene archives.

Images may use JPG, JPEG or PNG, with an extension matching their actual
encoding. Preserve RGB mode, the original camera names and 1120x1120 resolution.
Submit predictions and the two parameter JSONs only. Exclude model weights,
`0-pred.json`, training images, depth, masks, metric reports, caches and optional
MP4 videos from the archives. The on-disk submission stage is named `reinit`,
even though the command/script mode is named `resimulate`.

### Physical parameter JSONs

`materials_configs.json` contains exactly one uppercase material object for
the case. All four baselines use the same inference export format:

| Material | Required parameters |
| --- | --- |
| ELASTIC | E, nu |
| SAND | friction_angle |
| NEWTONIAN | mu, kappa |
| NON_NEWTONIAN | mu, kappa, yield, visc |
| PLASTICINE | E, nu, yield |

`E`, `mu`, `kappa`, `yield` and `visc`, where required for the material,
must be strictly positive because the scorer compares them in log10 space.
Zero is invalid for these parameters; do not replace predictions with an
epsilon to pass validation. This restriction does not apply to velocity
components, which may be zero or negative.

The following examples use simple illustrative initial values, not estimated
results or reference answers. Each code block represents a separate
`materials_configs.json`:

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

`initvelocity.json` contains exactly three finite numeric components in the
baseline's PAC simulation coordinates, retaining their signs:

```json
{"vx": 0.0, "vy": -0.1, "vz": 0.2}
```

## Check before uploading

The offline [check.py](check.py) uses the bundled
[60-case submission manifest](dataset/submission_manifest.json). The manifest
contains scene identifiers, material labels and camera membership. Common
frame targets and image dimensions are stored once under `defaults`; individual
scenes may override them. No source paths, hashes, boundary flags or FPS records
are included, since those are not used to validate the submitted files. It contains no target material
values, target velocities or reference RGB. Keep the checker and manifest
together; no GT, network, authentication or GPU is needed.

From this directory:

```bash
pip install pillow numpy opencv-python-headless
python check.py ./submissions --report ./ppe_check_report.json
```

| Argument | Purpose |
| --- | --- |
| submission_dir | Folder containing the 60 scene ZIPs; searched recursively |
| --report PATH | Write a detailed report to a new JSON file |
| --verbose | Print every error and warning |
| --fail-on-warning | Treat image-content warnings as failures |
| --manifest PATH | Use another release manifest; defaults to the bundled single60 file |

The checker validates coverage, ZIP integrity, safe paths, both parameter
JSONs, material-specific fields and ranges, exact novel/reinit cameras, frame
names, image encoding, full decoding and resolution. It checks images with
Pillow and OpenCV. Entirely repeated sequences and uniform black/white frames
produce warnings. Progress shows elapsed time and estimated time remaining.

The checker enforces this tutorial's recommended submission format, not every
input variation the Worker can parse. In particular, it requires four-digit
frame names (`0000` through `0089`), exact JSON fields and RGB images at the
release resolution. The Worker parses frame indices as integers, so unpadded
names such as `0.png` may pass its structural checks but are intentionally
rejected here. Use the documented format rather than relying on looser parsing.

JPEG/PNG sequences have no FPS field, so the checker verifies their 90 frame
identifiers; it cannot prove that they were simulated at 150 FPS. Likewise,
valid parameter ranges cannot establish parameter accuracy or detect every
unit/log-space mistake. The script does not compute leaderboard metrics or
modify prediction archives.

Passing these checks does not guarantee a correct simulation, an accurate
prediction, a valid final score or successful online execution; resource and
network failures are outside the checker's scope.

Exit codes: **0** = valid, possibly with warnings; **1** = submission validation
failed; **2** = command, dependency or configuration error. Inspect warnings
before uploading. Existing report files are not overwritten. Upload the
validated scene archives to the repository/subdirectory selected in the
Leaderboard submission form; private-reference scoring is separate.

## Checker regression tests

[test_check.py](test_check.py) creates small synthetic archives and validates
success and failure cases, including outputs from the parameter-export
functions of all four methods. GPU simulation and rendering are not required:

```bash
python -m unittest discover -s . -p test_check.py -v
```
