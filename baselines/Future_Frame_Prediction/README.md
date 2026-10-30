# Long-Term Future Frame Prediction

Predict how a physical scene evolves **after its observed video ends**. The task
evaluates future RGB predictions at previously observed camera viewpoints and,
optionally, at held-out viewpoints. It is temporal extrapolation, not
reconstruction of the observed frames.

## Task setting

The evaluation set comprises **103 scenes: 84 Unreal Engine scenes, 10 Doris
scenes, and 9 MPM scenes**. The standardized per-scene setting is:

| Dimension | Setting |
| --- | --- |
| Cameras | 12 static cameras: 10 training views and 2 novel views |
| Observed interval | First half of the scene's timeline |
| Public RGB input | Observed-interval RGB from the 10 training views only |
| Prediction interval | Second half of the timeline |
| Required output | Future RGB for all 10 training views |
| Optional output | Future RGB for both novel views |
| Moving camera | Excluded from this task |
| Ground truth | Private; not part of the public download |

**Camera membership and frame targets come from the official scene JSON files,
not from assumed camera numbers or a hard-coded video length.** Use the scene
list, archives, and transforms from the same release.

Here, **training view** means an observed camera within an evaluation scene, not
the dataset training split. Novel cameras have public poses but no input RGB.
Their camera numbers vary between scenes.

### Observed frames versus future frames

For example, a 150-frame video provides `0000–0074` and requires `0075–0149`;
a 90-frame video uses `0000–0044` and `0045–0089` instead. Read exact targets from
`transforms_test.json`; preserve frame identifiers, poses, and timestamps.
Do not renumber future frames from zero or assume a fixed 75-frame horizon.

## Public inputs and metadata

| File | Purpose | Does it provide target RGB? |
| --- | --- | --- |
| `transforms_train.json` | Observed frames and poses for training cameras | Only its public observed RGB is available |
| `transforms_val.json` | Held-out camera membership and associated pose/time metadata | No |
| `transforms_test.json` | Exact future camera/frame targets to predict | No; its RGB paths describe targets, not downloadable images |

Future RGB and novel-camera RGB are intentionally absent from public inputs.
JSON target paths do not make those images available for training. Use only the
public observed frames; do not obtain hidden RGB from another task or scene variant.

## Download and run a baseline

Use the [Hugging Face Leaderboard download guide](https://huggingface.co/datasets/vLAR/PhysInOne/blob/main/Utils/LEADERBOARD_DOWNLOAD.md)
for the public download tools and current release. With `scripts/download_data.py`
available, run:

```bash
python scripts/download_data.py \
  --task future-prediction \
  --output-dir ./data \
  --extract
```

The local task directory is `data/leaderboard/future_prediction/`, with physics
category subdirectories and one extracted directory per scene. No access token
is required for these public inputs. The `--subset mini/large` option belongs to
Video Generation, not Future Prediction.

Baseline-specific installation, training, checkpoint, and inference instructions:

| Baseline | Usage guide |
| --- | --- |
| FreeGave | [FreeGave README](Long-term_Future_Frame_Prediction/FreeGave/README.md) |
| DefGS | [DefGS README](Long-term_Future_Frame_Prediction/Def3G/README.md) (legacy code directory: `Def3G`) |
| TRACE | [TRACE README](Long-term_Future_Frame_Prediction/TRACE/README.md) |
| TiNeuVox | [TiNeuVox README](Long-term_Future_Frame_Prediction/TiNeuVox/README.md) |
| MAGI-1 | [MAGI-1 README](Long-term_Future_Frame_Prediction/MAGI-1/README.md) |
| ExtDM | [ExtDM README](Long-term_Future_Frame_Prediction/ExtDM/README.md) |

Scene-optimization methods must fit only the public observed frames. Video
models may condition on those frames without per-scene training. Any model
pretraining or fine-tuning must respect the held-out evaluation data boundary.
Inference should save predictions only; private-reference scoring is performed
by the leaderboard.

## Package predictions

Produce one scene archive containing camera directories and predicted future
RGB. For example, for the 150-frame setting:

```text
<scene_id>.zip
├── CineCamera_<training_view_a>/
│   └── rgb/
│       ├── 0075.jpg
│       ├── ...
│       └── 0149.jpg
├── ...                         # all other training cameras
├── CineCamera_<novel_view_a>/   # optional: include both novel cameras or neither
│   └── rgb/...
└── CineCamera_<novel_view_b>/
    └── rgb/...
```

Use the official scene identifier (including `_trajectory`) for the archive name,
and the exact camera names and frame identifiers in `transforms_test.json`.
Images may be **JPG, JPEG, or PNG**; PNG avoids additional lossy compression.
Preserve the source image resolution. Submit RGB predictions, not ground truth,
observed input frames, comparison panels, depth, segmentation, or model weights.
Do not include metadata or moving-camera files. Camera directories may be at the
ZIP root or inside one outer scene directory. Physics-category directories
outside the ZIPs are optional.

Two submission modes are supported:

1. **Training-view future prediction:** provide every required future frame for
   all training cameras. Omit novel-camera directories entirely. Novel-view
   metrics are unavailable and displayed as `-`, not as zero scores.
2. **Training- and novel-view future prediction:** additionally provide every
   required future frame for both novel cameras in every scene.

Missing training cameras or target frames fail validation. A partially supplied
novel-camera group also fails; an empty novel-camera folder is not an omission.
For an aggregate novel-view score, the complete novel group must be supplied
across all evaluated scenes. Pure video models such as MAGI-1 and ExtDM can use
the first mode without inventing novel-view results.

## Check before uploading

The offline [check.py](check.py) uses the bundled
[submission manifest](dataset/submission_manifest.json): scene IDs, camera names,
exact target frame identifiers, and image sizes only. It needs **no GT, network,
authentication, or GPU**. Keep the script and manifest together.

Run from the repository root:

```bash
pip install pillow numpy opencv-python-headless

python baselines/Future_Frame_Prediction/check.py ./predictions \
  --report ./fp_check_report.json
```

| Argument | Purpose |
| --- | --- |
| `submission_dir` | Folder containing the 103 scene ZIPs; searched recursively |
| `--report PATH` | Write detailed per-scene results to a new JSON file |
| `--verbose` | Print every error and warning |
| `--fail-on-warning` | Treat content warnings as failures |
| `--manifest PATH` | Use a different official release manifest; defaults to the bundled file |

The checker validates scene coverage, camera groups, exact future frame names,
ZIP integrity, image decoding, and resolution. Missing, duplicate, or extra scene
ZIPs and unexpected files are errors. JPG/PNG images are fully decoded with Pillow
and OpenCV. Black/white frames and entirely repeated videos produce warnings.
Progress includes elapsed time and estimated time remaining.

Exit codes: **0** = valid (possibly with warnings), **1** = validation failure,
**2** = command, dependency, or configuration error. Inspect warnings before
uploading. Format validation cannot establish prediction accuracy or prove that
an image is not GT. Prediction archives are never modified.

After checking, upload the prediction ZIPs and select their dataset repository
and subdirectory in the Leaderboard submission form.

## Evaluation

Scores are reported separately for future frames at training and novel views.
The evaluator includes PSNR, SSIM, LPIPS, PMF, and three PhysicsIQ motion-mask
IoU components (spatial, spatiotemporal, and weighted spatial IoU). The three
PhysicsIQ components are not a single composite score. A `train` score here is
still a score on **hidden future frames**, not on the observed training images.
