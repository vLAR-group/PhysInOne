# Long-Term Future Frame Prediction on PhysInOne

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
list, archives, and transforms from the same release. If an older local copy has
different camera counts or temporal splits, check the current release before
running; do not randomly redefine its evaluation views or targets yourself.

### Training views versus novel views

- **Training views** are the cameras whose past RGB observations are available.
  You may use those observations for conditioning or per-scene optimization.
  Their future RGB is withheld and is an evaluation target.
- **Novel views** are held-out static cameras. Their camera poses and timestamps
  are provided as metadata, but their RGB is not supplied as training input.
  Predicting their future video tests both temporal extrapolation and viewpoint
  generalization.
- Do not assume that `CineCamera_0` and `CineCamera_1` are always the novel views.
  Read `transforms_val.json`, and preserve the original camera names.

Here, **training view** describes a camera split within an evaluation scene; it
does not mean that the entire scene video belongs to the dataset's training set.
Likewise, `transforms_val.json` identifies held-out cameras; it does not grant
access to validation RGB or permission to optimize against private images.

### Observed frames versus future frames

For a zero-indexed sequence with an even total length `N`, the intended split is:

- Observed frames: `0` through `N / 2 - 1`.
- Future frames: `N / 2` through `N - 1`.

| Total frames per camera | Observed frame names | Future frame names | Frames per half |
| ---: | --- | --- | ---: |
| 90 | `0000.jpg`–`0044.jpg` | `0045.jpg`–`0089.jpg` | 45 |
| 120 | `0000.jpg`–`0059.jpg` | `0060.jpg`–`0119.jpg` | 60 |
| 130 | `0000.jpg`–`0064.jpg` | `0065.jpg`–`0129.jpg` | 65 |
| 150 | `0000.jpg`–`0074.jpg` | `0075.jpg`–`0149.jpg` | 75 |
| 300 | `0000.jpg`–`0149.jpg` | `0150.jpg`–`0299.jpg` | 150 |

These are examples, **not a fixed 75-frame prediction requirement**. In code,
read the exact observed entries from `transforms_train.json` and the exact
prediction entries from `transforms_test.json`. Keep their frame identifiers,
camera poses, and timestamps; do not renumber the future interval from zero or
replace the supplied time values with an assumed frame rate.

## Public inputs and metadata

| File | Purpose | Does it provide target RGB? |
| --- | --- | --- |
| `transforms_train.json` | Observed frames and poses for training cameras | Only its public observed RGB is available |
| `transforms_val.json` | Held-out camera membership and associated pose/time metadata | No |
| `transforms_test.json` | Exact future camera/frame targets to predict | No; its RGB paths describe targets, not downloadable images |

The test targets include both training and novel cameras. Consequently,
`transforms_train.json` and `transforms_test.json` need not contain the same
number of entries, even though the timeline is split equally.

For a 150-frame example, an extracted public scene looks like this:

```text
<scene>_trajectory/
├── transforms_train.json
├── transforms_val.json
├── transforms_test.json
├── CineCamera_<training_view_a>/
│   └── rgb/
│       ├── 0000.jpg
│       ├── ...
│       └── 0074.jpg
└── ...                         # the other nine training cameras
```

The absence of future RGB and novel-camera RGB is intentional. A loader must
not try to open every image referenced by the three JSON files during training
or inference. In particular, do not obtain hidden images from another task,
another dataset copy, related scene variants, or private organizer resources.

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
is required for these public inputs. The `--subset mini/full` option belongs to
Video Generation, not Future Prediction.

Baseline-specific installation, training, checkpoint, and inference instructions:

| Baseline | Usage guide |
| --- | --- |
| FreeGave | [FreeGave README](Long-term_Future_Frame_Prediction/FreeGave/README.md) |
| Def3G | [Def3G README](Long-term_Future_Frame_Prediction/Def3G/README.md) |
| TRACE | [TRACE README](Long-term_Future_Frame_Prediction/TRACE/README.md) |
| TiNeuVox | [TiNeuVox README](Long-term_Future_Frame_Prediction/TiNeuVox/README.md) |
| MAGI-1 | [MAGI-1 README](Long-term_Future_Frame_Prediction/MAGI-1/README.md) |
| ExtDM | [ExtDM README](Long-term_Future_Frame_Prediction/ExtDM/README.md) |

Scene-optimization methods must fit only the public observed frames. Video
models may condition on those frames without per-scene training. Any model
pretraining or fine-tuning must respect the held-out evaluation data boundary.
Inference should save predictions only; private-reference scoring is performed
by the leaderboard.

## Prediction layout and evaluation

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

Use the official scene identifier for the archive name, and the exact camera
and frame names in `transforms_test.json` inside it. Preserve the source image
resolution specified by the release. Submit RGB predictions, not ground truth,
observed input frames, comparison panels, depth, segmentation, or model weights.

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

Scores are reported separately for future frames at training and novel views.
The evaluator includes PSNR, SSIM, LPIPS, PMF, and three PhysicsIQ motion-mask
IoU components (spatial, spatiotemporal, and weighted spatial IoU). The three
PhysicsIQ components are not a single composite score. A `train` score here is
still a score on **hidden future frames**, not on the observed training images.

Before submission, check that all scene IDs match the official task list, each
required frame is present and decodable, and no private GT has entered either
your model inputs or your output archives.
