# create_dataset.py User Guide

Convert PhysInOne origin render sequences into datasets for PAC-NeRF/GIC. The script supports frame sampling with `gap`, three camera split modes, foreground image generation, and mappings from output camera IDs to original camera names.

## Environment Setup

The script requires `numpy`. Generating foreground and transparent images also requires `opencv-python`.

```powershell
& C:\Users\11630\miniconda3\python.exe -m pip install numpy opencv-python
```

Use `--no-copy-images` to generate metadata only. This does not require OpenCV or segmentation data, but the script still lists the original RGB files to identify available views and sample frames.

All commands below are single-line PowerShell commands. Adjust the Python executable, script, input, and output paths for your environment.

## Input Data

A typical case has the following structure. `CineCamera_N` represents a static camera; camera IDs do not need to be consecutive or limited to 0 through 14.

```text
<case>_trajectory/
|-- reinit.json
|-- particle_materials.json
|-- recorder_stats.json
|-- static_camera_list.txt
|-- transforms_val.json
|-- blender_CineCamera_0.json
|-- blender_CineCamera_1.json
|-- ...
`-- CineCamera_N/
    |-- rgb/
    |   |-- 0000.jpg
    |   `-- ...
    |-- seg/
    |   |-- 0000.npz
    |   `-- ...
    `-- bg/
        `-- bg_0001.jpg
```

File requirements:

| File or Directory | Purpose |
| --- | --- |
| `blender_CineCamera_N.json` | Camera intrinsics and extrinsics. Must contain `camera_angle_x`, `img_w`, `img_h`, and a nonempty `frames` list. The first frame must contain `transform_matrix`. |
| `CineCamera_N/rgb/` | Supports `.jpg`, `.jpeg`, and `.png`. Frames are sampled after sorting filenames in natural order. |
| `CineCamera_N/seg/<RGB-stem>.npz` | Used for foreground image generation. The array key must be `seg`; for example, `0000.jpg` uses `0000.npz`. |
| `CineCamera_N/bg/bg_0001.jpg` | Optional background image, included by default when present. It is not subject to `gap` sampling. |
| `static_camera_list.txt` | Preferred camera source for `val-json` and `random`, with one camera name per line. If absent, cameras are discovered from camera JSON files. |
| `transforms_val.json` | Defines test cameras for `val-json`. The script reads the first directory component of `frames[*].file_path`, such as `CineCamera_0/rgb/0000`. |
| `particle_materials.json`, `recorder_stats.json` | Used to determine foreground segmentation IDs automatically. Not required with `--no-masks`, `--no-copy-images`, or manually specified foreground IDs. |
| `reinit.json` | Identifies origin cases. Cases without this file are also accepted as origin inputs. |

When `reinit.json` exists, the script reads `case_type`, falling back to `type` if that key is absent. The value must be `ori` or `origin`, ignoring case and surrounding whitespace. Explicitly selecting a reinit case raises an error; batch processing skips such cases.

`CineCamera_Moving` is excluded. The script currently processes static cameras only and uses the first frame's camera matrix for all output frames from that camera. The input `depth/` directory is not used.

## Camera Split Modes

| `--test-split-mode` | Test Camera Source | Suitable Layout |
| --- | --- | --- |
| `val-json` (default) | Cameras listed in `transforms_val.json`. | Complete datasets with rendered training/test RGB images and a validation JSON file. |
| `train-only` | Cameras with camera JSON files but no RGB images. | Reduced datasets with training RGB images and test camera directories containing only `bg/` or no RGB images. |
| `random` | Randomly selected static cameras with both camera JSON files and RGB images. | Datasets that need a new split with a specified number of test cameras. |

### val-json

The script reads the camera list and assigns each camera to the training or test set using the validation JSON. Cameras in both sets require RGB images. Cameras without RGB images are skipped with a warning.

If the validation JSON is missing, the test camera set is empty and all available cameras are assigned to training.

```powershell
& C:\Users\11630\miniconda3\python.exe D:\Physinone\baseline\create_dataset.py --case-path "P:\doris\disk36\Rendered\Videos\Test-all\SinglePhysics\ElasticFall__bg851__wBxhzP_trajectory" --output-path "D:\Physinone\baseline\output_val" --gap 50 --test-split-mode val-json
```

### train-only

The script scans `blender_CineCamera_*.json` directly in the case directory. It does not use `static_camera_list.txt` or `transforms_val.json`.

Cameras with RGB images are assigned to training; cameras without RGB images are assigned to testing. For test cameras, metadata is generated from `frames[::gap]` in the camera JSON, without generating the corresponding foreground RGB, mask, or transparent images. Background images, when present, are still copied and recorded by default.

For example, the local sample has 15 camera JSON files. `CineCamera_0` and `CineCamera_1` have backgrounds only, while the other 13 cameras have RGB images. This produces 13 training views and 2 test views.

```powershell
& C:\Users\11630\miniconda3\python.exe D:\Physinone\baseline\create_dataset.py --case-path "D:\Physinone\baseline\ElasticFall__bg851__aZnA3h_trajectory" --output-path "D:\Physinone\baseline\output_train_only" --gap 50 --test-split-mode train-only
```

### random

Set the number of test cameras with `--test-camera-count`. Use `--random-seed` to reproduce the split; the default seed is `42`.

Candidates come from the static camera list, or from camera JSON discovery if the list is absent. Camera names are deduplicated and sorted in natural order. Cameras without camera JSON files or RGB images are excluded, so views with backgrounds only do not participate in the random split.

The test camera count must be greater than 0 and smaller than the available candidate count, leaving at least one training camera. This mode does not read the validation JSON. The same candidate set, seed, and Python environment reproduce the split. Changing `gap` does not change the camera split.

The following command randomly selects 3 test cameras and assigns the remaining cameras to training:

```powershell
& C:\Users\11630\miniconda3\python.exe D:\Physinone\baseline\create_dataset.py --case-path "P:\doris\disk36\Rendered\Videos\Test-all\SinglePhysics\ElasticFall__bg851__wBxhzP_trajectory" --output-path "D:\Physinone\baseline\output_random" --gap 50 --test-split-mode random --test-camera-count 3 --random-seed 42
```

## Metadata-Only Checks

Add `--no-copy-images` to create the output directories, both `all_data.json` files, and `camera_mapping.json` without copying RGB or background images or generating masks. Use this to inspect camera splits and output IDs before generating the full dataset.

```powershell
& C:\Users\11630\miniconda3\python.exe D:\Physinone\baseline\create_dataset.py --case-path "P:\doris\disk36\Rendered\Videos\Test-all\SinglePhysics\ElasticFall__bg851__wBxhzP_trajectory" --output-path "D:\Physinone\baseline\output_check" --gap 50 --test-split-mode random --test-camera-count 3 --random-seed 42 --no-copy-images
```

Remove this flag to generate the full dataset. Alternatively, `--no-masks` copies RGB and background images without generating foreground mask or transparent images.

## Batch Processing and Multiple Gaps

`--origin-path` accepts either a single case directory or a parent directory containing cases as direct children. Batch processing scans direct children only; it does not recursively traverse groups such as `SinglePhysics` and `DoublePhysics`.

The following command processes all origin cases under `SinglePhysics`, generating separate `gap30` and `gap50` datasets:

```powershell
& C:\Users\11630\miniconda3\python.exe D:\Physinone\baseline\create_dataset.py --origin-path "P:\doris\disk36\Rendered\Videos\Test-all\SinglePhysics" --output-path "D:\Physinone\baseline\output_batch" --gap 30 50 --test-split-mode random --test-camera-count 3 --random-seed 42
```

Use `--case-name "ElasticFall__bg851__wBxhzP"` with `--origin-path` to process one case under the parent directory. The `_trajectory` suffix is optional. `--case-path` takes precedence; normally, provide either `--case-path` or `--origin-path`.

Random mode uses the same seed for each case. Cases with identical candidate camera sets receive the same selected camera names.

## Output Structure and Camera Mapping

```text
<output-path>/
`-- <dataset-name>_gap50/
    |-- camera_mapping.json
    |-- train/
    |   |-- all_data.json
    |   `-- data/
    |       |-- r_0_-1.jpg
    |       |-- r_0_0.jpg
    |       |-- m_0_0.png
    |       |-- a_0_0.png
    |       `-- ...
    `-- test/
        |-- all_data.json
        `-- data/
            `-- ...
```

By default, `dataset-name` is the input case directory name. Override it with `--dataset-name`.

| Output File | Description |
| --- | --- |
| `r_<camera>_<frame>.<ext>` | Copy of an original RGB image, retaining its extension in lowercase. |
| `r_<camera>_-1.jpg` | Copy of a background image, generated when present unless backgrounds are disabled. |
| `m_<camera>_<frame>.png` | Three-channel image preserving the object's original colors and setting the background to white. This is not a binary mask. |
| `a_<camera>_<frame>.png` | Four-channel transparent image with foreground alpha 255 and background alpha 0. |
| `all_data.json` | RGB/background paths, camera parameters, and timestamps for the current split. Does not include entries for `m_` or `a_` files. |
| `camera_mapping.json` | Training and test camera ID mappings for the current case and gap. |

Training and test cameras are numbered independently and consecutively from 0. These IDs appear in the `<camera>` part of output filenames and may differ from the original camera IDs. The mapping includes only cameras for which data was actually generated. A simplified example:

```json
{
    "train": {
        "0": "CineCamera_2",
        "1": "CineCamera_3"
    },
    "test": {
        "0": "CineCamera_0",
        "1": "CineCamera_1"
    }
}
```

In this example, `train/data/r_0_0.jpg` corresponds to `CineCamera_2`, while `test/data/r_0_0.jpg` corresponds to `CineCamera_0`.

## Frames, Time, and Camera Parameters

RGB files are sorted in natural filename order and sampled with `[::gap]`. Output frame IDs are reassigned consecutively from 0. For example, with 90 input frames and `gap=50`, list positions 0 and 50 are selected and assigned output frame IDs 0 and 1.

The effective frame rate and output timestamps are calculated as follows:

```text
effective_fps = original_fps / gap
time = output_frame_index / effective_fps
background_time = -1 / effective_fps
```

`--original-fps` defaults to `150`. The script does not automatically read the camera JSON's `fps` or time fields. Set this parameter explicitly if the original render frame rate differs; use a positive value.

Each `all_data.json` is a list of records containing `file_path`, `time`, `c2w`, and `intrinsic`. Paths are relative to the containing `train/` or `test/` directory, such as `./data/r_0_0.jpg`. Test entries in `train-only` mode and entries from metadata-only runs may reference images that do not exist in the output.

`c2w` is a 3 x 4 matrix obtained by left-multiplying the input 4 x 4 camera matrix by the matrix below and taking the first three rows. A 3 x 4 input matrix is first extended with a final homogeneous row:

```text
[ 1  0  0  0 ]
[ 0  0  1  0 ]
[ 0 -1  0  0 ]
[ 0  0  0  1 ]
```

`intrinsic` is a 3 x 3 matrix with `fx = fy = 0.5 * img_w / tan(0.5 * camera_angle_x)` and principal point `(img_w / 2, img_h / 2)`. Angles are in radians.

## Segmentation

By default, object mesh names are taken from the keys of `particle_materials.json`. The script finds matching meshes in the `solid_object`, `interactable_object`, and `breakable_object` groups of `recorder_stats.json`. For each matching mesh, it uses the first non-null `custom_stencil_value` in `components`. The union of the matching IDs defines the foreground.

Use `--foreground-stencil-values 128 129` to override the foreground IDs manually. Segmentation images are resized with nearest-neighbor interpolation when their resolution differs from RGB. A missing per-frame segmentation file produces a warning and skips that frame's `m_` and `a_` images; RGB and metadata are still generated. Failure to find any foreground IDs automatically raises an error.

## Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--case-path` | None | Explicit origin case directory. Provide at least one of this option and `--origin-path`. |
| `--origin-path` | None | A single case directory or a parent containing cases as direct children. |
| `--case-name` | None | Select one case under `--origin-path`; the `_trajectory` suffix is optional. |
| `--output-path` | Required | Output root directory. |
| `--dataset-name` | Case directory name | Custom output name; supported for a single case only. |
| `--gap` | Required | One or more positive integers, such as `--gap 30 50`. |
| `--test-split-mode` | `val-json` | `val-json`, `train-only`, or `random`. |
| `--test-camera-count` | None | Positive test camera count required in `random` mode; must be smaller than the available camera count. Unused in other modes. |
| `--random-seed` | `42` | Random seed, used only in `random` mode. |
| `--static-camera-list-path` | Case directory's `static_camera_list.txt` | Override the camera list for a single case. Unused in `train-only` mode. |
| `--val-json-path` | Case directory's `transforms_val.json` | Override the validation JSON for a single case. Used only in `val-json` mode. |
| `--original-fps` | `150.0` | Original frame rate used to calculate output timestamps. |
| `--foreground-stencil-values` | Automatic lookup | One or more foreground segmentation IDs. |
| `--no-copy-images` | Disabled | Generate metadata and camera mapping only, without copying images or generating masks. |
| `--no-masks` | Disabled | Skip `m_` and `a_` images while still copying RGB and background images. |
| `--no-background` | Disabled | Exclude background images and their metadata entries. |
| `--overwrite` | Disabled | Delete the current `<dataset-name>_gap<gap>/` directory before regenerating it. |
| `-h`, `--help` | N/A | Show command-line help. |

## Reruns and Troubleshooting

- **How do I rerun with a different mode, seed, or test camera count?** Output directory names contain only the case name and gap, without the split mode or seed. Use a different `--output-path` or, for a single case, `--dataset-name`. Alternatively, check the target directory and use `--overwrite`.
- **What happens without `--overwrite`?** JSON files and images with matching names are overwritten, but other existing images are not removed. Changing the split can leave old files behind. `--no-copy-images` also does not remove existing images.
- **Why does the test directory contain only backgrounds while its JSON lists other frames?** In `train-only` mode, test cameras without RGB images receive camera and time metadata for those frames without corresponding output images.
- **Why does `val-json` produce no test views?** Check the validation JSON path and its camera names, then verify that those cameras have RGB images. A missing validation JSON assigns all available cameras to training.
- **Why does batch processing not work directly from `Test-all`?** Only direct child directories are scanned. Run the script separately for groups such as `SinglePhysics`, `DoublePhysics`, and `TriplePhysics`.
- **How do I resolve `No module named 'cv2'`?** Install `opencv-python` using the same Python executable that runs the script, or disable mask generation with `--no-masks` or `--no-copy-images`.
