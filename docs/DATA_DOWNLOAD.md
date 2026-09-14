# PhysInOne Public Data Download

The PhysInOne public downloader retrieves the released Leaderboard inputs and
validation 3D assets from the public `vLAR/PhysInOne` dataset repository. It
uses only the Python standard library and anonymous HTTPS requests.

Ground-truth outputs for the Leaderboard are private and are not included.

## Requirements

- Python 3.9 or newer
- Enough free disk space for the selected release
- Unreal Engine 5.5.4 when using the 3D asset project

No Python package installation or account configuration is required.

## Get the downloader

Clone the GitHub repository and run the script from the project root:

```bash
git clone https://github.com/vLAR-group/PhysInOne.git
cd PhysInOne
python scripts/download_data.py --help
```

The maintained download lists are stored in `scripts/download_lists/`. The
script reads them locally and does not fetch utility files at runtime.

## Quick start

Download all four public Leaderboard tasks:

```bash
python scripts/download_data.py \
  --task video_generation future_prediction \
         physical_properties_estimation motion_transfer \
  --output-dir ./PhysInOne_data
```

Download one task:

```bash
python scripts/download_data.py \
  --task future_prediction \
  --output-dir ./PhysInOne_data
```

Download and assemble the validation 3D assets:

```bash
python scripts/download_data.py \
  --task 3d_assets \
  --output-dir ./PhysInOne_data
```

ZIP extraction is enabled by default for `3d_assets`. Leaderboard ZIP files
remain compressed unless `--extract` is specified.

## Task values

| `--task` value | Released public content | File count |
| --- | --- | ---: |
| `video_generation` | Initial frames, camera metadata, and captions | 75,865 |
| `future_prediction` | Fixed-view input sequences and split metadata | 103 ZIP files |
| `physical_properties_estimation` | Scene inputs and shared supplementary files | 72 scene ZIP files plus 2 shared files |
| `motion_transfer` | Source videos, reference frames, and captions | 217 ZIP files |
| `3d_assets` | Validation Unreal Engine project assets for 1,000 scenes | 4,299 files |
| `all` | Every item above | 80,558 files |

Multiple task values can follow one `--task` option. Hyphenated names and the
short aliases shown by `--help` are also accepted.

## Command-line options

| Option | Description |
| --- | --- |
| `--task NAME [NAME ...]` | Select one or more tasks, or `all`. |
| `--output-dir PATH` | Set the local output root. The path must not contain whitespace. |
| `--scene TEXT` | Keep paths containing a scene name or six-character scene ID. Repeatable. |
| `--workers N` | Set concurrent downloads. Default: `4`. |
| `--revision REV` | Select a dataset revision. Default: `main`. |
| `--retries N` | Set retries after each failed request. Default: `3`. |
| `--timeout SECONDS` | Set the timeout for each network request. Default: `60`. |
| `--extract` | Safely validate and extract selected ZIP files. |
| `--no-extract` | Keep all ZIP files compressed, including 3D asset archives. |
| `--delete-zip-after-extract` | Remove a ZIP only after successful extraction. |
| `--no-resume` | Restart incomplete `.part` files instead of resuming. |
| `--force` | Replace completed local downloads. |
| `--list-only` | Print the selected public URLs and exit. |
| `--dry-run` | Show counts and output locations without downloading. |

## Examples

Download Video Generation inputs:

```bash
python scripts/download_data.py \
  --task video_generation \
  --output-dir ./PhysInOne_data
```

Download and extract Future Prediction inputs:

```bash
python scripts/download_data.py \
  --task future_prediction \
  --extract \
  --output-dir ./PhysInOne_data
```

Download selected Motion Transfer scenes by ID:

```bash
python scripts/download_data.py \
  --task motion_transfer \
  --scene B15ta6 \
  --scene 7omQio \
  --output-dir ./PhysInOne_data
```

Preview Physical Properties Estimation without downloading:

```bash
python scripts/download_data.py \
  --task physical_properties_estimation \
  --dry-run \
  --output-dir ./PhysInOne_data
```

Download 3D assets and remove archives after verified extraction:

```bash
python scripts/download_data.py \
  --task 3d_assets \
  --delete-zip-after-extract \
  --output-dir ./PhysInOne_data
```

## Full rendered dataset selection

The repository also includes a filter and downloader for selecting scene
archives from the 16 public rendered-data shards.

Create a selection file:

```bash
python scripts/filter_cases.py \
  --split train \
  --activity_type double \
  --phenomena FrictionStop \
  --num 100 \
  --output selected_cases.json
```

Download the selected archives:

```bash
python scripts/download_selected.py \
  --selection selected_cases.json \
  --output-dir ./PhysInOne_data/full_dataset \
  --keep-shard-structure
```

`filter_cases.py` reads the bundled
`scripts/download_lists/repo_assignment.txt` index. `download_selected.py`
reads the bundled `repo_map.json` and uses the same anonymous, resumable HTTPS
implementation as the task downloader.

| Selection option | Description |
| --- | --- |
| `--split train|val|test` | Filter by dataset split. |
| `--activity_type single|double|triple` | Filter by interaction complexity. |
| `--phenomena NAME [NAME ...]` | Require one or more physical phenomena. |
| `--match_mode contains|exact` | Control phenomenon matching. |
| `--num N` | Randomly sample at most `N` matching cases. |
| `--seed N` | Set the sampling seed. |
| `--output PATH` | Write the selection JSON. |

| Selected-download option | Description |
| --- | --- |
| `--selection PATH` | Read a selection JSON produced by `filter_cases.py`. |
| `--output-dir PATH` | Set the whitespace-free archive destination. |
| `--keep-shard-structure` | Preserve part, split, and activity directories. |
| `--workers N` | Set concurrent downloads. |
| `--retries N` | Set retries after a failed request. |
| `--no-resume` | Restart incomplete `.part` files. |
| `--overwrite` | Replace completed local archives. |
| `--dry-run` | Preview transfers without downloading. |

## Local directory layout

All directories created by the downloader have names without spaces:

```text
PhysInOne_data/
├── leaderboard/
│   ├── video_generation/
│   ├── future_prediction/
│   ├── physical_properties_estimation/
│   └── motion_transfer/
├── 3d_assets/
│   └── PhysicBenchmark/
└── _download_logs/
    └── YYYYMMDD_HHMMSS/
        ├── downloads.tsv
        ├── extractions.tsv
        ├── failures.tsv
        └── summary.json
```

Published object paths are URL-encoded by the maintained lists. The downloader
maps them to the no-space local layout above, so shell commands do not require
special quoting for task directories.

## Resume and verification

Downloads are written to `.part` files and renamed atomically after completion.
Run the same command again to reuse completed files and resume incomplete ones.
Use `--force` only when a completed local file must be replaced.

ZIP archives are checked for unsafe paths, symbolic links, and CRC errors before
extraction. A completion marker is written only after successful extraction.
Detailed progress and failures are recorded under `_download_logs/`.

For slower or rate-limited networks, reduce concurrency:

```bash
python scripts/download_data.py \
  --task all \
  --workers 2 \
  --output-dir ./PhysInOne_data
```

## 3D asset setup

The validation release contains Unreal Engine 5.5.4 resources for 1,000 scenes.
Windows is recommended for the simplest setup; Linux is also supported with
additional Unreal Engine configuration.

After the default 3D asset download finishes, launch:

```text
PhysInOne_data/3d_assets/PhysicBenchmark/PhysInOne.uproject
```

Do not rename directories inside `PhysicBenchmark/`. Unreal Engine resolves
assets by project-relative package paths, including `BackGrounds_raw`,
`__ExternalActors__`, and `__ExternalObjects__`.
