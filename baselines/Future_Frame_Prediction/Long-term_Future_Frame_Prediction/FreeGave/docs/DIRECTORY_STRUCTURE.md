# Directory Structure

## Source release

```text
FreeGave/
├── README.md
├── LICENSE
├── THIRD_PARTY_NOTICES.md
├── environment.yml
├── infer_physinone.py
├── train_physinone.py
├── arguments/
├── gaussian_renderer/
├── scene/
├── utils/
├── submodules/
│   ├── depth-diff-gaussian-rasterization/
│   └── simple-knn/
├── checkpoints/
│   └── README.md
├── configs/
│   └── physinone.yaml
└── manifests/
    └── scenes_103.txt
```

### Entry points

- `infer_physinone.py`: prediction-only rendering. It uses test camera metadata
  but does not read future RGB ground truth.
- `train_physinone.py`: FreeGave training on public observed RGB frames. It
  explicitly skips validation and test image loading.

### Core modules

- `arguments/`: command-line parameter groups and JSON configuration loading.
- `gaussian_renderer/`: differentiable Gaussian rendering code.
- `scene/`: cameras, dataset readers, Gaussian models, and deformation models.
- `utils/`: geometry, image, loss, pose, and velocity-field utilities.
- `submodules/`: source for the CUDA rasterizer and nearest-neighbor extension.

### Metadata and weights

- `configs/physinone.yaml`: human-readable parameters used by the released
  checkpoints. The command-line interfaces remain the source of runtime values.
- `manifests/scenes_103.txt`: expected checkpoint scene set.
- `checkpoints/`: destination for the independently distributed checkpoint
  files.

## Checkpoint release

```text
checkpoints/
├── SinglePhysics/
│   └── <scene>_trajectory/
├── DoublePhysics/
│   └── <scene>_trajectory/
└── TriplePhysics/
    └── <scene>_trajectory/
```

Every scene directory contains exactly four files needed for inference:

```text
<scene>_trajectory/
├── point_cloud/iteration_25000/point_cloud.ply
└── deform/iteration_25000/
    ├── deform.pth
    ├── code_field.pth
    └── vel.pth
```

## Files intentionally excluded

The release does not contain:

- private future RGB ground truth;
- rendered `train`, `val`, or `test` output folders;
- PMF or image-quality evaluation scripts;
- intermediate checkpoints at iterations 4,000, 10,000, or 20,000;
- TensorBoard event files or metric summaries;
- machine-specific batch scripts, progress databases, IDE settings, or logs;
- absolute local paths from the original `cfg_args` files;
- generated CUDA build directories, Python caches, or package metadata.

This separation keeps the public baseline reproducible without exposing
evaluation ground truth or machine-specific research artifacts.

