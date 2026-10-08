# Directory Structure

```text
ExtDM/
├── infer_physinone.py
├── config/
├── data/
├── model/
├── scripts/
├── utils/
├── checkpoints/
├── manifests/
└── README.md
```

Only source code, portable configuration, documentation, attribution, and
scene identifiers are staged here. Model checkpoints are distributed
separately. Private future RGB, generated outputs, metrics, caches, compiled
extensions, local progress files, and machine-specific batch scripts are
intentionally excluded.
