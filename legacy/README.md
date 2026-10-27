# Historical prototype

`prototype/` preserves the early generic model registry, YOLOv8/RT-DETR wrappers,
illustrative v0/v1 manifests and their configuration files. These are not the
YOLO26 D0–D3 training pipeline and do not reproduce the published measurements.
Some prototype manifests have five or seven classes; the current task has four.
Their example source images are not distributed with this repository.

The generic wrappers are retained as project history, not as tested support for
additional model families. Their original packaging metadata is also historical;
the current project is run from a checkout using its documented requirements.

Three compatibility entry points remain:

```bash
python -m runners.train --model dummy --version v0
python -m datasets.build_version --help
python -m datasets.yolo_datayaml --help
```

The dummy command checks the lightweight Python setup and configuration loader;
it does not train a detector. A real v0 build still requires its external example
images. Current model training is `python -m runners.train_d3`, and frozen dataset
construction is `python -m datasets.build_d3`.
