# Repository map

This is a research and embedded-deployment project run from a source checkout.
Use Python 3.12. It is not currently distributed as an installable Python library.

| Directory | What belongs here |
|---|---|
| `datasets/` | Frozen dataset construction, split/label checks and the small-box ablation |
| `runners/` | Actual YOLO26 training, common-holdout evaluation, RKNN export and material-abstention analysis |
| `configs/` | Fixed experiment recipes and separate training/export dependency locks |
| `manifests/` | Reproducible dataset definitions and source-content hashes, without images |
| `deployment/luckfox/` | C/C++ camera pipeline, decoder, board smoke program and build/install scripts |
| `scripts/` | Report generation, artifact packaging and standalone training/transport supervisors |
| `reports/` | Preserved input metrics, epoch CSVs, derived tables, figures and experiment notes |
| `reference/taco/` | Category mapping and upstream provenance |
| `tests/` | Dataset integrity, queue behavior, diagnostic matching, rental policy and decoder checks |
| `docs/` | Reproduction, methods, interpretation limits and deployment instructions |
| `legacy/` | Early prototype retained for history, separate from the current pipeline |

Ignored local `data/pool/`, `data/versions/`, `storage/`, `runs/`, `dist/` and
`notes/recovery/` are not part of a clean clone. See the
[local storage guide](local-storage.md) for their roles on the development machine.

## Start here

The [README](../README.md) gives the task, results and their limitations.
[D3 reproduction](d3.md) explains how to obtain the required source pool, build
the frozen version and run the pinned training/evaluation recipe. The source
images must be supplied separately; they are not downloaded by the smoke test.
[Luckfox deployment](luckfox.md) covers the native runtime and external SDKs.

| Task | Entry point |
|---|---|
| Build frozen D3 | `python -m datasets.build_d3 --help` |
| Train Nano or Medium | `python -m runners.train_d3 --help` |
| Evaluate a completed checkpoint | `python -m runners.evaluate_d3 --help` |
| Export Nano320 to RV1103 INT8 | `python -m runners.export_rknn --help` |
| Reproduce historical tables | `python scripts/summarize_history.py` |
| Reproduce D3 tables | `python scripts/summarize_d3.py --root reports/d3/inputs --output reports/d3` |
| Package checked model artifacts | `python scripts/package_d3.py --help` |
| Run the sequential Nano ablations | `python -m scripts.overnight_queue --help` |

The `train_job`, `evaluate_job`, MSI backup and Vast control scripts document the
execution machinery used for these experiments. They are operational helpers,
not a general cluster scheduler. In particular, the MSI backup helper retains
the original SSH alias/Windows workspace; inspect that helper before reusing it
on another machine. The offline scientific results do not require cloud access.

## Verification scope

CI runs Python unit tests, the dummy compatibility smoke, host C++ decoder tests
and exact regeneration of the published metric tables. These checks need neither
the private data pool nor a GPU. GPU smokes, full training, ONNX comparisons,
RKNN compilation and board measurements are separate recorded evidence; a green
CI run is not a new hardware measurement.

The old `runners.train` / `datasets.build_version` entry points are compatibility
wrappers around [the prototype](../legacy/README.md). Use the D3 entry points above
for the current four-material detector.
