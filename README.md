# trash-vision-lab

Research pipeline for **multiclass trash detection** with:
- dataset cleaning + versioning (reproducible dataset ladder)
- comparing model families (YOLO / DETR / two-stage detectors)
- hard negative/positive mining + mini active learning loop
- limited, targeted synthetic data (~10-15% of train) + QA filters
- final deployment as an inference microservice + Telegram bot

## Goal

Build a **reproducible, experiment-driven** pipeline that answers:
1) Which dataset version/filters + which model family gives best real-world performance?
2) How much do synthetic / catalog anchors help rare classes (glass/other/metal)?
3) How to reduce false positives for class `other` (hard negatives and thresholding)?
4) How stable are results on **unseen real backgrounds** (strict hold-out)?

**Important rule:** validation/test is **real-only** (hold-out with unseen locations/backgrounds). No leakage across splits by background/location.

---

## Classes (5)

- `plastic`
- `paper` (paper + cardboard merged)
- `metal`
- `glass`
- `other`

**Other policy (strict):**
- YES: rags, rubber chunks (not tires), mixed mass, plywood/DSP fragments, torn black bags.
- NO: shoes, tires, car parts, people/animals/body parts -> those go to **hard-negative** pool (images without boxes).

---

## Data sources

- **TACO official** (hard revision)
- **TACO community** (major cleanup: removed ~1400 images, removed ~3300 bad "plastic" boxes, fixed many wrong labels/boxes)
- **Real-life** (own photos; ~600+ images, ~1100 objects, annotated in CVAT)
- **Catalog anchors** (objects on table/plain background; used selectively with light augment only)
- **Crops / second-chance** (cropped parts of successful scenes to boost rare classes)

Every image has `source={real,catalog,crop,synthetic}` metadata (mandatory for analysis).

---

## What's already done

### Global dataset cleanup
- Removed ultra-small boxes (min ~2% of short side or ~32 px at ~1600 long side)
- Removed blurry frames (Variance of Laplacian thresholding)
- Removed duplicates (perceptual hash, distance ~5-8)
- Fixed/deleted wrong annotations (class mistakes, oversized boxes, bad images)

### Annotation workflow (CVAT)
- Class policy defined + `hardcase_check` tagging
- Hotkeys / process tuned
- Source tagging enforced

### Current counts (pre-final re-balance)
Before final balancing, total objects ~6927. Target is to reduce weak plastic by ~500 (to ~2700) and add second-chance + synthetic/catalog to rare classes.

---

## Dataset versions (research ladder)

Versions are **manifests**, not copies of data.

Example ladder:
1. `taco_official`
2. `+ community`
3. `+ quality filters (blur/tiny/extreme res)`
4. `+ annotation fixes + bad image deletions`
5. `+ catalog anchors (light aug only on catalog)`
6. `dedup + class balance + light aug`
7. `+ synthetic (targeted, ~10-15% train)`
8. Hard Neg/Pos mining #1 + mini-AL
9. Hard Neg/Pos mining #2 + AL loop + small object retarget
10. Semi-supervised (high threshold pseudo-labels; **no `other`**)
11. Curriculum training (easy/mid/hard phases)

**Validation/test:** always real-only, with strict hold-out (unseen backgrounds).

**Local smoke dataset (not in git):** `sec_chance_v0` is a quick local manifest built from `sec_chance_img/` for smoke runs. The folder is ignored and can be replaced freely.

---

## Synthetic data policy (short)

Synthetic is **not the core**, it's a spice:
- prefer inpainting/placement on **real backgrounds**
- track `background_id` and never split same background across train/val/test
- reuse background 2-3 times (max 5 if strongly different)
- post-QA: blur filter, auto-detect + size filter, pHash anti-dup

Generation is outsourced for now; we integrate results through the same filters + manifests.

---

## Repository philosophy

We keep **one true pool** of data and build dataset versions reproducibly.

- `data/pool/`    - single source of truth (images + annotations), NOT in git
- `data/manifests/` - version definitions (YAML, uses `splits`), in git
- `data/versions/`  - materialized artifacts per version (split lists, data.yaml, optional symlinks), NOT in git

Training code is model-agnostic: one entrypoint calls model wrappers.

---

## Project structure (current)

```
trash-vision-lab/
  README.md
  pyproject.toml
  configs/                # YAML configs (base + model overrides)
  data/
    pool/                 # source of truth (images + labels), NOT in git
      images/
      labels/
    manifests/            # dataset version definitions, in git
    versions/             # built artifacts per version, NOT in git
  datasets/
    build_version.py      # builds version artifacts from manifest
    yolo_datayaml.py      # generates YOLO data.yaml from version artifacts
  models/
    base.py               # Model base + TrainRequest
    registry.py           # model registry (@register + get)
    dummy.py              # minimal training stub
  runners/
    train.py              # unified CLI entrypoint
  utils/
    paths.py
    seed.py
    logging.py            # optional: mlflow/wandb
  scripts/                # QC/metrics/utils
```

---

## How training is called (target UX)

Single command, same interface for all models:
```bash
python -m runners.train   --model yolo_v8m   --version v1   --config configs/base.yaml   --override configs/yolo8m.yaml
```

Inside each model wrapper we define:
- which dataset artifacts to use for a version
- model-specific config
- outputs (checkpoints, logs, metrics)
- tracking backend (W&B or MLflow)

Metrics we track:
- mAP@0.5 and mAP@0.5:0.95 per-class
- PR curves, confusion matrix
- per-source metrics (real/crop/catalog/synth)
- AP by object size (S/M/L) if supported
- inference latency + memory (CPU/GPU)
- threshold calibration per-class (esp. higher threshold for `other`)

Notes for agents (Codex/Claude Code)

Do NOT:
- duplicate dataset files per version (use manifests + splits/symlinks)
- mix synthetic/catalog into validation/test
- allow background leakage across splits

Do:
- keep code modular: dataset versioning, model wrappers, unified runner
- implement minimal end-to-end path first (toy manifest -> version artifacts -> one model train)
