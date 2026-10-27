# Sequential Nano experiments

The user authorized this fixed two-experiment queue on the MSI RTX 4060 Laptop
GPU on 2026-10-27. It runs independently of the SSH connection. These are
follow-up experiments, separate from the completed `d3-seed42` release.

| Order | Experiment | Image size | Train boxes | Recipe |
|---|---|---:|---:|---|
| 1 | Full D3, higher training resolution | 640 | 5807 | [Nano640](../configs/overnight-nano640.json) |
| 2 | D3 with train boxes below 8px omitted | 320 | 5663 | [Nano320/drop-lt8](../configs/overnight-nano320-drop-lt8.json) |

Both use 150 epochs, seed 42, batch 16, AdamW, weight decay .001 and the same
pretrained YOLO26n checkpoint and augmentation recipe as the original D3 Nano.
Neither is fine-tuned from the completed D3 weights. The first changes input
resolution; the second changes the train label policy. They are not independent
replicates, and a single seed does not establish statistical significance.

The second variant preserves all 3431 train images and every split membership.
It removes 144 boxes from 118 train images, using
`min(box_width * image_width, box_height * image_height) * 320 / max(image_width, image_height) < 8`.
Exactly 8px is retained. The comparison is made before augmentation. All images
are independent byte-identical copies; val/test labels are byte-identical to
D3. This is label deletion, not an ignore-region loss. Original D3 is unmodified.
The [frozen variant manifest](../manifests/d3-drop-lt8.json) has fingerprint
`bcf2fb051cf451a0610775c26167a98b4617a60348f961788333d09fecedb896`.

## Worker behavior

[The queue](../configs/overnight_queue.json) first builds and hash-verifies the
variant and runs a one-epoch smoke check for each experiment. It then runs the
full first training and its common-holdout validation, followed by the second
training and its validation. The GPU jobs never overlap. Both evaluations use
the complete original D3 validation labels; no test evaluation or threshold
search is scheduled overnight.

The worker keeps Windows awake while active. A per-experiment smoke/training/
validation failure is recorded and the next independent experiment proceeds.
A `control-night/stop-request` file or the 20-hour overall deadline stops the
whole queue; training stage limits are 12 hours for Nano640 and 6 for Nano320.
These are bounds, not completion estimates. A Windows reboot or explicit
shutdown still interrupts the job; resumption requires inspecting its state.

Existing directories/status files are never overwritten. `best.pt`, `last.pt`
and checkpoints every 10 epochs remain in separate run directories. The JSON
status includes the current subprocess, latest completed epoch and metrics.
After the queue finishes it creates `night-results.tar`, containing logs,
recipes, manifests, run artifacts and a file checksum manifest, plus an archive
SHA256 receipt. No dataset images are included.

## Reproduce and inspect

Use an isolated checkout with the pinned D3 training environment. The base
workspace supplies `data/versions/d3` and the original `weights/yolo26n.pt`.

```bash
python -m scripts.overnight_queue --base-root /path/to/base-workspace
```

For just the reproducible dataset transform:

```bash
python -m datasets.build_small_box_variant \
  --source /path/to/base-workspace/data/versions/d3 \
  --manifest manifests/d3-drop-lt8.json \
  --output data/versions/d3-drop-lt8 \
  --report control-night/label-deletion.json
```

Inspect `control-night/status.json`, per-stage logs, and
`runs/night/<run_name>/results.csv`. The deletion report records every removed
label with source hash, original line and projected size. Runtime logs and
weights remain outside Git. The [size diagnostic](../reports/object-sizes/README.md)
explains the hypothesis and why the result should not be assumed in advance.
