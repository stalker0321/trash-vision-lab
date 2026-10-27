# Nano resolution and small-label experiments

Both training runs completed 150 epochs on the RTX 4060 Laptop GPU. Each
started from the same pretrained Nano weights as the original D3 run, using
seed 42 and the [fixed recipes](../../docs/overnight-nano-experiments.md).
The table uses the same reviewed validation set: 553 images / 1067 boxes.
All numbers are mAP50–95 percentages from best-checkpoint evaluations.

| Training | Evaluation input | Validation mAP50–95 | Change from D3 Nano320 (pp) |
|---|---:|---:|---:|
| [Original D3 Nano](../d3/inputs/nano_d3_val/summary.json), 320 | 320 | 29.14% | — |
| [D3 Nano, omit train boxes below 8px](evaluations/nano320_drop_lt8_val/summary.json), 320 | 320 | 28.87% | -0.27 |
| [D3 Nano](evaluations/nano640_val/summary.json), 640 | 640 | 41.97% | +12.83 |
| [Same 640-trained checkpoint](evaluations/nano640_at320_val/summary.json) | 320 | 27.42% | -1.72 |

## What these experiments establish

Removing 144 small boxes from 118 training images did not improve Nano320 in
this run. All 3431 training images and the complete validation annotations were
retained. Deleted labels leave the objects visible as unlabelled content;
this is not an ignore-region loss or removal of the objects' pixels. The result
does not establish that all small-object filtering policies are ineffective.

Training and evaluating at 640 improved validation AP substantially. Reusing
that checkpoint at the Pico's 320 input did not retain the gain. This check
used the checkpoint selected at 640, without further adaptation to 320.
It does not establish that every possible high-resolution pretraining or
distillation scheme would fail.

Neither result supports replacing the current D2 Nano320 deployment candidate.
Training at 640 leaves the Nano architecture unchanged, but these accuracy
results depend on inference resolution. No new RKNN/INT8 accuracy or board
performance result is implied by this table.

These are single-seed follow-ups with validation-based selection, not estimates
of statistical significance. The new checkpoints have not been evaluated on
test. Training took about 5h09m for Nano640 and 3h30m for the filtered Nano320;
these are training times, not deployment latency.

## Evidence and reproduction

[evidence.json](evidence.json) indexes both complete training CSVs/run records,
the three evaluation summaries and their SHA256 hashes. Both best checkpoints
were verified locally against the completed run metadata. The original D3
baseline is linked above; its evidence remains in `reports/d3/`.

The [queue instructions](../../docs/overnight-nano-experiments.md) reproduce the
two training runs. Once the queue is finished, the additional resolution check
uses its unchanged Nano640 best checkpoint:

```bash
python -m runners.evaluate_d3 \
  --weights runs/night/yolo26n_d3_640_s42_wd001/weights/best.pt \
  --imgsz 320 --split val --data-root /path/to/base-workspace/data/versions/d3 \
  --recipe configs/d3_training.json \
  --output runs/evaluation/nano640_at320_val --device 0
```

Use the pinned training environment. Weights and source images are external
artifacts; they are not part of Git or the earlier `d3-seed42` release.
