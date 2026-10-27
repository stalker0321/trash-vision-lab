# Recovered Nano adapt30 on the reviewed holdout

The historical 30-epoch control fine-tune was recovered and evaluated without
new training. Its fixed best checkpoint was selected during the original D2
run (epoch 3), then compared with its parent using the same reviewed D3 labels
and the same 320-input evaluation protocol.

Both models were trained on D2, which includes the main manual curation of
scenes and boxes. The D3 labels used here come from the later post-hard-mining
annotation review. This changes their evaluation reference, not their training
data: adapt30 has not been fine-tuned on D3.

| Split | Images / boxes | Parent D2 Nano mAP50–95 | adapt30 mAP50–95 | Change (pp) |
|---|---:|---:|---:|---:|
| Validation | 553 / 1067 | 29.48% | 30.38% | +0.90 |
| Test | 552 / 1044 | 30.23% | 30.93% | +0.70 |
| Hard subset of test | 84 / 359 | 18.77% | 19.32% | +0.55 |

The validation result was checked first. The unchanged checkpoint was then
evaluated once on test and its existing hard subset, with no new training,
threshold search or test-based checkpoint selection. The hard subset overlaps
test and is not independent additional evidence.

The modest improvement makes this checkpoint worth checking for deployment.
It does not establish statistical significance: there is one seed, and the
historical checkpoint selection and dataset-review limitations still apply.
The [project history](../../docs/project-history.md) documents those limits.

The existing D2 RKNN export remains the board-tested artifact. This adapt30
checkpoint has not yet been exported/tested through INT8 RKNN on Pico; the
table is PyTorch detector accuracy, not on-board accuracy or speed. A useful
next step is export agreement and real-device testing, rather than more training
before establishing whether this saved improvement survives deployment.

## Evidence

- [Original training record](../historical-followups/training/yolo26n_d2_s42_adapt30/run.json)
  and [all 30 epochs](../historical-followups/training/yolo26n_d2_s42_adapt30/results.csv).
- Fixed checkpoint SHA256:
  `c0b71a7613d7d20c97ac17597d126f91e7722d68dfdc45098e17646bd6f2bf89`.
- Evaluation summaries: [validation](val/summary.json), [test](test/summary.json),
  [hard test](hard_test/summary.json). Parent summaries remain in
  [reports/d3/inputs](../d3/inputs/).
- [evidence.json](evidence.json) records the comparison and hashes. Evaluation
  summaries and full predictions were copied locally and verified against MSI
  SHA256 receipts; prediction dumps remain outside Git.

The pinned D3 evaluation environment and frozen data are required. Repeat with
`--split test` or `--split hard_test` and a distinct output directory as needed:

```bash
python -m runners.evaluate_d3 \
  --weights /path/to/yolo26n_d2_s42_adapt30/weights/best.pt \
  --imgsz 320 --split val --data-root /path/to/data/versions/d3 \
  --recipe configs/d3_training.json \
  --output runs/evaluation/adapt30_d3_val --device 0
```
