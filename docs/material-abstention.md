# Experimental material-label abstention

A detector can retain a box while declining to name its material. The proposed
UI would show `unknown material (suggested: plastic)` when class evidence is
ambiguous. This is a selective classification experiment, not a fifth trained
class or an unknown-object detector.

The experiment uses the Nano320 normalized FP32 ONNX graph from the portable
export. It preserves every box produced by confidence 0.25 and class-wise NMS
at IoU 0.45. Only the material name can be withheld.

Two rules are compared:

- **Margin:** withhold the material when the best class score minus the
  second-best class score is below a threshold.
- **Confidence:** withhold the material when its best class score is below a
  threshold. This is a useful simpler comparator.

Class scores are not calibrated probabilities or a separate objectness score.
Neither rule can guarantee that an uncertain box contains litter.

## Fixed evaluation procedure

The source is [runners/material_abstention.py](../runners/material_abstention.py).
It verifies the ONNX/dataset hashes and reads all 553 validation images first.
A fixed threshold grid is searched for each rule, maximizing material accuracy
among named, matched boxes while retaining at least 90% of validation matches
and at least 50 named matches. Ties prefer greater coverage, then a lower threshold.

The selected rules are written to disk before any of the 552 test images are
processed. Predictions are matched to ground truth greedily in confidence
order, with class-agnostic IoU ≥0.5 and one-to-one assignment. This measures
material classification conditional on a located object; it is not detection AP.

```bash
# In the separate RKNN/ONNX CPU environment:
python -m runners.material_abstention \
  --export-root runs/export/d3-nano --data-root data/versions/d3 \
  --output runs/abstention/d3
```

The output includes the protocol, locked validation selection, aggregate
results and private prediction records. Existing experiments are not overwritten.

## D2 pilot on the reviewed holdout

The historical D2 Nano checkpoint provides an initial full-pipeline check.
Both D2 and D3 use the same reviewed holdout and fixed selection procedure.
[Aggregate D2 results](../reports/abstention/d2.json) and the
[validation selection](../reports/abstention/d2-selection.json) are preserved.

| Test result | Always name | Margin ≥0.20 | Confidence ≥0.35 |
| --- | ---: | ---: | ---: |
| Matched-box material coverage | 100% | 93.26% | 88.62% |
| Material accuracy among named matches | 67.84% | 70.33% | 70.68% |
| Incorrect material names withheld | 0 | 32 | 44 |
| Correct material names withheld | 0 | 16 | 37 |
| All boxes labelled unknown | 0 | 150 | 218 |

All three views retain the same 1021 detected boxes, including 712 matches
and 309 unmatched detections. Unmatched detections can also receive an unknown
material label; they are still false or unannotated boxes in this evaluation.
The confidence rule met 90% coverage on validation but retained 88.62% on test,
illustrating that the validation constraint is not a test guarantee.

The margin rule improves conditional material accuracy modestly while
withholding fewer correct names than the selected confidence rule. It does
not repair the underlying detector or eliminate confidently wrong materials.

## D3 repeat

The completed D3 Nano export was evaluated once using the same procedure.
[Aggregate D3 results](../reports/abstention/d3.json) and its independently
locked [validation selection](../reports/abstention/d3-selection.json) are preserved.
Validation selected the same thresholds, margin 0.20 and confidence 0.35.

| Test result | Always name | Margin ≥0.20 | Confidence ≥0.35 |
| --- | ---: | ---: | ---: |
| Matched-box material coverage | 100% | 93.50% | 89.52% |
| Material accuracy among named matches | 65.78% | 66.52% | 67.11% |
| Incorrect material names withheld | 0 | 22 | 36 |
| Correct material names withheld | 0 | 27 | 43 |
| All boxes labelled unknown | 0 | 158 | 241 |

D3 produces 1147 boxes: 754 matches and 393 unmatched detections. Margin-based
abstention withholds 49 matched material names, including more correct names
than incorrect ones. Its conditional accuracy gain is only 0.74 percentage
points. This is a modest trade-off, not evidence of reliable uncertainty or a
reason to enable the option by default. It preserves the same 1147 boxes.
The different match populations also prevent interpreting the D2/D3
conditional accuracy difference as classification on an identical object set.

## Limits and deployment status

The host calculation was cross-checked against Ultralytics LetterBox/NMS on
18 test frames, and aggregate counts and unique ground-truth assignments were
independently recomputed. The thresholds are still exploratory: one checkpoint per dataset version,
a retrospectively reviewed holdout and no uncertainty estimate.

Only four known materials have ground truth. Novel materials and open-set
objects were not evaluated. INT8 score rounding may also change threshold
behavior, so FP32 results must not be presented as a board measurement.
The camera runtime has an optional material display, disabled by default:

```sh
TVL_MATERIAL_MARGIN=0.20 ./camera_pipeline model.rknn --field /userdata/field
```

A rejected material is shown in gray as `unknown (plastic)` (or the other
suggested class). Boxes and class-wise NMS remain unchanged. Field mode writes
`detections-material.csv`, recording the displayed class, suggested class,
confidence, margin and threshold; the original `detections.csv` format is used
when the option is off. Set the value to `0` or leave it unset to disable it.

The value 0.20 was selected in both FP32 experiments above; it is not a
validated INT8 operating point. On the 32 stored D2 INT8 tensors, it defers 6 of 52 names while
preserving every detection. Quantizer, near-tie and disabled-mode host tests
pass, and the complete runtime cross-compiles. No fresh hardware test has
been performed and the installed board application has not been changed.
