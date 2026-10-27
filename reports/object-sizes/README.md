# Object size diagnostic

This validation-only follow-up examines whether D3's extra small-object labels
explain the Nano320 regression. It does **not** establish that explanation. No
labels were removed and no additional training was launched for this audit.
The completed D3 release and its frozen dataset remain unchanged. After this
audit, the user authorized the separate [overnight two-experiment queue](../../docs/overnight-nano-experiments.md).

## Size at the model input

Box dimensions are projected to a 320px longest image side, preserving aspect
ratio, before augmentation. Bins use the shorter box side; upper endpoints are
exclusive. Medium640 predictions use the same 320px bins to compare the same
objects. Mosaic, scaling, cropping and pixel rounding are not simulated.

| Short side at 320px | D2 train | D3 train | Net change | Validation boxes |
|---|---:|---:|---:|---:|
| <4px | 2 | 3 | +1 | 2 |
| 4–8px | 91 | 141 | +50 | 32 |
| 8–16px | 978 | 1082 | +104 | 230 |
| 16–32px | 1877 | 1950 | +73 | 414 |
| ≥32px | 2538 | 2631 | +93 | 389 |
| Total | 5486 | 5807 | +321 | 1067 |

Only 144/5807 D3 training boxes (2.48%) have a short side below 8px; 1226
(21.11%) are below 16px. The net increase below 8px is 51, out of +321 overall.
These are net bucket changes, not identities of newly added objects: revised
coordinates, replacements and removals can also change bucket membership.

Short-side filtering also removes long, thin objects that may remain visible.
An area criterion differs substantially: only 41 boxes have projected area
below 64px², and 713 below 256px². The JSON includes both short-side and
square-root-area distributions and affected image counts.

## Existing validation predictions

The table shows correctly classified, localized objects / ground-truth objects
at confidence ≥0.25 and IoU ≥0.5. Predictions are matched in descending score
order, one-to-one. These diagnostic recalls are **not mAP**, and thresholds
need not have the same calibration across models.

| Short side | Nano D2 | Nano D3 | Medium D2 | Medium D3 |
|---|---:|---:|---:|---:|
| <4px (2 objects) | 0/2 | 0/2 | 0/2 | 0/2 |
| 4–8px (32 objects) | 3/32 | 6/32 | 11/32 | 13/32 |
| 8–16px (230 objects) | 80/230 | 95/230 | 121/230 | 121/230 |
| 16–32px (414 objects) | 196/414 | 209/414 | 262/414 | 268/414 |
| ≥32px (389 objects) | 230/389 | 243/389 | 244/389 | 273/389 |

At this fixed threshold, Nano D3 recovers more labeled objects across all
nontrivial size bins. It also emits more false positives: 656 unmatched/wrong
class detections out of 1209, versus D2's 525 out of 1034. Correct matches rise
from 509 to 553. This is consistent with a precision/recall trade-off at this
operating point; it does not identify why aggregate AP changes. The JSON also
includes confidence 0.001 and class-agnostic localization diagnostics.

This evidence challenges the strong claim that Nano cannot see the added
objects. It does not rule out insufficient detail for material recognition,
training interference, noisy labels, capacity limits, or seed variability.
Medium's validation recall gains are not confined to the smallest bins.

## Useful next experiments

1. **Isolate resolution:** train Nano on the unchanged D3 at 640 using the same
   starting checkpoint, seed and recipe, then compare with Nano320. The earlier
   [640 inference-only probe](../resolution/README.md) did not improve validation
   AP, but it did not train at 640. This isolates training/inference resolution
   within one architecture. It remains an exploratory one-seed comparison.
2. **Test the proposed annotation policy:** a separate Nano320 variant could
   omit train boxes with projected short side <8px (144 boxes in 118 images).
   Keep all 3431 training images, the original 150-epoch recipe and pretrained
   initialization. Preserve validation/test and their small boxes verbatim.
   Evaluate full validation AP, precision/recall and size groups before one
   final test evaluation. Report the new fingerprint and every omitted label.
   This would test the consequences of label deletion, not prove invisibility.

Deleting an annotation while retaining its object does not create an ignore
region in standard YOLO training: locations without assigned targets contribute
negative class targets. See the [Ultralytics detection loss](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/loss.py).
An actual ignore-region loss would be a separate implementation and experiment;
masking image pixels or excluding whole images would introduce other changes.
The 8px value is a proposed conservative ablation, not a measured visibility
boundary or a chosen production threshold. Do not sweep thresholds on test.

## Reproduce

```bash
python -m scripts.analyze_object_sizes \
  --data-root data/versions/d3 \
  --baseline-root /path/to/restored-d2 \
  --evaluations /path/to/verified/runs/evaluation \
  --output /tmp/object-size-audit.json
```

The script verifies frozen manifest and label hashes, unchanged D2/D3 training
membership, validation membership/counts, and the prediction category convention
of the pinned Ultralytics 8.4.143 serializer (JSON IDs 1–4). It records prediction
and checkpoint hashes. Source data and per-image predictions remain local;
only the aggregate [audit](validation-audit.json) is published. Unit tests cover
aspect-ratio projection, boundary bins, duplicate predictions, wrong material
and localization misses. These diagnostics do not alter published D3 metrics.
