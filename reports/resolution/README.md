# Validation-only inference resolution probe

The same Nano checkpoints trained at 320 px were evaluated at 320 and 640 px.
No weights, thresholds, or checkpoints were selected in this probe. No new test
evaluation was run. The [protocol](../../configs/nano_resolution_probe.json)
was recorded before the 640 px runs; the 320 px anchors come from the fixed
common-holdout evaluation. All other evaluation settings remain unchanged.

| Checkpoint | Validation AP50–95 at 320 | At 640 | Change (pp) |
| --- | ---: | ---: | ---: |
| D2 Nano | 29.48% | 26.87% | −2.61 |
| D3 Nano | 29.14% | 28.53% | −0.61 |

Increasing inference resolution alone did not improve validation AP for either
320-trained checkpoint. This does not test training Nano at 640, and it does
not isolate the cause of the D3 result. The Pico artifact remains fixed at 320.
The comparison includes one checkpoint per dataset version and no uncertainty
estimate. [Complete summaries](nano-validation.json) preserve per-class metrics,
input populations, checkpoint hashes and runtime settings.

Reproduce the extra evaluation with the pinned training environment:

```bash
python -m runners.evaluate_d3 --weights /path/to/nano-best.pt \
  --imgsz 640 --split val --output runs/resolution/nano_640_val
```
