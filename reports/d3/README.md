# D2 and D3 on the same reviewed holdout

Fixed seed42 checkpoints, evaluated against the frozen D3 labels.

All AP values below are percentages.

## Validation

| Configuration | AP50 | AP50–95 |
| --- | ---: | ---: |
| nano_d2, 320 px | 42.51 | 29.48 |
| nano_d3, 320 px | 42.43 | 29.14 |
| medium_d2, 640 px | 60.05 | 47.98 |
| medium_d3, 640 px | 59.71 | 48.22 |

## Test

| Configuration | AP50 | AP50–95 |
| --- | ---: | ---: |
| nano_d2, 320 px | 42.68 | 30.23 |
| nano_d3, 320 px | 40.45 | 28.51 |
| medium_d2, 640 px | 57.04 | 45.53 |
| medium_d3, 640 px | 59.02 | 47.64 |

## Hard subset of test

| Configuration | AP50 | AP50–95 |
| --- | ---: | ---: |
| nano_d2, 320 px | 30.63 | 18.77 |
| nano_d3, 320 px | 30.62 | 18.56 |
| medium_d2, 640 px | 51.73 | 38.85 |
| medium_d3, 640 px | 56.83 | 43.80 |

## Test AP50–95 by material

| Configuration | plastic | paper | metal | glass |
| --- | ---: | ---: | ---: | ---: |
| nano_d2 | 37.72 | 29.20 | 32.36 | 21.65 |
| nano_d3 | 35.30 | 32.18 | 27.53 | 19.02 |
| medium_d2 | 48.58 | 40.98 | 51.70 | 40.87 |
| medium_d3 | 51.93 | 42.57 | 54.80 | 41.25 |

All per-class AP values are retained in [per-class.csv](https://github.com/stalker0321/trash-vision-lab/blob/main/reports/d3/per-class.csv).

## Interpretation limits

- One seed per model; no estimate of run-to-run uncertainty.
- Historical D2 best checkpoints were selected using older validation labels.
- Hard test is a subset of test, not an independent evaluation.
- The real-image pool was involved in historical mining/review; this is a retrospective holdout.
- Nano uses 320 px and Medium 640 px; this does not isolate architecture alone.
- Medium resumed from an optimizer checkpoint after a network interruption; resumed input/augmentation order need not match an uninterrupted run.
