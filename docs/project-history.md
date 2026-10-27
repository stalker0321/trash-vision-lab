# Project history and evidence

The repository brings together the dataset, training and embedded-deployment
workspaces. The recovery audit distinguishes completed experiments from plans;
a discussion alone is not counted as a completed model run.

## Development stages

| Stage | Completed work |
| --- | --- |
| April–June 2025 | Six-class classifier and Telegram prototype, followed by object detection. Early incorrect polygon/category conversions are excluded from later comparisons. |
| Dataset curation | TACO official/community, additional real glass images, personal street photographs, catalog photographs and object crops; manual cleanup and grouping of related images. |
| March 2026 hard mining | Three YOLO26L models, 120 epochs at 640 px, with each of 3683 real images held out in exactly one fold. Catalog/crop images stayed in training. |
| April 2026 review | Model proposals were reviewed in CVAT. The retained reviewed pool has 5464 images and 8881 boxes, versus 8387 boxes in the earlier label snapshot. |
| September 2026 benchmark | Nano320 and Medium640 trained on D0/D1/D2 with seeds 42 and 123, plus separately identified follow-up runs. |
| September 2026 deployment | D2 Nano exported to RKNN INT8; camera/NPU pipeline, button/LED field capture, startup and recovery of saved-photo counters on Luckfox Pico Mini A. |
| October 2026 D3 | Same selected D2 images and splits, final reviewed annotations, reproducible manifest, new Nano/Medium training and a fixed common-holdout evaluation queue. |
| October 2026 Nano follow-ups | Full Nano640 training and Nano320 with tiny train labels omitted; validation of both and a 640-trained checkpoint evaluated at 320. |

The source group named “my_photos” includes 597 personal photographs plus
27 images from dataset_ninja_glass and 41 from small_glass_dataset.
It would be inaccurate to describe all 665 as personally photographed.

Catalog means simpler real object photographs under controlled conditions.
The frozen D2/D3 training recipe uses 650 catalog photographs and 210
identified-parent crops; neither appears in validation/test. No generated
synthetic images are included in that frozen recipe.

## Hard mining and manual review

The three real-image validation folds contain 1228, 1228 and 1227 images.
This was a review-selection workflow without an independent test set.
Its fold scores are not comparable to the later D0/D1/D2 benchmark.

Predictions were compared with existing annotations to identify potential
missing boxes and class conflicts. Temporary CVAT classes were used to keep
review proposals distinct from accepted annotations.

| Source | Review images | Candidate boxes |
| --- | ---: | ---: |
| TACO official | 590 | 830 |
| TACO community | 752 | 1150 |
| Personal/additional glass group | 344 | 530 |
| Total | 1686 | 2510 |

These are candidate counts, not automatically accepted annotations.
The retained evidence includes all three training CSVs/checkpoints and the
source-specific review packages. The original final prediction dumps were
not all recovered, so the repository does not claim a complete replay of
that historical selection step. Rebuilding D3 from the frozen manifest does
not require CVAT.

## Why D3 exists

The audit found that the September benchmark used the earlier label snapshot,
although the later reviewed pool already existed. The retained evidence does
not establish why that older snapshot was selected.

D3 preserves D2 image identities, groups and splits, and changes the labels.
Within the selected training split, boxes increase from 5486 to 5807.
Validation becomes 553 images / 1067 boxes, and test 552 / 1044.
The 84-image hard subset is contained in test.

This separates the historical benchmark from training on the final reviewed
annotations. Historical datasets and checkpoints remain unchanged;
[baseline retention policy](retraining-plan.md) explains the decision.

## Evidence available in this repository

- [Frozen D3 manifest](../manifests/d3.json), including image/label hashes and splits.
- [Historical experiment records](../reports/historical/experiments.json):
  12 main runs and the selected Nano WD=0.001 run, each with its complete
  150-epoch CSV and checksum.
- [Recomputed historical results](../reports/historical/summary.json) and
  [plot](../reports/historical/validation.svg).
- [Historical follow-ups](../reports/historical-followups/README.md): weight
  decay, real-scene fine-tuning, a matched control/distillation pair and 256-input
  adaptation, with completed run records and full training curves.
- [D3 Nano resolution and tiny-label results](../reports/nano-ablations/README.md),
  including the negative results and the limits of each comparison.
- [Original TACO snapshot and class mapping](taco-provenance.md).
- [D3 training and evaluation protocol](d3.md).
- [Recovered Luckfox runtime and measurements](luckfox.md).

The earlier Nano D0/D1 seed42 CSVs were recovered; their best checkpoints
were not found in the inspected local snapshots. Historical table entries
therefore describe recorded validation curves, not newly reevaluated weights.

## Interpretation limits and engineering lessons

Known crop families and duplicates were grouped before splitting, but the
recovered groups do not prove that every related capture sequence was found.
Historical mining/review also involved the broader real-image pool, limiting
claims of a completely independent retrospective test.

D0/D1/D2 change complete data recipes and sample counts. Nano and Medium also
differ in resolution and, in the original series, weight decay.
Historical D2 best checkpoints were selected on older validation labels;
evaluating them on reviewed labels does not create a perfectly controlled
annotation-only retraining experiment.

A later RKNN export rewrote 72 training/calibration JPEG files in the old
workspace. Their intact source copies were recovered and verified; the new
D3 materialization uses independent copies with the frozen hashes.
Export dependencies also changed the old ML environment, so training and
RKNN export environments are kept separate.

The field audit found false positives, duplicate detections and paper/plastic
confusion. Its saved JPEGs already contain board-drawn boxes and are not a
clean ground-truth benchmark. No accuracy improvement is claimed from the
discussed blur/area heuristics or the interrupted Medium640 field comparison.

Raw images, private conversation exports and recovery logs remain outside Git.

## Earlier attempts and unexecuted plans

The private conversation recovery also records a catalog-only curriculum
warm-up followed by mixed-data training, alongside changes from five/six target
classes to four. The precise checkpoint/configuration and final third curriculum
stage were not established. These attempts belong to the development history;
their chat/screenshot metrics are not added to the reproducible benchmark tables.

The archived September plan also proposed Medium D0/D1 retraining on historical
`without_small_obj` variants. The inspected queue/run records do not establish
that these jobs ran. That proposal is distinct from the completed October D3
Nano320 experiment with a precisely defined 8px threshold.
