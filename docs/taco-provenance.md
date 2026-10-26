# Original TACO and the four-material baselines

The original TACO annotations, a legacy six-class YOLO export, and this
project's four-material baselines are distinct representations.

| Representation | Meaning |
| --- | --- |
| Upstream TACO COCO | 60 categories, 28 supercategories, segmentation polygons and separate bounding boxes |
| Legacy six-class YOLO export | Earlier derived labels; the folder name “original” does not make its taxonomy original |
| Historical D0/D1 | COCO bounding boxes mapped to four materials, with group separation and documented geometry rules |

The [upstream project](https://github.com/pedropro/TACO#readme) describes its
COCO annotations and marks community annotations as not yet reviewed by its
authors. The exact source commit, download URLs and SHA256 values are retained
in [provenance.json](../reference/taco/provenance.json).

| Source snapshot | Images | Annotations | Nonempty segmentations | Target boxes after mapping, before splitting |
| --- | ---: | ---: | ---: | ---: |
| Official | 1500 | 4784 | 4784 | 3554 |
| Community | 3831 | 8334 | 8319 | 6892 |

## Bounding boxes

The restored September preparation used the COCO bounding-box field in
xywh form. It did not reinterpret the first polygon coordinates as a box.
Coordinates were normalized against the declared COCO frame and clipped at
image boundaries; no additional minimum-box-size filter was applied.

An independent polygon-envelope audit found no difference larger than
1.01 pixels from the corresponding COCO box, including multipart polygons.
Reconversion matched all 4401 retained raw training label files:
1094 official and 3307 community. D0 is contained in that audited set.

Correct conversion does not repair incorrect source coordinates. The prepared
pool contains 179 community images with a difference between COCO dimensions
and the oriented image dimensions. Some normalize correctly; the reviewed
frame 000010 is a confirmed counterexample. Twelve community images were
excluded for invalid target geometry and 20 raw boxes were clipped.

These are **TACO-derived four-material baselines**, not the untouched
60-category benchmark.

## Class conventions

The complete historical mapping is preserved in
[mapping_60_to_4.json](../reference/taco/mapping_60_to_4.json), alongside the
[source taxonomy](../reference/taco/taxonomy_60_classes.csv).

| Examples | Historical target | Interpretation |
| --- | --- | --- |
| Paper, cardboard, drink carton | paper | A visual packaging convention, not a composition analysis |
| Foam, foam cup, plastic straw | plastic | Includes the relevant fine-grained source categories |
| Aluminium/card-backed blister packaging | metal | Composite objects; differs from the reviewed plastic convention |
| Squeezable tube | plastic | Material can be ambiguous for individual objects |
| Bottle and bottle cap | Separate boxes of the respective material | Object/part policy affects target counts |
| Battery, food, rope, shoe, Unlabeled litter, cigarette | Excluded | Pixels remain in the scene without these target boxes |

The earlier six-class export placed some Unlabeled litter into plastic.
The actual restored D0/D1 training mapping excluded source category 58.
Consequently that earlier remapping error must not be attributed to the
historical D0/D1 weights.

The current reviewed blister convention is plastic. The old raw mapping has
6 blister boxes in D0 and 25 in D1; the five examples discussed during visual
review were a matched sample, not the total count. Historical runs are retained
with this limitation rather than silently relabelled.

Some source Unlabeled litter objects have individually reviewed material labels.
That does not justify assigning the entire original category to one material.
A small exploratory model-assisted classification pilot was not applied to
the frozen datasets.

## What the comparisons can establish

D0/D1/D2 compare complete data pipelines, including differences in image
membership, annotations and inclusion policies. Their gains cannot isolate
annotation cleaning, additional photos or catalog sampling individually.

D3 preserves the selected D2 images and updates the annotations. The old/new
weights are evaluated on a common reviewed holdout, while the historical
checkpoint-selection and mining limitations remain documented.
See [D3](d3.md), [project history](project-history.md), and
[baseline retention](retraining-plan.md).
