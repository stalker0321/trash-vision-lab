# Historical baseline retention policy

Only D3 Nano320 and Medium640 are part of the current retraining work.
Historical D0/D1/D2 data and checkpoints are preserved.

An initial proposal to rerun all versions after removing images containing
TACO Unlabeled litter was cancelled after inspecting the actual training
labels. The restored D0/D1 mapping already excluded that category from target
boxes. Its older accidental assignment to plastic belongs to an earlier
six-class export, not to these historical checkpoints.

A separate convention mismatch remains for blister packaging: the historical
raw mapping used metal, while the reviewed convention uses plastic.
There are 6 such source boxes in D0 and 25 in D1, with D0 contained in D1.
The effect on metrics has not been measured; it is not asserted to be zero
or statistical noise.

Historical limitations are documented in [the TACO audit](taco-provenance.md).
New training follows [the D3 protocol](d3.md), and comparisons use the same
reviewed validation/test annotations. Reviewed labels are not retroactively
overlaid onto the historical D0/D1 training data.
