# Historical adaptation, distillation and regularization experiments

These September D2 experiments were recovered from completed run records,
full epoch CSVs and retained weights. They are separate from the main
D0/D1/D2 comparison and from the October D3 follow-ups.

**The table reports the maximum recorded epoch validation mAP50–95 on the old
D2 labels: 553 images / 961 boxes.** It is not a fresh evaluation on the reviewed
D3 validation labels (1067 boxes), and it must not be joined to the D3 table as
though the annotation sets were identical. All runs below use seed 42.

| Experiment / reference | Training input | Epochs in this run | Best recorded validation mAP50–95 | Best epoch |
|---|---:|---:|---:|---:|
| Original Nano, WD=0.0005 | 320 | 150 | 28.26% | 111 |
| Nano, WD=0.001, trained from pretrained weights | 320 | 150 | 29.73% | 84 |
| Original Medium, WD=0.0005 | 640 | 150 | 44.80% | 82 |
| Medium, WD=0.001, trained from pretrained weights | 640 | 150 | 46.67% | 120 |
| Nano: real-scenes-only fine-tuning (`street20`) | 320 | 20 | 27.02% | 18 |
| Nano: control fine-tuning (`adapt30`) | 320 | 30 | 30.39% | 3 |
| Nano: Medium-teacher distillation (`kd30`) | 320 | 30 | 29.62% | 1 |
| Nano: low-resolution fine-tuning (`lowres30`) | 256 | 30 | 25.63% | 2 |

## Medium → Nano distillation

`adapt30`, `kd30` and `lowres30` all start from the same completed Nano WD=0.001
best checkpoint. They use full D2 training data, batch 16, AdamW, learning rate
0.0001, weight decay 0.001, cosine decay, one warmup epoch, mosaic 0.5 and
close-mosaic 10. These are 30 additional epochs, not training from scratch.

The control and KD protocols differ by the addition of the Medium teacher and
the distillation loss. KD uses Ultralytics 8.4.143 feature distillation with
`dis=6.0`; the teacher is Medium D2/seed42/WD=0.001. Both student and frozen
teacher receive the same 320 input. The retained CSV records positive finite
`train/dis_loss` throughout all 30 epochs. The teacher's separate
[320 preflight evaluation](evaluations/teacher_m320/summary.json) is 31.83% AP.

KD reached 29.62%, versus 30.39% for the matched control and 29.73% for its
parent. This attempt did not demonstrate a distillation benefit. It tests one
teacher, loss weight and adaptation recipe; it does not rule out distillation
in general. The early best epochs and lower final scores are preserved in the
CSV files, rather than reporting only a successful-looking intermediate value.

## Other completed follow-ups

- **Regularization:** changing weight decay from 0.0005 to 0.001 increased
  recorded best validation AP for both Nano and Medium in these seed42 runs.
  This is a single-seed observation across different GPU environments.
- **Real-scene fine-tuning:** `street20` starts from the original Nano
  WD=0.0005 checkpoint, not the later WD=0.001 parent. It uses 2571 real training
  images, excluding catalog photos and crops, with mosaic disabled and a lower
  learning rate. Its 27.02% is below the parent's 28.26%. Several aspects of
  the recipe changed, so this does not isolate the value of catalog images.
- **256-input adaptation:** `lowres30` reached 25.63%. Before adaptation, the
  same parent evaluated at 256 scored [25.16%](evaluations/parent_n256/summary.json).
  That reference is a separate preflight evaluation, while 25.63% is selected
  from the training validation curve. No fresh common-protocol comparison,
  statistical claim or on-board speed gain is established here.

The historical `adapt30` control has a higher recorded validation peak than
the selected deployment model. A subsequent
[common-label evaluation](../adapt30-common/README.md) found 30.38% validation
and 30.93% test AP, versus 29.48% / 30.23% for its parent. Those are new reviewed-label
evaluations, separate from the historical table above. RKNN/on-board testing of
adapt30 is still pending; its historical peak alone did not establish a
replacement deployment model.

## Preserved evidence and limits

[evidence.json](evidence.json) contains a row for every run, exact best/last
recorded AP, checkpoint hashes and SHA256 hashes of every included input file.
`training/` retains the original run metadata and full CSVs, including parent and
teacher identities and training protocols. Seven corresponding best checkpoints
were rehashed successfully during recovery; the original Nano reference here
uses its preserved run record/CSV. Weights and raw conversations remain outside
Git. This is evidence recovery, not a new training or test-evaluation campaign.

The archived training entry point was `trashcv.train` with adaptation helpers;
the present `runners.train_d3` does not implement these historical KD variants.
Run records preserve their recipes, but the current checkout does not claim
a one-command rerun of those legacy training jobs.
