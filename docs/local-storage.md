# Local data and model storage

The repository contains source code, frozen manifests and aggregate reports.
Dataset images, materialized versions, local run artifacts and maintenance
receipts stay outside Git. The local workstation is organized under this
project directory; old workspace paths remain compatibility symlinks.

| Location | Purpose |
|---|---|
| `data/pool/images/sha256/` | Immutable image bytes, addressed by SHA256; one stored inode per identical image |
| `data/pool/reviewed/` | Full reviewed source pool: 5464 images and 8881 boxes, used to build D3 |
| `data/versions/d3/` | Frozen selected D3: 3431 train, 553 validation and 552 test images |
| `manifests/` | Public, versioned D3 and tiny-box-ablation manifests |
| `storage/history/d0-d2-workspace/` | Preserved historical preparation code, raw data, labels, splits, configs and runs |
| `storage/history/transport-snapshots/` | Retained contents of old transport archives; images reference the shared pool |
| `storage/models/` | Named links to model runs and board exports, without extra copies |
| `storage/catalog.json` | Local index of dataset roles, fingerprints and model locations |
| `notes/recovery/` | Private audit receipts, remote-job control records and recovered run artifacts |
| `dist/` | Small model release archives, separate from dataset transport bundles |

## Which annotations are D2 and D3?

The old `trash-lab-dataset-v2/trash_dataset_4_class` pool contains the annotations
used for D2. The old standalone `trash-vision-dataset` pool contains the later
reviewed annotations used for D3. The folders are source pools; D2/D3 themselves
are the frozen selections and split memberships recorded in their manifests.

The two full pools have 5464 byte-identical images, checked with SHA256.
Their box counts are 8387 and 8881. There are 874 byte-different label files;
548 remain different after sorting numeric rows and rounding to eight decimal
places. Both annotation histories are retained independently. For every one of
the 4536 selected clean images, normalized old-pool labels match the frozen D2
labels and reviewed-pool labels match D3.

The D2 training split has 5486 boxes; D3 has 5807 on the same 3431 images.
Validation/test also have their own preserved historical and reviewed labels.
See [D3 provenance](d3.md) and the [common-holdout comparison](../reports/d3/README.md).

## Image sharing and reproducibility

On this workstation, existing image paths are **read-only hardlinks** to the
SHA256 pool. Filenames, bytes, image IDs and split membership remain unchanged;
annotations are separate files. The pool's `index.json` maps existing image
views to their hashes. This prevents duplicate image payloads from occupying
disk space in every historical/materialized version.

The historical prepared workspace also contained 72 JPEGs previously rewritten
by an old board export, already identified in the earlier integrity audit.
Those image paths were restored to the exact original SHA256 values in the
frozen historical manifest. The rewritten bytes remain preserved under
`storage/history/jpeg-export-rewrites/` with a repair receipt. No annotations
or training results were changed.

Image files have no write bits. Make an independent writable copy before
editing a photograph; do not change permissions on a shared inode. Frozen
dataset labels and manifests must likewise be revised through a new version,
not silently overwritten. The normal D3 builder still makes independent output
copies for a new training/export workspace, and its read-only image verifier
does not rewrite source JPEGs.

The reviewed source root is now `data/pool/reviewed`. For example:

```bash
python -m datasets.build_d3 --from-manifest \
  --reviewed-root data/pool/reviewed \
  --manifest manifests/d3.json --output /path/to/new-d3-workspace
```

Historical dataset manifests, labels, references and absolute-path aliases are
preserved. Validation/test remain real-only. Directory sizes viewed separately
can count the same hardlinked images repeatedly; `du -sh` on the whole project
counts each inode once.

## Retention policy

Retain source image content, every annotation version, split/group manifests,
pretrained weights, `best.pt`, `last.pt`, complete epoch CSVs, run configs,
evaluation outputs and board-export provenance. Old completed-run `epoch*.pt`
snapshots are expendable when best/last and the experiment history have been
verified. Do not prune an active training directory.

Large uncompressed transport archives are temporary. Before deleting their
containers, retain every payload: image members by content hash, all other
members in a small snapshot, and the member inventory with checksums and
original metadata. This preserves historical code/configuration/annotation
snapshots that may differ from current files.

The 2026-10-27 cleanup has private before/after inventories and preservation
checks under `notes/recovery/storage-cleanup-20261027`. It only changes local
storage organization; it does not change model weights, annotations, reported
metrics, the published D3 release, or the separate MSI training workspaces.
