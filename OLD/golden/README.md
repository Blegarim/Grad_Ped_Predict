# OLD golden reference artifacts

This folder is the **golden artifact reference** for Phase-A behavior-preserving porting. It is vendored
into this repo (not gitignored) so that remote / sandbox Claude Code sessions have the reference without
needing the workstation's local disk.

## What lives here

| File | Provenance |
|---|---|
| `sequences_train_sample.pkl` | first 500 sequences of `sequences_train.pkl` |
| `sequences_val_sample.pkl`   | first 500 sequences of `sequences_val.pkl` |
| `sequences_test_sample.pkl`  | first 500 sequences of `sequences_test.pkl` |

Each sample is a `list[dict]` with keys `images` (list[20]), `bboxes` (list[20]), `actions`, `looks`,
`crosses` (ints). `bboxes` is present here because it is only dropped later, at the LMDB-writer stage (1.2).

These 500-sequence slices are deliberately small (~1.2 MB each) — enough to verify pipeline **shape /
window-structure / label-logic** parity without committing 446 MB of full pkls (which also exceed GitHub's
100 MB-per-file limit and would need Git LFS).

## What was intentionally NOT vendored

- **Full `sequences_{train,val,test}.pkl`** (219 / 52 / 175 MB) — live on the workstation at
  `…/Undergrad_Project/Undergrad_thesis_project/`. The full label-rate table is documented in `CLAUDE.md`
  (the committed table matches these pkls exactly), so the full files are only needed for a full data
  re-run, not for parity checks.
- **Trained weights** `model_outputs/best_model_epoch28_0122_1511.pth` (full) and
  `…_epoch18_0205_1626_motion_only.pth` (motion_only) — **excluded as stale and not parity-usable.** They
  were saved Jan 22 / Feb 5, but `models/` was refactored in 5+ later commits (through May 27): the fusion
  LayerNorm went single `norm` → separate `image_norm`/`motion_norm`, and the ViT relative-position-bias
  handling changed (B2). Against the current OLD code they only load with `strict=False`, leaving the
  fusion norms randomly initialised — so a forward pass is **not** numerically faithful. A golden weight
  fixture, if ever needed, should be re-trained/re-captured from the current OLD code, not reused from these.

## Regenerating these samples

```
.venv/Scripts/python.exe - <<'PY'
import pickle
ORIG = r"c:/Users/LENOVO/Desktop/Undergrad_Project/Undergrad_thesis_project"
for split in ("train","val","test"):
    d = pickle.load(open(f"{ORIG}/sequences_{split}.pkl","rb"))[:500]
    pickle.dump(d, open(f"OLD/golden/sequences_{split}_sample.pkl","wb"), protocol=pickle.HIGHEST_PROTOCOL)
PY
```
