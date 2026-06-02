# MIGRATION.md

Running log of the Phase-A ground-up rebuild (behavior-preserving restructure). One row per ported
module. See [REBUILD_SCHEMATIC.md](REBUILD_SCHEMATIC.md) for the prompts and [CLAUDE.md](CLAUDE.md) for the
architecture, band-aid table (B1–B13), and imbalance policy. The porting workflow itself is the
`behavior-preserving-port` skill.

**OLD repo (read-only reference):** `c:/Users/LENOVO/Desktop/Undergrad_Project/Undergrad_thesis_project`

## How to use this file

For each module you port:
1. Capture a golden fixture from the OLD repo **before** writing new code.
2. Port into the target file(s); resolve the band-aids the prompt lists.
3. Add a parity test; record the result here.
4. Fill in the row: golden fixture path, band-aids resolved, behavior changes flagged, parity result.

**Parity legend:** ✅ exact (within tol) · ⚠️ intentional behavior change (justified in Notes) ·
❌ failing/blocked · — not started.

## Progress

| Prompt | Module(s) | Source (OLD) | Golden fixture | Band-aids resolved | Parity | Notes |
|---|---|---|---|---|---|---|
| 0.1 | repo scaffold, `pyproject.toml`, `.gitignore` | `requirements.txt`, repo root | n/a | B11, B12 | ✅ | gate green on py3.10: ruff + 3 smoke tests + editable src-layout import |
| 0.2 | `config/schema.py`, `config/loader.py`, `configs/*.yaml` | `config.py`, hardcoded args | `tests/fixtures/golden/legacy_config.json` | B1, B6, B7 | ✅ | `vit_kwargs()`/`motion_kwargs()` reproduce OLD `config.py` dicts exactly; 23 config tests green on py3.10. See Config decisions below. |
| 0.3 | `utils/{seed,device,amp,memory,logging}.py`, `paths.py` | `train.py` perf/AMP/mem idioms | n/a | B8, part B1/B9 | — | |
| 1.1 | `data/pie_sequences.py` | `scripts/generate_sequences.py` | | B5 | — | verify dataset-stat table |
| 1.2 | `data/lmdb_writer.py`, `data/transforms.py` | `scripts/preprocess_data_lmdb.py`, `PIE_sequence_Dataset_1.py` | | B5, upstream B7 | — | document 8-dim motion channels |
| 1.3 | `data/balance.py` | `scripts/balance_sequences.py`, `split_balance_sequences_all.py` | | B3 (offline), B5 | — | imbalance policy (w/ 1.6, 3.1) |
| 1.4 | `data/augment.py` | `scripts/augment_sequences.py` | | B5 | — | flip negates motion[:,2] — verify index |
| 1.5 | `data/lmdb_dataset.py`, `data/collate.py` | `scripts/lmdb_dataset.py`, `train_utils.py` | | B7 | — | worker-safe LMDB env |
| 1.6 | `data/sampler.py` | `train.py:34-123` | | B3 (online, dedup scans) | — | single metadata scan (w/ 1.3, 3.1) |
| 1.7 | `data/stats.py`, `scripts/count_labels.py` | `label_count.py` | | — | — | drift check vs stat table |
| 2.1 | `models/vit.py` | `models/Vision_Transformer.py` | | B2, B13, B6 | — | eager params → strict=True load |
| 2.2 | `models/motion_encoder.py` | `models/Motion_Encoder.py` | | — | — | T≤200 guard |
| 2.3 | `models/cross_attention.py`, `models/heads.py` | `models/Cross_Attention_Module.py` | | B4 | — | crosses_pooled decision |
| 2.4 | `models/ensemble.py`, `models/registry.py` | `models/Unified_Module.py`, `scripts/model_utils.py` | | B10 | — | typed factory + forward adapter |
| 2.5 | `models/ablations.py` | `models/AblationModels.py` | | B11 | — | per-ablation output keys |
| 3.1 | `losses/multitask.py` | `train.py:144-153,341-345` | | B3 (loss), part B1 | — | imbalance policy (w/ 1.3, 1.6) |
| 3.2 | `training/metrics.py` | `train.py` val, `test.py` eval | | B1 | — | shared by train+test |
| 4.1 | `training/trainer.py` | `train.py:125-175,236-632` | | B1, B2 (consumer), B8 | — | no dummy-forward |
| 4.2 | `training/chunk_loader.py` | `train.py:368-504`, `train_utils.py:80-98` | | B9 | — | crash-safe ChunkPrefetcher |
| 4.3 | `training/callbacks.py` | `train.py` ckpt/early-stop/sched | | B2 (load), B11, B1 | — | full-state resume, strict=True |
| 4.4 | two-phase schedule on Trainer | `train_two_phase.py` | | B1 | — | phases as config, not god-script |
| 4.5 | `utils/logging.py` CSV conventions | `train.py`/`test.py` CSV writers | n/a | B11, B1 | — | run-dir + index.csv |
| 5.1 | `eval/evaluate.py`, `scripts/evaluate.py` | `test.py` | | B1, B10 | — | |
| 5.2 | `eval/benchmark.py` | `Vision_Transformer.py` fvcore use | n/a | — | — | params/FLOPs/latency/FPS/VRAM |
| 5.3 | `eval/inference.py` | `main.py`, `extract_frames.py`, `pedestrian_detection.py` | | — | — | reuse Phase-1 preprocessing |
| 6.1 | `viz/plots.py`, `scripts/visualize.py` | `scripts/plot_results.py` | n/a | — | — | consume new CSV schema |
| 6.2 | `viz/qualitative.py` | `visualize_comparison.py`, `visualize_gt.py` | n/a | B11 | — | temporal_weights overlays |
| 7.1 | `export/onnx.py`, `scripts/export_onnx.py` | `onnx/onnx_export.py` | | B2 | — | onnxruntime parity check |
| 8.1 | `tests/`, CI gate | OLD `test_*.py` ad-hoc scripts | n/a | B12 | — | golden fixtures + ruff/pytest |
| 8.2 | `CLAUDE.md`, `README.md`, docstrings | OLD `CLAUDE.md`, `README.md`, `GUIDELINE.md` | n/a | — | — | keep stat table in sync |
| 9.1 | cutover & legacy retirement | whole OLD repo | n/a | B11 | — | parity gate per model_type |

## Decisions Log

Record cross-cutting decisions here as they're made (so coupled prompts stay consistent):

- **Imbalance policy** (1.3 / 1.6 / 3.1): _TBD — which of offline-balance / sampler / loss-weight is the default; which are opt-in._
- **`crosses_pooled` fate** (B4, Prompt 2.3): _TBD — auxiliary-diagnostic vs config-gated-off._
- **8-dim motion channel definition** (1.2 / 1.4 / 2.2): _TBD — document each channel; confirm flip-negated index._
- **Sequence-length policy** (1.5): _TBD — truncate vs pad vs variable._
- **Custom chunk prefetch vs torch DataLoader** (4.2): _default to preserving custom prefetch this phase._

### Config decisions (Prompt 0.2)

Locked so the later prompts that consume config stay consistent:

- **Defaults source = `config.py` + `train.py`/`test.py` literals**, NOT the `__main__` smoke-test kwargs.
  B6 drift recorded: `Vision_Transformer.__main__` used `d_model=224`, `stage_dims=[48,96,168,96]`,
  `head_nums=[2,4,7,4]`, `dropout=0.1`; `Motion_Encoder.__main__` used `hidden_dim=224`. These never fed
  training (train.py imports `vit_args_config()`/`motion_enc_args_config()`); `ModelCfg` mirrors the latter.
  The drifting `__main__` blocks get rebuilt to consume `ModelCfg` in Prompts 2.1/2.2 — drift closes there.
- **Q1 — dict overrides REPLACE** the whole dict (no deep-merge), e.g. `train.loss_weight={crosses:2.0}`
  drops `actions`/`looks`. Documented in `loader.py`; asserted by `test_override_dict_replaces_not_merges`.
- **Q2 — `TrainCfg.use_amp: bool` is the *request*;** runtime ANDs it with CUDA availability in
  `utils/amp.py` (Prompt 0.3). Schema stores intent, not the resolved value.
- **Q3 — `EvalCfg.bench_context_scale = 3.0`** kept as the single, deliberate synthetic-benchmark scale
  (FLOPs/latency only; real-data eval uses `DataCfg.context_scale = 2.0`, which is fixed by how the LMDB
  crops were physically written). Treated as an intentional value, not a "drift to fix later".
  ⚠️ *Confirm with user:* if "unify to 3.0" meant something other than "keep the benchmark at 3.0", revisit.
- **Q4 — `DataCfg.chunk_size = 5000`** is canonical (OLD `preprocess_data_lmdb.main()` default was 4500).
  Affects only future LMDB re-writes, never model parity.
- **Q5 — frozen dataclasses with `slots=True`;** list-like fields are `tuple`s (immutable, hashable),
  `dict` defaults via `field(default_factory=...)`. Adapters convert tuples back to lists for the OLD
  model constructors (the parity surface). Verified on py3.10.
- **B7 closure (partial):** `MAX_SEQ_LEN → DataCfg.max_seq_len`; the `motions[..., :8]` slice → `motion_dim`
  on both `DataCfg` and `ModelCfg`, cross-checked equal in `validate_config`. The collate slice is deleted
  in Prompt 1.5 once the writer (1.2) is confirmed to emit exactly `motion_dim` channels.
- **Override channel:** CLIs use a repeatable `--set section.field=value` (via `build_argparser`), not
  `argparse.REMAINDER`, so overrides can't swallow real subcommand flags.

## Parity Gate (Phase A → cutover)

Before retiring the legacy repo (Prompt 9.1): the new repo must reproduce the OLD test metrics per
`model_type` within tolerance, using ported weights. Targets tracked in the `experiment-tracking` skill's
`references/baseline-results.md`.

## OLD Top-Level File Disposition (Prompt 0.1 audit trail)

Disposition contract for every OLD top-level file: **PORT** (logic migrated), **FOLD-INTO-TESTS**
(behavior captured as a test), **DROP** (superseded/dead/artifact), **DECIDE** (open). No file is moved
in 0.1 — this table is the contract the later prompts execute against.

| OLD path | Disposition | New location / reason |
|---|---|---|
| `config.py` | PORT | `config/schema.py` + `loader.py` (0.2). B6. |
| `train.py` (635-line god-script) | PORT (split) | `training/{trainer,chunk_loader,callbacks,metrics}.py` + `losses/multitask.py` + `data/sampler.py` (P3/P4). B1. |
| `train_two_phase.py` | PORT | phase-schedule on `training/trainer.py` (4.4). B1. |
| `test.py` | PORT | `eval/evaluate.py` + `scripts/evaluate.py` (5.1). B1. |
| `main.py` | PORT | `eval/inference.py` (5.3). |
| `label_count.py` | PORT | `data/stats.py` + `scripts/count_labels.py` (1.7). |
| `class_imbalance_strategies.py` | DROP / fold | unified imbalance policy in `data/sampler.py` + `losses/multitask.py` (B3). Salvage formulas, then drop. |
| `imbalance_config.py` | DROP / fold | merged into `TrainCfg` imbalance fields (B3). |
| `models/Vision_Transformer.py` | PORT | `models/vit.py` (2.1). B2, B13, B6. |
| `models/Motion_Encoder.py` | PORT | `models/motion_encoder.py` (2.2). |
| `models/Cross_Attention_Module.py` | PORT | `models/cross_attention.py` + `heads.py` (2.3). B4. |
| `models/Unified_Module.py` | PORT | `models/ensemble.py` (2.4). |
| `models/AblationModels.py` | PORT | `models/ablations.py` (2.5). |
| `models/__init__.py` | DROP | replaced by `models/registry.py` typed factory (B10). |
| `scripts/generate_sequences.py` | PORT | `data/pie_sequences.py` (1.1). B5. |
| `scripts/preprocess_data_lmdb.py` | PORT | `data/lmdb_writer.py` (1.2). B5. |
| `scripts/PIE_sequence_Dataset_1.py` | PORT | `data/lmdb_writer.py` + `transforms.py` (1.2). B5. |
| `scripts/balance_sequences.py` | PORT | `data/balance.py` (1.3). B3/B5. |
| `scripts/split_balance_sequences_all.py` | PORT | `data/balance.py` (1.3). B5. |
| `scripts/augment_sequences.py` | PORT | `data/augment.py` (1.4). B5. |
| `scripts/lmdb_dataset.py` | PORT | `data/lmdb_dataset.py` (1.5). |
| `scripts/train_utils.py` | PORT (split) | collate→`data/collate.py`, EarlyStopping→`training/callbacks.py`, mp prefetch→`training/chunk_loader.py`, memory poll→`utils/memory.py` (B7/B9). |
| `scripts/model_utils.py` | DROP / replace | `models/registry.py` typed dispatch (B10). |
| `scripts/plot_results.py` | PORT | `viz/plots.py` + `scripts/visualize.py` (6.1). |
| `scripts/pedestrian_detection.py` | PORT (conditional) | `eval/inference.py` helper or external `[infer]` extra (decide in 5.3). |
| `scripts/preprocess_data.py` | DROP | dead non-LMDB variant (B5). |
| `onnx/onnx_export.py` | PORT | `export/onnx.py` + `scripts/export_onnx.py` (7.1). |
| `ablation_usage_example.py` | DROP | usage doc only; superseded by README/docstrings. B11. |
| `final_ablation_verification.py` | FOLD-INTO-TESTS | `tests/test_model_shapes.py` (2.5/8.1). B11/B12. |
| `test_ablation_models.py` | FOLD-INTO-TESTS | `tests/test_model_shapes.py`. B12. |
| `test_ablation_structure_clean.py` | FOLD-INTO-TESTS | `tests/test_model_shapes.py`. B12. |
| `test_imbalance_setup.py` | FOLD-INTO-TESTS | `tests/test_losses.py` / test_sampler (1.6/3.1). B12. |
| `visualize_comparison.py` | PORT | `viz/qualitative.py` (6.2). B11. |
| `visualize_gt.py` | PORT | `viz/qualitative.py` (6.2). B11. |
| `extract_frames.py` | PORT / fold | helper for `eval/inference.py` (5.3). B11. |
| `run_env.bat` | DROP | replaced by `pip install -e .[dev]` + README. B11. |
| `requirements.txt` | DROP (absorb) | deps moved into `pyproject.toml`. |
| `CLAUDE.md` / `README.md` / `GUIDELINE.md` | PORT (rewrite) | regenerated in 8.2; new CLAUDE.md/README already exist. |
| `training_log/*.csv`, `*.xlsx` | DROP (gitignore) | run artifacts; archive originals out-of-repo (B11). |
| `plots/*.png`, `qualitative_visualize/*.jpg` | DROP (gitignore) | generated artifacts (B11). |
| `model_outputs/`, `best_model_outputs/`, `venv/` | DROP (gitignore) | weights/env never tracked (B11). |
| `.claude/`, `.understand-anything/`, `.vscode/` | DECIDE → resolved | keep `.claude/skills` tracked; gitignore `.claude/settings.local.json` + `.understand-anything/` caches (open question 5). |

### Dropped from OLD `requirements.txt`

- **Unused (0 imports):** `yacs`, `torchview`.
- **Platform shim:** `pywin32` (Windows-only; left to the resolver / installed ad hoc).
- **Pure transitive** (~25 pins) left to the resolver per the "deliberately minimal" mandate:
  `certifi`, `charset-normalizer`, `colorama`, `contourpy`, `cycler`, `filelock`, `fonttools`, `fsspec`,
  `huggingface-hub`, `idna`, `iopath`, `Jinja2`, `joblib`, `kiwisolver`, `MarkupSafe`, `mpmath`,
  `networkx`, `packaging`, `portalocker`, `py-cpuinfo`, `pyparsing`, `requests`, `safetensors`, `sympy`,
  `termcolor`, `threadpoolctl`, `typing_extensions`, `ultralytics-thop`, `urllib3`.

### Deliberate additions (not behavior changes)

- `onnx` / `onnxruntime` (`[export]` extra) — schematic mandates an onnxruntime parity check (P7);
  absent from OLD requirements.
- `ruff`, `pytest`, `pytest-cov` (`[dev]` extra) — lint + test gate (B12).

### `tabulate` / `timm` watch

- `tabulate==0.9.0` — kept as core; **verify usage in 1.7**, demote/drop if unused.
- `timm==1.0.20` — imported by `Vision_Transformer.py`; keep until 2.1 confirms which symbols are used,
  candidate to drop in Phase B.

### Prompt 0.1 verification checklist

Verified on the repo `.venv` (Python 3.10.0):

- [x] Editable src-layout install succeeds (`pip install -e .` → `pedpredict 0.0.0`), with the full pinned
      core stack installed (`torch==2.7.1+cpu`, `numpy`, …) — the earlier `--no-deps` shortcut is no longer
      needed now that `.venv` carries the real dependencies.
- [x] `ruff check .` exits 0 ("All checks passed!").
- [x] `pytest -m "not slow"` exits 0 (3 passed).
- [x] `import pedpredict` from an arbitrary cwd resolves to `src/pedpredict/__init__.py`, version
      `0.0.0` (proves src-layout install, not accidental cwd import).

> **Environment note:** the repo's `.venv` is now **Python 3.10.0** with `pedpredict` editable-installed —
> the canonical env for all tests/lint; run `.venv/Scripts/python.exe` directly (no `PYTHONPATH`, no side
> venvs). It satisfies `requires-python = ">=3.10,<3.13"`; the pinned `torch==2.7.1` / `numpy==2.2.6` have
> no wheels for 3.13+. (Bare `py` / `python` on this machine still resolve to 3.14 — invoke the venv
> interpreter explicitly.)

