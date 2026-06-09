"""Training entry point (Prompt 4.4, B1).

Dispatches to either:
  * Single-phase: ``Trainer.fit()``              — ``schedule.enabled = false`` (default)
  * Multi-phase:  ``run_phase_schedule()``        — ``schedule.enabled = true``

Usage:
    # single-phase (plain Trainer):
    python scripts/train.py --config-dir configs

    # enable the 3-phase schedule (override enabled flag):
    python scripts/train.py --set schedule.enabled=true

    # override a phase LR via dotted path (uses the schedule.yaml values as base):
    python scripts/train.py --set schedule.enabled=true

    # standard config override:
    python scripts/train.py --set train.lr=5e-5 --set train.num_epochs=5
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.paths import resolve_paths
from pedpredict.training.chunk_loader import ChunkPrefetcher, gather_lmdb_chunks
from pedpredict.training.schedule import run_phase_schedule
from pedpredict.training.trainer import build_trainer
from pedpredict.utils.device import get_device
from pedpredict.utils.logging import create_run_dir, make_run_id


def _build_chunk_builders(cfg, *, device, scan_cache):
    """Return a dict mapping data_source names to zero-arg ChunkProvider factories.

    All factories share one ``LabelScanCache`` (so each chunk is scanned at most once across the
    loss-weight scan and the per-chunk sampler scans) and resolve LMDB dirs via ``resolve_paths`` so
    training works from any cwd. ``augmented`` reuses ``ChunkPrefetcher.from_config`` (base + opt-in
    aug train dirs); ``balanced`` points at the opt-in ``lmdb_train_balanced`` dirs.
    """
    resolved = resolve_paths(cfg.paths)
    val_paths = gather_lmdb_chunks([resolved.lmdb_val])
    balanced_dirs = [
        d if d.is_absolute() else resolved.root / d
        for d in (Path(p) for p in cfg.paths.lmdb_train_balanced)
    ]
    balanced_paths = gather_lmdb_chunks(balanced_dirs)
    pin = device.type == "cuda"

    def _augmented():
        return ChunkPrefetcher.from_config(cfg, scan_cache=scan_cache, pin_memory=pin)

    def _balanced():
        return ChunkPrefetcher(
            cfg,
            train_lmdb_paths=balanced_paths,
            val_lmdb_paths=val_paths,
            scan_cache=scan_cache,
            pin_memory=pin,
        )

    return {"augmented": _augmented, "balanced": _balanced}


def main(argv=None) -> int:
    parser = build_argparser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config_dir, args.overrides)

    device = get_device()

    if cfg.schedule.enabled:
        # ---------------------------------------------------------------- multi-phase schedule
        from pedpredict.data.sampler import LabelScanCache, class_weights_ce
        from pedpredict.losses.multitask import build_multitask_loss
        from pedpredict.models.registry import build_model
        from pedpredict.training.callbacks import CheckpointManager
        from pedpredict.training.trainer import Trainer
        from pedpredict.utils.device import enable_perf_flags

        enable_perf_flags(device)
        run_id = make_run_id(cfg.eval.model_type, "schedule")
        run_dir = create_run_dir(Path(cfg.paths.runs_dir), run_id)

        # Build loss once from the augmented (full) train set — shared across all phases.
        scan_cache = LabelScanCache()
        resolved = resolve_paths(cfg.paths)
        augmented_paths = gather_lmdb_chunks(resolved.lmdb_train)
        counts = scan_cache.aggregate_counts(augmented_paths)
        class_weights = class_weights_ce(counts, device=device)
        loss = build_multitask_loss(cfg.train, class_weights).to(device)

        model = build_model(cfg)
        chunk_builders = _build_chunk_builders(cfg, device=device, scan_cache=scan_cache)

        # The trainer starts with augmented chunks; reset_for_phase will swap them per phase.
        initial_chunks = chunk_builders["augmented"]()
        trainer = Trainer(
            cfg, model, device, initial_chunks,
            loss=loss,
            scan_cache=scan_cache,
            # Placeholder (None dir -> no-op saves); run_phase_schedule replaces it per phase.
            checkpointer=CheckpointManager(None, run_id=run_id, model_type=cfg.eval.model_type),
            logger=None,                            # will be replaced per phase
            run_dir=run_dir,
        )

        results = run_phase_schedule(
            cfg, trainer, cfg.schedule, chunk_builders, run_dir=run_dir
        )
        print(f"Schedule complete. {len(results)} phases, run dir: {run_dir}")
        for pr in results:
            n_ep = len(pr.epoch_results)
            best = pr.epoch_results[-1].val_loss if pr.epoch_results else float("nan")
            print(f"  {pr.phase_name}: {n_ep} epoch(s), final val_loss={best:.4f}, "
                  f"best_ckpt={pr.best_ckpt}")

    else:
        # ---------------------------------------------------------------- single-phase
        chunks = ChunkPrefetcher.from_config(cfg, pin_memory=(device.type == "cuda"))
        trainer = build_trainer(cfg, chunks, device=device)
        results = trainer.fit()
        print(f"Training complete. {len(results)} epoch(s), run dir: {trainer.run_dir}")

    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
