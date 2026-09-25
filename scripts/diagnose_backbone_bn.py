"""Checks (a) + (b) of docs/RECIPE_V2_PLAN.md on a trained timm-backbone checkpoint. Needs data + checkpoint.

(a) Drift: the BatchNorm running statistics saved in the checkpoint vs the pretrained timm backbone, per
    layer, plus a check that the frozen *weights* really are unchanged.
(b) Re-score one split three ways — as saved | pretrained BN statistics restored | BN re-estimated from
    ``--batches`` training batches under ``--draws`` different sampler draws. The draws reproduce what the
    end of any training epoch does to a frozen-but-train-mode backbone, so their spread is the score jitter
    caused by BN statistics alone.

Writes ONLY to ``--out-dir`` (``bn_drift_<ckpt>.csv``, ``rescore.json``, ``summary.md``) — never the run dir,
so no ``eval_log.csv`` / thresholds / ``index.csv`` side effects. The "as saved" pass must reproduce the
checkpoint's stored ``eval_log.csv`` row for the same split and protocol; a mismatch is reported loudly.

Usage (research PC):
    python scripts/diagnose_backbone_bn.py --out-dir outputs/diagnostics/r1_bn \\
        --checkpoint outputs/runs/<run>/checkpoints/best.pth \\
        --extra-checkpoint outputs/runs/<run>/checkpoints/last.pth
"""

from __future__ import annotations

import csv
import dataclasses
import json
import multiprocessing as mp
import random
import sys
from collections.abc import Iterator
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader

from pedpredict.config import build_argparser, load_config, merge_eval_config, validate_config
from pedpredict.config.schema import RootCfg
from pedpredict.data.augment import RuntimeAugmentor
from pedpredict.data.collate import build_collate
from pedpredict.data.lmdb_dataset import LMDBChunkDataset
from pedpredict.data.pose import pose_motion_transform
from pedpredict.data.sampler import LabelScanCache, build_weighted_sampler
from pedpredict.eval import diagnostics as dg

# Private on purpose: these are the exact loaders `evaluate.py` scores with, so "as saved" is comparable.
from pedpredict.eval.evaluate import _eval_chunk_loaders, _split_chunk_paths, evaluate_model, load_eval_weights
from pedpredict.losses.onset import crosses_metric_keys
from pedpredict.models.registry import build_model
from pedpredict.models.timm_backbone import TimmBackbone
from pedpredict.paths import protocol_lmdb_dirs, resolve_paths
from pedpredict.training.chunk_loader import gather_lmdb_chunks
from pedpredict.utils.amp import resolve_amp
from pedpredict.utils.device import enable_perf_flags, get_device
from pedpredict.utils.seed import set_seed

_PARITY_TOL = 1e-3


def _load_cfg(args) -> RootCfg:
    """Runtime config + the checkpoint's own architecture (same inheritance as scripts/evaluate.py)."""
    cfg = load_config(args.config_dir, args.overrides)
    run_config = Path(args.checkpoint).resolve().parent.parent / "resolved_config.yaml"
    if not run_config.exists():
        raise FileNotFoundError(f"no resolved_config.yaml next to the checkpoint's run dir: {run_config}")
    cfg = merge_eval_config(cfg, run_config, args.overrides)
    validate_config(cfg)
    return cfg


def _load_model(cfg: RootCfg, checkpoint: str, device: torch.device) -> torch.nn.Module:
    model = build_model(cfg, cfg.eval.model_type).to(device)
    load_eval_weights(model, checkpoint, device=device)
    if not isinstance(getattr(model, "vit", None), TimmBackbone):
        raise TypeError(f"{checkpoint}: expected a timm visual backbone at model.vit (model.vit_backbone != legacy)")
    return model.eval()


def _write_drift_csv(path: Path, rows: list[dg.BNDrift]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[f.name for f in dataclasses.fields(dg.BNDrift)])
        writer.writeheader()
        for row in rows:
            fields = dataclasses.asdict(row)
            writer.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in fields.items()})


def _score(model: torch.nn.Module, cfg: RootCfg, device: torch.device, split: str) -> dict[str, float]:
    """One evaluate.py-identical pass; crosses metrics + where the probabilities sit."""
    artifacts = evaluate_model(
        model,
        _eval_chunk_loaders(cfg, _split_chunk_paths(cfg, split), device),
        device,
        cfg.eval,
        use_amp=resolve_amp(cfg.train.use_amp, device),
        collect_predictions=True,
        active_tasks=cfg.train.ordered_active_tasks(),
        output_keys=crosses_metric_keys(cfg.model),
    )
    flat = artifacts.metrics.as_flat_dict()
    preds = artifacts.predictions or {}
    return {
        "n": float(artifacts.n_samples),
        "auc": flat["crosses_auc"],
        "f1_at_0.5": flat["crosses_f1"],
        "precision_at_0.5": flat["crosses_precision"],
        "recall_at_0.5": flat["crosses_recall"],
        "best_f1_same_split": artifacts.oracle["crosses_f1"],
        "best_threshold_same_split": artifacts.oracle["crosses_threshold"],
        **dg.score_summary(preds["crosses_true"], preds["crosses_prob_1"]),
    }


def _stored_row(checkpoint: str, protocol: str, split: str) -> dict[str, str] | None:
    """The checkpoint run's own eval_log.csv row for this protocol/split (latest wins), if any."""
    log = Path(checkpoint).resolve().parent.parent / "eval_log.csv"
    if not log.exists():
        return None
    name = Path(checkpoint).name
    rows = [
        r for r in csv.DictReader(log.open(encoding="utf-8"))
        if r.get("protocol") == protocol and r.get("split") == split and r.get("checkpoint", "").endswith(name)
    ]
    return rows[-1] if rows else None


def _parity(saved: dict[str, float], stored: dict[str, str] | None) -> dict[str, object]:
    if stored is None:
        return {"checked": False, "reason": "no stored eval_log row for this checkpoint/protocol/split"}
    diffs = {k: abs(saved[k] - float(stored[col])) for k, col in (("auc", "crosses_auc"), ("f1_at_0.5", "crosses_f1"))}
    return {"checked": True, "passed": all(d <= _PARITY_TOL for d in diffs.values()), "abs_diff": diffs}


def _train_context_batches(cfg: RootCfg, seed: int, n_batches: int, device: torch.device) -> Iterator[torch.Tensor]:
    """``n_batches`` context-crop batches drawn exactly as training draws them: one random train chunk, the
    weighted sampler, runtime augmentation. Mirrors ``ChunkPrefetcher._build_train_loader`` without persistent
    workers, so stopping early shuts the workers down cleanly."""
    train_dirs, _, _ = protocol_lmdb_dirs(resolve_paths(cfg.paths), cfg.data.protocol)
    path = random.Random(seed).choice(gather_lmdb_chunks(train_dirs))
    augmentor = RuntimeAugmentor(cfg.augment, cfg.data.source_width) if cfg.augment.runtime else None
    dataset = LMDBChunkDataset.from_config(
        path, cfg.data, augmentor=augmentor, aug_seed=cfg.train.seed * 1_000_003 + seed,
        pose_transform=pose_motion_transform(cfg),
    )
    sampler = None
    if cfg.train.use_weighted_sampler:
        sampler = build_weighted_sampler(LabelScanCache().get(path, dataset.seq_ids), cfg.train)
    torch.manual_seed(seed)                     # WeightedRandomSampler draws from the global torch RNG
    loader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, sampler=sampler, shuffle=sampler is None,
        num_workers=cfg.train.num_workers, collate_fn=build_collate(cfg.data), pin_memory=device.type == "cuda",
    )
    iterator = iter(loader)
    try:
        for _ in range(n_batches):
            yield next(iterator)[1].to(device, non_blocking=True)
    finally:
        del iterator
        dataset.close()


def _drift_section(model, pretrained, args, out_dir: Path, device: torch.device, cfg: RootCfg) -> dict:
    """Check (a): drift tables for the checkpoint (and the optional extra one) + the frozen-weight check."""
    reference = dg.bn_state(pretrained)
    report: dict[str, object] = {"max_frozen_weight_difference": dg.max_param_difference(model.vit.net, pretrained)}
    checkpoints = [args.checkpoint] + ([args.extra_checkpoint] if args.extra_checkpoint else [])
    for ckpt in checkpoints:
        net = model.vit.net if ckpt == args.checkpoint else _load_model(cfg, ckpt, device).vit.net
        rows = dg.bn_drift(reference, dg.bn_state(net))
        _write_drift_csv(out_dir / f"bn_drift_{Path(ckpt).stem}.csv", rows)
        report[Path(ckpt).stem] = dg.summarize_drift(rows)
    return report


def _rescore_section(model, pretrained, args, cfg: RootCfg, device: torch.device) -> dict:
    """Check (b): as saved | pretrained BN | BN re-estimated from training batches, per draw."""
    net, saved = model.vit.net, dg.bn_state(model.vit.net)
    reference = dg.bn_state(pretrained)
    results: dict[str, object] = {}
    results["as_saved"] = _score(model, cfg, device, args.split)
    stored = _stored_row(args.checkpoint, cfg.data.protocol, args.split)
    results["parity_with_stored_eval_row"] = _parity(results["as_saved"], stored)
    print(f"[diagnose_bn] as saved: {results['as_saved']}")
    print(f"[diagnose_bn] parity: {results['parity_with_stored_eval_row']}")

    dg.load_bn_state(net, reference)
    results["pretrained_bn"] = _score(model, cfg, device, args.split)
    print(f"[diagnose_bn] pretrained BN: {results['pretrained_bn']}")

    use_amp = resolve_amp(cfg.train.use_amp, device)
    for draw in range(args.draws):
        dg.load_bn_state(net, saved)
        seen = dg.reestimate_bn(
            model.vit, _train_context_batches(cfg, 1000 + draw, args.batches, device),
            use_amp=use_amp, device_type=device.type,
        )
        model.eval()
        entry = _score(model, cfg, device, args.split)
        entry["batches"] = float(seen)
        entry["drift_from_pretrained"] = dg.summarize_drift(dg.bn_drift(reference, dg.bn_state(net)))
        results[f"reestimated_draw{draw}"] = entry
        print(f"[diagnose_bn] re-estimated draw {draw}: {entry}")
    dg.load_bn_state(net, saved)
    return results


def _summary_markdown(report: dict) -> str:
    lines = ["# Backbone BN diagnostics", "", "```json", json.dumps(report["drift"], indent=2), "```", ""]
    rescore = report.get("rescore")
    if rescore:
        keys = ("auc", "f1_at_0.5", "best_f1_same_split", "mean_prob", "share_predicted_positive")
        lines += ["| variant | " + " | ".join(keys) + " |", "|---" * (len(keys) + 1) + "|"]
        for name, entry in rescore.items():
            if isinstance(entry, dict) and "auc" in entry:
                lines.append(f"| {name} | " + " | ".join(f"{entry[k]:.4f}" for k in keys) + " |")
        lines += ["", f"Parity with stored eval row: `{rescore['parity_with_stored_eval_row']}`"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = build_argparser()
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to diagnose (usually best.pth).")
    parser.add_argument("--extra-checkpoint", default="", help="Optional second checkpoint for the drift table only.")
    parser.add_argument("--out-dir", required=True, help="Where to write the diagnostics (never the run dir).")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--draws", type=int, default=3, help="Independent BN re-estimation draws.")
    parser.add_argument("--batches", type=int, default=50, help="Training batches per re-estimation draw.")
    parser.add_argument("--skip-rescore", action="store_true", help="Only write the drift tables (check a).")
    args = parser.parse_args(argv)

    cfg = _load_cfg(args)
    set_seed(cfg.train.seed)
    device = get_device()
    enable_perf_flags(device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_model(cfg, args.checkpoint, device)
    pretrained = timm.create_model(
        cfg.model.vit_backbone, pretrained=True, num_classes=0, global_pool="avg", in_chans=cfg.model.in_channels
    ).eval()
    report: dict[str, object] = {
        "checkpoint": args.checkpoint, "protocol": cfg.data.protocol, "split": args.split,
        "drift": _drift_section(model, pretrained, args, out_dir, device, cfg),
    }
    if not args.skip_rescore:
        report["rescore"] = _rescore_section(model, pretrained, args, cfg, device)
    (out_dir / "rescore.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    print(f"[diagnose_bn] wrote {out_dir}")
    parity = (report.get("rescore") or {}).get("parity_with_stored_eval_row", {})
    return 1 if parity.get("checked") and not parity.get("passed") else 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
