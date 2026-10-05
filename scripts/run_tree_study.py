#!/usr/bin/env python
"""The pre-registered tree study (``outputs/diagnostics/tree_study/PREREG.md``) — runs on any CPU machine.

Three subcommands, in order:

    # 1. gates: pinned counts, row alignment with every v4 dump, probe reproduction, and (optional) a deep
    #    checkpoint re-scored on the exported inputs must reproduce its own stored dump
    python scripts/run_tree_study.py verify --exports outputs/features/pose58 \\
        --checkpoint outputs/runs/<c4_r2s_s42 run>/checkpoints/best.pth
    # 2. fit arms (restartable; a finished seed is skipped). --tags picks parts: primary horizon cues curve anchored
    python scripts/run_tree_study.py run --exports outputs/features/pose58 --tags primary
    # 3. verdicts + tables -> outputs/diagnostics/tree_study/{results.json,report.md}
    python scripts/run_tree_study.py report

Inputs come from ``scripts/export_window_features.py`` (research PC). Nothing here needs a GPU or an LMDB.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from pedpredict.eval import tree_study as ts
from pedpredict.eval.onset_timing import load_dump
from pedpredict.eval.tree_study_report import write_report
from pedpredict.eval.window_features import check_against_dump

_OUT = Path("outputs/diagnostics/tree_study")
_DIAG = Path("outputs/diagnostics")
_PIXEL_FREE = ("eval.model_type=pose_kinematics", "pose.enabled=true", "model.motion_norm=none",
               "data.motion_dim=58", "model.motion_dim=58", "data.visual_input=none")


def _alignment(exports: Path) -> dict[str, list[str]]:
    """Every local streaming-test dump (v4 deep arms + the GBM probe) against the exported test split."""
    test, _ = load_dump(exports / ts.EXPORT_FILES["test"])
    dumps = sorted(_DIAG.glob("*_c4_*/onset_test.npz")) + sorted((_DIAG / "probe").glob("gbm_s*.npz"))
    return {str(p): check_against_dump(test, load_dump(p)[0]) for p in dumps}


def _deep_reproduction(checkpoint: Path, exports: Path) -> dict[str, float]:
    """Re-score a v4 checkpoint on the exported test inputs (its run's standardization applied) and compare
    with that run's stored dump: proves the export is exactly what the deep model consumed."""
    import torch

    from pedpredict.config import load_config, merge_eval_config
    from pedpredict.eval.evaluate import load_eval_weights
    from pedpredict.models.registry import build_model, forward_model

    run_dir = checkpoint.resolve().parent.parent
    cfg = merge_eval_config(load_config("configs", list(_PIXEL_FREE)), run_dir / "resolved_config.yaml",
                            list(_PIXEL_FREE))
    model = build_model(cfg, cfg.eval.model_type)
    load_eval_weights(model, checkpoint, device=torch.device("cpu"))
    model.eval()
    X = torch.from_numpy(load_dump(exports / ts.EXPORT_FILES["test"])[0]["X"])
    if cfg.pose.input_mean:
        mean, std = torch.tensor(cfg.pose.input_mean), torch.tensor(cfg.pose.input_std)
        X = ((X - mean) / std).clamp(-cfg.pose.input_clip, cfg.pose.input_clip)
    probs = []
    with torch.inference_mode():
        for xb in torch.split(X, 2048):
            empty = torch.empty(xb.shape[0], xb.shape[1], 0)
            out = forward_model(model, empty, empty, xb)
            probs.append(torch.softmax(out["crosses_frame"].float(), dim=1)[:, 1].numpy())
    p = np.concatenate(probs)
    stored = load_dump(_DIAG / run_dir.name / "onset_test.npz")[0]["p_frame"]
    return {"run": run_dir.name, "max_abs_diff": float(np.abs(p - stored).max()),
            "mean_abs_diff": float(np.abs(p - stored).mean()),
            "rank_corr": float(np.corrcoef(np.argsort(np.argsort(p)), np.argsort(np.argsort(stored)))[0, 1])}


def cmd_verify(args) -> int:
    report: dict[str, object] = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    data = ts.load_exports(args.exports, ["train", "censored", "val", "test"])
    report["label_gate"] = ts.label_gate(data)
    alignment = _alignment(args.exports)
    report["alignment_failures"] = {k: v for k, v in alignment.items() if v}
    report["alignment_checked"] = len(alignment)
    report["probe_reproduction"] = ts.probe_reproduction(data, _DIAG / "probe")
    if args.checkpoint:
        report["deep_reproduction"] = _deep_reproduction(Path(args.checkpoint), args.exports)
    ok = (all(v["ok"] for v in report["label_gate"].values()) and not report["alignment_failures"]
          and len(report["label_gate"]) == len(ts.PINNED_COUNTS)
          and (not args.checkpoint or report["deep_reproduction"]["max_abs_diff"] < 1e-3))
    report["gate"] = "PASS" if ok else "FAIL"
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "verify.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print(f"[verify] {report['gate']} -> {args.out / 'verify.json'}")
    return 0 if ok else 1


def cmd_run(args) -> int:
    arms = [a for a in ts.ARMS if (not args.tags or set(a.tags) & set(args.tags))
            and (not args.arms or a.name in args.arms)]
    needed = {"train", "val", "test", "anc_test", "anc_val"}
    needed |= {s for a in arms for s in a.train}
    needed |= {"anc_train"} if any(a.n_windows or a.protocol == "anchored" for a in arms) else set()
    data = ts.load_exports(args.exports, sorted(needed))
    missing = sorted(needed - set(data) - {"anc_test", "anc_val"})
    if missing:
        print(f"[tree] missing exports: {missing}")
        return 1
    print(f"[tree] {len(arms)} arms: {[a.name for a in arms]}")
    ts.run_arms(arms, data, args.out)
    return 0


def cmd_report(args) -> int:
    json_path, md_path = write_report(args.out, _DIAG)
    print(md_path.read_text(encoding="utf-8"))
    print(f"[report] -> {json_path}, {md_path}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("verify", "run", "report"):
        p = sub.add_parser(name)
        p.add_argument("--out", type=Path, default=_OUT)
        if name != "report":
            p.add_argument("--exports", type=Path, default=Path("outputs/features/pose58"))
    sub.choices["verify"].add_argument("--checkpoint", default="")
    sub.choices["run"].add_argument("--tags", nargs="*", default=[])
    sub.choices["run"].add_argument("--arms", nargs="*", default=[])
    args = parser.parse_args(argv)
    return {"verify": cmd_verify, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
