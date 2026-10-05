"""Dump per-window predictions + onset labels for the timing report. Needs data + checkpoint (research PC).

Thin wrapper over :func:`pedpredict.eval.onset_timing.dump_predictions`. Saves, per window, the binary
head's probability, the onset head's readout probability and hazard logits (when the checkpoint has the
head), the stored ``crosses`` label, the three S1 onset fields and ``track_id``. Writes only ``--out``;
nothing in the run dir changes. The architecture is inherited from the checkpoint's ``resolved_config.yaml``
exactly as ``scripts/evaluate.py`` does it, so the split's scores match the stored eval rows.

Usage (research PC), then copy the .npz files anywhere and run scripts/report_detection_curve.py:
    python scripts/dump_onset_predictions.py --split val \\
        --checkpoint outputs/runs/<run>/checkpoints/best.pth --out outputs/diagnostics/<run>/onset_val.npz
    python scripts/dump_onset_predictions.py --split test \\
        --checkpoint outputs/runs/<run>/checkpoints/best.pth --out outputs/diagnostics/<run>/onset_test.npz
"""

from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

from pedpredict.config import build_argparser, load_config, merge_eval_config, validate_config
from pedpredict.data.feature_cache import verify_model_cache
from pedpredict.eval import onset_timing as ot
from pedpredict.eval.evaluate import _split_chunk_paths, load_eval_weights
from pedpredict.models.registry import build_model
from pedpredict.utils.amp import resolve_amp
from pedpredict.utils.device import enable_perf_flags, get_device
from pedpredict.utils.seed import set_seed


def main(argv=None) -> int:
    parser = build_argparser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--out", required=True, help="Output .npz path.")
    args = parser.parse_args(argv)

    cfg = load_config(args.config_dir, args.overrides)
    run_config = Path(args.checkpoint).resolve().parent.parent / "resolved_config.yaml"
    if not run_config.exists():
        raise FileNotFoundError(f"no resolved_config.yaml for the checkpoint: {run_config}")
    cfg = merge_eval_config(cfg, run_config, args.overrides)
    validate_config(cfg)
    set_seed(cfg.train.seed)

    device = get_device()
    enable_perf_flags(device)
    model = build_model(cfg, cfg.eval.model_type).to(device)
    load_eval_weights(model, args.checkpoint, device=device)
    chunk_paths = _split_chunk_paths(cfg, args.split)
    verify_model_cache(cfg, model, chunk_paths)

    arrays = ot.dump_predictions(
        model,
        ot.dump_loaders(cfg, chunk_paths, device),
        device,
        use_amp=resolve_amp(cfg.train.use_amp, device),
    )
    meta = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "protocol": cfg.data.protocol,
        "model_type": cfg.eval.model_type,
        "onset_head": cfg.model.onset_head,
        "onset_lookahead": cfg.model.onset_lookahead,
        "onset_bin_width": cfg.model.onset_bin_width,
        "onset_horizon": cfg.model.onset_horizon,
        "onset_report_crosses": cfg.model.onset_report_crosses,
    }
    path = ot.save_dump(args.out, arrays, meta)
    print(f"[dump_onset] {len(arrays['crosses'])} windows ({', '.join(sorted(arrays))}) -> {path}")
    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
