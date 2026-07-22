"""One-command model-arm runner: train on BOTH protocols, evaluate each on the full 2×2 matrix.

The cross-protocol matrix axis is train-protocol × eval-protocol (setup.md §9), so a complete model
arm is TWO training runs — one on the anchored benchmark LMDBs, one on the streaming LMDBs — each
evaluated on both protocols. That is ten near-identical command lines per arm:

    for train_protocol in {anchored, streaming}:
        1. train.py    --set data.protocol=<train_protocol>       (+ all your --set overrides)
        2. evaluate.py --split val  data.protocol=anchored        (tune + store anchored thresholds)
        3. evaluate.py --split test data.protocol=anchored        (report at anchored thresholds)
        4. evaluate.py --split val  data.protocol=streaming       (tune + store streaming thresholds)
        5. evaluate.py --split test data.protocol=streaming       (report at streaming thresholds)

Typing those by hand is meticulous and error-prone (one mistyped ``--set`` and the arm is wrong).
This wrapper takes **exactly the train.py argument surface** — ``--config-dir`` plus repeatable
``--set section.field=value`` — and drives all ten steps. ``data.protocol`` is OWNED by the runner:
each training appends its own ``--set data.protocol=<train_protocol>`` last (later ``--set`` tokens
win), and each eval cell appends its protocol the same way — so a ``data.protocol`` in your ``--set``
is ignored (a warning says so). Eval inherits the model architecture from each checkpoint's
``resolved_config.yaml`` (evaluate.py's ``merge_eval_config``), so the training ``--set`` stays the
single source of truth.

Everything is recorded per the normal run-dir convention: each training run gets its own
``outputs/runs/{run_id}`` (run ids are tagged ``..._trainanchored`` / ``..._trainstreaming``, plus
your ``--tag`` if given) holding ``resolved_config.yaml`` / ``train_log.csv`` / ``thresholds.json`` /
``checkpoints/``; all four eval cells append rows to that run's ``eval_log.csv``; the cross-run
``index.csv`` gets one row per training run.

Order matters and is fixed: within each protocol the ``val`` pass runs before ``test`` because val
sweeps and stores the per-task F1 thresholds that the test pass loads and applies (M2). The runner is
fail-fast: the first non-zero step aborts everything after it (a failed train means no checkpoint to
evaluate; a failed val means test would apply stale/missing thresholds).

Usage (identical --set surface to train.py; e.g. the frozen-ViT pose_full arm):
    python scripts/run_arm.py \
        --set model.vit_backbone=tiny_vit_5m_224 --set "train.active_tasks=[crosses]" \
        --set eval.model_type=pose_full \
        --set model.vit_pretrained=true --set model.freeze_vit_backbone=true \
        --set train.use_weighted_sampler=false --set augment.runtime=true \
        --set train.num_epochs=30 --set train.warmup_epochs=4 \
        --set pose.enabled=true --set model.motion_norm=none \
        --set data.motion_dim=58 --set model.motion_dim=58

Eval passthrough flags (forwarded to every evaluate.py call):
    --save-predictions --save-temporal-weights --benchmark --no-config-inherit

Other flags:
    --tag <name>                       Run-id suffix for both training runs (each also gets its
                                       train-protocol suffix), e.g. --tag posev2 ->
                                       ..._pose_full_posev2_trainanchored.
    --protocols anchored streaming     Restrict which train arms run (default: both). Useful to
                                       rerun one half after a failure.
    --skip-train --checkpoint <path>   Evaluate an existing checkpoint on the 4-cell eval matrix
                                       only (no training; --protocols is ignored).
    --dry-run                          Print every command that WOULD run, then exit.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from pedpredict.config import build_argparser, load_config
from pedpredict.utils.logging import CONFIG_SNAPSHOT_FILENAME

#: Both training arms of the cross-protocol matrix, in canonical order.
_TRAIN_PROTOCOLS: tuple[str, ...] = ("anchored", "streaming")

#: Fixed per-checkpoint evaluation matrix. val-before-test per protocol (val tunes the thresholds
#: test applies, M2); anchored before streaming to match the canonical matrix order.
_EVAL_MATRIX: tuple[tuple[str, str], ...] = (
    ("val", "anchored"),
    ("test", "anchored"),
    ("val", "streaming"),
    ("test", "streaming"),
)

_SCRIPTS_DIR = Path(__file__).resolve().parent
_TRAIN_PY = _SCRIPTS_DIR / "train.py"
_EVALUATE_PY = _SCRIPTS_DIR / "evaluate.py"

#: The line train.py prints on completion: "... run dir: <path>" (both single-phase and schedule).
_RUN_DIR_MARKER = "run dir:"


def _overrides_to_set_args(overrides: Sequence[str]) -> list[str]:
    """Re-materialize a parsed ``--set`` override list into CLI ``--set k=v`` tokens.

    ``build_argparser`` collapses each ``--set A=B`` into one ``"A=B"`` string in ``args.overrides``
    (and ``--set A B`` normalizes there too), so one ``--set`` token per entry round-trips exactly.
    """
    args: list[str] = []
    for token in overrides:
        args += ["--set", token]
    return args


def _base_argv(config_dir: str, overrides: Sequence[str]) -> list[str]:
    """The ``--config-dir ... --set ...`` tail shared by every train/eval subprocess."""
    return ["--config-dir", config_dir, *_overrides_to_set_args(overrides)]


def _print_step(label: str, cmd: Sequence[str]) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n{'=' * 88}\n[run_arm] {label}\n  $ {printable}\n{'=' * 88}", flush=True)


def _run(cmd: Sequence[str], *, label: str, dry_run: bool) -> int:
    """Print + run one subprocess, streaming its output live. Returns the child exit code (0 if dry)."""
    _print_step(label, cmd)
    if dry_run:
        return 0
    return subprocess.run(list(cmd), check=False).returncode


def _snapshot_run_ids(runs_dir: Path) -> set[str]:
    """Names of the run dirs currently under ``runs_dir`` (empty set if it does not exist yet)."""
    if not runs_dir.exists():
        return set()
    return {p.name for p in runs_dir.iterdir() if p.is_dir()}


def _train_and_capture_run_dir(
    config_dir: str,
    overrides: Sequence[str],
    runs_dir: Path,
    *,
    protocol: str,
    tag: str,
    dry_run: bool,
) -> Path | None:
    """Run one train.py arm on ``protocol``; return the run dir it produced.

    The runner OWNS ``data.protocol`` for training: ``--set data.protocol=<protocol>`` is appended
    after the user's overrides so it always wins (parse_overrides is last-wins). The run is tagged
    ``[{tag}_]train{protocol}`` so the two arms are distinguishable in run ids and index.csv.

    Primary run-dir discovery is train.py's stdout ``run dir: <path>`` line (print() -> stdout;
    tqdm -> stderr, so stdout stays clean). A snapshot diff of ``runs_dir`` before/after is the
    fallback if stdout parsing yields nothing (e.g. a future train.py wording change).
    """
    run_tag = f"{tag}_train{protocol}" if tag else f"train{protocol}"
    cmd = [
        sys.executable,
        str(_TRAIN_PY),
        *_base_argv(config_dir, overrides),
        "--set",
        f"data.protocol={protocol}",
        "--tag",
        run_tag,
    ]
    _print_step(f"TRAIN  protocol={protocol}", cmd)
    if dry_run:
        return runs_dir / f"<run_id_{run_tag}>"

    before = _snapshot_run_ids(runs_dir)
    # Tee stdout so the user still sees train.py's completion summary while we scan it for the run dir.
    proc = subprocess.run(list(cmd), check=False, capture_output=True, text=True)
    sys.stdout.write(proc.stdout)
    sys.stderr.write(proc.stderr)
    sys.stdout.flush()
    if proc.returncode != 0:
        return None

    run_dir = _parse_run_dir(proc.stdout)
    if run_dir is None or not run_dir.exists():
        run_dir = _newest_new_run_dir(runs_dir, before)
    return run_dir


def _parse_run_dir(stdout: str) -> Path | None:
    """Extract the path after the last ``run dir:`` marker in train.py's stdout, or None."""
    hit: str | None = None
    for line in stdout.splitlines():
        idx = line.find(_RUN_DIR_MARKER)
        if idx != -1:
            hit = line[idx + len(_RUN_DIR_MARKER):].strip()
    if not hit:
        return None
    path = Path(hit)
    return path if path.exists() else None


def _newest_new_run_dir(runs_dir: Path, before: set[str]) -> Path | None:
    """The most-recently-modified run dir that appeared since ``before`` (fallback discovery)."""
    new = [p for p in runs_dir.iterdir() if p.is_dir() and p.name not in before] if runs_dir.exists() else []
    if not new:
        return None
    return max(new, key=lambda p: p.stat().st_mtime)


def _resolve_checkpoint(run_dir: Path) -> Path:
    """The best-checkpoint path inside a run dir (``<run_dir>/checkpoints/best.pth``)."""
    return run_dir / "checkpoints" / "best.pth"


def _eval_command(
    config_dir: str,
    overrides: Sequence[str],
    checkpoint: Path,
    split: str,
    protocol: str,
    passthrough: Sequence[str],
) -> list[str]:
    """Build one evaluate.py argv: training --set first, then this cell's protocol (wins), + passthrough.

    The training overrides are forwarded so runtime fields (paths, batch size, seed) match the arm; the
    trailing ``--set data.protocol=<protocol>`` selects this cell's eval LMDBs (later ``--set`` wins).
    Model architecture is inherited from the checkpoint config by evaluate.py itself.
    """
    return [
        sys.executable,
        str(_EVALUATE_PY),
        *_base_argv(config_dir, overrides),
        "--set",
        f"data.protocol={protocol}",
        "--split",
        split,
        "--checkpoint",
        str(checkpoint),
        *passthrough,
    ]


def _run_eval_matrix(
    config_dir: str,
    overrides: Sequence[str],
    checkpoint: Path,
    passthrough: Sequence[str],
    *,
    arm_label: str,
    dry_run: bool,
) -> int:
    """Run the fixed 4-cell eval matrix against one checkpoint. Returns the first non-zero exit (or 0)."""
    for split, protocol in _EVAL_MATRIX:
        cmd = _eval_command(config_dir, overrides, checkpoint, split, protocol, passthrough)
        rc = _run(cmd, label=f"EVAL  {arm_label}  split={split}  eval_protocol={protocol}", dry_run=dry_run)
        if rc != 0:
            print(
                f"[run_arm] eval step FAILED ({arm_label}, split={split}, eval_protocol={protocol}, "
                f"exit={rc}); aborting.",
                file=sys.stderr,
            )
            return rc
    return 0


def _eval_passthrough(args) -> list[str]:
    """Collect the evaluate.py-only flags the user opted into, in evaluate.py's own flag spelling."""
    flags: list[str] = []
    if args.save_predictions:
        flags.append("--save-predictions")
    if args.save_temporal_weights:
        flags.append("--save-temporal-weights")
    if args.benchmark:
        flags.append("--benchmark")
    if args.no_config_inherit:
        flags.append("--no-config-inherit")
    return flags


def main(argv=None) -> int:
    parser = build_argparser()      # same --config-dir / repeatable --set surface as train.py
    parser.add_argument(
        "--tag", default="",
        help="Run-id suffix for both training runs (each also gets a train{protocol} suffix).",
    )
    parser.add_argument(
        "--protocols", nargs="+", choices=list(_TRAIN_PROTOCOLS), default=list(_TRAIN_PROTOCOLS),
        metavar="PROTOCOL",
        help="Which train arms to run (default: both, anchored first). E.g. --protocols streaming "
             "to rerun only the streaming-trained half.",
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip training; run the 4-cell eval matrix on an existing checkpoint "
             "(requires --checkpoint; --protocols is ignored).",
    )
    parser.add_argument(
        "--checkpoint", default=None,
        help="With --skip-train: the best.pth to evaluate. Ignored when training runs (each freshly "
             "trained checkpoint is used).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print every command that would run, then exit without executing anything.",
    )
    # evaluate.py passthrough flags (forwarded to all eval calls).
    parser.add_argument("--save-predictions", action="store_true", help="evaluate.py --save-predictions.")
    parser.add_argument(
        "--save-temporal-weights", action="store_true", help="evaluate.py --save-temporal-weights.",
    )
    parser.add_argument("--benchmark", action="store_true", help="evaluate.py --benchmark.")
    parser.add_argument(
        "--no-config-inherit", action="store_true", help="evaluate.py --no-config-inherit.",
    )
    args = parser.parse_args(argv)

    if any(o.lstrip("-").startswith("data.protocol") for o in args.overrides):
        print(
            "[run_arm] WARNING: --set data.protocol=... is ignored — the runner owns the protocol "
            "(it trains on BOTH and appends its own override last, which wins).",
            file=sys.stderr,
        )

    # Validate the training --set once, up front — surfaces a bad override before any subprocess runs
    # (and gives us the resolved runs_dir for run-dir discovery). validate=True is the default.
    cfg = load_config(args.config_dir, args.overrides)
    runs_dir = Path(cfg.paths.runs_dir)
    passthrough = _eval_passthrough(args)

    # ------------------------------------------------------------------ eval-only mode
    if args.skip_train:
        if not args.checkpoint:
            parser.error("--skip-train requires --checkpoint <path/to/best.pth>")
        checkpoint = Path(args.checkpoint)
        if not (args.dry_run or checkpoint.exists()):
            parser.error(f"--checkpoint does not exist: {checkpoint}")
        print(f"[run_arm] skipping training; evaluating checkpoint: {checkpoint}")
        return _run_eval_matrix(
            args.config_dir, args.overrides, checkpoint, passthrough,
            arm_label="ckpt", dry_run=args.dry_run,
        )

    # ------------------------------------------------------------------ full cross-protocol matrix
    if cfg.schedule.enabled:
        print(
            "[run_arm] WARNING: schedule.enabled=true — a multi-phase run writes per-phase "
            "checkpoints, not <run_dir>/checkpoints/best.pth. This runner targets single-phase "
            "training; pass --skip-train --checkpoint <phase best.pth> to evaluate a schedule run.",
            file=sys.stderr,
        )

    completed: list[tuple[str, Path, Path]] = []   # (train_protocol, run_dir, checkpoint)
    for train_protocol in args.protocols:
        run_dir = _train_and_capture_run_dir(
            args.config_dir, args.overrides, runs_dir,
            protocol=train_protocol, tag=args.tag, dry_run=args.dry_run,
        )
        if run_dir is None:
            print(
                f"[run_arm] training FAILED (protocol={train_protocol}: non-zero exit or no run dir "
                "found); aborting.",
                file=sys.stderr,
            )
            return 1
        checkpoint = _resolve_checkpoint(run_dir)
        print(f"[run_arm] train[{train_protocol}] produced run dir: {run_dir}")
        if not args.dry_run and not checkpoint.exists():
            print(
                f"[run_arm] ERROR: no checkpoint at {checkpoint} after training (was a best.pth ever "
                f"saved? check that {run_dir / CONFIG_SNAPSHOT_FILENAME} exists). Aborting.",
                file=sys.stderr,
            )
            return 1

        rc = _run_eval_matrix(
            args.config_dir, args.overrides, checkpoint, passthrough,
            arm_label=f"train={train_protocol}", dry_run=args.dry_run,
        )
        if rc != 0:
            return rc
        completed.append((train_protocol, run_dir, checkpoint))

    print(f"\n[run_arm] DONE — {len(completed)} training arm(s), each evaluated on the full 2×2 matrix:")
    for train_protocol, run_dir, checkpoint in completed:
        print(f"  train={train_protocol:<9} run_dir={run_dir}")
        print(f"  {'':<15} ckpt={checkpoint}  (eval rows -> {run_dir / 'eval_log.csv'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
