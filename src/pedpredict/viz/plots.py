"""Quantitative thesis figures (Prompt 6.1) — ports OLD ``scripts/plot_results.py``.

The same four figure families as the OLD script — (1) training curves, (2) PR + threshold sweep,
(3) ablation bars, (4) temporal attention — re-pointed at the NEW run-dir artifacts:

* **train curves** read ``RunDir.train_log_path`` (``train_log.csv`` / ``TRAIN_LOG_COLUMNS``, 4.5) — the
  OLD TitleCase columns (``Epoch`` / ``Avg Train Loss`` / ``Actions F1`` …) are now snake_case
  (``epoch`` / ``train_loss`` / ``actions_f1`` …). The new schema always emits per-task F1 + ``macro_f1``,
  so OLD's accuracy-fallback branch is dropped (unreachable, documented in MIGRATION.md 6.1).
* **PR / threshold** read ``predictions.npz`` (``{task}_true/_prob_1`` …) written by
  ``eval.save_predictions_npz`` (5.1) — replaces OLD's per-sample predictions CSV.
* **ablation bars** read one ``eval_log.csv`` row per model (``EVAL_LOG_COLUMNS`` — has ``model_type``,
  ``{task}_f1``, ``{task}_auc``). OLD's fragile ``_parse_summary_table`` header-sniffing is deleted.
* **temporal attention** read ``temporal_weights.npz`` (``temporal_weights`` + ``{task}_true``) written by
  ``eval.save_temporal_weights_npz`` (5.1) — structurally unchanged from OLD.

Design: every figure is a PURE ``data-in -> Figure-out`` function (no I/O), so figures are testable and
regenerable; ``save_figure`` owns the path + ``savefig`` + ``close``. Paths flow from the typed ``RunDir``
handle (``utils.logging``) / ``PathsCfg`` — no hardcoded ``plots/`` literal in module code (B11).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: no display needed to render PNGs

import csv  # noqa: E402  (stdlib; after backend pin to keep matplotlib first)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    f1_score,
    precision_recall_curve,
)

from pedpredict.utils.logging import (  # noqa: E402
    EVAL_LOG_FILENAME,
    RunDir,
    read_index,
)

#: The three binary tasks, in contract order. Mirrors ``losses.multitask.TASKS`` / the output contract
#: (B4), declared locally so this CSV/NPZ-driven layer needn't import the loss/metrics modules just for a
#: string tuple (the quantitative figures read serialized columns, never the model).
TASKS: tuple[str, ...] = ("actions", "looks", "crosses")

__all__ = [
    "DPI",
    "HEAD_COLORS",
    "ABLATION_COLORS",
    "load_train_log",
    "load_predictions_npz",
    "load_eval_row",
    "load_ablation_metrics",
    "figure_loss_curves",
    "figure_per_head_f1_curves",
    "figure_pr_curves",
    "figure_f1_threshold",
    "figure_ablation_bars",
    "figure_temporal_attention",
    "save_figure",
    "generate_run_figures",
    "generate_ablation_figures",
]

DPI: int = 150

# Consistent palette (OLD plot_results.py:70-89).
_COLOR_TRAIN = "#1f77b4"
_COLOR_VAL = "#d62728"
_COLOR_MACRO = "#9467bd"
HEAD_COLORS: dict[str, str] = {"actions": "#1f77b4", "looks": "#ff7f0e", "crosses": "#2ca02c"}
ABLATION_COLORS: dict[str, str] = {
    "full": "#1f77b4",
    "motion_only": "#ff7f0e",
    "visual_only": "#2ca02c",
    "vanilla_concat": "#d62728",
}

#: Canonical PNG filenames per family (one place so writers/tests agree).
LOSS_CURVES_PNG = "loss_curves.png"
PER_HEAD_F1_PNG = "per_head_f1_curves.png"
PR_CURVES_PNG = "pr_curves.png"
F1_THRESHOLD_PNG = "f1_threshold.png"
ABLATION_F1_PNG = "ablation_f1.png"
ABLATION_AUC_PNG = "ablation_auc.png"
TEMPORAL_ATTENTION_PNG = "temporal_attention.png"

#: Artifact filenames produced by eval (5.1) inside ``RunDir.plots_dir``.
PREDICTIONS_NPZ = "predictions.npz"
TEMPORAL_WEIGHTS_NPZ = "temporal_weights.npz"


# --------------------------------------------------------------------------- loaders (schema adapters)


def load_train_log(path: str | Path) -> dict[str, np.ndarray]:
    """Read a ``train_log.csv`` (4.5 ``TRAIN_LOG_COLUMNS``) into ``{column: float ndarray}``.

    Replaces OLD ``_load_training_log`` (TitleCase headers). Blank cells are skipped so a column with
    sparse values still loads; non-numeric columns raise (the train log is all-numeric).
    """
    data: dict[str, list[float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            for key, value in row.items():
                if value is None or value == "":
                    continue
                data.setdefault(key, []).append(float(value))
    return {k: np.asarray(v, dtype=float) for k, v in data.items()}


def load_predictions_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Read ``predictions.npz`` (5.1) into a plain dict of arrays (``{task}_true/_prob_0/_prob_1/_pred``)."""
    with np.load(path) as npz:
        return {key: npz[key] for key in npz.files}


def load_eval_row(path: str | Path) -> dict[str, float]:
    """Read ONE ``eval_log.csv`` row (``EVAL_LOG_COLUMNS``) as a flat metric dict.

    Picks the LAST row (most recent evaluation; the append-only ``CsvLogger`` may hold several). Numeric
    cells are coerced to ``float``; context strings (``model_type`` / ``checkpoint`` / ``split``) pass
    through as ``str`` so callers keep them.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"eval log has no data rows: {path}")
    out: dict[str, float] = {}
    for key, value in rows[-1].items():
        if value is None or value == "":
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value  # type: ignore[assignment]  # context column (model_type, ...)
    return out


def load_ablation_metrics(
    eval_logs: Mapping[str, str | Path],
) -> dict[str, dict[str, float]]:
    """Map ``{model_type: eval_log.csv}`` -> ``{model_type: flat metric dict}`` (one row each).

    Replaces OLD ``_parse_summary_table`` text-scraping: ``{task}_f1`` / ``{task}_auc`` are plain columns.
    """
    return {name: load_eval_row(path) for name, path in eval_logs.items()}


# --------------------------------------------------------------------------- family 1: training curves


def figure_loss_curves(train_log: Mapping[str, np.ndarray]) -> Figure:
    """Train vs val loss over epochs with a best-(min-val)-epoch marker (OLD ``plot_loss_curves``)."""
    epochs = train_log["epoch"]
    train_loss = train_log["train_loss"]
    val_loss = train_log["val_loss"]
    best_epoch = epochs[int(np.argmin(val_loss))]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, train_loss, color=_COLOR_TRAIN, label="Train loss")
    ax.plot(epochs, val_loss, color=_COLOR_VAL, label="Val loss")
    ax.axvline(best_epoch, ls="--", color="gray", lw=0.9, label=f"Best val epoch {int(best_epoch)}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def figure_per_head_f1_curves(
    train_log: Mapping[str, np.ndarray], tasks: Sequence[str] = TASKS
) -> Figure:
    """Per-task validation F1 (+ macro-F1) over epochs (OLD ``plot_per_head_f1_curves``).

    The new train log always carries ``{task}_f1`` + ``macro_f1`` (3.2 ``METRIC_COLUMNS``), so OLD's
    accuracy-fallback branch is intentionally dropped (MIGRATION.md 6.1).
    """
    epochs = train_log["epoch"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for task in tasks:
        ax.plot(epochs, train_log[f"{task}_f1"], color=HEAD_COLORS[task], label=f"{task.capitalize()} F1")
    if "macro_f1" in train_log:
        ax.plot(epochs, train_log["macro_f1"], color=_COLOR_MACRO, ls="--", label="Macro F1")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- family 2: PR + threshold


def _precision_at(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    return tp / (tp + fp + 1e-12)


def _recall_at(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp / (tp + fn + 1e-12)


def figure_pr_curves(predictions: Mapping[str, np.ndarray], tasks: Sequence[str] = TASKS) -> Figure:
    """Per-task precision-recall curve with AP, t=0.5 dot, and F1-optimal star (OLD ``plot_pr_curves``)."""
    fig, axes = plt.subplots(1, len(tasks), figsize=(14, 4))
    for ax, task in zip(np.atleast_1d(axes), tasks, strict=True):
        y_true = predictions[f"{task}_true"].astype(int)
        y_prob = predictions[f"{task}_prob_1"]
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, color=HEAD_COLORS[task], lw=1.5)
        ax.annotate(f"AP = {ap:.3f}", xy=(0.02, 0.02), xycoords="axes fraction", fontsize=9)

        pred_05 = (y_prob >= 0.5).astype(int)
        ax.plot(_recall_at(y_true, pred_05), _precision_at(y_true, pred_05), "o", ms=7,
                color="black", label="t=0.5")
        f1s = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
        best = int(np.argmax(f1s))
        ax.plot(recall[best], precision[best], "*", ms=11, color="red",
                label=f"best F1 t={thresholds[best]:.2f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.legend(fontsize=7, frameon=False, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(task.capitalize(), fontsize=10)
    fig.tight_layout()
    return fig


def figure_f1_threshold(
    predictions: Mapping[str, np.ndarray],
    tasks: Sequence[str] = TASKS,
    *,
    lo: float = 0.05,
    hi: float = 0.96,
    step: float = 0.01,
) -> Figure:
    """Per-task F1 vs decision threshold, marking the optimum (OLD ``plot_f1_threshold``).

    The fine ``0.01`` sweep is for a smooth curve; it is deliberately finer than the coarse
    ``EvalCfg`` grid that ``metrics.optimal_threshold_metrics`` uses for *logged* numbers (different
    purpose, documented so they don't look like accidental divergence).
    """
    thresholds = np.arange(lo, hi, step)
    fig, axes = plt.subplots(1, len(tasks), figsize=(14, 4))
    for ax, task in zip(np.atleast_1d(axes), tasks, strict=True):
        y_true = predictions[f"{task}_true"].astype(int)
        y_prob = predictions[f"{task}_prob_1"]
        f1s = np.array([f1_score(y_true, (y_prob >= t).astype(int), zero_division=0) for t in thresholds])
        best = int(np.argmax(f1s))
        best_t, best_f1 = float(thresholds[best]), float(f1s[best])
        ax.plot(thresholds, f1s, color=HEAD_COLORS[task], lw=1.5)
        ax.axvline(best_t, ls="--", color="gray", lw=0.9)
        ax.annotate(f"t={best_t:.2f}\nF1={best_f1:.3f}", xy=(best_t, best_f1), fontsize=8,
                    xytext=(best_t + 0.08, best_f1 - 0.08),
                    arrowprops=dict(arrowstyle="->", color="gray"))
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1")
        ax.set_xlim(lo - 0.02, hi - 0.01)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.set_title(task.capitalize(), fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- family 3: ablation bars


def figure_ablation_bars(
    ablation: Mapping[str, Mapping[str, float]],
    metric: str,
    ylabel: str,
    tasks: Sequence[str] = TASKS,
) -> Figure:
    """Grouped per-task bars across model types, with a full-model macro reference line.

    ``metric`` is a ``METRIC_COLUMNS`` suffix (``"f1"`` / ``"auc"``); each bar reads ``{task}_{metric}``
    from the model's flat eval-log dict (OLD ``_ablation_bar_chart`` + ``plot_ablation_{f1,auc}``).
    """
    model_names = list(ablation.keys())
    n_bars = len(model_names)
    bar_w = 0.8 / max(n_bars, 1)
    x = np.arange(len(tasks))

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for i, name in enumerate(model_names):
        vals = [float(ablation[name].get(f"{t}_{metric}", 0.0)) for t in tasks]
        ax.bar(x + i * bar_w, vals, bar_w, label=name.replace("_", " "),
               color=ABLATION_COLORS.get(name, f"C{i}"))
    if "full" in ablation:
        macro = float(np.mean([float(ablation["full"].get(f"{t}_{metric}", 0.0)) for t in tasks]))
        ax.axhline(macro, ls="--", lw=0.9, color="gray", label=f"Full macro {ylabel} ({macro:.2f})")
    ax.set_xticks(x + bar_w * (n_bars - 1) / 2)
    ax.set_xticklabels([t.capitalize() for t in tasks])
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- family 4: temporal attention


def figure_temporal_attention(
    temporal_weights: np.ndarray,
    labels: Mapping[str, np.ndarray],
    tasks: Sequence[str] = TASKS,
) -> Figure:
    """Mean +/- std per-frame softmax weight, split by class, per task (OLD ``plot_temporal_attention``).

    A task's subplot is hidden when its ``{task}_true`` labels are absent (temporal weights are
    full-model-only and the eval writer only stores labels it has).
    """
    frames = np.arange(temporal_weights.shape[1])
    fig, axes = plt.subplots(1, len(tasks), figsize=(15, 4))
    for ax, task in zip(np.atleast_1d(axes), tasks, strict=True):
        label_key = f"{task}_true"
        if label_key not in labels:
            ax.set_visible(False)
            continue
        y_true = labels[label_key].astype(int)
        color = HEAD_COLORS.get(task, "gray")
        for cls_val, ls in ((0, "-"), (1, "--")):
            mask = y_true == cls_val
            if mask.sum() == 0:
                continue
            sub = temporal_weights[mask]
            mean, std = sub.mean(axis=0), sub.std(axis=0)
            ax.plot(frames, mean, ls=ls, color=color, lw=1.5, label=f"{task.capitalize()}={cls_val}")
            ax.fill_between(frames, mean - std, mean + std, color=color, alpha=0.12)
        ax.set_xlabel("Frame index")
        ax.set_ylabel("Mean softmax weight")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(task.capitalize(), fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- save wrapper + drivers


def save_figure(fig: Figure, path: str | Path, *, dpi: int = DPI) -> Path:
    """Write ``fig`` to ``path`` (parents created), close it, and return the path."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def generate_run_figures(run_dir: RunDir, *, which: Sequence[str] | None = None) -> list[Path]:
    """Regenerate the per-run figures from a run's artifacts into ``run_dir.plots_dir``.

    Families run only when their input artifact exists (preserves OLD's "only the supplied phases"
    behavior). ``which`` optionally restricts to a subset of
    ``{"loss", "f1", "pr", "threshold", "temporal"}``.
    """
    want = set(which) if which is not None else None

    def enabled(name: str) -> bool:
        return want is None or name in want

    written: list[Path] = []
    plots = run_dir.plots_dir
    if run_dir.train_log_path.exists() and (enabled("loss") or enabled("f1")):
        log = load_train_log(run_dir.train_log_path)
        if enabled("loss"):
            written.append(save_figure(figure_loss_curves(log), plots / LOSS_CURVES_PNG))
        if enabled("f1"):
            written.append(save_figure(figure_per_head_f1_curves(log), plots / PER_HEAD_F1_PNG))

    preds_path = plots / PREDICTIONS_NPZ
    if preds_path.exists() and (enabled("pr") or enabled("threshold")):
        preds = load_predictions_npz(preds_path)
        if enabled("pr"):
            written.append(save_figure(figure_pr_curves(preds), plots / PR_CURVES_PNG))
        if enabled("threshold"):
            written.append(save_figure(figure_f1_threshold(preds), plots / F1_THRESHOLD_PNG))

    tw_path = plots / TEMPORAL_WEIGHTS_NPZ
    if tw_path.exists() and enabled("temporal"):
        data = load_predictions_npz(tw_path)  # same NPZ-to-dict adapter
        weights = data.pop("temporal_weights")
        written.append(save_figure(figure_temporal_attention(weights, data), plots / TEMPORAL_ATTENTION_PNG))
    return written


def generate_ablation_figures(
    eval_logs: Mapping[str, str | Path], out_dir: str | Path
) -> list[Path]:
    """Write ``ablation_f1.png`` + ``ablation_auc.png`` from per-model ``eval_log.csv`` into ``out_dir``."""
    ablation = load_ablation_metrics(eval_logs)
    out = Path(out_dir)
    return [
        save_figure(figure_ablation_bars(ablation, "f1", "F1"), out / ABLATION_F1_PNG),
        save_figure(figure_ablation_bars(ablation, "auc", "AUC"), out / ABLATION_AUC_PNG),
    ]


def ablation_logs_from_index(runs_root: str | Path) -> dict[str, Path]:
    """Discover ``{model_type: eval_log.csv}`` from the cross-run ``index.csv`` (latest eval per type).

    Convenience for ``--ablation-from-index``: scans ``kind == 'eval'`` rows (index rows are appended in
    run order, so the last per ``model_type`` wins) and resolves each ``run_dir/eval_log.csv``.
    """
    latest: dict[str, Path] = {}
    for row in read_index(runs_root):
        if row.get("kind") != "eval":
            continue
        run_dir = row.get("run_dir")
        if not run_dir:
            continue
        eval_log = Path(run_dir) / EVAL_LOG_FILENAME
        if eval_log.exists():
            latest[row["model_type"]] = eval_log
    return latest
