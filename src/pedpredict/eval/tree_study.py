"""The pre-registered tree-model study: data, arms, fitting and dumps (``outputs/diagnostics/tree_study``).

``PREREG.md`` in that directory fixes every choice below before any number exists; this module is its
executable form and :mod:`pedpredict.eval.tree_study_report` turns the dumps into its verdicts. Inputs are the
exports of ``scripts/export_window_features.py``. Every fit is written as the standard prediction dump
(:func:`pedpredict.eval.onset_timing.save_dump`), so ``scripts/report_detection_curve.py`` and every other
dump reader work on it unchanged.

Selection reads the validation split only. Per arm the class weighting (``none`` | ``balanced``) is chosen
once, on the first seed's best validation AUC (arms marked ``weighting_from`` inherit their parent's choice,
so they differ from it in one thing); per fit the boosting iteration count is the one with the best
validation AUC of the arm's reported score. The test split is read only to score a finished model.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from pedpredict.baselines.tree import (
    SUMMARY_STATS,
    TreeParams,
    cue_groups,
    fit_binary,
    fit_hazard,
    hazard_logits,
    horizon_label,
    person_period,
    staged_binary_scores,
    staged_hazard_readout,
    summary_features,
    truncate,
)
from pedpredict.data.onset_target import OnsetSpec
from pedpredict.data.pie_sequences import ONSET_FIELDS
from pedpredict.eval.onset_timing import hazard_horizon_prob, load_dump, save_dump

__all__ = [
    "EXPORT_FILES", "SEEDS", "CURVE_SEEDS", "ArmSpec", "ARMS", "SplitData", "Fit", "load_split", "load_exports",
    "concat", "fit_arm_seed", "score_split", "run_arm", "run_arms", "score_key", "label_gate", "PINNED_COUNTS",
    "probe_reproduction",
]

#: Export file per split (``outputs/features/pose58/`` by default).
EXPORT_FILES: dict[str, str] = {
    "train": "train.npz", "censored": "train_censored.npz", "val": "val.npz", "test": "test.npz",
    "anc_train": "train_benchmark.npz", "anc_val": "val_benchmark.npz", "anc_test": "test_benchmark.npz",
}
SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)          # primary arms, and every binary arm (cheap)
CURVE_SEEDS: tuple[int, ...] = (42, 43, 44)            # secondary hazard arms; learning-curve draws
#: The v2 Dataset Statistics table (CLAUDE.md) — N and crosses=1 per streaming split; censored = 7,470.
PINNED_COUNTS: dict[str, tuple[int, int | None]] = {
    "train": (88214, 2530), "val": (20490, 569), "test": (69875, 2140), "censored": (7470, None),
}
_LABEL_KEYS: tuple[str, ...] = ("crosses", "track_id", *ONSET_FIELDS, "tte")
#: Which dumps a fit writes: (split, file). Missing splits are skipped.
_DUMPS: tuple[tuple[str, str], ...] = (("test", "onset_test.npz"), ("val", "onset_val.npz"),
                                       ("anc_test", "anchored_test.npz"), ("anc_val", "anchored_val.npz"))


@dataclass(frozen=True)
class ArmSpec:
    """One arm of the study. Defaults are the binary twin of the deep R2 arm."""

    name: str
    objective: str = "binary"              # "binary" | "hazard"
    train: tuple[str, ...] = ("train",)    # export splits concatenated for training (streaming protocol)
    horizon: int = 32                      # reported horizon H (frames)
    label_mode: str = "stored"             # binary: "stored" crosses | horizon_label "drop" | "zero"
    lookahead: int = 96                    # hazard L
    bin_width: int = 4                     # hazard w
    cues: str = "all"                      # channel group (baselines.tree.cue_groups)
    protocol: str = "streaming"            # "anchored": train anc_train, select on anc_val
    fraction: float = 1.0                  # stratified share of training pedestrians kept (learning curve)
    n_windows: int = 0                     # -1: random subsample of len(anc_train) windows (matched size)
    weighting_from: str = ""               # inherit this arm's chosen weighting instead of selecting
    seeds: tuple[int, ...] = SEEDS
    tags: tuple[str, ...] = field(default=())

    @property
    def onset_spec(self) -> OnsetSpec:
        return OnsetSpec(lookahead=self.lookahead, bin_width=self.bin_width, horizon=self.horizon)


def _arms() -> tuple[ArmSpec, ...]:
    """Every pre-registered arm, parents before the arms that inherit their weighting."""
    cens = ("train", "censored")
    arms = [
        ArmSpec("B", tags=("primary",)),
        ArmSpec("Hz", objective="hazard", tags=("primary",)),
        ArmSpec("HzC", objective="hazard", train=cens, tags=("primary", "horizon")),
    ]
    for h, (look, width) in {32: (96, 4), 64: (128, 8), 160: (224, 16)}.items():
        arms += [ArmSpec(f"Bd{h}", train=cens, horizon=h, label_mode="drop", tags=("horizon",)),
                 ArmSpec(f"Bz{h}", train=cens, horizon=h, label_mode="zero", tags=("horizon",))]
        if h != 32:   # H = 32 is HzC: same geometry, same data
            arms.append(ArmSpec(f"Hz{h}", objective="hazard", train=cens, horizon=h, lookahead=look,
                                bin_width=width, seeds=CURVE_SEEDS, tags=("horizon",)))
    arms += [ArmSpec(f"B_{g}", cues=g, weighting_from="B", tags=("cues",))
             for g in ("no_ego", "box_ego", "box", "pose", "ego")]
    arms.append(ArmSpec("Hz_no_ego", objective="hazard", cues="no_ego", weighting_from="Hz", seeds=CURVE_SEEDS,
                        tags=("cues",)))
    for f in (0.125, 0.25, 0.5):
        arms += [ArmSpec(f"B_f{f:g}", fraction=f, weighting_from="B", seeds=CURVE_SEEDS, tags=("curve",)),
                 ArmSpec(f"Hz_f{f:g}", objective="hazard", fraction=f, weighting_from="Hz", seeds=CURVE_SEEDS,
                         tags=("curve",))]
    arms += [ArmSpec("A", protocol="anchored", train=("anc_train",), tags=("anchored",)),
             ArmSpec("B_matched", n_windows=-1, weighting_from="B", tags=("anchored",))]
    return tuple(arms)


ARMS: tuple[ArmSpec, ...] = _arms()


def score_key(spec: ArmSpec) -> str:
    """The dump key an arm is scored on: the binary head, or the hazard readout at ``spec.horizon``."""
    return "p_frame" if spec.objective == "binary" else "p_readout"


# --------------------------------------------------------------------------- data


@dataclass
class SplitData:
    """One export split reduced to summary features (all channels) + its label arrays."""

    name: str
    features: np.ndarray
    n_channels: int
    labels: dict[str, np.ndarray]

    def __len__(self) -> int:
        return len(self.features)

    def columns(self, cues: str) -> np.ndarray:
        """Feature columns of a channel group (summary block ``s`` of channel ``c`` is column ``s * C + c``)."""
        chans = cue_groups(self.n_channels)[cues]
        return np.asarray([s * self.n_channels + c for s in range(len(SUMMARY_STATS)) for c in chans])

    def take(self, index: np.ndarray) -> SplitData:
        labels = {k: v[index] for k, v in self.labels.items()}
        return SplitData(self.name, self.features[index], self.n_channels, labels)


def load_split(name: str, path: str | Path) -> SplitData:
    arrays, _ = load_dump(path)
    X = arrays.pop("X")
    labels = {k: arrays[k] for k in _LABEL_KEYS if k in arrays}
    return SplitData(name, summary_features(X), X.shape[2], labels)


def load_exports(directory: str | Path, names: Iterable[str] | None = None) -> dict[str, SplitData]:
    """Every export present in ``directory`` (or just ``names``), keyed by split name."""
    directory = Path(directory)
    wanted = list(names) if names is not None else list(EXPORT_FILES)
    return {n: load_split(n, directory / EXPORT_FILES[n]) for n in wanted if (directory / EXPORT_FILES[n]).exists()}


def concat(parts: Sequence[SplitData]) -> SplitData:
    """Row-concatenate splits on the label keys they all carry."""
    keys = set.intersection(*(set(p.labels) for p in parts))
    return SplitData("+".join(p.name for p in parts), np.concatenate([p.features for p in parts]),
                     parts[0].n_channels, {k: np.concatenate([p.labels[k] for p in parts]) for k in sorted(keys)})


def label_gate(data: Mapping[str, SplitData]) -> dict[str, dict[str, int | bool]]:
    """Counts vs the pinned Dataset Statistics, and stored ``crosses`` vs the onset-field rule at H = 32."""
    out = {}
    for name, (n_pin, pos_pin) in PINNED_COUNTS.items():
        if name not in data:
            continue
        lab = data[name].labels
        derived = (lab["onset_offset"] >= 0) & (lab["onset_offset"] < 32)
        out[name] = {"n": len(data[name]), "n_pinned": n_pin, "positives": int(lab["crosses"].sum()),
                     "positives_pinned": -1 if pos_pin is None else pos_pin,
                     "crosses_vs_onset_rule_mismatches": int((derived != (lab["crosses"] > 0)).sum())}
        out[name]["ok"] = n_pin == out[name]["n"] and (pos_pin is None or pos_pin == out[name]["positives"])
    return out


# --------------------------------------------------------------------------- fitting


@dataclass
class Fit:
    model: HistGradientBoostingClassifier
    n_iter: int
    val_auc: float
    weighting: str
    n_rows: int
    seconds: float
    val_auc_curve: list[float]


def _pedestrian_subsample(split: SplitData, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Windows of a stratified random ``fraction`` of pedestrians (crossers and never-crossers separately)."""
    tracks, inverse = np.unique(split.labels["track_id"], return_inverse=True)
    crosser = np.zeros(len(tracks), dtype=bool)
    crosser[inverse[split.labels["onset_offset"] >= 0]] = True
    kept: list[int] = []
    for flag in (True, False):
        pool = np.flatnonzero(crosser == flag)
        kept += list(rng.choice(pool, max(1, round(fraction * len(pool))), replace=False))
    return np.flatnonzero(np.isin(inverse, kept))


def _train_split(spec: ArmSpec, data: Mapping[str, SplitData], seed: int) -> SplitData:
    split = concat([data[s] for s in spec.train])
    rng = np.random.default_rng(seed)
    if spec.fraction < 1.0:
        split = split.take(_pedestrian_subsample(split, spec.fraction, rng))
    if spec.n_windows:
        n = len(data["anc_train"]) if spec.n_windows < 0 else spec.n_windows
        split = split.take(np.sort(rng.choice(len(split), n, replace=False)))
    return split


def _binary_target(spec: ArmSpec, split: SplitData) -> tuple[np.ndarray, np.ndarray]:
    if spec.label_mode == "stored":
        return split.labels["crosses"].astype(np.int64), np.ones(len(split), dtype=bool)
    return horizon_label(split.labels["onset_offset"], split.labels["future_observed"], spec.horizon, spec.label_mode)


def _selection_set(spec: ArmSpec, data: Mapping[str, SplitData]) -> tuple[SplitData, np.ndarray]:
    """Validation windows + their target: the stored label at H = 32, else the knowable label at H."""
    if spec.protocol == "anchored":
        val = data["anc_val"]
        return val, val.labels["crosses"].astype(np.int64)
    val = data["val"]
    if spec.horizon == 32:
        return val, val.labels["crosses"].astype(np.int64)
    label, keep = horizon_label(val.labels["onset_offset"], val.labels["future_observed"], spec.horizon, "drop")
    return val.take(np.flatnonzero(keep)), label[keep]


def fit_arm_seed(spec: ArmSpec, data: Mapping[str, SplitData], seed: int, weighting: str,
                 params: TreeParams) -> Fit:
    """Fit one arm at one seed, then cut it at the validation-best boosting iteration."""
    train = _train_split(spec, data, seed)
    cols = train.columns(spec.cues)
    feats = train.features[:, cols]
    val, y_val = _selection_set(spec, data)
    val_feats = val.features[:, cols]
    start = time.time()
    if spec.objective == "binary":
        y, keep = _binary_target(spec, train)
        model = fit_binary(feats[keep], y[keep], weighting=weighting, params=params, seed=seed)
        stages, n_rows = staged_binary_scores(model, val_feats), int(keep.sum())
    else:
        lab = train.labels
        fields = (lab["onset_offset"], lab["future_observed"], lab["track_crosses"])
        model = fit_hazard(feats, *fields, spec.onset_spec, weighting=weighting, params=params, seed=seed)
        stages = staged_hazard_readout(model, val_feats, spec.onset_spec.horizon_bins)
        n_rows = len(person_period(*fields, spec.onset_spec)[0])
    curve = [float(roc_auc_score(y_val, s)) for s in stages]
    best = int(np.argmax(curve)) + 1
    truncate(model, best)
    return Fit(model, best, curve[best - 1], weighting, n_rows, time.time() - start, [round(a, 5) for a in curve])


def score_split(fit: Fit, spec: ArmSpec, split: SplitData) -> dict[str, np.ndarray]:
    feats = split.features[:, split.columns(spec.cues)]
    if spec.objective == "binary":
        return {"p_frame": fit.model.predict_proba(feats.astype(np.float64))[:, 1].astype(np.float32)}
    logits = hazard_logits(fit.model, feats, spec.onset_spec.num_bins)
    readout = hazard_horizon_prob(logits, spec.onset_spec.horizon_bins).astype(np.float32)
    return {"p_readout": readout, "hazard_logits": logits}


def _write_fit(fit: Fit, spec: ArmSpec, seed: int, data: Mapping[str, SplitData], seed_dir: Path) -> None:
    seed_dir.mkdir(parents=True, exist_ok=True)
    meta = {"arm": spec.name, "seed": seed, "objective": spec.objective, "horizon": spec.horizon,
            "onset_lookahead": spec.lookahead, "onset_bin_width": spec.bin_width, "weighting": fit.weighting,
            "n_iter": fit.n_iter, "source": "tree_study"}
    for split_name, filename in _DUMPS:
        if split_name in data:
            split = data[split_name]
            save_dump(seed_dir / filename, {**split.labels, **score_split(fit, spec, split)},
                      {**meta, "split": split_name})
    record = {"spec": asdict(spec), "seed": seed, "weighting": fit.weighting, "n_iter": fit.n_iter,
              "val_auc": fit.val_auc, "n_rows": fit.n_rows, "seconds": round(fit.seconds, 1),
              "val_auc_curve": fit.val_auc_curve}
    (seed_dir / "fit.json").write_text(json.dumps(record, indent=1))   # written last = the done marker


def _choose_weighting(spec: ArmSpec, data: Mapping[str, SplitData], arm_dir: Path, params: TreeParams,
                      chosen: Mapping[str, str]) -> str:
    """The arm's class weighting: inherited, or selected on the first seed's validation AUC (and that
    seed's winning fit written, so it is not fitted twice)."""
    if spec.weighting_from:
        return chosen[spec.weighting_from]
    record = arm_dir / "arm.json"
    if record.exists():
        return json.loads(record.read_text())["weighting"]
    seed = spec.seeds[0]
    fits = {w: fit_arm_seed(spec, data, seed, w, params) for w in ("none", "balanced")}
    weighting = max(fits, key=lambda w: (fits[w].val_auc, w == "none"))   # tie -> unweighted
    _write_fit(fits[weighting], spec, seed, data, arm_dir / f"s{seed}")
    arm_dir.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps({"weighting": weighting, "selection_seed": seed,
                                  "val_auc": {w: f.val_auc for w, f in fits.items()}}, indent=1))
    return weighting


def run_arm(spec: ArmSpec, data: Mapping[str, SplitData], out_dir: Path, params: TreeParams,
            chosen: dict[str, str], log=print) -> None:
    """Fit every seed of ``spec`` not already on disk (restartable: ``fit.json`` marks a finished seed)."""
    arm_dir = out_dir / spec.name
    chosen[spec.name] = _choose_weighting(spec, data, arm_dir, params, chosen)
    for seed in spec.seeds:
        seed_dir = arm_dir / f"s{seed}"
        if (seed_dir / "fit.json").exists():
            continue
        fit = fit_arm_seed(spec, data, seed, chosen[spec.name], params)
        _write_fit(fit, spec, seed, data, seed_dir)
        log(f"[tree] {spec.name} s{seed}: weighting={fit.weighting} n_iter={fit.n_iter} "
            f"val_auc={fit.val_auc:.4f} rows={fit.n_rows} {fit.seconds:.0f}s")


def run_arms(arms: Sequence[ArmSpec], data: Mapping[str, SplitData], out_dir: Path,
             params: TreeParams | None = None, log=print) -> dict[str, str]:
    """Run ``arms`` in order; a parent named in ``weighting_from`` must run first (ARMS is ordered so)."""
    params = params or TreeParams()
    chosen: dict[str, str] = {}
    for spec in arms:
        if spec.weighting_from and spec.weighting_from not in chosen:
            parent = next(a for a in ARMS if a.name == spec.weighting_from)
            run_arm(replace(parent, seeds=parent.seeds[:1]), data, out_dir, params, chosen, log)
        run_arm(spec, data, out_dir, params, chosen, log)
    return chosen


# --------------------------------------------------------------------------- gates


def probe_reproduction(data: Mapping[str, SplitData], probe_dir: Path,
                       seeds: Sequence[int] = (42, 43, 44)) -> list[dict[str, float]]:
    """Refit the 2026-09-25 probe as best recorded (sklearn defaults, ``class_weight="balanced"``, summary
    features, base train windows) and compare with its stored dumps. A tolerance gate, not an exact one:
    the probe's own script was lost, so its hyperparameters are known only as "defaults"."""
    from scipy.stats import spearmanr

    from pedpredict.eval.detection_curve import detection_curve

    train, test = data["train"], data["test"]
    rows = []
    for seed in seeds:
        model = HistGradientBoostingClassifier(class_weight="balanced", random_state=seed)
        model.fit(train.features.astype(np.float64), train.labels["crosses"])
        p = model.predict_proba(test.features.astype(np.float64))[:, 1]
        rates = [pt.detection_rate for pt in detection_curve({**test.labels, "p": p}, "p", budgets=(205, 41))]
        row = {"seed": seed, "auc": float(roc_auc_score(test.labels["crosses"], p)),
               "det205": rates[0], "det41": rates[1]}
        stored_path = probe_dir / f"gbm_s{seed}.npz"
        if stored_path.exists():
            stored = load_dump(stored_path)[0]
            old = [pt.detection_rate for pt in detection_curve(stored, "p_frame", budgets=(205, 41))]
            row.update(stored_auc=float(roc_auc_score(stored["crosses"], stored["p_frame"])),
                       stored_det205=old[0], stored_det41=old[1],
                       spearman_vs_stored=float(spearmanr(p, stored["p_frame"]).statistic))
            row["reproduces"] = abs(row["auc"] - row["stored_auc"]) <= 0.01 and abs(rates[0] - old[0]) <= 0.03
        rows.append(row)
    return rows
