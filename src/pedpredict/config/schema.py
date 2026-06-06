"""Typed config schema (Prompt 0.2).

Frozen dataclasses that replace OLD ``config.py`` + every hardcoded hyperparameter
scattered through ``train.py`` / ``test.py`` / ``scripts/train_utils.py``. This module is
pure data: no I/O, no argparse (that lives in ``loader.py``).

Defaults mirror the *training-effective* values from the OLD repo (``config.py`` and
``train.py:280-296``), NOT the drifting ``Vision_Transformer.__main__`` /
``Motion_Encoder.__main__`` smoke-test kwargs (band-aid B6 — see MIGRATION.md).

``ModelCfg.vit_kwargs()`` / ``motion_kwargs()`` reproduce the OLD ``vit_args_config()`` /
``motion_enc_args_config()`` dicts byte-for-byte; ``tests/test_config.py`` asserts that
parity against ``tests/fixtures/golden/legacy_config.json``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class PathsCfg:
    """Run-relative artifact locations (resolved against the cwd / run dir, not hardcoded)."""

    pie_root: str = "data"                 # PIE toolkit data_path (cloned into repo); images at pie_root/images/...
    sequences_dir: str = "data/sequences"  # generate_sequences pkl output home (gitignored under data/)
    lmdb_train: tuple[str, ...] = ("preprocessed_train", "preprocessed_train_aug")
    lmdb_train_balanced: tuple[str, ...] = ("preprocessed_train_balanced",)  # Phase 1 balanced-warmup source
    lmdb_val: str = "preprocessed_val"
    lmdb_test: str = "preprocessed_test"
    log_dir: str = "training_log"          # legacy flat dirs (kept for reading OLD artifacts)
    ckpt_dir: str = "best_model_outputs"   # legacy flat dirs
    run_ckpt_dir: str = "model_outputs"    # legacy flat dirs
    runs_dir: str = "outputs/runs"         # new per-run home: outputs/runs/{run_id}/ (Prompt 0.3, Q-A)


@dataclass(frozen=True, slots=True)
class DataCfg:
    """Data-layer constants. B7: ``MAX_SEQ_LEN`` and the ``motions[..., :8]`` slice land here."""

    # B7: magic constants relocated from scripts/train_utils.py
    max_seq_len: int = 20            # was train_utils.MAX_SEQ_LEN
    motion_dim: int = 8              # was the [..., :8] slice; writer must emit exactly this many channels
    img_height: int = 128            # write+read tight size; also read-tight model input
    img_width: int = 128
    # read-time context model input (OLD train.py:362 Resize((224,224))) — distinct from the write-time
    # context size (img_*  * context_scale = 384); stored 384 crops are re-decoded and shrunk to this.
    read_context_height: int = 224
    read_context_width: int = 224
    context_scale: float = 3.0       # context crop = scale * tight bbox (uniform 3.0; flex for ablation)
    jpeg_quality: int = 90
    chunk_size: int = 5000           # Q4: canonical 5000 (OLD main() default was 4500); does not affect parity
    # LMDB map_size heuristic (lmdb_writer.compute_map_size) — defaults reproduce OLD preprocess literals
    lmdb_map_size_bytes: int | None = None   # explicit override; None -> heuristic
    lmdb_map_size_floor_gib: float = 4.0     # OLD floor: 4 * 1024**3
    lmdb_map_size_safety: float = 1.5        # OLD safety multiplier
    # offline writer DataLoader parallelism (preprocessing only — behavior-neutral)
    preprocess_num_workers: int = 8
    preprocess_prefetch_factor: int = 2
    # sequence generation — sliding-window params (generate_sequences.py)
    seq_len: int = 20
    stride: int = 3
    future_offset: int = 30
    tol: int = 2
    # PIE source opts (generate_data_trajectory_sequence) — defaults mirror the OLD data_opts literals
    min_track_size: int = 10
    fstride: int = 1
    data_split_type: str = "default"
    seq_type: str = "all"
    squarify_ratio: float = 0.0
    height_min: float = 0.0
    height_max: float | None = None  # None -> float('inf'); PIE height_rng upper bound
    # ImageNet normalization
    norm_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    norm_std: tuple[float, float, float] = (0.229, 0.224, 0.225)


@dataclass(frozen=True, slots=True)
class ModelCfg:
    """Model hyperparameters. Single source of truth replacing the drifting ``__main__`` blocks (B6)."""

    d_model: int = 128               # get_unified_dim_model() — one value shared by ALL modules
    in_channels: int = 3
    motion_dim: int = 8              # must equal DataCfg.motion_dim (cross-checked in validate_config, B7)
    # ViT — mirror config.vit_args_config() EXACTLY
    stage_dims: tuple[int, ...] = (36, 36, 288, 36)
    layer_nums: tuple[int, ...] = (2, 4, 5, 7)
    head_nums: tuple[int, ...] = (2, 2, 16, 2)
    window_size: tuple[int | None, ...] = (8, 4, 2, None)
    mlp_ratio: tuple[int, ...] = (4, 4, 4, 4)
    drop_path: float = 0.15
    attn_dropout: float = 0.15
    proj_dropout: float = 0.15
    dropout: float = 0.15
    # MotionEncoder — mirror config.motion_enc_args_config() EXACTLY
    motion_hidden_dim: int = 168
    motion_num_layers: int = 2
    motion_num_heads: int = 8
    motion_dropout: float = 0.3
    # head wiring (get_model(..., dropout=0.1))
    head_dropout: float = 0.1
    num_classes: dict[str, int] = field(default_factory=lambda: {"actions": 2, "looks": 2, "crosses": 2})
    # CrossAttentionModule — mirror scripts/model_utils.get_model()'s full-model wiring EXACTLY.
    # NOTE: get_model passes num_heads=4 (NOT the legacy class default 8) and does NOT forward dropout
    # (so cross_attn + classifier use head_dropout=0.1, the class default).
    cross_attn_num_heads: int = 4
    use_frame_crosses: bool = True
    frame_pool: str = "logsumexp"       # {"logsumexp", "max", "mean"}
    # B4: crosses_pooled is the pooled-feature crosses head. Legacy ALLOCATED it but never called it
    # (dead param). Here it is LIVE-but-unsupervised (default on): emitted as an auxiliary diagnostic kept
    # ready to swap in for crosses_frame, never fed to the loss. See MIGRATION.md (2.3) / CLAUDE.md.
    emit_crosses_pooled: bool = True

    def vit_kwargs(self) -> dict:
        """Reproduce OLD ``vit_args_config()`` — values as lists (parity surface)."""
        return {
            "in_channels": self.in_channels,
            "stage_dims": list(self.stage_dims),
            "layer_nums": list(self.layer_nums),
            "head_nums": list(self.head_nums),
            "window_size": list(self.window_size),
            "d_model": self.d_model,
            "mlp_ratio": list(self.mlp_ratio),
            "drop_path": self.drop_path,
            "attn_dropout": self.attn_dropout,
            "proj_dropout": self.proj_dropout,
            "dropout": self.dropout,
        }

    def motion_kwargs(self) -> dict:
        """Reproduce OLD ``motion_enc_args_config()`` (parity surface)."""
        return {
            "motion_dim": self.motion_dim,
            "hidden_dim": self.motion_hidden_dim,
            "d_model": self.d_model,
            "num_layers": self.motion_num_layers,
            "num_heads": self.motion_num_heads,
            "dropout": self.motion_dropout,
        }

    def cross_kwargs(self) -> dict:
        """Reproduce OLD ``get_model()``'s ``CrossAttentionModule(...)`` call (parity surface).

        ``emit_crosses_pooled`` is intentionally excluded (constructor-only, like ``img_size`` /
        ``max_positions`` in the ViT / MotionEncoder ports) so this stays a pure legacy-parity surface.
        """
        return {
            "d_model": self.d_model,
            "num_heads": self.cross_attn_num_heads,
            "num_classes_dict": dict(self.num_classes),
            "dropout": self.head_dropout,
            "use_frame_crosses": self.use_frame_crosses,
            "frame_pool": self.frame_pool,
        }


@dataclass(frozen=True, slots=True)
class TrainCfg:
    """Training hyperparameters — replaces train.py:280-296,346 literals (B1)."""

    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 4
    num_epochs: int = 30
    num_workers: int = 4
    use_amp: bool = True             # request; runtime-gated by CUDA availability in utils/amp.py (Q2)
    loss_weight: dict[str, float] = field(
        default_factory=lambda: {"actions": 0.8, "looks": 0.8, "crosses": 1.2}
    )
    use_weighted_sampler: bool = True
    sampler_powers: dict[str, float] = field(
        default_factory=lambda: {"crosses": 1.5, "actions": 0.3, "looks": 0.7}
    )
    sampler_min_weight: float = 1e-6   # floor for per-sample sampler weights (OLD build_sampler_weights)
    grad_clip_max_norm: float = 1.0    # clip_grad_norm_ bound (OLD train.py:158,163 literal — B1)
    early_stop_patience: int = 15
    early_stop_min_delta: float = 0.001
    sched_factor: float = 0.5
    sched_patience: int = 2
    sched_threshold: float = 1e-4
    # chunk prefetch loader (Prompt 4.2, B9) — OLD train.py:367-498 literals.
    chunk_preload_depth: int = 3            # OLD min(3, n) warm-ahead window
    chunk_warm_ram_threshold: float = 96.0  # OLD wait_for_memory(threshold=96)
    chunk_warm_mem_interval: float = 1.0    # OLD wait_for_memory(interval=1)
    chunk_warm_mem_timeout: float | None = None   # opt-in cap on the legacy infinite RAM wait
    chunk_queue_timeout: float = 300.0      # OLD queue.get(timeout=300) skip-on-timeout
    dataloader_prefetch_factor: int = 2     # OLD loader_kwargs['prefetch_factor'] = 2 (num_workers>0)


@dataclass(frozen=True, slots=True)
class EvalCfg:
    """Evaluation / benchmark hyperparameters — replaces test.py:308-322 literals (B1)."""

    batch_size: int = 16
    num_workers: int = 4
    model_type: str = "full"
    # Efficiency benchmark methodology (Prompt 5.2). Input SHAPES come from DataCfg (the eager ViT is
    # bound to read_context_height, so benchmarking uses the real inference resolution, not a synthetic
    # scale). These fields are only the timing knobs.
    bench_batch_size: int = 1            # benchmark batch (OLD compute_flops/inference_latency used 1)
    bench_warmup: int = 10               # latency warmup iterations (OLD inference_latency warmup loop)
    latency_trials: int = 50
    threshold_sweep_lo: float = 0.10
    threshold_sweep_hi: float = 0.90
    threshold_sweep_step: float = 0.05


@dataclass(frozen=True, slots=True)
class BalanceCfg:
    """Offline class-balancing (Prompt 1.3) — the OPT-IN majority-downsample lever, OFF by default.

    Defaults = the recommended *enabled* behavior (corrected 30/70). The two legacy scripts are
    reproduced by the ``balance.BALANCE_EQUAL`` / ``BALANCE_RATIO_30_70`` presets, not these defaults.
    See ``data/balance.py`` and the CLAUDE.md / MIGRATION.md imbalance policy.
    """

    enabled: bool = False                  # OFF by default (online sampler + loss weights handle imbalance)
    cross_pos_ratio: float = 0.30          # target crosses=1 fraction (Q2: 30/70, flexible)
    target_action_rate: float = 0.5        # target actions=1 fraction in the balanced subset
    target_look_rate: float = 0.5          # target looks=1 fraction in the balanced subset
    x11_select: str = "lower"              # "lower" | "upper" — which end of the feasible x11 interval
    subsample_cross1: bool = True          # priority-subsample cross=1 to n1 (vs keep all)
    allow_approx: bool = True              # greedy fallback when the exact solve is infeasible
    on_infeasible: str = "empty"           # "raise" | "empty" — behavior when no subset solves
    legacy_x00_sign_bug: bool = False      # reproduce OLD solve_exact sign bug (parity only); Phase-B drop
    seed: int = 0


@dataclass(frozen=True, slots=True)
class AugmentCfg:
    """Offline minority-class augmentation (Prompt 1.4). Defaults = OLD ``SequenceAugmenter`` literals.

    The DEFAULT imbalance lever (policy 1.3): ``enabled=True``. Produces the ``preprocessed_train_aug``
    LMDB (``PathsCfg.lmdb_train[1]``) of minority records + their single-transform augmented copies
    (negatives are NOT included — they already live in ``preprocessed_train``). Top-level section (not
    ``data.augment``) because ``apply_overrides`` caps overrides at ``section.field``.
    """

    enabled: bool = True              # offline augmentation is the default imbalance lever (policy 1.3)
    # per-call compose: OLD random.randint(2, 4) single-transform variants drawn from the 4 below
    n_augs_min: int = 2
    n_augs_max: int = 4
    # per-transform probabilities (OLD SequenceAugmenter.__init__)
    p_flip: float = 0.5
    p_color: float = 0.4
    p_noise: float = 0.3
    p_erase: float = 0.2
    # ColorJitter params (OLD T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1))
    color_brightness: float = 0.2
    color_contrast: float = 0.2
    color_saturation: float = 0.3
    color_hue: float = 0.1
    motion_noise_std: float = 0.02    # OLD motion_noise(noise_std=0.02)
    erase_n_frames: int = 2           # OLD random_erase_frames(n_frames=2)
    # oversampling multipliers (OLD augment_minority_sequences)
    crosses_multiplier: int = 6
    looks_multiplier: int = 3
    seed: int = 42


@dataclass(frozen=True, slots=True)
class PhaseCfg:
    """One phase in a training schedule (Prompt 4.4, B1).

    Exactly matches the hardcoded constants from OLD ``train_two_phase.py`` when assembled into
    ``ScheduleCfg``'s default ``phases`` tuple (see ``_default_phases()`` below).
    """

    name: str                            # human label used in log filenames ("balanced_warmup", …)
    data_source: str                     # "balanced" -> lmdb_train_balanced | "augmented" -> lmdb_train
    lr: float                            # Adam LR for this phase (fresh optimizer, no momentum carry-over)
    max_epochs: int                      # hard epoch cap; EarlyStopping may end sooner
    early_stop_patience: int
    early_stop_min_delta: float = 0.001
    weight_decay: float = 1e-5           # same OLD constant; flexible for ablation
    sched_factor: float = 0.5
    sched_patience: int = 2
    sched_threshold: float = 1e-4
    freeze_backbone: bool = False        # True -> Phase 3 "decouple classifiers" (OLD freeze_backbone())
    reload_best: bool = False            # True -> strict-load prev phase best.pth before starting


def _default_phases() -> tuple[PhaseCfg, ...]:
    """Return the canonical 3-phase tuple matching OLD train_two_phase.py hardcoded values exactly."""
    return (
        PhaseCfg(
            name="balanced_warmup",
            data_source="balanced",
            lr=1e-4,
            max_epochs=10,
            early_stop_patience=5,
            freeze_backbone=False,
            reload_best=False,
        ),
        PhaseCfg(
            name="full_finetune",
            data_source="augmented",
            lr=1e-5,
            max_epochs=20,
            early_stop_patience=5,
            freeze_backbone=False,
            reload_best=True,
        ),
        PhaseCfg(
            name="decouple_classifiers",
            data_source="augmented",
            lr=5e-5,
            max_epochs=5,
            early_stop_patience=3,
            freeze_backbone=True,
            reload_best=True,
        ),
    )


@dataclass(frozen=True, slots=True)
class ScheduleCfg:
    """Configurable multi-phase training schedule (Prompt 4.4, B1).

    ``enabled=False`` (default) -> plain ``Trainer.fit()`` single-phase path.
    ``enabled=True``            -> ``run_phase_schedule()`` in ``training/schedule.py``.

    The default ``phases`` exactly reproduce OLD ``train_two_phase.py`` behavior:
    balanced-subset warmup -> full fine-tune -> decouple classifiers.
    """

    enabled: bool = False
    phases: tuple[PhaseCfg, ...] = field(default_factory=_default_phases)


@dataclass(frozen=True, slots=True)
class RootCfg:
    """Top-level config tree. Built by ``loader.load_config``."""

    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    balance: BalanceCfg = field(default_factory=BalanceCfg)
    augment: AugmentCfg = field(default_factory=AugmentCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
