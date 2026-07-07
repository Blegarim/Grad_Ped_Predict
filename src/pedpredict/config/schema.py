"""Typed config schema.

Frozen dataclasses holding every hyperparameter and path — pure data, no I/O or argparse
(that lives in ``loader.py``). Overridable from the CLI via ``--set section.field=value``.

``ModelCfg.vit_kwargs()`` / ``motion_kwargs()`` / ``cross_kwargs()`` build the constructor dicts
for the model modules and are pinned against ``tests/fixtures/golden/legacy_config.json`` by
``tests/test_config.py`` — treat them as a parity surface (don't rename/reorder casually).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Writer stores all 9 channels; runtime slices to data.motion_dim ("store wide, slice narrow").
# Channel semantics in transforms.compute_motion.
MOTION_STORE_DIM: int = 9
MOTION_CHANNELS: tuple[str, ...] = ("cx", "cy", "dx", "dy", "w", "h", "dw", "dh", "ego_speed")


@dataclass(frozen=True, slots=True)
class PathsCfg:
    """Run-relative artifact locations (resolved against the cwd / run dir, not hardcoded)."""

    pie_root: str = "data"                 # PIE toolkit data_path; frames at pie_root/images/...
    sequences_dir: str = "data/sequences"  # make_sequences pkl output
    lmdb_train: tuple[str, ...] = ("preprocessed_train", "preprocessed_train_aug")
    lmdb_train_balanced: tuple[str, ...] = ("preprocessed_train_balanced",)  # balanced-warmup source
    lmdb_val: str = "preprocessed_val"
    lmdb_test: str = "preprocessed_test"
    lmdb_test_benchmark: str = "preprocessed_test_benchmark"  # anchored-protocol eval set (test split)
    # Anchored-protocol train/val; selected together with lmdb_test_benchmark when
    # data.protocol="anchored" (resolved in paths.protocol_lmdb_dirs).
    lmdb_train_benchmark: tuple[str, ...] = ("preprocessed_train_benchmark",)
    lmdb_val_benchmark: str = "preprocessed_val_benchmark"
    log_dir: str = "training_log"          # legacy flat dirs (kept for reading old artifacts)
    ckpt_dir: str = "best_model_outputs"
    run_ckpt_dir: str = "model_outputs"
    runs_dir: str = "outputs/runs"         # per-run home: outputs/runs/{run_id}/


@dataclass(frozen=True, slots=True)
class DataCfg:
    """Data-layer constants and sequence-generation params."""

    max_seq_len: int = 20
    motion_dim: int = 8              # consumed motion width, sliced from the 9 stored; 8 = no ego, 9 = with
    # PIE source-frame pixel dims — used by flip augmentation, cross-checked vs model.motion_norm_image_size.
    source_width: int = 1920
    source_height: int = 1080
    img_height: int = 128            # tight-crop write/read size (also the tight model input)
    img_width: int = 128
    read_context_height: int = 224   # read-time context input; distinct from the stored 384 crop
    read_context_width: int = 224
    context_scale: float = 3.0       # context crop = scale * tight bbox
    jpeg_quality: int = 90
    chunk_size: int = 5000
    # LMDB map_size: explicit 4 GiB (Windows pre-allocates the file); None -> heuristic below.
    lmdb_map_size_bytes: int | None = 4 * 1024**3
    lmdb_map_size_floor_gib: float = 4.0     # heuristic floor when bytes is None
    lmdb_map_size_safety: float = 1.5        # heuristic safety multiplier when bytes is None
    preprocess_num_workers: int = 8          # offline writer DataLoader parallelism (behavior-neutral)
    preprocess_prefetch_factor: int = 2
    # sequence generation — sliding-window params
    seq_len: int = 20
    stride: int = 3
    future_offset: int = 30
    tol: int = 2
    # Anchored benchmark eval set: fixed-TTE windows around the PIE crossing_point, labeled by the
    # crossing event. obs_len matches streaming seq_len; sampling stride = round(obs_len * (1 - overlap)).
    benchmark_obs_len: int = 20
    benchmark_tte_min: int = 30
    benchmark_tte_max: int = 60
    benchmark_overlap: float = 0.7
    # Which LMDB set train/eval read: "streaming" (dense-window v2 dirs, ~37:1) | "anchored" (benchmark
    # dirs, ~2.5:1). Resolved via paths.protocol_lmdb_dirs; does not change windowing.
    protocol: str = "streaming"
    # PIE source opts (generate_data_trajectory_sequence)
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
    """Model hyperparameters — single source of truth shared across all modules."""

    d_model: int = 128               # one value shared by ALL modules
    in_channels: int = 3
    motion_dim: int = 8              # must equal DataCfg.motion_dim (checked in validate_config)
    # In-forward motion norm: "image" = fixed per-channel scale, keeps absolute geometry |
    # "per_sequence" = legacy z-norm ablation arm (golden-pinned).
    motion_norm: str = "image"
    motion_norm_image_size: tuple[int, int] = (1920, 1080)  # (W, H); must equal data.source_width/height
    ego_speed_scale: float = 50.0    # km/h scale for the ego channel under "image" norm
    # ViT stage schedule: monotonic dims, real 7x7 windows (dim/head = 24). The collapsed legacy
    # [36,36,288,36] + 2x2 windows is golden-pinned, not the default.
    stage_dims: tuple[int, ...] = (48, 96, 192, 384)
    layer_nums: tuple[int, ...] = (2, 2, 6, 2)
    head_nums: tuple[int, ...] = (2, 4, 8, 16)
    window_size: tuple[int | None, ...] = (7, 7, 7, None)
    mlp_ratio: tuple[int, ...] = (4, 4, 4, 4)
    drop_path: float = 0.15
    attn_dropout: float = 0.15
    proj_dropout: float = 0.15
    dropout: float = 0.15
    # Visual-backbone swap (docs/BACKBONE_STUDY.md): "legacy" = the from-scratch ViT_Hierarchical above
    # (default, golden-pinned) | a timm model name (e.g. "tiny_vit_5m_224") drops in a pretrained backbone
    # behind the same [B,T,3,H,W]->[B,T,d_model] contract. The stage_dims/window_size/... fields configure
    # the legacy ViT only and are inert when a timm backbone is selected.
    vit_backbone: str = "legacy"
    vit_pretrained: bool = True      # load timm ImageNet weights (ignored when vit_backbone="legacy")
    # Freeze the visual backbone for the whole run: requires_grad=False on all `vit.*` params, so the
    # optimizer trains only motion + fusion + heads. The field-standard PIE recipe on the small anchored
    # set (~4.9k windows) — frozen pretrained visual features stop the ViT memorizing. DISTINCT from
    # ScheduleCfg.freeze_backbone (freezes ALL but the task heads); this freezes ONLY the ViT.
    freeze_vit_backbone: bool = False
    # MotionEncoder
    motion_hidden_dim: int = 168
    motion_num_layers: int = 2
    motion_num_heads: int = 8
    motion_dropout: float = 0.3
    # head wiring
    head_dropout: float = 0.1
    num_classes: dict[str, int] = field(default_factory=lambda: {"actions": 2, "looks": 2, "crosses": 2})
    cross_attn_num_heads: int = 4       # CrossAttentionModule heads (not the class default 8)
    use_frame_crosses: bool = True
    frame_pool: str = "logsumexp"       # {"logsumexp", "max", "mean"}
    emit_crosses_pooled: bool = True    # live-but-unsupervised aux crosses head; never fed to the loss
    # Fusion lever (default on): adds the motion query as a residual so motion content reaches the heads.
    # False = no-residual ablation (golden-pinned).
    fusion_residual: bool = True

    def vit_kwargs(self) -> dict:
        """Build the ViT_Hierarchical constructor dict (parity surface — values as lists)."""
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
        """Build the MotionEncoder constructor dict (parity surface)."""
        return {
            "motion_dim": self.motion_dim,
            "hidden_dim": self.motion_hidden_dim,
            "d_model": self.d_model,
            "num_layers": self.motion_num_layers,
            "num_heads": self.motion_num_heads,
            "dropout": self.motion_dropout,
        }

    def cross_kwargs(self) -> dict:
        """Build the CrossAttentionModule constructor dict (parity surface).

        ``emit_crosses_pooled`` is intentionally excluded (constructor-only) to keep this a pure
        legacy-parity surface.
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
    """Training hyperparameters."""

    lr: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 4
    # step the optimizer every accum_steps micro-batches (effective batch = batch_size * accum_steps);
    # accum_steps=1 is legacy per-batch stepping (golden-pinned). Remainder flushes at the chunk boundary.
    accum_steps: int = 8
    num_epochs: int = 30
    num_workers: int = 4
    use_amp: bool = True             # request; runtime-gated by CUDA availability in utils/amp.py
    seed: int = 42                   # global RNG seed; set_seed() at the top of train/evaluate scripts
    # picks best.pth + drives early stop: {"val_loss", "macro_f1", "crosses_f1"} (F1s maximized).
    # Independent of the LR schedule below.
    selection_metric: str = "macro_f1"
    loss_weight: dict[str, float] = field(
        default_factory=lambda: {"actions": 0.8, "looks": 0.8, "crosses": 1.2}
    )
    use_class_weights: bool = False  # imbalance lever 3: inverse-freq CE weights (off in run #2)
    use_weighted_sampler: bool = True
    # imbalance lever 2, tuned down (run #2 canonical): ~26% effective crosses vs 2.8% base
    sampler_powers: dict[str, float] = field(
        default_factory=lambda: {"crosses": 0.5, "actions": 0.3, "looks": 0.3}
    )
    sampler_min_weight: float = 1e-6   # floor for per-sample sampler weights
    grad_clip_max_norm: float = 1.0    # clip_grad_norm_ bound
    early_stop_patience: int = 20      # wide enough for warmup_cosine to traverse its full curve
    early_stop_min_delta: float = 0.001
    sched_factor: float = 0.5          # ReduceLROnPlateau knobs (lr_schedule="plateau" only)
    sched_patience: int = 2
    sched_threshold: float = 1e-4
    # LR schedule: "warmup_cosine" (default) = linear warmup -> cosine to lr_min, val_loss never drives
    # the LR | "plateau" = ReduceLROnPlateau on val_loss (sched_* + lr_min apply).
    lr_schedule: str = "warmup_cosine"
    warmup_epochs: int = 1             # linear-warmup length in epochs (warmup_cosine only; 0 = none)
    warmup_start_factor: float = 0.1   # first warmup epoch runs at warmup_start_factor * lr
    lr_min: float = 1e-6               # cosine eta_min / plateau min_lr floor
    # chunk prefetch loader
    chunk_preload_depth: int = 3            # warm-ahead window
    chunk_warm_ram_threshold: float = 96.0  # wait_for_memory RAM % threshold
    chunk_warm_mem_interval: float = 1.0
    chunk_warm_mem_timeout: float | None = None   # opt-in cap on the RAM wait
    chunk_queue_timeout: float = 300.0      # queue.get skip-on-timeout
    dataloader_prefetch_factor: int = 2     # applies when num_workers > 0


@dataclass(frozen=True, slots=True)
class EvalCfg:
    """Evaluation / benchmark hyperparameters."""

    batch_size: int = 16
    num_workers: int = 4
    model_type: str = "full"
    # Efficiency benchmark: input shapes come from DataCfg (the eager ViT is bound to read_context_height,
    # so benchmarking uses the real inference resolution). These fields are only the timing knobs.
    bench_batch_size: int = 1
    bench_warmup: int = 10               # latency warmup iterations
    latency_trials: int = 50
    threshold_sweep_lo: float = 0.10
    threshold_sweep_hi: float = 0.90
    threshold_sweep_step: float = 0.05


@dataclass(frozen=True, slots=True)
class InferenceCfg:
    """Video-inference knobs.

    Only the detect/track/window/render knobs live here; the model-input *geometry* (``seq_len``,
    ``context_scale``, tight/context sizes, ``motion_dim``, ImageNet norm) is reused from
    :class:`DataCfg` so inference uses the same geometry the model trained on.
    """

    detector_weights: str = "yolo11n.pt"
    detector_class_idx: int = 0            # pedestrian class
    detector_conf: float = 0.3
    window_stride: int = 1                 # slide every frame; the data pipeline uses 3
    batch_size: int = 32
    default_fps: float = 30.0              # DirFrameSource fps when frames carry no container fps
    draw_color_chips: bool = True


@dataclass(frozen=True, slots=True)
class BalanceCfg:
    """Offline class-balancing — the opt-in majority-downsample lever, off by default.

    Defaults are the recommended *enabled* behavior (30/70). See ``data/balance.py`` and the CLAUDE.md
    imbalance policy.
    """

    enabled: bool = False                  # off by default (online sampler + loss weights handle imbalance)
    cross_pos_ratio: float = 0.30          # target crosses=1 fraction
    target_action_rate: float = 0.5        # target actions=1 fraction in the balanced subset
    target_look_rate: float = 0.5          # target looks=1 fraction in the balanced subset
    x11_select: str = "lower"              # "lower" | "upper" — which end of the feasible x11 interval
    subsample_cross1: bool = True          # priority-subsample cross=1 to n1 (vs keep all)
    allow_approx: bool = True              # greedy fallback when the exact solve is infeasible
    on_infeasible: str = "empty"           # "raise" | "empty" — behavior when no subset solves
    legacy_x00_sign_bug: bool = False      # reproduce OLD solve_exact sign bug (parity only)
    seed: int = 0


@dataclass(frozen=True, slots=True)
class AugmentCfg:
    """Offline minority-class augmentation.

    The default imbalance lever (``enabled=True``): produces the ``preprocessed_train_aug`` LMDB
    (``PathsCfg.lmdb_train[1]``) of minority records + their single-transform augmented copies
    (negatives already live in ``preprocessed_train``). Top-level section, not ``data.augment``,
    because ``apply_overrides`` caps overrides at ``section.field``.
    """

    enabled: bool = True
    # On-the-fly train-time augmentation: a fresh random composition of the four transforms per sample
    # each epoch, in the dataset read path (train split only). Independent of `enabled`: `runtime`
    # preserves the class ratio, so it is a data-scarcity regularizer, not an imbalance lever.
    runtime: bool = False
    # per-call compose: draw n_augs in [min, max] single-transform variants from the 4 below
    n_augs_min: int = 2
    n_augs_max: int = 4
    # per-transform probabilities
    p_flip: float = 0.5
    p_color: float = 0.4
    p_noise: float = 0.3
    p_erase: float = 0.2
    # ColorJitter params
    color_brightness: float = 0.2
    color_contrast: float = 0.2
    color_saturation: float = 0.3
    color_hue: float = 0.1
    motion_noise_std: float = 0.02
    erase_n_frames: int = 2
    # minority oversampling multipliers
    crosses_multiplier: int = 6
    looks_multiplier: int = 3
    seed: int = 42


@dataclass(frozen=True, slots=True)
class PoseCfg:
    """Pose-keypoint motion arm (docs/POSE_ENCODER.md).

    ``enabled`` gates both the writer (store raw ``[T, 23, 3]`` keypoints in the meta pickle) and the
    read path (build the pose feature block and concat it onto the image-normalized motion vector, so
    ``motions`` leaves the dataset as ``[T, 9 + feature_dim()]``). Joint layout and feature math live
    in ``data/pose.py``.
    """

    enabled: bool = False
    extractor: str = "dwpose"      # "dwpose" | "alphapose_halpe" (extraction script; dwpose implemented)
    include_arms: bool = False     # keep elbows+wrists: 19 kept joints instead of 15
    conf_channel: bool = True      # feed per-joint confidence scores
    smooth_window: int = 5         # temporal smoothing window at extraction time (frames)
    min_conf: float = 0.3          # below -> joint treated as missing and interpolated (extraction)
    cache_dir: str = "pose_cache"  # extraction output root: {cache_dir}/{set}/{video}.npz

    def feature_dim(self) -> int:
        """Per-frame pose feature width: 2n coords + n confidences + 2 angle (sin, cos) pairs."""
        n = 19 if self.include_arms else 15
        return 2 * n + (n if self.conf_channel else 0) + 4


@dataclass(frozen=True, slots=True)
class PhaseCfg:
    """One phase in a training schedule (assembled into ScheduleCfg by ``_default_phases()``)."""

    name: str                            # human label used in log filenames ("balanced_warmup", …)
    data_source: str                     # "balanced" -> lmdb_train_balanced | "augmented" -> lmdb_train
    lr: float                            # Adam LR for this phase (fresh optimizer, no momentum carry-over)
    max_epochs: int                      # hard epoch cap; EarlyStopping may end sooner
    early_stop_patience: int
    early_stop_min_delta: float = 0.001
    weight_decay: float = 1e-5
    sched_factor: float = 0.5
    sched_patience: int = 2
    sched_threshold: float = 1e-4
    freeze_backbone: bool = False        # True -> freeze backbone (Phase 3 "decouple classifiers")
    reload_best: bool = False            # True -> strict-load prev phase best.pth before starting


def _default_phases() -> tuple[PhaseCfg, ...]:
    """Return the canonical 3-phase tuple: balanced warmup -> full fine-tune -> decouple classifiers."""
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
    """Configurable multi-phase training schedule.

    ``enabled=False`` (default) -> single-phase ``Trainer.fit()``; ``enabled=True`` ->
    ``run_phase_schedule()`` in ``training/schedule.py``.
    """

    enabled: bool = False
    phases: tuple[PhaseCfg, ...] = field(default_factory=_default_phases)


@dataclass(frozen=True, slots=True)
class ExportCfg:
    """ONNX export knobs."""

    opset: int = 17                         # ONNX opset version; keep in sync with ort compatibility
    output_dir: str = "outputs/onnx"        # export destination (relative to cwd or absolute)
    include_temporal_weights: bool = False  # full model only; False = 3-key legacy-compatible graph
    parity_atol: float = 1e-4              # abs tolerance for onnxruntime parity assertion (CPU fp32 math)
    parity_rtol: float = 1e-4              # rel tolerance
    parity_batch_size: int = 2             # dummy batch for parity run (> 1 exercises batch axis)
    parity_seq_len: int = 4                # dummy T for parity run (short — keeps it fast)


@dataclass(frozen=True, slots=True)
class RootCfg:
    """Top-level config tree. Built by ``loader.load_config``."""

    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
    infer: InferenceCfg = field(default_factory=InferenceCfg)
    balance: BalanceCfg = field(default_factory=BalanceCfg)
    augment: AugmentCfg = field(default_factory=AugmentCfg)
    pose: PoseCfg = field(default_factory=PoseCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    export: ExportCfg = field(default_factory=ExportCfg)
