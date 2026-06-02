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

    lmdb_train: tuple[str, ...] = ("preprocessed_train", "preprocessed_train_aug")
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
    img_height: int = 128
    img_width: int = 128
    context_scale: float = 2.0       # LMDB crops were physically written at this scale (do not change)
    jpeg_quality: int = 90
    chunk_size: int = 5000           # Q4: canonical 5000 (OLD main() default was 4500); does not affect parity
    # sequence generation (generate_sequences.py)
    seq_len: int = 20
    stride: int = 3
    future_offset: int = 30
    tol: int = 2
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
    early_stop_patience: int = 15
    early_stop_min_delta: float = 0.001
    sched_factor: float = 0.5
    sched_patience: int = 2
    sched_threshold: float = 1e-4


@dataclass(frozen=True, slots=True)
class EvalCfg:
    """Evaluation / benchmark hyperparameters — replaces test.py:308-322 literals (B1)."""

    batch_size: int = 16
    num_workers: int = 4
    model_type: str = "full"
    bench_context_scale: float = 3.0     # Q3: deliberate benchmark scale (synthetic FLOPs/latency only)
    bench_img_size: int = 128
    latency_trials: int = 50
    threshold_sweep_lo: float = 0.10
    threshold_sweep_hi: float = 0.90
    threshold_sweep_step: float = 0.05


@dataclass(frozen=True, slots=True)
class RootCfg:
    """Top-level config tree. Built by ``loader.load_config``."""

    paths: PathsCfg = field(default_factory=PathsCfg)
    data: DataCfg = field(default_factory=DataCfg)
    model: ModelCfg = field(default_factory=ModelCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    eval: EvalCfg = field(default_factory=EvalCfg)
