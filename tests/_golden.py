"""Golden-fixture registry + loader.

Single source of truth for the golden **characterization** fixtures: every module with a captured
numeric reference is registered here, mapping its fixture file to the ``_capture/`` script that
regenerates it and a one-line note on what it guards. The fixtures pin each module's output numerics;
a change that moves them is surfaced as a test failure rather than slipping through silently.

``tests/test_golden_outputs.py`` turns this registry into a coverage gate — leaving an orphan fixture,
shipping a fixture with no regenerator, or registering one no test references becomes a failure.

The ``_capture/`` regenerators were originally written against the legacy reference repo, which now
lives only in the ``legacy-archive`` git tag. To re-capture a fixture, ``git checkout legacy-archive``
first; on ``main`` the committed fixtures are the source of truth.

Existing per-module tests keep their own ``torch.load`` calls (co-located assertions are the better
pattern); ``load_golden`` is offered for new tests and is used by the meta-test itself.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

GOLDEN_DIR: Path = Path(__file__).resolve().parent / "fixtures" / "golden"
CAPTURE_DIR: Path = Path(__file__).resolve().parent / "_capture"


@dataclass(frozen=True)
class GoldenSpec:
    """One captured legacy reference: its fixture, its regenerator, and what it guards."""

    fixture: str          # filename under tests/fixtures/golden/
    capture: str | None   # regenerator under tests/_capture/, or None if captured offline
    note: str             # the module / behavior this fixture protects


# Keyed by short logical name. `capture=None` marks fixtures captured offline (no in-repo
# regenerator) — recorded explicitly so the meta-test does not flag them as missing scripts.
GOLDEN_MANIFEST: dict[str, GoldenSpec] = {
    "vit": GoldenSpec(
        "vit.pt", "capture_vit_golden.py", "ViT_Hierarchical forward parity (2.1, B2 no-lazy-params)"
    ),
    "motion_encoder": GoldenSpec(
        "motion_encoder.pt", "capture_motion_golden.py", "MotionEncoder forward parity (2.2)"
    ),
    "cross_attention": GoldenSpec(
        "cross_attention.pt", "capture_cross_attention_golden.py", "CrossAttentionModule parity (2.3)"
    ),
    "ensemble": GoldenSpec(
        "ensemble.pt",
        "capture_ensemble_golden.py",
        "EnsembleModel + ablations + eval + ONNX parity (2.4/2.5/5.x/7.1)",
    ),
    "losses": GoldenSpec(
        "losses_cases.pt", "capture_losses_golden.py", "multitask loss numeric parity (3.x)"
    ),
    "metrics": GoldenSpec(
        "metrics_cases.pt", "capture_metrics_golden.py", "MetricAccumulator parity (3.x)"
    ),
    "sampler": GoldenSpec(
        "sampler_cases.json", "capture_sampler_golden.py", "WeightedRandomSampler weights (1.6/3.1)"
    ),
    "augment": GoldenSpec(
        "augment_cases.pt", "capture_augment_golden.py", "offline augmentation parity, flip idx-2 (1.4)"
    ),
    "balance": GoldenSpec(
        "balance_cases.json", "capture_balance_golden.py", "offline balance solver parity (1.3)"
    ),
    "lmdb_writer": GoldenSpec(
        "lmdb_process_record.pt", "capture_lmdb_golden.py", "crop/motion record + transforms parity (1.2)"
    ),
    "lmdb_dataset": GoldenSpec(
        "lmdb_dataset_cases.pt", "capture_lmdb_dataset_golden.py", "LMDBChunkDataset per-item parity"
    ),
    "trainer_step": GoldenSpec(
        "trainer_step.pt", "capture_trainer_golden.py", "trainer step / callbacks / schedule parity (4.x)"
    ),
    "pie_sequences": GoldenSpec(
        "pie_sequences_counts.json", None, "sequence-generation label counts (offline capture, 1.1)"
    ),
    "legacy_config": GoldenSpec(
        "legacy_config.json", None, "legacy config -> ModelCfg/DataCfg parity (offline capture, B6)"
    ),
}


def golden_path(name: str) -> Path:
    """Absolute path to a registered fixture (raises KeyError on an unknown name)."""
    return GOLDEN_DIR / GOLDEN_MANIFEST[name].fixture


def load_golden(name: str, *, weights_only: bool = False) -> Any:
    """Load a registered fixture by logical name (``.json`` parsed, ``.pt`` via ``torch.load``)."""
    path = golden_path(name)
    if not path.exists():
        spec = GOLDEN_MANIFEST[name]
        hint = f"run tests/_capture/{spec.capture}" if spec.capture else "captured offline"
        raise FileNotFoundError(f"missing golden fixture {path} ({hint})")
    if path.suffix == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    return torch.load(path, weights_only=weights_only)
