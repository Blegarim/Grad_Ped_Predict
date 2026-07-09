"""Streamlit control panel — a UI front door over the same config + scripts the CLI uses.

Personal-use only. This does NOT replace argparse: it is a second way to produce the exact same
``--set section.field=value`` override list that ``scripts/*.py`` already consume. The form widgets are
**auto-generated** by introspecting the frozen dataclass schema (``src/pedpredict/config/schema.py``),
so the panel never goes stale when a config field is added — a new field shows up as a widget on its own.
Live validation calls the real ``validate_config`` (the same invariants the CLI enforces), and the action
buttons launch the existing ``scripts/*.py`` as subprocesses and stream their stdout into the page.

Run it from the repo root (needs the ``ui`` extra: ``pip install -e .[ui]``):

    streamlit run scripts/ui.py

Anything that touches the data / GPU (train, evaluate, build_lmdb, distribution) must run on the lab PC,
so run this panel there for those actions. On the personal PC it is still useful for building + validating
a config and browsing ``outputs/runs/``.
"""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import types
import typing
from pathlib import Path

import streamlit as st
import yaml

# Repo layout: this file is scripts/ui.py, so the package root is ../src.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:  # allow `streamlit run scripts/ui.py` without an editable install on PATH
    sys.path.insert(0, str(_SRC))

from pedpredict.config import RootCfg, load_config  # noqa: E402
from pedpredict.config.loader import (  # noqa: E402  (internal, but this IS the config UI)
    _FRAME_POOLS,
    _PROTOCOLS,
    _SECTIONS,
    _SELECTION_METRICS,
    ConfigError,
    validate_config,
)
from pedpredict.models.registry import ModelType  # noqa: E402

# --------------------------------------------------------------------------- enum choices
# String fields the loader validates via `in {...}` inside validate_config get an explicit dropdown.
# Everything else is inferred purely from the declared field type. Keep this table in step with
# validate_config's membership checks (it is the ONLY hand-maintained mapping in this file).
_CHOICES: dict[str, tuple[str, ...]] = {
    "eval.model_type": tuple(m.value for m in ModelType),
    "data.protocol": tuple(sorted(_PROTOCOLS)),
    "train.selection_metric": tuple(sorted(_SELECTION_METRICS)),
    "train.lr_schedule": ("warmup_cosine", "plateau"),
    "model.frame_pool": tuple(sorted(_FRAME_POOLS)),
    "model.motion_norm": ("image", "per_sequence", "none"),
    "pose.extractor": ("dwpose", "alphapose_halpe"),
    "balance.x11_select": ("lower", "upper"),
    "balance.on_infeasible": ("raise", "empty"),
}

# Sections that are heavy / structural and better edited via raw --set than a generated form.
# `schedule.phases` is a tuple of nested dataclasses; `paths` is filesystem plumbing. Both are shown
# read-only, with an escape hatch (the extra-overrides box) for the rare case you need to touch them.
_SKIP_SECTIONS = frozenset({"schedule"})
_COLLAPSED_SECTIONS = frozenset({"paths", "export"})

# Actions -> (script, whether it needs extra positional flags the form can't supply on its own).
# Each maps onto an existing thin CLI; we only append the shared `--set` channel + config-dir.
_DATA_DEPENDENT = "⚠ needs data/GPU — run on the lab PC"


# --------------------------------------------------------------------------- introspection helpers


def _is_union(tp: object) -> bool:
    origin = typing.get_origin(tp)
    return origin is typing.Union or origin is getattr(types, "UnionType", object())


def _base_type(declared: object) -> object:
    """Strip Optional[...] to the first non-None member; leave other types as-is."""
    if _is_union(declared):
        members = [a for a in typing.get_args(declared) if a is not type(None)]
        return members[0] if members else str
    return declared


def _scalar_widget(key: str, declared: object, current: object):
    """Render one field as the widget its type implies; return the widget's value."""
    choices = _CHOICES.get(key)
    if choices is not None:
        idx = choices.index(current) if current in choices else 0
        return st.selectbox(key, choices, index=idx)

    base = _base_type(declared)
    if base is bool:
        return st.toggle(key, value=bool(current))
    if base is int:
        return st.number_input(key, value=int(current), step=1, format="%d")
    if base is float:
        # text_input keeps scientific notation (1e-4) legible instead of a spinner's 0.0001.
        raw = st.text_input(key, value=repr(current))
        try:
            return float(raw)
        except ValueError:
            st.caption(f":red[not a float: {raw!r}]")
            return current
    if base is str:
        # plain string field — edit the value verbatim (no yaml wrapping, which would append `\n...`).
        return st.text_input(key, value=str(current))
    # tuple / dict fall through to a yaml text box; parsed back like the CLI does.
    return st.text_input(key, value=yaml.safe_dump(current, default_flow_style=True).strip())


def _render_section(name: str, cfg_section, defaults_section) -> dict[str, str]:
    """Render every scalar field of one section; return {dotted_key: override_str} for CHANGED fields."""
    hints = typing.get_type_hints(type(cfg_section))
    overrides: dict[str, str] = {}
    for f in dataclasses.fields(cfg_section):
        declared = hints[f.name]
        base = _base_type(declared)
        current = getattr(cfg_section, f.name)
        default = getattr(defaults_section, f.name)
        key = f"{name}.{f.name}"

        # tuple/dict fields the form doesn't edit inline are shown read-only (use the raw box for those).
        if typing.get_origin(base) in (tuple, dict) and key not in _CHOICES:
            st.text_input(key, value=str(current), disabled=True,
                          help="container field — edit via the raw overrides box below")
            continue

        value = _scalar_widget(key, declared, current)
        if value != default:
            # Emit as the CLI would receive it: strings for scalars (loader coerces by declared type).
            overrides[key] = value if isinstance(value, str) else _emit(value)
    return overrides


def _emit(value: object) -> str:
    """Render a scalar override value the way the CLI expects it as a --set token argument."""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


# --------------------------------------------------------------------------- subprocess runner


def _run_script(script: str, extra_args: list[str], overrides: dict[str, str], config_dir: str):
    """Launch scripts/<script> with the shared --set channel + any extra flags; stream stdout live."""
    cmd = [sys.executable, str(_REPO_ROOT / "scripts" / script), "--config-dir", config_dir]
    for key, val in overrides.items():
        cmd += ["--set", f"{key}={val}"]
    cmd += extra_args

    st.code(" ".join(cmd), language="bash")
    log_area = st.empty()
    lines: list[str] = []
    proc = subprocess.Popen(
        cmd, cwd=str(_REPO_ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line.rstrip("\n"))
        log_area.code("\n".join(lines[-400:]), language="text")  # tail; keep the DOM bounded
    code = proc.wait()
    (st.success if code == 0 else st.error)(f"{script} exited with code {code}")


# --------------------------------------------------------------------------- page


def main() -> None:
    st.set_page_config(page_title="pedpredict control panel", layout="wide")
    st.title("pedpredict — control panel")
    st.caption(
        "A form over the same config + scripts the CLI uses. Widgets are auto-generated from the "
        "dataclass schema; only fields you change become `--set` overrides."
    )

    config_dir = st.sidebar.text_input("config dir", value="configs")
    try:
        defaults = load_config(config_dir, overrides=None, validate=False)
    except Exception as exc:  # noqa: BLE001 — surface any load failure in-page, don't crash the panel
        st.error(f"Could not load configs from {config_dir!r}: {exc}")
        st.stop()
        return

    st.sidebar.markdown("### Sections")
    # One expander per section; collapsed for the plumbing-y ones.
    overrides: dict[str, str] = {}
    for name in _SECTIONS:
        if name in _SKIP_SECTIONS:
            continue
        expanded = name not in _COLLAPSED_SECTIONS
        with st.expander(f"`{name}`", expanded=expanded):
            section = getattr(defaults, name)
            overrides |= _render_section(name, section, section)

    with st.expander("raw extra overrides (one `section.field=value` per line)", expanded=False):
        raw_extra = st.text_area(
            "for tuple/dict/schedule fields the form can't edit inline", value="", height=100,
            label_visibility="collapsed",
        )
        for line in raw_extra.splitlines():
            line = line.strip()
            if line and "=" in line:
                k, v = line.split("=", 1)
                overrides[k.strip()] = v.strip()

    # ------------------------------------------------------------------ live validation
    st.markdown("### Effective overrides")
    if overrides:
        st.code("\n".join(f"--set {k}={v}" for k, v in overrides.items()), language="bash")
    else:
        st.caption("_(none — all values at their config defaults)_")

    override_tokens = [f"{k}={v}" for k, v in overrides.items()]
    valid = False
    try:
        cfg: RootCfg = load_config(config_dir, override_tokens, validate=False)
        validate_config(cfg)
        st.success("✓ config is valid")
        valid = True
    except ConfigError as exc:
        st.error(f"✗ invalid config: {exc}")
    except Exception as exc:  # noqa: BLE001 — coercion errors etc. also belong in-page
        st.error(f"✗ could not build config: {exc}")

    # ------------------------------------------------------------------ actions
    st.markdown("### Run")
    st.caption(_DATA_DEPENDENT + " for train / evaluate / build_lmdb / distribution.")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Train", disabled=not valid, width="stretch"):
            _run_script("train.py", [], overrides, config_dir)
        if st.button("Count labels", disabled=not valid, width="stretch"):
            _run_script("count_labels.py", [], overrides, config_dir)

    with col2:
        ckpt = st.text_input("checkpoint (for evaluate)", value="", key="ckpt")
        split = st.selectbox("split", ("val", "test"), key="split")
        eval_ready = valid and bool(ckpt.strip())
        if st.button("Evaluate", disabled=not eval_ready, width="stretch"):
            _run_script("evaluate.py", ["--checkpoint", ckpt, "--split", split], overrides, config_dir)
        if not ckpt.strip():
            st.caption("_enter a checkpoint path to enable Evaluate_")

    with col3:
        if st.button("Distribution report", disabled=not valid, width="stretch"):
            _run_script("report_distribution.py", [], overrides, config_dir)

    # ------------------------------------------------------------------ runs browser
    st.markdown("### Recent runs")
    index = _REPO_ROOT / cfg.paths.runs_dir / "index.csv" if valid else None
    if index and index.exists():
        import csv

        with open(index, encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        st.dataframe(rows[-25:], width="stretch")
    else:
        st.caption(f"_no run index yet (expected at {cfg.paths.runs_dir}/index.csv)_" if valid else "")


if __name__ == "__main__":
    main()
