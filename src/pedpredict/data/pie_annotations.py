"""Read PIE behaviour annotations straight from the XML set (frames not required).

Why this exists: the *label* half of the v2 contract depends only on PIE's per-frame behaviour tags,
which ship as a ~25 MB XML tree — small enough to keep on a laptop, unlike the frames. Parsing them
here lets :mod:`pedpredict.data.onset_stats` reproduce the whole sequence-label contract with no PIE
toolkit, no images, and no built LMDB. Verified 2026-08-20: the tracks this yields, fed through
:func:`~pedpredict.data.pie_sequences.window_track`, reproduce the golden per-split counts exactly
(``tests/fixtures/golden/pie_sequences_counts.json``).

Two facts about PIE's XML that this relies on, both read out of the toolkit rather than assumed:

* Only tracks labelled ``pedestrian`` carry behaviour tags, and every one of them does — the 1842
  tracks across the six sets are exactly PIE's behaviour-annotated pedestrians. There is no
  bounding-box-only pedestrian class in these files, so no track arrives with fabricated labels.
* Boxes flagged ``outside="1"`` are skipped, matching ``pie_data._get_annotations``.

**Labels only.** ``ego_speed`` is zero-filled here (OBD speed lives in a separate annotation set), so
these tracks are for *counting*, never for building training data — that is ``make_sequences.py``.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Iterator
from pathlib import Path

from pedpredict.data.pie_sequences import PieTrack

__all__ = ["DEFAULT_SPLIT_SETS", "iter_annotation_tracks", "load_annotation_tracks"]

#: PIE's ``data_split_type='default'`` set assignment (mirrors ``pie_data._get_image_set_ids``).
DEFAULT_SPLIT_SETS: dict[str, tuple[str, ...]] = {
    "train": ("set01", "set02", "set04"),
    "val": ("set05", "set06"),
    "test": ("set03",),
}

#: Text -> scalar for the three behaviour tags this module reads (mirrors ``pie_data._map_text_to_scalar``).
_TAG_MAPS: dict[str, dict[str, int]] = {
    "action": {"standing": 0, "walking": 1},
    "look": {"not-looking": 0, "looking": 1},
    "cross": {"not-crossing": 0, "crossing": 1, "crossing-irrelevant": -1},
}


def _iter_xml_sources(path: Path, sets: tuple[str, ...]) -> Iterator[tuple[str, bytes]]:
    """Yield ``(name, xml_bytes)`` for every ``*_annt.xml`` under ``path`` belonging to ``sets``.

    ``path`` is either the zipped annotation archive or an unpacked directory of set folders, so the
    same call works on the laptop (zip) and on the lab PC (extracted tree).
    """
    if path.is_file() and path.suffix == ".zip":
        with zipfile.ZipFile(path) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith("_annt.xml") and any(f"/{s}/" in f"/{name}" for s in sets):
                    yield name, archive.read(name)
        return
    if not path.is_dir():
        raise FileNotFoundError(f"PIE annotations not found at {path} (expected a .zip or a directory)")
    for set_id in sets:
        for xml_path in sorted((path / set_id).glob("*_annt.xml")):
            yield str(xml_path), xml_path.read_bytes()


def _tag(box: ET.Element, name: str) -> int:
    """Read one behaviour attribute off a ``<box>`` and map its text to PIE's scalar."""
    node = box.find(f'./attribute[@name="{name}"]')
    if node is None or node.text is None:
        raise ValueError(f"box is missing the {name!r} attribute — not a behaviour-annotated pedestrian")
    try:
        return _TAG_MAPS[name][node.text]
    except KeyError as exc:
        raise ValueError(f"unknown {name} value {node.text!r} in PIE annotation") from exc


def _parse_track(elem: ET.Element) -> PieTrack | None:
    """One ``<track label="pedestrian">`` -> :class:`PieTrack`; ``None`` for any other object class."""
    if elem.get("label") != "pedestrian":
        return None
    all_boxes = elem.findall("./box")
    boxes = [b for b in all_boxes if int(b.get("outside", "0")) == 0]
    if not boxes:
        return None
    id_node = all_boxes[0].find('./attribute[@name="id"]')
    track_id = "" if id_node is None or id_node.text is None else id_node.text
    return PieTrack(
        track_id=track_id,
        images=[f"{track_id}/{b.get('frame')}" for b in boxes],  # placeholder: no frames on disk
        bboxes=[[float(b.get("xtl")), float(b.get("ytl")),  # type: ignore[arg-type]
                 float(b.get("xbr")), float(b.get("ybr"))] for b in boxes],  # type: ignore[arg-type]
        ego_speed=[0.0] * len(boxes),  # labels-only: OBD speed is a separate annotation set
        actions=[_tag(b, "action") for b in boxes],
        looks=[_tag(b, "look") for b in boxes],
        crosses=[_tag(b, "cross") for b in boxes],
    )


def iter_annotation_tracks(path: str | Path, split: str) -> Iterator[PieTrack]:
    """Yield every behaviour-annotated pedestrian track in ``split`` from the PIE annotation XMLs.

    ``path`` is the ``annotations.zip`` archive or an unpacked annotations directory; ``split`` is
    ``train`` / ``val`` / ``test`` under PIE's default set assignment.
    """
    try:
        sets = DEFAULT_SPLIT_SETS[split]
    except KeyError as exc:
        raise ValueError(f"unknown split {split!r} (expected one of {sorted(DEFAULT_SPLIT_SETS)})") from exc
    for _name, payload in _iter_xml_sources(Path(path), sets):
        for elem in ET.fromstring(payload).findall("./track"):
            track = _parse_track(elem)
            if track is not None:
                yield track


def load_annotation_tracks(path: str | Path, split: str) -> list[PieTrack]:
    """Eager :func:`iter_annotation_tracks` (the split fits comfortably in memory: <1000 tracks)."""
    return list(iter_annotation_tracks(path, split))
