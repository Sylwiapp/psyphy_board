# -*- coding: utf-8 -*-
"""BrainVision .vmrk: parse markers; segment starts from type New Segment."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BVMarker:
    """One marker row (position = sample index in data file)."""

    index: int
    mk_type: str
    description: str
    position: int
    size: int
    channel: int


def _parse_marker_line(line: str) -> BVMarker | None:
    mm = re.match(r"Mk(\d+)=(.+)", line.strip())
    if not mm:
        return None
    idx = int(mm.group(1))
    rhs = mm.group(2).strip()
    # Strip trailing long session id after channel (Recorder quirk).
    rhs = re.sub(r",\d{12,}$", "", rhs)
    m = re.search(r",(\d+),(\d+),(\d+)$", rhs)
    if not m:
        return None
    pos, size, ch = int(m.group(1)), int(m.group(2)), int(m.group(3))
    prefix = rhs[: m.start()].strip()
    if "," in prefix:
        mk_type, desc = prefix.split(",", 1)
    else:
        mk_type, desc = prefix, ""
    return BVMarker(
        index=idx,
        mk_type=mk_type.strip(),
        description=desc.strip(),
        position=pos,
        size=size,
        channel=ch,
    )


def parse_vmrk(path: Path) -> list[BVMarker]:
    """Read [Marker Infos] MkN= lines."""
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    in_marker_infos = False
    out: list[BVMarker] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_marker_infos = stripped.casefold() == "[marker infos]"
            continue
        if not in_marker_infos or not stripped.startswith("Mk"):
            continue
        mk = _parse_marker_line(line)
        if mk is not None:
            out.append(mk)
    return sorted(out, key=lambda x: (x.position, x.index))


def new_segment_times_seconds(markers: list[BVMarker], sampling_hz: float) -> list[float]:
    """Start times (s) for markers of type New Segment; (position-1)/fs like time_s column."""
    fs = float(sampling_hz)
    if fs <= 0:
        return []
    times: list[float] = []
    for m in markers:
        if m.mk_type.casefold() != "new segment":
            continue
        t = max(0.0, (float(m.position) - 1.0) / fs)
        times.append(t)
    return sorted(set(round(t, 9) for t in times))


def resolve_vmrk_path(vhdr_path: Path, marker_file: str | None) -> Path | None:
    """Path to .vmrk next to vhdr, or None."""
    parent = vhdr_path.parent
    if marker_file and marker_file.strip():
        p = parent / marker_file.strip()
        if p.is_file():
            return p
    guess = vhdr_path.with_suffix(".vmrk")
    if guess.is_file():
        return guess
    return None
