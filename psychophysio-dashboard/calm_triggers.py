# -*- coding: utf-8 -*-
"""CALM-Bogna: parsowanie logu triggerow (port rownolegly) i przedzialy analizy z par kodow."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from session_geom import SessionGeom

# Code pairs (start, end) and short label; order matches data/CALM_Bogna_EEG_trigger_codes.txt
CALM_ANALYSIS_CODE_PAIRS: tuple[tuple[int, int, str], ...] = (
    (40, 41, "Rest - baseline"),
    (120, 121, "Rest - boundary 1"),
    (140, 141, "Rest - boundary 2"),
    (20, 21, "Wspomnienia - baseline"),
    (100, 101, "Wspomnienia - boundary"),
)

_TS_LINE = re.compile(
    r"^(\d{8}T\d{6}(?:\.\d+)?)\s+(\d+)\s+",
)


def _parse_log_timestamp(s: str) -> float:
    """Sekundy od epoki (tylko do roznic czasu miedzy liniami w jednym logu)."""
    s = s.strip()
    m = re.match(r"^(\d{8}T\d{6})(\.\d+)?$", s)
    if not m:
        raise ValueError(f"niepoprawny timestamp: {s!r}")
    main = m.group(1)
    frac_v = float(m.group(2)) if m.group(2) else 0.0
    dt = datetime.strptime(main, "%Y%m%dT%H%M%S")
    return dt.timestamp() + frac_v


def parse_calm_trigger_log(text: str) -> dict[int, float]:
    """
    Zwraca kod triggera -> czas w sekundach od triggera 1 (Experiment start).
    Ostatnia linia z danym kodem wygrywa (gdyby powtorzenia).
    """
    origin: float | None = None
    raw: dict[int, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = _TS_LINE.match(line)
        if not m:
            continue
        ts_s = _parse_log_timestamp(m.group(1))
        code = int(m.group(2))
        raw[code] = ts_s
        if code == 1:
            origin = ts_s
    if origin is None:
        raise ValueError(
            "Brak triggera kod 1 (Experiment start); nie da sie zsynchronizowac osi czasu."
        )
    return {c: t - origin for c, t in raw.items()}


def load_calm_trigger_times(path: Path) -> dict[int, float]:
    return parse_calm_trigger_log(path.read_text(encoding="utf-8", errors="replace"))


def build_calm_analysis_geom(session_s: float, code_times: dict[int, float]) -> SessionGeom | None:
    """
    Buduje SessionGeom z rozlacznymi oknami [t_start, t_end] dla skonfigurowanych par kodow.
    Brakujace pary sa pomijane; gdy zadnej pelnej pary, zwrot None.
    """
    session_s = float(session_s)
    windows: list[tuple[float, float]] = []
    labels: list[str] = []
    for c0, c1, lab in CALM_ANALYSIS_CODE_PAIRS:
        if c0 not in code_times or c1 not in code_times:
            continue
        t0, t1 = float(code_times[c0]), float(code_times[c1])
        lo, hi = (t0, t1) if t0 <= t1 else (t1, t0)
        lo = max(0.0, lo)
        hi = min(session_s, hi)
        if hi > lo + 1e-3:
            windows.append((lo, hi))
            labels.append(lab)
    if not windows:
        return None
    return SessionGeom(
        session_s=session_s,
        segment_edges=(0.0, session_s),
        analysis_windows=tuple(windows),
        segment_labels=tuple(labels),
    )
