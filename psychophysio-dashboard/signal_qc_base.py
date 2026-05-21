# -*- coding: utf-8 -*-
"""Shared QC scaffolding for all psychophysiological signals in PsyPhy Datalab.

Single source of truth for helpers used by `ecg_qc.py`, `eda_qc.py`, `rsp_qc.py`
and similar modules, so that the same numerical primitives (`impute_finite`,
`estimate_fs_from_time`, window aggregation) live in one place.

Nothing here assumes anything about the signal domain (ECG vs EDA vs Resp);
the module provides pure-numerical helpers plus a generic `WindowSegment`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np


try:
    import neurokit2 as nk  # noqa: F401 - re-export NK_AVAILABLE / NK_VERSION

    NK_AVAILABLE = True
    NK_VERSION = getattr(nk, "__version__", "unknown")
except Exception:  # noqa: BLE001 - environment may not have NK2 installed
    nk = None  # type: ignore[assignment]
    NK_AVAILABLE = False
    NK_VERSION = "not installed"


FLAT_REL_STD_MAX_DEFAULT = 1e-5
NAN_WARN_RATIO_DEFAULT = 0.02
NAN_BAD_RATIO_DEFAULT = 0.15


def impute_finite(x: np.ndarray) -> np.ndarray:
    """Replace NaN/Inf with the median; preserve shape and float dtype."""
    if x.size == 0:
        return x
    mask = np.isfinite(x)
    if mask.all():
        return x.astype(float, copy=False)
    med = float(np.nanmedian(x)) if np.any(mask) else 0.0
    out = x.astype(float, copy=True)
    out[~mask] = med
    return out


def estimate_fs_from_time(t: np.ndarray) -> float:
    """Sampling rate inferred from a monotonic time vector (median 1/dt)."""
    if t.size < 2:
        return 1.0
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 1.0
    med = float(np.median(dt))
    return 1.0 / med if med > 0 else 1.0


def is_flat(x: np.ndarray, rel_std_max: float = FLAT_REL_STD_MAX_DEFAULT) -> bool:
    """Heuristic: signal looks flat (std / peak-to-peak below threshold)."""
    if x.size == 0:
        return True
    amp = float(np.max(x) - np.min(x))
    if amp <= 1e-15:
        return True
    sd = float(np.std(x))
    return (sd / (amp + 1e-15)) < rel_std_max


def nan_fraction(x: np.ndarray) -> float:
    """Fraction of non-finite samples (NaN/Inf); 1.0 for empty input."""
    if x.size == 0:
        return 1.0
    return float(np.mean(~np.isfinite(x)))


def clip_fraction(x: np.ndarray, margin_ratio: float = 0.995) -> float:
    """Fraction of samples close to absolute maximum (ADC saturation proxy)."""
    if x.size == 0:
        return 0.0
    x_clean = impute_finite(x)
    amax = float(np.max(np.abs(x_clean)))
    if amax <= 0:
        return 0.0
    thr = amax * margin_ratio
    return float(np.mean(np.abs(x_clean) >= thr))


@dataclass(frozen=True)
class WindowSegment:
    """One time window of local QC - generic.

    `metrics` holds extra per-signal numbers (e.g. R-count for ECG, SCR-count
    for EDA, breath-rate for Resp) so the dashboard can render uniformly.
    """

    t_start_s: float
    t_end_s: float
    ok: bool
    reasons_pl: tuple[str, ...]
    nan_fraction: float
    flat_window: bool
    metrics: dict[str, float] = field(default_factory=dict)


def iter_windows(
    t: np.ndarray, window_sec: float
) -> Iterator[tuple[float, float, np.ndarray]]:
    """Yield (t_start, t_end, mask) over an evenly-split time axis.

    The last window may be shorter; empty windows are skipped.
    """
    if t.size < 2:
        return
    win = max(float(window_sec), 1.0)
    t0, t1 = float(t[0]), float(t[-1])
    span = max(t1 - t0, 1e-9)
    n_seg = max(1, int(np.ceil(span / win)))
    edges = np.linspace(t0, t1, n_seg + 1)
    for wi in range(len(edges) - 1):
        a, b = float(edges[wi]), float(edges[wi + 1])
        m = (t >= a) & (t < b)
        if not np.any(m):
            continue
        yield a, b, m


def label_from_reasons(
    n_bad: int,
    nan_frac: float,
    flat_signal: bool,
    window_ok_fraction: float,
    n_windows: int,
    *,
    nan_warn_ratio: float = NAN_WARN_RATIO_DEFAULT,
    nan_bad_ratio: float = NAN_BAD_RATIO_DEFAULT,
) -> str:
    """Summary label: 'dobry' / 'ostroznie' / 'slaby'.

    Heuristic kept consistent across ECG/EDA/Resp modules so dashboard
    summaries are comparable between signals.
    """
    if (
        n_bad == 0
        and nan_frac < nan_warn_ratio
        and (n_windows < 3 or window_ok_fraction >= 0.7)
        and not flat_signal
    ):
        return "dobry"
    if n_bad >= 2 or nan_frac >= nan_bad_ratio or flat_signal:
        return "slaby"
    return "ostroznie"
