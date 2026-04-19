# -*- coding: utf-8 -*-
"""ECG QC / preprocessing for PsyPhy Datalab (no Streamlit).

Column `ecg_mv` as in BV pipeline / synthetic data. Heuristic thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks


ECG_COLUMN = "ecg_mv"


@dataclass
class EcgQcOptions:
    use_bandpass: bool = True
    bandpass_low_hz: float = 5.0
    bandpass_high_hz: float = 40.0
    detrend_linear: bool = True
    r_peak_min_distance_s: float = 0.28
    r_peak_prominence_factor: float = 0.35
    rr_min_ms: float = 300.0
    rr_max_ms: float = 2000.0
    window_sec: float = 60.0
    flat_rel_std_max: float = 1e-5
    nan_warn_ratio: float = 0.02
    nan_bad_ratio: float = 0.15
    clip_margin_ratio: float = 0.995
    clip_warn_frac: float = 0.001


@dataclass
class EcgQcReport:
    fs_hz: float
    n_samples: int
    duration_min: float
    nan_fraction: float
    flat_signal: bool
    clip_fraction: float
    n_peaks: int
    median_rr_ms: float | None
    mean_hr_bpm: float | None
    rr_outside_frac: float
    n_rr: int
    rr_ms_list: list[float] = field(default_factory=list)
    window_sec: float = 0.0
    n_windows: int = 0
    n_windows_ok: int = 0
    window_ok_fraction: float = 0.0
    overall_label: str = ""
    notes: list[str] = field(default_factory=list)


def _bandpass(x: np.ndarray, fs: float, low: float, high: float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size < 10 or fs <= 0:
        return x
    nyq = fs * 0.5
    lo = max(low / nyq, 1e-5)
    hi = min(high / nyq, 0.99)
    if lo >= hi:
        return x
    b, a = butter(3, [lo, hi], btype="band")
    mask = np.isfinite(x)
    if not np.any(mask):
        return x
    out = np.full_like(x, np.nan, dtype=float)
    xm = x.copy()
    xm[~mask] = np.nanmedian(x[mask])
    try:
        out[mask] = filtfilt(b, a, xm, method="pad")
    except ValueError:
        out = x
    return out


def _detrend_linear(x: np.ndarray) -> np.ndarray:
    t = np.arange(len(x), dtype=float)
    mask = np.isfinite(x)
    if np.sum(mask) < 3:
        return x
    coef = np.polyfit(t[mask], x[mask], 1)
    return x - (coef[0] * t + coef[1])


def detect_r_peaks(signal: np.ndarray, fs: float, opt: EcgQcOptions) -> np.ndarray:
    s = np.asarray(signal, dtype=float)
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)
    if opt.detrend_linear:
        s = _detrend_linear(s)
    if opt.use_bandpass:
        s = _bandpass(s, fs, opt.bandpass_low_hz, opt.bandpass_high_hz)
    std = float(np.std(s)) if s.size else 0.0
    prom = max(std * opt.r_peak_prominence_factor, 1e-12)
    dist = max(int(opt.r_peak_min_distance_s * fs), 1)
    peaks, _ = find_peaks(s, distance=dist, prominence=prom)
    return peaks


def compute_ecg_qc_report(
    time_s: np.ndarray,
    ecg: np.ndarray,
    fs_hz: float,
    opt: EcgQcOptions,
) -> EcgQcReport:
    notes: list[str] = []
    t = np.asarray(time_s, dtype=float).ravel()
    x = np.asarray(ecg, dtype=float).ravel()
    n = min(len(t), len(x))
    t, x = t[:n], x[:n]
    duration_min = (float(t[-1] - t[0]) / 60.0) if n > 1 else 0.0

    nan_frac = float(np.mean(~np.isfinite(x))) if n else 1.0
    x_clean = np.nan_to_num(x, nan=np.nanmedian(x[np.isfinite(x)]) if np.any(np.isfinite(x)) else 0.0)

    st_all = float(np.std(x_clean[np.isfinite(x_clean)])) if np.any(np.isfinite(x_clean)) else 0.0
    amp = float(np.max(x_clean) - np.min(x_clean)) if n else 0.0
    flat = amp <= 1e-15 or (st_all / (amp + 1e-15)) < opt.flat_rel_std_max

    amax = float(np.max(np.abs(x_clean))) if n else 0.0
    thr_clip = amax * opt.clip_margin_ratio if amax > 0 else 0.0
    clip_frac = float(np.mean(np.abs(x_clean) >= thr_clip)) if thr_clip > 0 else 0.0

    peaks = detect_r_peaks(x_clean, fs_hz, opt)
    n_peaks = len(peaks)
    rr_ms = np.array([])
    if n_peaks >= 2:
        dt = np.diff(t[peaks]) * 1000.0
        rr_ms = dt[(dt > 0) & np.isfinite(dt)]

    median_rr: float | None = float(np.median(rr_ms)) if rr_ms.size else None
    mean_hr = (60000.0 / median_rr) if median_rr is not None and median_rr > 0 else None

    if rr_ms.size:
        bad = (rr_ms < opt.rr_min_ms) | (rr_ms > opt.rr_max_ms)
        rr_out = float(np.mean(bad))
    else:
        rr_out = 1.0

    win = max(float(opt.window_sec), 1.0)
    n_win = 0
    n_ok = 0
    if n > 1 and duration_min * 60 >= win * 0.5:
        t0, t1 = float(t[0]), float(t[-1])
        edges = np.arange(t0, t1 + 1e-9, win)
        for i in range(len(edges) - 1):
            a, b = edges[i], edges[i + 1]
            m = (t >= a) & (t < b)
            if not np.any(m):
                continue
            seg = x_clean[m]
            n_win += 1
            nf = float(np.mean(~np.isfinite(seg)))
            st = float(np.std(seg[np.isfinite(seg)])) if np.any(np.isfinite(seg)) else 0.0
            amp_w = float(np.max(seg) - np.min(seg)) if seg.size else 0.0
            flat_w = amp_w <= 1e-12 or (st / (amp_w + 1e-15)) < opt.flat_rel_std_max
            ok = nf < opt.nan_bad_ratio and not flat_w
            if ok:
                n_ok += 1
    win_frac = (n_ok / n_win) if n_win else 0.0

    bad_reasons: list[str] = []
    if nan_frac >= opt.nan_bad_ratio:
        bad_reasons.append("high_nan")
    if flat:
        bad_reasons.append("flat")
    if clip_frac >= opt.clip_warn_frac:
        bad_reasons.append("clip")
    min_expected_peaks = max(8, int(duration_min * 40))
    if rr_out > 0.25 or (duration_min > 0.2 and n_peaks < min_expected_peaks):
        bad_reasons.append("rr_peaks")
    if win_frac < 0.5 and n_win >= 3:
        bad_reasons.append("windows")

    if not bad_reasons and nan_frac < opt.nan_warn_ratio and rr_out < 0.1 and win_frac >= 0.7:
        label = "dobry"
    elif len(bad_reasons) >= 2 or nan_frac >= opt.nan_bad_ratio or flat:
        label = "slaby"
    else:
        label = "ostroznie"

    notes.append(
        "R-peak detection: bandpass + scipy.find_peaks (heuristic). "
        "Validate for publication (e.g. manual check or clinical tool)."
    )
    if not opt.use_bandpass:
        notes.append("Bandpass off: peaks may be worse with strong baseline drift.")

    return EcgQcReport(
        fs_hz=fs_hz,
        n_samples=n,
        duration_min=duration_min,
        nan_fraction=nan_frac,
        flat_signal=flat,
        clip_fraction=clip_frac,
        n_peaks=n_peaks,
        median_rr_ms=median_rr,
        mean_hr_bpm=mean_hr,
        rr_outside_frac=rr_out,
        n_rr=int(rr_ms.size),
        rr_ms_list=[float(x) for x in rr_ms.tolist()] if rr_ms.size else [],
        window_sec=opt.window_sec,
        n_windows=n_win,
        n_windows_ok=n_ok,
        window_ok_fraction=win_frac,
        overall_label=label,
        notes=notes,
    )


def extract_ecg_series(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if df is None or df.empty or ECG_COLUMN not in df.columns:
        return np.array([]), np.array([])
    d = df.sort_values("time_s").reset_index(drop=True)
    return d["time_s"].to_numpy(dtype=float), d[ECG_COLUMN].to_numpy(dtype=float)


def preprocess_visible(ecg: np.ndarray, fs: float, opt: EcgQcOptions) -> np.ndarray:
    s = np.asarray(ecg, dtype=float)
    s = np.nan_to_num(s, nan=np.nanmedian(s[np.isfinite(s)]) if np.any(np.isfinite(s)) else 0.0)
    if opt.detrend_linear:
        s = _detrend_linear(s)
    if opt.use_bandpass:
        s = _bandpass(s, fs, opt.bandpass_low_hz, opt.bandpass_high_hz)
    return s


def estimate_fs_from_time(t: np.ndarray) -> float:
    if t.size < 2:
        return 1.0
    dt = np.diff(t)
    dt = dt[dt > 0]
    if dt.size == 0:
        return 1.0
    med = float(np.median(dt))
    return 1.0 / med if med > 0 else 1.0
