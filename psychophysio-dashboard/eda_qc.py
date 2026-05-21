# -*- coding: utf-8 -*-
"""EDA / GSR QC and preprocessing for PsyPhy Datalab -- backed by NeuroKit2.

Thin, testable wrappers around `neurokit2`:
- cleaning via `nk.eda_clean` (LP Butterworth around 3 Hz),
- tonic/phasic decomposition via `nk.eda_phasic` (default `method="highpass"`;
  cvxEDA is also available but requires the external `cvxopt` package),
- SCR detection via `nk.eda_peaks` on the phasic signal,
- window-level and session-level aggregates (number of SCRs, mean amplitude,
  SCL slope).

References:
- Boucsein, W. (2012). Electrodermal Activity. Springer (2nd ed.).
- Greco, A., Valenza, G., Citi, L., & Scilingo, E. P. (2016). cvxEDA: A convex
  optimization approach to electrodermal activity processing. IEEE TBME
  63(4):797-804.
- Society for Psychophysiological Research Ad Hoc Committee (Boucsein et al.,
  2012). Publication recommendations for electrodermal measurements.
  Psychophysiology 49(8):1017-1034.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from signal_qc_base import (
    NK_AVAILABLE,
    NK_VERSION,
    WindowSegment,
    estimate_fs_from_time,
    impute_finite,
    is_flat,
    iter_windows,
    label_from_reasons,
    nan_fraction,
)

if NK_AVAILABLE:
    import neurokit2 as nk  # noqa: F401
else:
    nk = None  # type: ignore[assignment]


EDA_COLUMN = "eda_us"

CLEAN_METHODS: tuple[str, ...] = ("neurokit", "biosppy", "none")
PHASIC_METHODS: tuple[str, ...] = ("highpass", "smoothmedian", "cvxeda", "sparsEDA")


@dataclass(frozen=True)
class EdaQcOptions:
    """Immutable EDA QC options (safe to use as a cache key)."""

    clean_method: str = "neurokit"
    phasic_method: str = "highpass"
    # Minimum SCR amplitude (in uS) accepted by eda_peaks.
    scr_amplitude_min: float = 0.1
    window_sec: float = 60.0

    # Physiological SCL range (Boucsein 2012).
    scl_min_us: float = 0.5
    scl_max_us: float = 50.0
    # Movement/artefact rate threshold on the cleaned signal (uS per second).
    artefact_rate_us_per_s: float = 5.0

    nan_warn_ratio: float = 0.02
    nan_bad_ratio: float = 0.15


@dataclass
class ScrPeaksResult:
    """SCR detection output on the phasic EDA signal."""

    onsets: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    peaks: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    amplitudes_us: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    recovery_idx: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def n_scr(self) -> int:
        return int(self.peaks.size)


@dataclass
class EdaQcReport:
    fs_hz: float
    n_samples: int
    duration_min: float
    nan_fraction: float
    flat_signal: bool

    mean_scl_us: float | None
    median_scl_us: float | None
    scl_in_range_fraction: float
    slope_scl_us_per_min: float | None

    n_scr: int
    scr_rate_per_min: float | None
    mean_scr_amplitude_us: float | None

    # Fraction of samples whose first derivative exceeds the artefact rate.
    artefact_fraction: float

    window_sec: float = 0.0
    n_windows: int = 0
    n_windows_ok: int = 0
    window_ok_fraction: float = 0.0
    window_segments: tuple[WindowSegment, ...] = ()

    overall_label: str = ""
    notes: list[str] = field(default_factory=list)


def extract_eda_series(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Pull (time_s, eda_us) out of a session DataFrame."""
    if df is None or df.empty or EDA_COLUMN not in df.columns:
        return np.array([]), np.array([])
    d = df.sort_values("time_s").reset_index(drop=True)
    return d["time_s"].to_numpy(dtype=float), d[EDA_COLUMN].to_numpy(dtype=float)


def clean_eda(signal: np.ndarray, fs: float, opt: EdaQcOptions) -> np.ndarray:
    """LP-filtered EDA via NK2 `eda_clean`; fall back to the raw signal on errors."""
    s = impute_finite(np.asarray(signal, dtype=float))
    if s.size < 10 or fs <= 0:
        return s
    if opt.clean_method == "none" or not NK_AVAILABLE:
        return s
    try:
        return np.asarray(
            nk.eda_clean(s, sampling_rate=float(fs), method=opt.clean_method),
            dtype=float,
        )
    except Exception:  # noqa: BLE001 - last-resort: return raw signal
        return s


def decompose_eda(
    cleaned: np.ndarray, fs: float, opt: EdaQcOptions
) -> tuple[np.ndarray, np.ndarray]:
    """Return (tonic SCL, phasic SCR) via `nk.eda_phasic`.

    Fallback when NK2 is missing or the requested method fails: tonic is a
    centred rolling mean (~60 s), phasic is the residual.
    """
    s = impute_finite(np.asarray(cleaned, dtype=float))
    if s.size < int(2 * fs) or fs <= 0:
        return s, np.zeros_like(s)
    if NK_AVAILABLE:
        try:
            decomp = nk.eda_phasic(
                s, sampling_rate=float(fs), method=opt.phasic_method
            )
            tonic = np.asarray(decomp["EDA_Tonic"].values, dtype=float)
            phasic = np.asarray(decomp["EDA_Phasic"].values, dtype=float)
            return tonic, phasic
        except Exception:  # noqa: BLE001 - fall through to rolling-mean fallback
            pass
    win = max(3, int(round(60.0 * fs)))
    series = pd.Series(s)
    tonic = series.rolling(window=win, center=True, min_periods=1).mean().to_numpy()
    phasic = s - tonic
    return tonic, phasic


def find_scr_peaks(
    phasic: np.ndarray, fs: float, opt: EdaQcOptions
) -> ScrPeaksResult:
    """SCR detection on the phasic signal via `nk.eda_peaks`."""
    empty = ScrPeaksResult()
    s = impute_finite(np.asarray(phasic, dtype=float))
    if s.size < int(2 * fs) or fs <= 0 or not NK_AVAILABLE:
        return empty
    try:
        _df, info = nk.eda_peaks(
            s,
            sampling_rate=float(fs),
            method="neurokit",
            amplitude_min=float(opt.scr_amplitude_min),
        )
        onsets = np.asarray(info.get("SCR_Onsets", []), dtype=int)
        peaks = np.asarray(info.get("SCR_Peaks", []), dtype=int)
        amps = np.asarray(info.get("SCR_Amplitude", []), dtype=float)
        recov = np.asarray(info.get("SCR_RecoveryTime", []), dtype=float)
        recovery_idx = (
            (peaks + np.round(recov * float(fs))).astype(int)
            if recov.size == peaks.size
            else np.array([], dtype=int)
        )
        return ScrPeaksResult(
            onsets=onsets,
            peaks=peaks,
            amplitudes_us=amps,
            recovery_idx=recovery_idx,
        )
    except Exception:  # noqa: BLE001
        return empty


def compute_eda_qc_report(
    time_s: np.ndarray,
    eda_raw: np.ndarray,
    fs_hz: float,
    opt: EdaQcOptions,
    tonic: np.ndarray | None = None,
    phasic: np.ndarray | None = None,
    scr: ScrPeaksResult | None = None,
) -> EdaQcReport:
    """Aggregate EDA QC over the recording: SCL range, SCR count, artefacts, windows."""
    notes: list[str] = []
    t = np.asarray(time_s, dtype=float).ravel()
    x = np.asarray(eda_raw, dtype=float).ravel()
    n = min(len(t), len(x))
    t, x = t[:n], x[:n]
    duration_min = (float(t[-1] - t[0]) / 60.0) if n > 1 else 0.0

    nan_frac = nan_fraction(x)
    x_clean = impute_finite(x)
    flat = is_flat(x_clean)

    if tonic is None:
        tonic = x_clean
    tonic = np.asarray(tonic, dtype=float)

    if phasic is None:
        phasic = np.zeros_like(x_clean)
    phasic = np.asarray(phasic, dtype=float)

    in_range = (tonic >= opt.scl_min_us) & (tonic <= opt.scl_max_us)
    scl_in_range = float(np.mean(in_range)) if tonic.size else 0.0
    mean_scl = float(np.mean(tonic)) if tonic.size else None
    median_scl = float(np.median(tonic)) if tonic.size else None

    slope = None
    if tonic.size >= 2 and duration_min > 0:
        slope = float((tonic[-1] - tonic[0]) / duration_min)

    if x_clean.size >= 2 and fs_hz > 0:
        deriv = np.diff(x_clean) * float(fs_hz)
        artefact_frac = float(np.mean(np.abs(deriv) > opt.artefact_rate_us_per_s))
    else:
        artefact_frac = 0.0

    if scr is None:
        scr = ScrPeaksResult()
    n_scr = scr.n_scr
    scr_rate = (n_scr / duration_min) if duration_min > 0 else None
    mean_amp = (
        float(np.mean(scr.amplitudes_us)) if scr.amplitudes_us.size else None
    )

    seg_list: list[WindowSegment] = []
    n_win = 0
    n_ok = 0
    if n > 1 and duration_min * 60 >= opt.window_sec * 0.5:
        peak_times = t[scr.peaks] if scr.peaks.size else np.array([])
        for a, b, mask in iter_windows(t, opt.window_sec):
            n_win += 1
            seg = x_clean[mask]
            seg_phasic = phasic[mask] if phasic.size == n else seg
            nf = nan_fraction(x[mask])
            flat_w = is_flat(seg)
            n_scr_w = int(np.sum((peak_times >= a) & (peak_times < b)))

            reasons: list[str] = []
            ok = True
            if nf >= opt.nan_bad_ratio:
                ok = False
                reasons.append(
                    "Wysoki udzial NaN w oknie ({:.1f} %, prog {:.0f} %)".format(
                        nf * 100, opt.nan_bad_ratio * 100
                    )
                )
            if flat_w:
                ok = False
                reasons.append("Sygnal EDA w oknie wyglada na plaski (elektroda?)")
            seg_mean = float(np.mean(seg)) if seg.size else 0.0
            if seg_mean < opt.scl_min_us or seg_mean > opt.scl_max_us:
                ok = False
                reasons.append(
                    "Srednia SCL poza zakresem fizjologicznym ({:.2f} uS; prog {:.1f}-{:.1f})".format(
                        seg_mean, opt.scl_min_us, opt.scl_max_us
                    )
                )

            if ok:
                n_ok += 1

            seg_list.append(
                WindowSegment(
                    t_start_s=a,
                    t_end_s=b,
                    ok=ok,
                    reasons_pl=tuple(reasons),
                    nan_fraction=nf,
                    flat_window=flat_w,
                    metrics={
                        "mean_scl_us": seg_mean,
                        "n_scr": float(n_scr_w),
                        "phasic_rms_us": float(np.sqrt(np.mean(seg_phasic**2)))
                        if seg_phasic.size
                        else 0.0,
                    },
                )
            )
    win_frac = (n_ok / n_win) if n_win else 0.0

    bad_reasons: list[str] = []
    if nan_frac >= opt.nan_bad_ratio:
        bad_reasons.append("high_nan")
    if flat:
        bad_reasons.append("flat")
    if mean_scl is not None and (mean_scl < opt.scl_min_us or mean_scl > opt.scl_max_us):
        bad_reasons.append("scl_out_of_range")
    if artefact_frac > 0.05:
        bad_reasons.append("artefact")

    label = label_from_reasons(
        n_bad=len(bad_reasons),
        nan_frac=nan_frac,
        flat_signal=flat,
        window_ok_fraction=win_frac,
        n_windows=n_win,
        nan_warn_ratio=opt.nan_warn_ratio,
        nan_bad_ratio=opt.nan_bad_ratio,
    )

    if NK_AVAILABLE:
        notes.append(
            "NeuroKit2 v{}: clean='{}', phasic='{}', amplitude_min={} uS.".format(
                NK_VERSION, opt.clean_method, opt.phasic_method, opt.scr_amplitude_min
            )
        )
    else:
        notes.append(
            "NeuroKit2 nie jest zainstalowany - `py -3 -m pip install neurokit2`."
        )
    if artefact_frac > 0.05:
        notes.append(
            "Wysoki udzial szybkich zmian (>{} uS/s): {:.1f} % probek - sprawdz ruchy/elektrody.".format(
                opt.artefact_rate_us_per_s, artefact_frac * 100
            )
        )

    return EdaQcReport(
        fs_hz=fs_hz,
        n_samples=n,
        duration_min=duration_min,
        nan_fraction=nan_frac,
        flat_signal=flat,
        mean_scl_us=mean_scl,
        median_scl_us=median_scl,
        scl_in_range_fraction=scl_in_range,
        slope_scl_us_per_min=slope,
        n_scr=n_scr,
        scr_rate_per_min=scr_rate,
        mean_scr_amplitude_us=mean_amp,
        artefact_fraction=artefact_frac,
        window_sec=opt.window_sec,
        n_windows=n_win,
        n_windows_ok=n_ok,
        window_ok_fraction=win_frac,
        window_segments=tuple(seg_list),
        overall_label=label,
        notes=notes,
    )


def compute_eda_metrics(
    time_s: np.ndarray,
    tonic: np.ndarray,
    phasic: np.ndarray,
    scr: ScrPeaksResult,
    fs: float,
) -> pd.DataFrame:
    """Single-row DataFrame with session-level EDA features (export-friendly)."""
    if tonic.size == 0:
        return pd.DataFrame()
    duration_min = (
        float(time_s[-1] - time_s[0]) / 60.0 if time_s.size > 1 else 0.0
    )
    out = {
        "EDA_Duration_min": duration_min,
        "EDA_Mean_SCL_uS": float(np.mean(tonic)),
        "EDA_SD_SCL_uS": float(np.std(tonic)),
        "EDA_Median_SCL_uS": float(np.median(tonic)),
        "EDA_Slope_SCL_uS_per_min": float(
            (tonic[-1] - tonic[0]) / max(duration_min, 1e-9)
        ),
        "EDA_Phasic_RMS_uS": float(np.sqrt(np.mean(phasic**2))) if phasic.size else 0.0,
        "EDA_N_SCR": float(scr.n_scr),
        "EDA_SCR_Rate_per_min": (
            float(scr.n_scr / duration_min) if duration_min > 0 else float("nan")
        ),
        "EDA_Mean_SCR_Amplitude_uS": (
            float(np.mean(scr.amplitudes_us)) if scr.amplitudes_us.size else float("nan")
        ),
        "EDA_Sum_SCR_Amplitude_uS": (
            float(np.sum(scr.amplitudes_us)) if scr.amplitudes_us.size else 0.0
        ),
    }
    return pd.DataFrame([out])


def preprocess_visible(eda: np.ndarray, fs: float, opt: EdaQcOptions) -> np.ndarray:
    """Convenience: return only the tonic SCL (usually what we want on the plot)."""
    cleaned = clean_eda(eda, fs, opt)
    tonic, _phasic = decompose_eda(cleaned, fs, opt)
    return tonic
