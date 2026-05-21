# -*- coding: utf-8 -*-
"""Respiration QC and preprocessing for PsyPhy Datalab -- backed by NeuroKit2.

Thin wrappers around `neurokit2` for each of the two respiration belts (thoracic
Resp01T / abdominal Resp02B):
- cleaning via `nk.rsp_clean(method="khodadad2018")` (bandpass 0.05-3 Hz),
- phase + rate extraction via `nk.rsp_process` (Inhale/Exhale, Rate, Amplitude,
  RVT),
- Respiratory Rate Variability via `nk.rsp_rrv`,
- window-level and session-level aggregates; physiological breath-rate ranges.

`coupling_two_belts` returns Pearson r and best xcorr lag between the two belts;
this is useful for detecting paradoxical breathing (chest/abdomen asynchrony).

References:
- Khodadad, D., Nordebo, S., Mueller, B., ... (2018). Optimized breath detection
  algorithm in electrical impedance tomography. Physiological Measurement
  39:094001.
- Harrison, S. J., Bianchi, S., Heinzle, J., ... (2021). A Hilbert-transform-based
  approach to estimate respiratory volume per time (RVT). NeuroImage 230:117844.
- Quintana, D. S., & Heathers, J. A. (2014). Considerations in the assessment of
  heart rate variability in biobehavioral research. Frontiers in Psychology
  5:805.
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


# Default belt: thoracic, which tends to be more stable while the participant
# is speaking (abdomen moves due to speech production).
RSP_COLUMN_DEFAULT = "resp_t"
RSP_BELTS: tuple[str, ...] = ("resp_t", "resp_b")

CLEAN_METHODS: tuple[str, ...] = (
    "khodadad2018",
    "biosppy",
    "hampel",
    "power",
    "none",
)


@dataclass(frozen=True)
class RspQcOptions:
    """Immutable QC options for respiration."""

    clean_method: str = "khodadad2018"
    method_rvt: str = "harrison2021"
    window_sec: float = 60.0

    # Resting-state breath-rate range in cycles per minute (Quintana & Heathers 2014).
    rate_min_cpm: float = 6.0
    rate_max_cpm: float = 30.0

    nan_warn_ratio: float = 0.02
    nan_bad_ratio: float = 0.15


@dataclass
class RspProcessed:
    """`nk.rsp_process` output in a dashboard-friendly shape."""

    cleaned: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    rate_cpm: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    amplitude: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    rvt: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    peaks: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    troughs: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    @property
    def n_cycles(self) -> int:
        return int(self.peaks.size)


@dataclass
class RspQcReport:
    channel: str  # 'resp_t' or 'resp_b'
    fs_hz: float
    n_samples: int
    duration_min: float
    nan_fraction: float
    flat_signal: bool

    mean_rate_cpm: float | None
    median_rate_cpm: float | None
    rate_in_range_fraction: float
    n_cycles: int

    mean_amplitude: float | None
    rrv_sdbb_ms: float | None  # SD of breath-to-breath intervals in ms
    rrv_rmssd_ms: float | None

    window_sec: float = 0.0
    n_windows: int = 0
    n_windows_ok: int = 0
    window_ok_fraction: float = 0.0
    window_segments: tuple[WindowSegment, ...] = ()

    overall_label: str = ""
    notes: list[str] = field(default_factory=list)


def extract_rsp_series(
    df: pd.DataFrame, channel: str = RSP_COLUMN_DEFAULT
) -> tuple[np.ndarray, np.ndarray]:
    """Pull (time_s, signal) for the requested belt out of a session DataFrame."""
    if df is None or df.empty or channel not in df.columns:
        return np.array([]), np.array([])
    d = df.sort_values("time_s").reset_index(drop=True)
    return d["time_s"].to_numpy(dtype=float), d[channel].to_numpy(dtype=float)


def clean_rsp(signal: np.ndarray, fs: float, opt: RspQcOptions) -> np.ndarray:
    """Filtered respiration via NK2 `rsp_clean`; fall back to raw on errors."""
    s = impute_finite(np.asarray(signal, dtype=float))
    if s.size < 10 or fs <= 0:
        return s
    if opt.clean_method == "none" or not NK_AVAILABLE:
        return s
    try:
        return np.asarray(
            nk.rsp_clean(s, sampling_rate=float(fs), method=opt.clean_method),
            dtype=float,
        )
    except Exception:  # noqa: BLE001
        return s


def process_rsp(signal: np.ndarray, fs: float, opt: RspQcOptions) -> RspProcessed:
    """Full NK2 pipeline for one belt: clean -> peaks/troughs -> rate/amplitude/RVT."""
    empty = RspProcessed()
    s = impute_finite(np.asarray(signal, dtype=float))
    if s.size < int(3 * fs) or fs <= 0 or not NK_AVAILABLE:
        return empty
    try:
        df_rsp, info = nk.rsp_process(
            s,
            sampling_rate=float(fs),
            method=opt.clean_method,
            method_rvt=opt.method_rvt,
        )
        return RspProcessed(
            cleaned=np.asarray(df_rsp.get("RSP_Clean", s), dtype=float),
            rate_cpm=np.asarray(df_rsp.get("RSP_Rate", []), dtype=float),
            amplitude=np.asarray(df_rsp.get("RSP_Amplitude", []), dtype=float),
            rvt=np.asarray(df_rsp.get("RSP_RVT", []), dtype=float),
            peaks=np.asarray(info.get("RSP_Peaks", []), dtype=int),
            troughs=np.asarray(info.get("RSP_Troughs", []), dtype=int),
        )
    except Exception:  # noqa: BLE001
        return empty


def _safe_rrv(peaks: np.ndarray, fs: float) -> tuple[float | None, float | None]:
    """Return (SDBB_ms, RMSSD_ms) from breath-peak indices, or (None, None)."""
    if peaks.size < 3 or fs <= 0:
        return None, None
    bb_ms = np.diff(peaks) / float(fs) * 1000.0
    bb_ms = bb_ms[np.isfinite(bb_ms) & (bb_ms > 0)]
    if bb_ms.size < 2:
        return None, None
    sdbb = float(np.std(bb_ms))
    diff = np.diff(bb_ms)
    rmssd = float(np.sqrt(np.mean(diff**2))) if diff.size else None
    return sdbb, rmssd


def compute_rsp_qc_report(
    time_s: np.ndarray,
    rsp_raw: np.ndarray,
    fs_hz: float,
    opt: RspQcOptions,
    processed: RspProcessed | None = None,
    channel: str = RSP_COLUMN_DEFAULT,
) -> RspQcReport:
    """Aggregate QC over one respiration belt."""
    notes: list[str] = []
    t = np.asarray(time_s, dtype=float).ravel()
    x = np.asarray(rsp_raw, dtype=float).ravel()
    n = min(len(t), len(x))
    t, x = t[:n], x[:n]
    duration_min = (float(t[-1] - t[0]) / 60.0) if n > 1 else 0.0

    nan_frac = nan_fraction(x)
    x_clean = impute_finite(x)
    flat = is_flat(x_clean)

    if processed is None:
        processed = RspProcessed()

    rate = processed.rate_cpm
    rate_finite = rate[np.isfinite(rate)] if rate.size else np.array([])
    in_range_frac = (
        float(np.mean((rate_finite >= opt.rate_min_cpm) & (rate_finite <= opt.rate_max_cpm)))
        if rate_finite.size
        else 0.0
    )
    mean_rate = float(np.mean(rate_finite)) if rate_finite.size else None
    median_rate = float(np.median(rate_finite)) if rate_finite.size else None
    mean_amp = (
        float(np.mean(processed.amplitude[np.isfinite(processed.amplitude)]))
        if processed.amplitude.size
        else None
    )
    sdbb, rmssd = _safe_rrv(processed.peaks, fs_hz)

    seg_list: list[WindowSegment] = []
    n_win = 0
    n_ok = 0
    if n > 1 and duration_min * 60 >= opt.window_sec * 0.5:
        peak_times = t[processed.peaks] if processed.peaks.size else np.array([])
        for a, b, mask in iter_windows(t, opt.window_sec):
            n_win += 1
            seg = x_clean[mask]
            nf = nan_fraction(x[mask])
            flat_w = is_flat(seg)
            n_cycles_w = int(np.sum((peak_times >= a) & (peak_times < b)))

            cpm_w = (
                n_cycles_w / max((b - a) / 60.0, 1e-6) if n_cycles_w > 0 else 0.0
            )
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
                reasons.append(
                    "Sygnal oddechu w oknie wyglada na plaski (bezdech? offset?)"
                )
            if (
                n_cycles_w > 0
                and (cpm_w < opt.rate_min_cpm or cpm_w > opt.rate_max_cpm)
            ):
                ok = False
                reasons.append(
                    "Czestosc oddechu w oknie poza zakresem ({:.1f} cpm; prog {:.0f}-{:.0f})".format(
                        cpm_w, opt.rate_min_cpm, opt.rate_max_cpm
                    )
                )
            if n_cycles_w == 0 and (b - a) >= 20.0:
                ok = False
                reasons.append("Brak wykrytych cykli oddechu w oknie >=20 s")

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
                        "n_cycles": float(n_cycles_w),
                        "rate_cpm": float(cpm_w),
                    },
                )
            )
    win_frac = (n_ok / n_win) if n_win else 0.0

    bad_reasons: list[str] = []
    if nan_frac >= opt.nan_bad_ratio:
        bad_reasons.append("high_nan")
    if flat:
        bad_reasons.append("flat")
    if mean_rate is not None and (
        mean_rate < opt.rate_min_cpm or mean_rate > opt.rate_max_cpm
    ):
        bad_reasons.append("rate_out_of_range")

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
            "NeuroKit2 v{}: clean='{}', RVT='{}'.".format(
                NK_VERSION, opt.clean_method, opt.method_rvt
            )
        )
    else:
        notes.append(
            "NeuroKit2 nie jest zainstalowany - `py -3 -m pip install neurokit2`."
        )

    return RspQcReport(
        channel=channel,
        fs_hz=fs_hz,
        n_samples=n,
        duration_min=duration_min,
        nan_fraction=nan_frac,
        flat_signal=flat,
        mean_rate_cpm=mean_rate,
        median_rate_cpm=median_rate,
        rate_in_range_fraction=in_range_frac,
        n_cycles=processed.n_cycles,
        mean_amplitude=mean_amp,
        rrv_sdbb_ms=sdbb,
        rrv_rmssd_ms=rmssd,
        window_sec=opt.window_sec,
        n_windows=n_win,
        n_windows_ok=n_ok,
        window_ok_fraction=win_frac,
        window_segments=tuple(seg_list),
        overall_label=label,
        notes=notes,
    )


def compute_rsp_metrics(
    time_s: np.ndarray,
    processed: RspProcessed,
    fs: float,
    channel: str = RSP_COLUMN_DEFAULT,
) -> pd.DataFrame:
    """Single-row DataFrame with session-level respiration features."""
    if processed.cleaned.size == 0:
        return pd.DataFrame()
    duration_min = (
        float(time_s[-1] - time_s[0]) / 60.0 if time_s.size > 1 else 0.0
    )
    rate_finite = processed.rate_cpm[np.isfinite(processed.rate_cpm)]
    amp_finite = processed.amplitude[np.isfinite(processed.amplitude)]
    sdbb, rmssd = _safe_rrv(processed.peaks, fs)
    out = {
        "RSP_Channel": channel,
        "RSP_Duration_min": duration_min,
        "RSP_N_Cycles": float(processed.n_cycles),
        "RSP_Mean_Rate_cpm": float(np.mean(rate_finite)) if rate_finite.size else float("nan"),
        "RSP_SD_Rate_cpm": float(np.std(rate_finite)) if rate_finite.size else float("nan"),
        "RSP_Mean_Amplitude": float(np.mean(amp_finite)) if amp_finite.size else float("nan"),
        "RSP_RRV_SDBB_ms": sdbb if sdbb is not None else float("nan"),
        "RSP_RRV_RMSSD_ms": rmssd if rmssd is not None else float("nan"),
    }
    if NK_AVAILABLE and processed.peaks.size >= 4:
        try:
            rate_for_rrv = processed.rate_cpm if processed.rate_cpm.size else None
            if rate_for_rrv is not None and np.any(np.isfinite(rate_for_rrv)):
                rrv = nk.rsp_rrv(
                    rate_for_rrv, sampling_rate=float(fs), show=False, silent=True
                )
                for c in rrv.columns:
                    v = rrv.iloc[0][c]
                    out[c] = float(v) if pd.notna(v) else float("nan")
        except Exception:  # noqa: BLE001
            pass
    return pd.DataFrame([out])


def coupling_two_belts(
    resp_t: np.ndarray, resp_b: np.ndarray, fs: float
) -> dict[str, float]:
    """Coupling between thoracic and abdominal belts: Pearson r + best-xcorr lag (s).

    High positive r -> synchronous chest/abdomen breathing (normal).
    r close to 0 or negative -> paradoxical breathing or belt placement problem.
    """
    a = impute_finite(np.asarray(resp_t, dtype=float))
    b = impute_finite(np.asarray(resp_b, dtype=float))
    n = min(a.size, b.size)
    if n < int(2 * fs) or fs <= 0:
        return {"pearson_r": float("nan"), "lag_s_max_xcorr": float("nan")}
    a = a[:n] - np.mean(a[:n])
    b = b[:n] - np.mean(b[:n])
    denom = (np.std(a) * np.std(b))
    r = float(np.mean(a * b) / denom) if denom > 0 else float("nan")
    max_lag = min(int(5 * fs), n // 2)
    if max_lag <= 1:
        return {"pearson_r": r, "lag_s_max_xcorr": float("nan")}
    xcorr = np.correlate(a, b, mode="full")
    mid = xcorr.size // 2
    window = xcorr[mid - max_lag : mid + max_lag + 1]
    lag_idx = int(np.argmax(window)) - max_lag
    return {"pearson_r": r, "lag_s_max_xcorr": float(lag_idx / fs)}
