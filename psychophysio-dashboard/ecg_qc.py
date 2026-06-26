# -*- coding: utf-8 -*-
"""ECG QC / preprocessing for PsyPhy Datalab — backed by NeuroKit2.

Thin, testable wrappers around `neurokit2` so that:
- preprocessing pipelines (cleaning + R-peak detection + artefact correction)
  follow community-validated implementations (Pan-Tompkins 1985, Hamilton 2002,
  Elgendi 2010, NK2 default) instead of bespoke heuristics;
- automatic RR artefact correction uses the Lipponen-Tarvainen 2019 algorithm
  via NK2 (`correct_artifacts=True`, exposed in NK2 as the "Kubios" method);
- mean signal quality is reported via NK2 `ecg_quality` (0..1 averaged).

References:
- Pan, J. & Tompkins, W. J. (1985). IEEE TBME 32(3):230-236.
- Lipponen, J. A. & Tarvainen, M. P. (2019). J. Med. Eng. Technol. 43(3):173-181.
- Makowski, D. et al. (2021). NeuroKit2. Behav. Res. Methods 53:1689-1696.
- Quigley, K. S. et al. (2024). Psychophysiology 61:e14604.
- Laborde, S., Mosley, E., & Thayer, J. F. (2017). Front. Psychol. 8:213.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from signal_qc_base import (
    NK_AVAILABLE,
    NK_VERSION,
    estimate_fs_from_time as _estimate_fs_from_time,
    impute_finite as _shared_impute_finite,
)

if NK_AVAILABLE:
    import neurokit2 as nk  # noqa: F401
else:
    nk = None  # type: ignore[assignment]


ECG_COLUMN = "ecg_mv"

# Whitelists of method names accepted by NK2; centralised so the UI can offer them.
CLEAN_METHODS: tuple[str, ...] = (
    "neurokit",
    "pantompkins1985",
    "hamilton2002",
    "elgendi2010",
    "biosppy",
    "engzeemod2012",
    "none",
)

PEAK_METHODS: tuple[str, ...] = (
    "neurokit",
    "pantompkins1985",
    "hamilton2002",
    "elgendi2010",
    "engzeemod2012",
    "kalidas2017",
    "rodrigues2021",
    "promac",
)

# Metody korekcji artefaktów RR (Etap 5). `kubios` = Lipponen–Tarvainen 2019
# (klasyfikuje ectopic/missed/extra/longshort), `neurokit` = iteracyjna korekcja
# progowa odstępów RR (interval-based) wbudowana w NK2.
CORRECT_METHODS: tuple[str, ...] = ("kubios", "neurokit")


def _astropy_available() -> bool:
    try:
        import astropy  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


# Metody estymacji widma (PSD) dla HRV częstotliwościowego (Etap 7).
# welch = FFT z oknami (domyślna), burg = autoregresyjna (AR; Task Force 1996),
# lomb = Lomb–Scargle (bez interpolacji RR; wymaga pakietu `astropy`).
PSD_METHODS: tuple[str, ...] = ("welch", "burg") + (
    ("lomb",) if _astropy_available() else ()
)


@dataclass(frozen=True)
class EcgQcOptions:
    """Immutable preprocessing/QC options (safe to use as cache key)."""

    clean_method: str = "neurokit"
    peak_method: str = "neurokit"
    correct_artifacts: bool = True
    correct_method: str = "kubios"  # metoda korekcji RR: kubios | neurokit
    powerline_hz: float = 50.0

    rr_min_ms: float = 300.0
    rr_max_ms: float = 2000.0
    window_sec: float = 60.0

    flat_rel_std_max: float = 1e-5
    nan_warn_ratio: float = 0.02
    nan_bad_ratio: float = 0.15
    clip_margin_ratio: float = 0.995
    clip_warn_frac: float = 0.001

    # Stage 4 (SQI per epoch) — minimalna jakość okna z `nk.ecg_quality`.
    # Wartości <0.5 zwykle oznaczają, że kształt QRS w tym oknie odbiega
    # od szablonu zbudowanego z większości uderzeń (Quigley et al. 2024).
    sqi_min_window: float = 0.5
    # Stage 6 (exclusion) — segment z udziałem korekt RR (Lipponen-Tarvainen 2019)
    # przekraczającym ten próg jest oznaczany jako „nie-OK" do HRV.
    # Domyślnie 5 % zgodnie z konwencją przyjmowaną w SPR / NK2 literaturze.
    corrected_frac_max: float = 0.05


@dataclass
class RPeaksResult:
    """Wynik `nk.ecg_peaks` rozbity na piki po/przed korekcją + typy fix-ów."""

    peaks: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    peaks_uncorrected: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )
    fixes_ectopic: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    fixes_missed: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    fixes_extra: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    fixes_longshort: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=int)
    )

    @property
    def n_corrections(self) -> int:
        return int(
            self.fixes_ectopic.size
            + self.fixes_missed.size
            + self.fixes_extra.size
            + self.fixes_longshort.size
        )


@dataclass(frozen=True)
class EcgWindowSegment:
    """Jedno okno czasowe lokalnego QC (surowy sygnał + RR w obrębie okna)."""

    t_start_s: float
    t_end_s: float
    ok: bool
    reasons_pl: tuple[str, ...]
    nan_fraction: float
    flat_window: bool
    n_peaks_in_window: int
    rr_outside_frac_window: float | None  # None jeśli < 2 R w oknie
    mean_quality_window: float | None = None  # Stage 4: SQI per-epoch (NK2)
    corrected_frac_window: float | None = None  # Stage 5/6: % korekt RR w oknie


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
    window_segments: tuple[EcgWindowSegment, ...] = ()

    mean_quality: float | None = None
    n_corrections: int = 0
    corrected_frac: float = 0.0  # Stage 6: % korekt RR względem n_peaks (LT19)
    n_manual_added: int = 0
    n_manual_removed: int = 0

    overall_label: str = ""
    notes: list[str] = field(default_factory=list)


def extract_ecg_series(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if df is None or df.empty or ECG_COLUMN not in df.columns:
        return np.array([]), np.array([])
    d = df.sort_values("time_s").reset_index(drop=True)
    return d["time_s"].to_numpy(dtype=float), d[ECG_COLUMN].to_numpy(dtype=float)


# Re-eksport wspólnych helperów (kompatybilność wsteczna — app.py importuje
# `estimate_fs_from_time` z `ecg_qc`).
estimate_fs_from_time = _estimate_fs_from_time
_impute_finite = _shared_impute_finite


def clean_ecg(signal: np.ndarray, fs: float, opt: EcgQcOptions) -> np.ndarray:
    """Cleaned ECG via NK2 `ecg_clean`; falls back to raw input on errors."""
    s = _impute_finite(np.asarray(signal, dtype=float))
    if s.size < 10 or fs <= 0:
        return s
    if opt.clean_method == "none" or not NK_AVAILABLE:
        return s
    try:
        kwargs: dict = {"sampling_rate": float(fs), "method": opt.clean_method}
        if opt.clean_method == "neurokit":
            kwargs["powerline"] = float(opt.powerline_hz)
        return np.asarray(nk.ecg_clean(s, **kwargs), dtype=float)
    except Exception:  # noqa: BLE001 — return raw signal as last resort
        return s


def detect_r_peaks(
    cleaned: np.ndarray,
    fs: float,
    opt: EcgQcOptions,
) -> RPeaksResult:
    """Detect R-peaks; returns final peaks, uncorrected peaks and fix-type indices."""
    empty = RPeaksResult()
    s = _impute_finite(np.asarray(cleaned, dtype=float))
    if s.size < int(2 * fs) or fs <= 0 or not NK_AVAILABLE:
        return empty
    try:
        # Detekcja bez korekcji — zawsze mamy „surowe” piki jako punkt odniesienia.
        _df, info = nk.ecg_peaks(
            s,
            sampling_rate=float(fs),
            method=opt.peak_method,
            correct_artifacts=False,
        )
        uncorr = np.asarray(info.get("ECG_R_Peaks", []), dtype=int)
        peaks = uncorr
        fixes: dict[str, np.ndarray] = {
            "ectopic": np.array([], dtype=int),
            "missed": np.array([], dtype=int),
            "extra": np.array([], dtype=int),
            "longshort": np.array([], dtype=int),
        }

        if bool(opt.correct_artifacts) and uncorr.size >= 4:
            # Wybór metody korekcji RR (Etap 5).
            method = (
                "Kubios"
                if str(opt.correct_method).lower() == "kubios"
                else "neurokit"
            )
            fix_info, peaks_clean = nk.signal_fixpeaks(
                uncorr, sampling_rate=float(fs), method=method
            )
            peaks = np.asarray(peaks_clean, dtype=int)
            if isinstance(fix_info, dict):
                for k in fixes:
                    v = fix_info.get(k)
                    if v is None:
                        continue
                    try:
                        fixes[k] = np.asarray(list(v), dtype=int)
                    except (TypeError, ValueError):
                        pass

        return RPeaksResult(
            peaks=peaks,
            peaks_uncorrected=uncorr,
            fixes_ectopic=fixes["ectopic"],
            fixes_missed=fixes["missed"],
            fixes_extra=fixes["extra"],
            fixes_longshort=fixes["longshort"],
        )
    except Exception:  # noqa: BLE001
        return empty


def parse_times_input(text: str) -> list[float]:
    """Parse comma/semicolon/whitespace-separated floats; ignore garbage."""
    if not text:
        return []
    out: list[float] = []
    for chunk in text.replace("\n", ",").replace(";", ",").split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            v = float(chunk)
        except ValueError:
            continue
        if v >= 0:
            out.append(v)
    return sorted(set(out))


def apply_manual_edits(
    peaks_idx: np.ndarray,
    fs: float,
    add_times_s: list[float] | None,
    remove_times_s: list[float] | None,
    tol_s: float = 0.05,
) -> tuple[np.ndarray, int, int]:
    """Apply manual add / remove of R-peaks.

    Removal snaps to nearest detected peak within `tol_s`; ignored otherwise.
    Returns (final peaks, n_added, n_removed) where counts reflect actual edits.
    """
    peaks = np.asarray(peaks_idx, dtype=int).copy()
    tol = max(1, int(tol_s * float(fs)))
    n_removed = 0
    if remove_times_s and peaks.size:
        keep = np.ones(peaks.shape, dtype=bool)
        for tr in remove_times_s:
            tr_idx = int(round(float(tr) * fs))
            j = int(np.argmin(np.abs(peaks - tr_idx)))
            if abs(int(peaks[j]) - tr_idx) <= tol:
                if keep[j]:
                    keep[j] = False
                    n_removed += 1
        peaks = peaks[keep]
    n_added = 0
    if add_times_s:
        add_idx = np.array(
            [int(round(float(t) * fs)) for t in add_times_s if float(t) >= 0],
            dtype=int,
        )
        before = peaks.size
        peaks = np.unique(np.concatenate([peaks, add_idx]))
        n_added = int(peaks.size - before)
    peaks = peaks[peaks >= 0]
    return np.sort(peaks), n_added, n_removed


def compute_spectrum_welch(
    x: np.ndarray, fs: float, max_freq_hz: float = 80.0
) -> tuple[np.ndarray, np.ndarray]:
    """Spectral density estimation via Welch (scipy). Returns (freqs, psd) up to ``max_freq_hz``.

    Używane do dydaktycznego porównania widma surowy vs po filtracji w Stage 2:
    pozwala pokazać znikający peak 50 Hz oraz redukcję mocy w paśmie < 0.5 Hz.
    """
    from scipy import signal as _sps  # opóźniony import, scipy jest w requirements

    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 4 or fs <= 0:
        return np.array([]), np.array([])
    # nperseg dobrane tak, żeby dać rozdzielczość ~0.05 Hz (ale nie więcej niż 1/8 sygnału)
    nperseg = int(min(max(int(fs * 8), 256), max(256, x.size // 4)))
    nperseg = max(64, nperseg)
    try:
        freqs, psd = _sps.welch(
            x,
            fs=float(fs),
            nperseg=min(nperseg, x.size),
            noverlap=None,
            scaling="density",
        )
    except Exception:  # noqa: BLE001
        return np.array([]), np.array([])
    mask = freqs <= float(max_freq_hz)
    return freqs[mask], psd[mask]


def compute_baseline_drift_std(x: np.ndarray, fs: float, cutoff_hz: float = 0.5) -> float:
    """Std sygnału w paśmie < ``cutoff_hz`` — proxy dla amplitudy „baseline wandera".

    Implementacja: filtr dolnoprzepustowy 4-rzędowy Butterworth (`filtfilt` =>
    zerowe opóźnienie fazowe), następnie odchylenie standardowe wyniku. Im
    większa wartość, tym silniejsze wędrowanie linii bazowej w sygnale.
    """
    from scipy import signal as _sps

    x = np.asarray(x, dtype=float).ravel()
    x = x[np.isfinite(x)]
    if x.size < 16 or fs <= 0 or cutoff_hz <= 0 or cutoff_hz >= fs / 2:
        return 0.0
    try:
        nyq = 0.5 * float(fs)
        b, a = _sps.butter(4, float(cutoff_hz) / nyq, btype="low")
        y = _sps.filtfilt(b, a, x, padlen=min(len(x) - 1, 3 * max(len(a), len(b))))
        return float(np.std(y))
    except Exception:  # noqa: BLE001
        return 0.0


def compute_powerline_power(
    x: np.ndarray, fs: float, target_hz: float = 50.0, half_bw_hz: float = 2.0
) -> float:
    """Średnia gęstość spektralna w paśmie ``target_hz ± half_bw_hz`` (proxy dla zakłócenia sieciowego).

    Z założenia używana parami (raw/cleaned), żeby pokazać redukcję po notch
    filter w Stage 2. Wartość bezwzględna jest jednostkowo zależna od skali
    sygnału — to porównanie raw→cleaned ma sens, nie sama liczba.
    """
    freqs, psd = compute_spectrum_welch(x, fs, max_freq_hz=max(target_hz + 2 * half_bw_hz, 80.0))
    if freqs.size == 0:
        return 0.0
    band = (freqs >= target_hz - half_bw_hz) & (freqs <= target_hz + half_bw_hz)
    if not np.any(band):
        return 0.0
    return float(np.mean(psd[band]))


def signal_quality_series(
    cleaned: np.ndarray, peaks_idx: np.ndarray, fs: float
) -> np.ndarray | None:
    """Per-sample NK2 `ecg_quality` (0..1, 1 = best). None gdy NK2 niedostępny.

    Stage 4 z dokumentu (SQI) — porównuje każde QRS do szablonu zbudowanego
    z większości uderzeń. Zwracana seria ma długość `cleaned`; może zawierać
    NaN tam, gdzie NK2 nie był w stanie policzyć jakości.
    """
    if not NK_AVAILABLE or peaks_idx.size < 4 or cleaned.size < int(2 * fs):
        return None
    try:
        q = nk.ecg_quality(cleaned, rpeaks=peaks_idx, sampling_rate=float(fs))
        return np.asarray(q, dtype=float)
    except Exception:  # noqa: BLE001
        return None


def signal_quality(
    cleaned: np.ndarray, peaks_idx: np.ndarray, fs: float
) -> float | None:
    """Mean NK2 `ecg_quality` over the recording (0..1, 1 = best)."""
    q = signal_quality_series(cleaned, peaks_idx, fs)
    if q is None:
        return None
    q = q[np.isfinite(q)]
    return float(np.mean(q)) if q.size else None


def compute_ecg_qc_report(
    time_s: np.ndarray,
    ecg_raw: np.ndarray,
    fs_hz: float,
    opt: EcgQcOptions,
    peaks_idx: np.ndarray | None = None,
    n_corrections: int = 0,
    mean_quality: float | None = None,
    n_manual_added: int = 0,
    n_manual_removed: int = 0,
    quality_series: np.ndarray | None = None,
    peaks_uncorrected: np.ndarray | None = None,
    fix_indices_uncorr: np.ndarray | None = None,
) -> EcgQcReport:
    """Aggregate QC over the recording given already-detected (and possibly edited) peaks.

    Parametry opcjonalne (Stage 4–6 z dokumentu referencyjnego):
    - ``quality_series``: per-sample seria NK2 ``ecg_quality`` (długość jak ``ecg_raw``);
      jeżeli podana, raport oblicza średni SQI w oknie.
    - ``peaks_uncorrected``: piki R **przed** korekcją RR; konieczne do liczenia
      udziału korekt LT19 per-okno.
    - ``fix_indices_uncorr``: zbiorczy zestaw indeksów piku korygowanego (typy
      ``ectopic | missed | extra | longshort``) wskazujący do ``peaks_uncorrected``.
    """
    notes: list[str] = []
    t = np.asarray(time_s, dtype=float).ravel()
    x = np.asarray(ecg_raw, dtype=float).ravel()
    n = min(len(t), len(x))
    t, x = t[:n], x[:n]
    duration_min = (float(t[-1] - t[0]) / 60.0) if n > 1 else 0.0

    nan_frac = float(np.mean(~np.isfinite(x))) if n else 1.0
    x_clean = _impute_finite(x)

    st_all = float(np.std(x_clean)) if x_clean.size else 0.0
    amp = float(np.max(x_clean) - np.min(x_clean)) if n else 0.0
    flat = amp <= 1e-15 or (st_all / (amp + 1e-15)) < opt.flat_rel_std_max

    amax = float(np.max(np.abs(x_clean))) if n else 0.0
    thr_clip = amax * opt.clip_margin_ratio if amax > 0 else 0.0
    clip_frac = float(np.mean(np.abs(x_clean) >= thr_clip)) if thr_clip > 0 else 0.0

    if peaks_idx is None:
        peaks_idx = np.array([], dtype=int)
    n_peaks = int(peaks_idx.size)

    rr_ms = np.array([])
    if n_peaks >= 2:
        dt = np.diff(t[peaks_idx]) * 1000.0
        rr_ms = dt[(dt > 0) & np.isfinite(dt)]

    median_rr: float | None = float(np.median(rr_ms)) if rr_ms.size else None
    mean_hr = (
        (60000.0 / median_rr) if median_rr is not None and median_rr > 0 else None
    )

    if rr_ms.size:
        bad = (rr_ms < opt.rr_min_ms) | (rr_ms > opt.rr_max_ms)
        rr_out = float(np.mean(bad))
    else:
        rr_out = 1.0

    win = max(float(opt.window_sec), 1.0)
    n_win = 0
    n_ok = 0
    seg_list: list[EcgWindowSegment] = []
    peaks_idx_arr = np.asarray(peaks_idx, dtype=int).ravel()
    peak_times = t[peaks_idx_arr] if peaks_idx_arr.size else np.array([], dtype=float)

    q_series = (
        np.asarray(quality_series, dtype=float).ravel()
        if quality_series is not None
        else None
    )
    if q_series is not None and q_series.size > n:
        q_series = q_series[:n]

    peaks_uncorr_arr = (
        np.asarray(peaks_uncorrected, dtype=int).ravel()
        if peaks_uncorrected is not None
        else None
    )
    fix_idx_arr = (
        np.asarray(fix_indices_uncorr, dtype=int).ravel()
        if fix_indices_uncorr is not None
        else None
    )
    fix_times: np.ndarray = np.array([], dtype=float)
    peaks_uncorr_times: np.ndarray = np.array([], dtype=float)
    if peaks_uncorr_arr is not None and peaks_uncorr_arr.size:
        valid = (peaks_uncorr_arr >= 0) & (peaks_uncorr_arr < n)
        peaks_uncorr_arr = peaks_uncorr_arr[valid]
        peaks_uncorr_times = t[peaks_uncorr_arr] if peaks_uncorr_arr.size else fix_times
        if fix_idx_arr is not None and fix_idx_arr.size and peaks_uncorr_arr.size:
            in_range = (fix_idx_arr >= 0) & (fix_idx_arr < peaks_uncorr_arr.size)
            fix_local = fix_idx_arr[in_range]
            if fix_local.size:
                fix_sample_idx = peaks_uncorr_arr[fix_local]
                fix_sample_idx = fix_sample_idx[
                    (fix_sample_idx >= 0) & (fix_sample_idx < n)
                ]
                fix_times = t[fix_sample_idx] if fix_sample_idx.size else fix_times

    n_corr_in_total = (
        int(n_corrections) if n_corrections else int(fix_times.size)
    )
    corrected_frac_global = (
        (n_corr_in_total / n_peaks) if n_peaks else 0.0
    )

    if n > 1 and duration_min * 60 >= win * 0.5:
        t0s, t1s = float(t[0]), float(t[-1])
        span = max(t1s - t0s, 1e-9)
        n_seg = max(1, int(np.ceil(span / win)))
        edges = np.linspace(t0s, t1s, n_seg + 1)
        for wi in range(len(edges) - 1):
            a, b = float(edges[wi]), float(edges[wi + 1])
            m = (t >= a) & (t < b)
            if not np.any(m):
                continue
            seg = x_clean[m]
            n_win += 1
            nf = float(np.mean(~np.isfinite(seg)))
            sd = float(np.std(seg)) if seg.size else 0.0
            amp_w = float(np.max(seg) - np.min(seg)) if seg.size else 0.0
            flat_w = amp_w <= 1e-12 or (sd / (amp_w + 1e-15)) < opt.flat_rel_std_max

            reasons: list[str] = []
            ok = True
            if nf >= opt.nan_bad_ratio:
                ok = False
                reasons.append(
                    f"Wysoki udział NaN w oknie ({nf * 100:.1f} %, próg {opt.nan_bad_ratio * 100:.0f} %)"
                )
            if flat_w:
                ok = False
                reasons.append(
                    "Sygnał w oknie wygląda na płaski (std względem amplitudy poniżej progu)"
                )

            idx_in = peaks_idx_arr[(peak_times >= a) & (peak_times < b)]
            idx_in = idx_in[np.argsort(t[idx_in])]
            n_pw = int(idx_in.size)
            rr_win_bad: float | None = None
            if n_pw >= 2:
                dtm = np.diff(t[idx_in]) * 1000.0
                dtm = dtm[(dtm > 0) & np.isfinite(dtm)]
                if dtm.size:
                    rr_win_bad = float(
                        np.mean((dtm < opt.rr_min_ms) | (dtm > opt.rr_max_ms))
                    )
                    if rr_win_bad > 0.35:
                        ok = False
                        reasons.append(
                            f"Dużo odstępów RR poza progami w tym oknie ({rr_win_bad * 100:.0f} %; "
                            f"progi {opt.rr_min_ms:.0f}–{opt.rr_max_ms:.0f} ms)"
                        )
            elif (b - a) >= 15.0:
                ok = False
                reasons.append(
                    f"Za mało wykrytych R w oknie ({n_pw}; potrzebne ≥2 do oceny RR)"
                )

            mean_q_w: float | None = None
            if q_series is not None:
                q_seg = q_series[m]
                q_seg = q_seg[np.isfinite(q_seg)]
                if q_seg.size:
                    mean_q_w = float(np.mean(q_seg))
                    if mean_q_w < opt.sqi_min_window:
                        ok = False
                        reasons.append(
                            f"Niska jakość sygnału w oknie (NK2 SQI {mean_q_w:.2f} "
                            f"< próg {opt.sqi_min_window:.2f}; "
                            "morfologia QRS odbiega od szablonu)"
                        )

            corr_w: float | None = None
            if peaks_uncorr_times.size:
                n_uncorr_w = int(
                    np.sum((peaks_uncorr_times >= a) & (peaks_uncorr_times < b))
                )
                if n_uncorr_w > 0:
                    n_fix_w = int(
                        np.sum((fix_times >= a) & (fix_times < b))
                    )
                    corr_w = n_fix_w / n_uncorr_w
                    if corr_w > opt.corrected_frac_max:
                        ok = False
                        reasons.append(
                            f"Wysoki udział korekt RR (LT19) w oknie "
                            f"({corr_w * 100:.1f} % > próg "
                            f"{opt.corrected_frac_max * 100:.0f} %); rozważ "
                            "wykluczenie segmentu z HRV"
                        )

            if ok:
                n_ok += 1

            seg_list.append(
                EcgWindowSegment(
                    t_start_s=a,
                    t_end_s=b,
                    ok=ok,
                    reasons_pl=tuple(reasons),
                    nan_fraction=nf,
                    flat_window=flat_w,
                    n_peaks_in_window=n_pw,
                    rr_outside_frac_window=rr_win_bad,
                    mean_quality_window=mean_q_w,
                    corrected_frac_window=corr_w,
                )
            )
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
    if mean_quality is not None and mean_quality < 0.5:
        bad_reasons.append("low_quality")
    if corrected_frac_global > opt.corrected_frac_max:
        bad_reasons.append("high_corrected_frac")

    if (
        not bad_reasons
        and nan_frac < opt.nan_warn_ratio
        and rr_out < 0.1
        and win_frac >= 0.7
        and (mean_quality is None or mean_quality >= 0.7)
    ):
        label = "dobry"
    elif len(bad_reasons) >= 2 or nan_frac >= opt.nan_bad_ratio or flat:
        label = "slaby"
    else:
        label = "ostroznie"

    if NK_AVAILABLE:
        notes.append(
            f"NeuroKit2 v{NK_VERSION}: clean='{opt.clean_method}', "
            f"peaks='{opt.peak_method}', correct_artifacts={opt.correct_artifacts}."
        )
    else:
        notes.append(
            "NeuroKit2 nie jest zainstalowany — `py -3 -m pip install neurokit2`."
        )
    if n_corr_in_total:
        notes.append(
            "Algorytm korekcji artefaktów RR (Lipponen-Tarvainen 2019, 'Kubios' "
            f"w NK2) zmodyfikował {n_corr_in_total} pików "
            f"({corrected_frac_global * 100:.2f} % wszystkich R-peaków; "
            f"próg segmentowy {opt.corrected_frac_max * 100:.0f} %)."
        )
    if n_manual_added or n_manual_removed:
        notes.append(
            f"Ręczna edycja: +{n_manual_added} dodanych, "
            f"-{n_manual_removed} usuniętych R-peaków."
        )

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
        rr_ms_list=[float(v) for v in rr_ms.tolist()] if rr_ms.size else [],
        window_sec=opt.window_sec,
        n_windows=n_win,
        n_windows_ok=n_ok,
        window_ok_fraction=win_frac,
        window_segments=tuple(seg_list),
        mean_quality=mean_quality,
        n_corrections=n_corr_in_total,
        corrected_frac=float(corrected_frac_global),
        n_manual_added=n_manual_added,
        n_manual_removed=n_manual_removed,
        overall_label=label,
        notes=notes,
    )


def compute_hrv_metrics(
    peaks_idx: np.ndarray, fs: float, psd_method: str = "welch"
) -> pd.DataFrame:
    """One-row DataFrame with NK2 HRV metrics (time + frequency when possible).

    `psd_method` steruje estymacją widma dla metryk częstotliwościowych
    (welch | burg | lomb). Returns empty DataFrame on too-short recordings /
    NK2 unavailable.
    """
    if not NK_AVAILABLE or peaks_idx.size < 4 or fs <= 0:
        return pd.DataFrame()
    out: dict[str, float] = {}
    try:
        ht = nk.hrv_time(peaks_idx, sampling_rate=float(fs))
        for c in ht.columns:
            v = ht.iloc[0][c]
            out[c] = float(v) if pd.notna(v) else float("nan")
    except Exception:  # noqa: BLE001
        pass
    try:
        hf = nk.hrv_frequency(
            peaks_idx,
            sampling_rate=float(fs),
            show=False,
            psd_method=str(psd_method),
        )
        for c in hf.columns:
            v = hf.iloc[0][c]
            out[c] = float(v) if pd.notna(v) else float("nan")
    except Exception:  # noqa: BLE001
        pass
    try:
        # Metryki nieliniowe (Poincaré SD1/SD2, DFA alpha1/alpha2) — głównie dla
        # nagrań dłuższych; dla krótkich okien część zwróci NaN (to oczekiwane).
        hn = nk.hrv_nonlinear(peaks_idx, sampling_rate=float(fs))
        for c in hn.columns:
            v = hn.iloc[0][c]
            out[c] = float(v) if pd.notna(v) else float("nan")
    except Exception:  # noqa: BLE001
        pass
    if not out:
        return pd.DataFrame()
    return pd.DataFrame([out])


def preprocess_visible(ecg: np.ndarray, fs: float, opt: EcgQcOptions) -> np.ndarray:
    """Backward-compatible alias used in older callers — returns cleaned signal."""
    return clean_ecg(ecg, fs, opt)
