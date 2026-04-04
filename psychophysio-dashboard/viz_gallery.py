"""
Warianty wizualizacji pod PsyPhy Datalab — do porównań i rozmów z promotorką.
Dane: syntetyczne lub przetworzone DataFrame z kolumnami time_s, oddech, puls_bpm, ecg_mv, eda_us.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import signal as scipy_signal

N_SEG = 6


def _seg_layout(df: pd.DataFrame) -> tuple[float, float, int]:
    session_s = max(float(df["time_s"].max()), 1.0)
    seg_len = session_s / N_SEG
    return session_s, seg_len, N_SEG


def _with_segment(df: pd.DataFrame) -> pd.DataFrame:
    session_s, seg_len, n_seg = _seg_layout(df)
    d = df.copy()
    d["segment"] = (d["time_s"] // seg_len).astype(np.int64).clip(0, n_seg - 1)
    return d


def fig_segment_bar_summary(df: pd.DataFrame) -> go.Figure:
    """Średnie (±SD) HR i EDA w każdym z 6 segmentów — porównanie bloków eksperymentalnych."""
    d = _with_segment(df)
    g = d.groupby("segment", sort=True).agg(
        hr=("puls_bpm", "mean"),
        hr_sd=("puls_bpm", "std"),
        eda=("eda_us", "mean"),
        eda_sd=("eda_us", "std"),
    )
    seg_labels = [f"Seg. {i + 1}" for i in g.index]

    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Średni puls ± SD (bpm)", "Średnie EDA ± SD (µS)"),
        vertical_spacing=0.14,
    )
    fig.add_trace(
        go.Bar(
            x=seg_labels,
            y=g["hr"],
            error_y=dict(type="data", array=g["hr_sd"], visible=True),
            marker_color="#ff7f0e",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=seg_labels,
            y=g["eda"],
            error_y=dict(type="data", array=g["eda_sd"], visible=True),
            marker_color="#d62728",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_layout(
        title_text="Podsumowanie po segmentach (średnia ± SD w bloku 10 min)",
        height=520,
        margin=dict(t=80),
    )
    return fig


def fig_histograms_hr_eda(df: pd.DataFrame) -> go.Figure:
    """Rozkłady wartości HR i EDA w całej sesji (jakie stany dominują)."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Rozkład pulsu (bpm)", "Rozkład EDA (µS)"))
    fig.add_trace(go.Histogram(x=df["puls_bpm"], nbinsx=50, name="puls", marker_color="#ff7f0e"), row=1, col=1)
    fig.add_trace(go.Histogram(x=df["eda_us"], nbinsx=50, name="EDA", marker_color="#d62728"), row=1, col=2)
    fig.update_layout(height=380, showlegend=False, title_text="Histogramy (cała sesja)")
    return fig


def fig_correlation_heatmap(df: pd.DataFrame, max_points: int = 8000) -> go.Figure:
    """Macierz korelacji Pearsona między kanałami (po ewentualnym rzadszym próbkowaniu)."""
    d = df[["oddech", "puls_bpm", "ecg_mv", "eda_us"]].copy()
    if len(d) > max_points:
        d = d.iloc[:: len(d) // max_points].reset_index(drop=True)
    c = d.corr()
    labels = ["oddech", "puls", "ECG", "EDA"]
    fig = go.Figure(
        data=go.Heatmap(
            z=c.values,
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale="RdBu",
            reversescale=True,
            text=np.round(c.values, 2),
            texttemplate="%{text}",
            colorbar=dict(title="r"),
        )
    )
    fig.update_layout(title="Korelacje między kanałami (Pearson)", height=400)
    return fig


def fig_spectrogram(
    df: pd.DataFrame,
    column: str,
    fs_hz: float,
    title: str,
) -> go.Figure:
    """Spektrogram (STFT) — struktura częstotliwościowa w czasie; tu na oddechu lub ECG."""
    x = df["time_s"].values
    y = df[column].values
    if len(y) < 256:
        y = np.pad(y, (0, 256 - len(y)))
    nperseg = min(512, max(128, len(y) // 40))
    f, t_stft, Sxx = scipy_signal.spectrogram(y, fs=fs_hz, nperseg=nperseg, noverlap=nperseg // 2)
    # ogranicz pasmo do sensownego (np. 0–2 Hz dla oddechu)
    fmax = min(2.5, f.max())
    mask = f <= fmax
    f = f[mask]
    Sxx = Sxx[mask, :]
    t_abs = t_stft + float(x[0]) if len(x) else t_stft

    fig = go.Figure(
        data=go.Heatmap(
            z=10 * np.log10(Sxx + 1e-12),
            x=t_abs,
            y=f,
            colorscale="Viridis",
            colorbar=dict(title="dB"),
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Czas (s)",
        yaxis_title="Częstotliwość (Hz)",
        height=420,
    )
    return fig


def fig_rolling_hr_variability(df: pd.DataFrame, window_s: float = 60.0) -> go.Figure:
    """Zmienność pulsu w oknach przesuwnych (proxy zmienności HR / „HRV-like” na poziomie trendu)."""
    d = df.sort_values("time_s").reset_index(drop=True)
    dt = np.diff(d["time_s"].values)
    med_dt = float(np.median(dt)) if len(dt) else 1.0
    win = max(5, int(window_s / max(med_dt, 0.01)))
    roll_std = d["puls_bpm"].rolling(window=win, center=True, min_periods=win // 2).std()
    fig = go.Figure(
        go.Scatter(
            x=d["time_s"],
            y=roll_std,
            mode="lines",
            line=dict(width=1, color="#9467bd"),
            name="SD pulsu",
        )
    )
    fig.update_layout(
        title=f"Odchylenie standardowe pulsu w oknie ~{window_s:.0f} s (rolling)",
        xaxis_title="Czas (s)",
        yaxis_title="SD (bpm)",
        height=360,
    )
    session_s, _, _ = _seg_layout(df)
    fig.update_xaxes(range=(0, session_s))
    return fig


def fig_eda_tonic_phasic(df: pd.DataFrame, tonic_window_s: float = 60.0) -> go.Figure:
    """Rozkład EDA na wolnozmienną (toniczną) i szybszą (fazyczną) — uproszczony model."""
    d = df.sort_values("time_s").reset_index(drop=True)
    dt = np.diff(d["time_s"].values)
    med_dt = float(np.median(dt)) if len(dt) else 1.0
    w = max(7, int(tonic_window_s / max(med_dt, 0.01)))
    tonic = d["eda_us"].rolling(window=w, center=True, min_periods=1).mean()
    phasic = d["eda_us"] - tonic
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("EDA — składowa wolna (toniczna, ~LP)", "EDA — składowa szybsza (fazyczna, reszta)"),
    )
    fig.add_trace(go.Scatter(x=d["time_s"], y=tonic, line=dict(width=1), name="toniczna"), row=1, col=1)
    fig.add_trace(go.Scatter(x=d["time_s"], y=phasic, line=dict(width=0.8), name="fazyczna"), row=2, col=1)
    fig.update_layout(height=520, showlegend=False, title_text="EDA: prosty podział tonic / phasic (demo)")
    session_s, _, _ = _seg_layout(df)
    fig.update_xaxes(range=(0, session_s), row=2, col=1)
    return fig


def fig_scatter_hr_eda_timecolor(df: pd.DataFrame, max_points: int = 5000) -> go.Figure:
    """Puls vs EDA — kolor = czas (czy trajektoria „wędruje” po przestrzeni stanów)."""
    d = df
    if len(d) > max_points:
        d = d.iloc[:: len(d) // max_points].reset_index(drop=True)
    fig = go.Figure(
        go.Scatter(
            x=d["puls_bpm"],
            y=d["eda_us"],
            mode="markers",
            marker=dict(size=4, color=d["time_s"], colorscale="Turbo", showscale=True, colorbar=dict(title="s")),
        )
    )
    fig.update_layout(
        title="Puls vs EDA (kolor = czas od startu sesji)",
        xaxis_title="Puls (bpm)",
        yaxis_title="EDA (µS)",
        height=440,
    )
    return fig


def fig_box_by_segment(df: pd.DataFrame, metric: str) -> go.Figure:
    """Wykres pudełkowy jednej zmiennej w podziale na segmenty."""
    _, _, n_seg = _seg_layout(df)
    d = _with_segment(df)
    labels = {i: f"Seg.{i + 1}" for i in range(n_seg)}
    d["lab"] = d["segment"].map(labels)
    fig = go.Figure()
    for lab in [f"Seg.{i + 1}" for i in range(n_seg)]:
        sub = d[d["lab"] == lab][metric]
        fig.add_trace(go.Box(y=sub, name=lab, boxmean="sd"))
    ttl = {"puls_bpm": "Puls (bpm)", "eda_us": "EDA (µS)", "oddech": "Oddech (a.u.)"}[metric]
    fig.update_layout(title=f"Rozstrzelenie wartości: {ttl} · wg segmentu", height=420)
    return fig


def fig_parallel_coords(df: pd.DataFrame, max_rows: int = 1500) -> go.Figure:
    """Współrzędne równoległe — wielowymiarowy „profil” sesji (po min-max na kolumnach)."""
    d = df[["time_s", "oddech", "puls_bpm", "ecg_mv", "eda_us"]].copy()
    if len(d) > max_rows:
        d = d.iloc[:: len(d) // max_rows].reset_index(drop=True)
    for c in ["oddech", "puls_bpm", "ecg_mv", "eda_us"]:
        lo, hi = d[c].min(), d[c].max()
        if hi - lo > 1e-12:
            d[c] = (d[c] - lo) / (hi - lo)
        else:
            d[c] = 0.5
    fig = go.Figure(
        data=go.Parcoords(
            line=dict(color=d["time_s"], colorscale="Viridis", showscale=True, colorbar=dict(title="czas (s)")),
            dimensions=[
                dict(label="oddech", values=d["oddech"]),
                dict(label="puls", values=d["puls_bpm"]),
                dict(label="ECG", values=d["ecg_mv"]),
                dict(label="EDA", values=d["eda_us"]),
            ],
        )
    )
    fig.update_layout(title="Współrzędne równoległe (znormalizowane 0–1, kolor = czas)", height=480)
    return fig


def fig_radar_segment_profile(df: pd.DataFrame) -> go.Figure:
    """Profil segmentu w przestrzeni z-score (średnie) — szybkie porównanie 6 bloków."""
    d = _with_segment(df)
    _, _, n_seg = _seg_layout(df)
    metrics = ["oddech", "puls_bpm", "eda_us"]
    agg = d.groupby("segment")[metrics].mean()
    for m in metrics:
        mu, sigma = agg[m].mean(), agg[m].std()
        sigma = max(sigma, 1e-9)
        agg[m] = (agg[m] - mu) / sigma
    categories = ["oddech", "puls", "EDA"]
    fig = go.Figure()
    for i in range(n_seg):
        vals = list(agg.loc[i, metrics].values) + [agg.loc[i, metrics[0]]]
        cats = categories + [categories[0]]
        fig.add_trace(
            go.Scatterpolar(r=vals, theta=cats, fill="toself", name=f"Seg. {i + 1}", opacity=0.55)
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[-2, 2])),
        title="Radar (z-score średnich w segmencie) — kształt bloku",
        height=520,
        showlegend=True,
    )
    return fig
