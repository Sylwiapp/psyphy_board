"""
PsyPhy Datalab — dashboard: oddech, puls, ECG, EDA + segmenty sesji + transkrypt z kursorem czasu.
Surowe vs przetworzone (jak w typowym pipeline: QC vs przegląd).

Uruchomienie: py -3 -m streamlit run app.py
"""

from __future__ import annotations

import html as html_module
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from transcript_io import Utterance, load_transcript_auto, load_transcript_json_bytes
from bv_markers import new_segment_times_seconds, parse_vmrk, resolve_vmrk_path
from data_loader import BrainVisionMeta, find_vhdr_files, load_brainvision_auxiliary
from calm_triggers import build_calm_analysis_geom, load_calm_trigger_times
from session_geom import SessionGeom, session_geom_equal_split, session_geom_from_marker_starts
from data_validation import format_report_body, validate_brainvision_dataframe, validate_transcript
from ecg_qc import (
    CLEAN_METHODS,
    NK_AVAILABLE,
    NK_VERSION,
    PEAK_METHODS,
    EcgQcOptions,
    apply_manual_edits,
    clean_ecg,
    compute_ecg_qc_report,
    compute_hrv_metrics,
    detect_r_peaks,
    estimate_fs_from_time,
    extract_ecg_series,
    parse_times_input,
    signal_quality,
)
import viz_gallery as vg

APP_NAME = "PsyPhy Datalab"

# Stałe trybów nawigacji (jedno źródło prawdy — radio + logika osi X)
NAV_FULL = "Pełna sesja"
NAV_WINDOW = "Okno wokół kursora"
NAV_SEGMENT = "Wybrany segment"

# --- domyślna syntetyczna sesja: 60 min, 6 segmentów ---
DEFAULT_SESSION_S = 60 * 60
N_SEG_DEFAULT = 6


# „Surowe”: wyższa częstotliwość (symulacja); „przetworzone”: wygładzenie + rzadsze próbki
RAW_FS_HZ = 25.0
DISPLAY_FS_HZ = 4.0

SEGMENT_COLORS = [
    "rgba(230,240,255,0.55)",
    "rgba(255,245,230,0.55)",
    "rgba(235,255,235,0.55)",
    "rgba(255,235,245,0.55)",
    "rgba(245,240,255,0.55)",
    "rgba(240,250,250,0.55)",
]


def make_physiology_raw(seed: int, duration_s: float = DEFAULT_SESSION_S) -> pd.DataFrame:
    """Syntetyczne sygnały ~surowe (wysoka fs)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * RAW_FS_HZ)
    t = np.arange(n) / RAW_FS_HZ
    # oddech
    breath = np.sin(2 * np.pi * 0.2 * t) + 0.12 * rng.standard_normal(n)
    # puls (bpm) — wolnozmienny + szum
    hr = 70 + 8 * np.sin(2 * np.pi * t / 400) + 1.5 * rng.standard_normal(n)
    hr = np.clip(hr, 52, 110)
    # ECG — uproszczony (syntetyczny): komponent sinusoidalny + szum (nie jest to realistyczny EKG diagnostyczny)
    phase = np.cumsum(hr / 60.0 / RAW_FS_HZ) * 2 * np.pi
    ecg = np.sin(phase) * 0.4 + 0.15 * np.sin(3 * phase) + 0.08 * rng.standard_normal(n)
    # EDA (µS)
    eda = 4.0 + 0.5 * np.sin(2 * np.pi * t / 500) + np.cumsum(0.015 * rng.standard_normal(n))
    eda += 0.2 * rng.standard_normal(n)

    return pd.DataFrame(
        {
            "time_s": t,
            "oddech": breath,
            "puls_bpm": hr,
            "ecg_mv": ecg,
            "eda_us": eda,
        }
    )


def to_display(df_raw: pd.DataFrame, raw_fs_hz: float = RAW_FS_HZ) -> pd.DataFrame:
    """
    Typowy „widok przeglądowy”: filtracja dolnoprzepustowa przez średnią kroczącą + decymacja.
    (W publikacjach często pokazuje się przetworzone; surowe zostaje do QC i zoomów.)
    """
    x = df_raw.sort_values("time_s").reset_index(drop=True)
    w = max(3, int(raw_fs_hz / DISPLAY_FS_HZ))
    y = x.copy()
    for col in ("oddech", "puls_bpm", "ecg_mv", "eda_us"):
        y[col] = y[col].rolling(window=w, center=True, min_periods=1).mean()
    step = max(1, int(round(raw_fs_hz / DISPLAY_FS_HZ)))
    return y.iloc[::step].reset_index(drop=True)


def make_synthetic_utterances(geom: SessionGeom) -> list[Utterance]:
    """Mowa głównie w segmentach 1, 3, 5 (indeks 0,2,4) — reszta cisza / krótkie markery."""
    rng = np.random.default_rng(7)
    utt: list[Utterance] = []
    filler = (
        "To jest syntetyczny fragment wypowiedzi do prototypu.",
        "Druga fraza w bloku mowy.",
        "Krótsza wypowiedź.",
        "Tu może być Twój transkrypt z alignmentem.",
    )
    for seg_i in range(geom.n_seg):
        base, seg_end = geom.segment_bounds(seg_i)
        sl = seg_end - base
        if seg_i in (1, 3, 5):
            # segmenty „bez pełnej mowy” — rzadkie krótkie linie
            for k in range(3):
                t0 = base + 60 + k * 180 + rng.uniform(0, 20)
                utt.append(Utterance(t0, t0 + 8 + rng.uniform(0, 5), "[cisza / instrukcja — podmień tekstem]"))
            continue
        # segmenty z mową — więcej wypowiedzi
        t_cur = base + 10.0
        while t_cur < base + sl - 25:
            dur = rng.uniform(4.0, 22.0)
            text = filler[int(rng.integers(0, len(filler)))]
            utt.append(Utterance(t_cur, t_cur + dur, text))
            t_cur += dur + rng.uniform(0.8, 4.0)
    utt.sort(key=lambda u: u.start_s)
    return utt


def add_segment_shading(fig: go.Figure, rows: int, geom: SessionGeom) -> None:
    for i in range(geom.n_seg):
        x0, x1 = geom.segment_bounds(i)
        for r in range(1, rows + 1):
            fig.add_vrect(
                x0=x0,
                x1=x1,
                fillcolor=SEGMENT_COLORS[i % len(SEGMENT_COLORS)],
                layer="below",
                line_width=0,
                row=r,
                col=1,
            )


def add_cursor_vline(fig: go.Figure, cursor_s: float, rows: int) -> None:
    for r in range(1, rows + 1):
        fig.add_vline(
            x=cursor_s,
            line_width=2,
            line_color="rgba(200,50,50,0.85)",
            row=r,
            col=1,
        )


def add_segment_shading_single(fig: go.Figure, geom: SessionGeom) -> None:
    """Pasma segmentów na jednym panelu (nakładka)."""
    for i in range(geom.n_seg):
        x0, x1 = geom.segment_bounds(i)
        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=SEGMENT_COLORS[i % len(SEGMENT_COLORS)],
            layer="below",
            line_width=0,
        )


def clamp_window(center: float, half_width: float, session_s: float) -> tuple[float, float]:
    """Okno symetryczne wokół center, obcięte do [0, session_s]."""
    start = center - half_width
    end = center + half_width
    if start < 0:
        end -= start
        start = 0.0
    if end > session_s:
        start -= end - session_s
        end = float(session_s)
        start = max(0.0, start)
    return (start, end)


def compute_view_x_range(
    mode: str,
    cursor_s: float,
    window_s: float,
    segment_index: int,
    geom: SessionGeom,
) -> tuple[float, float]:
    """
    Tryby nawigacji po ciągłym czasie:
    - Pełna sesja — cała oś 0…3600 s
    - Okno wokół kursora — środek okna = kursor (ustawiasz klikając punkt na wykresie Plotly)
    - Wybrany segment — zakres jednego bloku: CALM z logu triggerów, New Segment w .vmrk, albo równe bloki
    """
    if mode == NAV_FULL:
        return (0.0, float(geom.session_s))
    if mode == NAV_WINDOW:
        half = max(1.0, window_s / 2.0)
        return clamp_window(cursor_s, half, geom.session_s)
    if mode == NAV_SEGMENT:
        lo, hi = geom.segment_bounds(segment_index)
        return (float(lo), float(hi))
    return (0.0, float(geom.session_s))


def apply_x_range_stacked(fig: go.Figure, rows: int, x_range: tuple[float, float]) -> None:
    for r in range(1, rows + 1):
        fig.update_xaxes(range=list(x_range), row=r, col=1)


def apply_x_range_overlay(fig: go.Figure, x_range: tuple[float, float]) -> None:
    fig.update_xaxes(range=list(x_range))


def min_max_norm(series: pd.Series | np.ndarray) -> np.ndarray:
    x = np.asarray(series, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


# Plotly + Streamlit serializują cały wykres do JSON — pełne nagranie 1 kHz (~4.5M pkt) przekracza domyślny limit (~200 MB).
MAX_PLOT_POINTS_PER_TRACE = 20_000


def envelope_downsample_pair(
    t: np.ndarray,
    y: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Skrócenie szeregu do wykresu: koperty min/max w kubełkach czasu — zachowuje piki lepiej niż prosty stride."""
    t = np.asarray(t, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    n = min(int(t.size), int(y.size))
    if n <= 0:
        return t[:0], y[:0]
    t, y = t[:n], y[:n]
    if n <= max_points or max_points < 4:
        return t, y
    n_buckets = max(1, max_points // 2)
    edges = np.linspace(0, n, n_buckets + 1)
    t_list: list[float] = []
    y_list: list[float] = []
    for b in range(n_buckets):
        lo = int(edges[b])
        hi = int(edges[b + 1])
        if hi <= lo:
            hi = min(lo + 1, n)
        if lo >= n:
            break
        hi = min(hi, n)
        idx = np.arange(lo, hi, dtype=int)
        valid = np.isfinite(y[idx])
        if not np.any(valid):
            continue
        idx = idx[valid]
        ya = y[idx]
        i_min = int(idx[int(np.argmin(ya))])
        i_max = int(idx[int(np.argmax(ya))])
        if i_min <= i_max:
            seq = ((t[i_min], y[i_min]), (t[i_max], y[i_max]))
        else:
            seq = ((t[i_max], y[i_max]), (t[i_min], y[i_min]))
        for tp, yp in seq:
            t_list.append(float(tp))
            y_list.append(float(yp))
    return np.asarray(t_list, dtype=float), np.asarray(y_list, dtype=float)


def fig_overlay_normalized(
    df: pd.DataFrame,
    cursor_s: float,
    title_suffix: str,
    x_range: tuple[float, float],
    geom: SessionGeom,
) -> go.Figure:
    """Wszystkie kanały na jednym wykresie po min–max (cała sesja) — porównanie kształtu w czasie."""
    fig = go.Figure()
    add_segment_shading_single(fig, geom)
    series_meta = (
        ("oddech", "oddech", "#1f77b4"),
        ("puls_bpm", "puls (bpm)", "#ff7f0e"),
        ("ecg_mv", "ECG (mV)", "#2ca02c"),
        ("eda_us", "EDA (µS)", "#d62728"),
    )
    t_full = df["time_s"].to_numpy(dtype=float)
    for col, label, color in series_meta:
        y_norm = min_max_norm(df[col])
        tx, yp = envelope_downsample_pair(t_full, y_norm, MAX_PLOT_POINTS_PER_TRACE)
        fig.add_trace(
            go.Scatter(
                x=tx,
                y=yp,
                name=label,
                mode="lines",
                line=dict(width=1.1, color=color),
                opacity=0.9,
            )
        )
    fig.add_vline(x=cursor_s, line_width=2, line_color="rgba(200,50,50,0.9)")
    fig.update_layout(
        title=dict(
            text=f"Nakładka (min–max na całej sesji) — {title_suffix}",
            font=dict(size=15),
        ),
        height=500,
        xaxis_title="Czas od startu sesji (s)",
        yaxis_title="0–1 (znormalizowane)",
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(l=50, r=20, t=80, b=40),
        uirevision="psyphy_datalab",
    )
    apply_x_range_overlay(fig, x_range)
    fig.update_yaxes(range=(-0.02, 1.02))
    return fig


def fig_stacked(
    df: pd.DataFrame,
    cursor_s: float,
    title_suffix: str,
    x_range: tuple[float, float],
    geom: SessionGeom,
    subplot_titles: tuple[str, str, str, str] | None = None,
) -> go.Figure:
    rows = 4
    stitles = subplot_titles or ("Oddech (a.u.)", "Puls (bpm)", "ECG (synt., mV)", "EDA (µS)")
    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=stitles,
    )
    add_segment_shading(fig, rows, geom)
    t_full = df["time_s"].to_numpy(dtype=float)
    pairs = [
        (df["oddech"].to_numpy(dtype=float), 1, "oddech", dict(width=0.8)),
        (df["puls_bpm"].to_numpy(dtype=float), 2, "puls", dict(width=0.8)),
        (df["ecg_mv"].to_numpy(dtype=float), 3, "ECG", dict(width=0.6)),
        (df["eda_us"].to_numpy(dtype=float), 4, "EDA", dict(width=0.8)),
    ]
    for y_arr, row, name, line_kw in pairs:
        tx, yy = envelope_downsample_pair(t_full, y_arr, MAX_PLOT_POINTS_PER_TRACE)
        fig.add_trace(
            go.Scatter(x=tx, y=yy, name=name, line=line_kw),
            row=row,
            col=1,
        )
    add_cursor_vline(fig, cursor_s, rows)
    fig.update_layout(
        height=920,
        showlegend=False,
        margin=dict(l=50, r=20, t=48, b=36),
        title=dict(text=f"Sygnały — {title_suffix}", font=dict(size=15)),
        uirevision="psyphy_datalab",
    )
    fig.update_xaxes(title_text="Czas od startu sesji (s)", row=4, col=1)
    apply_x_range_stacked(fig, rows, x_range)
    return fig


def find_active_utterance(utterances: list[Utterance], cursor_s: float) -> Utterance | None:
    """Przedziały domknięte-otwarte [start, end)."""
    for u in utterances:
        if u.start_s <= cursor_s < u.end_s:
            return u
    return None


def _transcript_utterance_blocks(utterances: list[Utterance], cursor_s: float) -> tuple[str, int]:
    """HTML akapitów + indeks aktywnej wypowiedzi (-1 gdy kursor w przerwie)."""
    parts: list[str] = []
    active_idx = -1
    for i, u in enumerate(utterances):
        is_active = u.start_s <= cursor_s < u.end_s
        if is_active:
            active_idx = i
        bg = "rgba(255,230,200,0.95)" if is_active else "rgba(255,255,255,0.92)"
        border = "2px solid #c44" if is_active else "1px solid #ccc"
        safe = html_module.escape(u.text)
        parts.append(
            f'<p id="utt-{i}" style="margin:6px 0;padding:8px;background:{bg};border:{border};'
            f'border-radius:6px;font-size:14px;line-height:1.35">'
            f'<span style="color:#666;font-size:12px">[{u.start_s:.1f} – {u.end_s:.1f} s]</span><br/>{safe}</p>'
        )
    body = "\n".join(parts) if parts else "<p>(brak wypowiedzi)</p>"
    return body, active_idx


def transcript_iframe_html(utterances: list[Utterance], cursor_s: float) -> str:
    """
    Pełny dokument HTML w iframe: podświetlenie + przewinięcie do aktywnej wypowiedzi (scrollIntoView).
    Przy każdej zmianie kursora Streamlit podmienia cały HTML — iframe się przeładowuje.
    """
    body, active_idx = _transcript_utterance_blocks(utterances, cursor_s)
    scroll_js = ""
    if active_idx >= 0:
        scroll_js = f"""
<script>
document.addEventListener("DOMContentLoaded", function() {{
  var el = document.getElementById("utt-{active_idx}");
  if (el) el.scrollIntoView({{ block: "center", behavior: "auto" }});
}});
</script>
"""
    inner = (
        f'<div id="tr-scroll" style="height:452px;overflow-y:auto;padding:10px;background:#fafafa;">'
        f"{body}</div>"
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width"/></head>
<body style="margin:0;font-family:system-ui,Segoe UI,sans-serif;">
{inner}
{scroll_js}
</body></html>"""


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    st.title(APP_NAME)
    st.caption("Sesja z transkryptem i segmentami · prototyp open source")

#     st.markdown(
#         """
# **Dane:** tryb syntetyczny albo **BrainVision** (`.vhdr` + `.eeg` w folderze `data`) — kanały pomocnicze Resp / GSR / HR.  
# **Segmenty:** domyślnie **6** bloków o równej długości (długość sesji zależy od nagrania).  
# **Nawigacja:** pełna oś · **okno** wokół kursora · **jeden segment**.  
# **Nakładka (4 krzywe, min–max)** jest na górze zakładki Sesja; **klik** w punkt krzywej w Plotly ustawia kursor przy **kolejnym** odświeżeniu (standard Streamlit).  
# **Galeria:** histogramy, spektrogramy, korelacje, boxploty itd.  
# **Transkrypt:** JSON/CSV — podświetlenie wg czasu; przygotujemy import **kwestionariuszy** (CSV) osobno.

# **Surowe vs przetworzone:** przegląd vs QC — oba widoki są w zakładkach obok siebie.
#         """
#     )

    if "cursor_s" not in st.session_state:
        st.session_state.cursor_s = 120.0

    seed = st.sidebar.number_input("Ziarno RNG (sygnały syntetyczne)", value=42, step=1)
    data_dir = Path(__file__).resolve().parent / "data"

    st.sidebar.markdown("**Źródło sygnałów**")
    src = st.sidebar.radio(
        "Dane",
        ["Syntetyczne (demo)", "BrainVision (.vhdr + .eeg w folderze data)"],
        index=0,
    )
    raw_fs_hz = RAW_FS_HZ
    data_note = ""
    loaded_bv = False
    meta_bv: BrainVisionMeta | None = None
    pick_vhdr: Path | None = None
    if src.startswith("Syntetyczne"):
        df_raw = make_physiology_raw(int(seed))
    else:
        vhdrs = find_vhdr_files(data_dir)
        if not vhdrs:
            st.sidebar.warning("Brak pliku `.vhdr` w `data`. Używam danych syntetycznych.")
            df_raw = make_physiology_raw(int(seed))
        else:
            pick_vhdr = st.sidebar.selectbox("Nagranie", vhdrs, format_func=lambda p: p.name)
            df_bv, msg, meta = load_brainvision_auxiliary(pick_vhdr)
            if df_bv is None:
                st.error(msg)
                if meta is not None:
                    eeg_path = pick_vhdr.parent / meta.data_file
                    if eeg_path.is_file():
                        st.caption(f"Powiązany plik próbek: `{eeg_path.name}` — rozmiar na dysku: **{eeg_path.stat().st_size:,} B**.")
                    else:
                        st.caption(f"Oczekiwany plik próbek: `{eeg_path}` — nie znaleziono.")
                st.caption("Używam danych **syntetycznych** do dalszego podglądu aplikacji.")
                df_raw = make_physiology_raw(int(seed))
                data_note = msg
            else:
                df_raw = df_bv
                loaded_bv = True
                meta_bv = meta
                raw_fs_hz = float(meta.sampling_hz) if meta else RAW_FS_HZ
                st.success(msg.split("\n\n")[0])
                if "\n\n" in msg:
                    st.markdown(msg.split("\n\n", 1)[1])
                data_note = msg

    session_s = max(float(df_raw["time_s"].max()), 1.0)
    geom = session_geom_equal_split(session_s, N_SEG_DEFAULT)
    segment_layout_caption = (
        f"Segmenty: **{geom.n_seg}** równych bloków czasu (brak pliku `.vmrk` lub brak markerów **New Segment**)."
    )
    calm_from_log = False
    if loaded_bv and pick_vhdr is not None:
        tlogs = sorted(pick_vhdr.parent.glob("*triggers*.log"))
        chosen_trigger: Path | None = None
        if len(tlogs) == 1:
            chosen_trigger = tlogs[0]
        elif len(tlogs) > 1:
            chosen_trigger = st.sidebar.selectbox(
                "Log triggerów CALM (`*triggers*.log`)",
                options=tlogs,
                format_func=lambda p: p.name,
                key="calm_trigger_log_pick",
            )
        if chosen_trigger is not None:
            try:
                code_times = load_calm_trigger_times(chosen_trigger)
                calm_geom = build_calm_analysis_geom(session_s, code_times)
                if calm_geom is not None:
                    geom = calm_geom
                    calm_from_log = True
                    segment_layout_caption = (
                        f"Segmenty: **{geom.n_seg}** przedziałów analizy **CALM** z `{chosen_trigger.name}` "
                        f"(czas względem triggera **1**; pary kodów w `calm_triggers.py`)."
                    )
                else:
                    st.sidebar.warning(
                        f"W logu `{chosen_trigger.name}` brakuje kompletnych par kodów CALM — "
                        "używam **.vmrk** / **New Segment** albo równych bloków."
                    )
            except Exception as exc:  # noqa: BLE001 — komunikat dla użytkownika Streamlit
                st.sidebar.warning(
                    f"Nie wczytano logu triggerów `{chosen_trigger.name}`: {exc}"
                )

    if not calm_from_log and loaded_bv and meta_bv is not None and pick_vhdr is not None:
        vmrk_path = resolve_vmrk_path(pick_vhdr, meta_bv.marker_file)
        if vmrk_path is not None:
            markers = parse_vmrk(vmrk_path)
            if any(m.mk_type.casefold() == "new segment" for m in markers):
                starts_s = new_segment_times_seconds(markers, raw_fs_hz)
                geom_marker = session_geom_from_marker_starts(session_s, starts_s)
                if geom_marker.n_seg > 1:
                    geom = geom_marker
                    segment_layout_caption = (
                        f"Segmenty: **{geom.n_seg}** bloków wg markerów **New Segment** w `{vmrk_path.name}` "
                        f"(czas z pozycji próbek, Fs = {raw_fs_hz:.1f} Hz)."
                    )
                else:
                    geom = session_geom_equal_split(session_s, N_SEG_DEFAULT)
                    segment_layout_caption = (
                        f"Segmenty: **{geom.n_seg}** równych bloków — w `{vmrk_path.name}` markery **New Segment** "
                        f"dają tylko **jeden** odcinek całej sesji; użyto podziału jak przy danych **syntetycznych**."
                    )
    st.session_state.cursor_s = float(np.clip(st.session_state.cursor_s, 0.0, geom.session_s))

    df_disp = to_display(df_raw, raw_fs_hz)

    default_path = data_dir / "transcript.example.json"
    st.sidebar.markdown("**Transkrypt**")
    use_upload = st.sidebar.checkbox("Wczytaj transkrypt z pliku (.json / .csv)", value=False)
    utterances: list[Utterance]
    transcript_from_user_file = False
    if use_upload:
        up = st.sidebar.file_uploader("Plik JSON lub CSV", type=["json", "csv"])
        if up is not None:
            utterances = load_transcript_auto(up, up.name)
            transcript_from_user_file = True
        else:
            utterances = make_synthetic_utterances(geom)
            st.sidebar.info("Brak pliku — używam syntetycznego transkryptu.")
    else:
        utterances = make_synthetic_utterances(geom)

    merged_example_transcript = False
    if default_path.exists() and st.sidebar.checkbox("Dopisz przykład z `data/transcript.example.json`", value=False):
        raw = default_path.read_bytes()
        extra = load_transcript_json_bytes(raw)
        utterances = sorted(utterances + extra, key=lambda u: u.start_s)
        merged_example_transcript = True

    # --- walidacja jakości wczytanych plików (QC heurystyki, nie „certyfikat” laboratoryjny) ---
    show_validation = (loaded_bv and meta_bv is not None) or transcript_from_user_file or merged_example_transcript
    if show_validation:
        with st.expander("Walidacja jakości wczytanych danych", expanded=bool(transcript_from_user_file or loaded_bv)):
            st.caption(
                "Automatyczne sprawdzenia: kompletność, typowa Fs, braki NaN, „płaskie” kanały, "
                "spójność transkryptu z długością sesji. **Nie zastępują** opisu metody ani decyzji labu."
            )
            if loaded_bv and meta_bv is not None:
                rep_bv = validate_brainvision_dataframe(df_raw, meta_bv, session_s=geom.session_s)
                if rep_bv.has_error:
                    st.error(rep_bv.summary_line())
                elif rep_bv.has_warn:
                    st.warning(rep_bv.summary_line())
                else:
                    st.success(rep_bv.summary_line())
                st.markdown(format_report_body(rep_bv))
            if transcript_from_user_file or merged_example_transcript:
                rep_tr = validate_transcript(utterances, session_s=geom.session_s)
                if rep_tr.has_error:
                    st.error(rep_tr.summary_line())
                elif rep_tr.has_warn:
                    st.warning(rep_tr.summary_line())
                else:
                    st.success(rep_tr.summary_line())
                st.markdown(format_report_body(rep_tr))

    st.sidebar.markdown("---")
    st.sidebar.caption(segment_layout_caption)
    st.sidebar.markdown("**Nawigacja po czasie**")
    nav_mode = st.sidebar.radio(
        "Tryb widoku osi X",
        [NAV_FULL, NAV_WINDOW, NAV_SEGMENT],
        index=0,
        help=(
            "Pełna sesja — cała oś czasu (zoom/pan w Plotly). "
            "Okno — wycinek wokół kursora (ustawiasz **klikając** punkt na krzywej). "
            "Wybrany segment — jeden przedział: z logu **CALM** (`*triggers*.log`), "
            "markerów **New Segment** w `.vmrk` albo równy podział czasu."
        ),
    )
    window_s = 120.0
    segment_index = 0
    if nav_mode == NAV_WINDOW:
        window_s = float(
            st.sidebar.select_slider(
                "Szerokość okna (s)",
                options=[30.0, 60.0, 120.0, 180.0, 300.0, 600.0, 900.0],
                value=120.0,
            )
        )
        st.sidebar.caption("**Kliknij** punkt na krzywej (Plotly), aby ustawić kursor — okno jest wyśrodkowane na nim.")
    elif nav_mode == NAV_SEGMENT:
        seg_choice = st.sidebar.selectbox(
            "Segment",
            list(range(1, geom.n_seg + 1)),
            format_func=lambda num: (
                f"{num}/{geom.n_seg} · {geom.segment_label(num - 1)} · "
                f"{geom.segment_bounds(num - 1)[0]:.0f}–{geom.segment_bounds(num - 1)[1]:.0f} s"
            ),
        )
        segment_index = int(seg_choice) - 1

    st.sidebar.caption(
        "Pliki `Data-*.txt` w `data` to logi impedancji, nie szereg czasowy."
    )
    with st.sidebar.expander("Kwestionariusze (planowane)"):
        st.caption(
            "Dodamy wczytywanie CSV (np. style supresji) i łączenie z `subject_id` / sesją — "
            "na razie przygotuj pliki i ustal z promotorką kolumny."
        )

    tab_session, tab_gallery, tab_ecg_qc = st.tabs(
        ["Sesja i transkrypt", "Galeria wizualizacji (warianty)", "QC / preprocessing — ECG"]
    )

    with tab_session:
        _render_session_tab(
            df_raw=df_raw,
            df_disp=df_disp,
            utterances=utterances,
            geom=geom,
            nav_mode=nav_mode,
            window_s=window_s,
            segment_index=segment_index,
            raw_fs_hz=raw_fs_hz,
            data_note=data_note,
            loaded_bv=loaded_bv,
        )

    with tab_gallery:
        _render_gallery_tab(df_raw, df_disp, raw_fs_hz, geom)

    with tab_ecg_qc:
        ecg_src = (
            str(pick_vhdr.resolve())
            if loaded_bv and pick_vhdr is not None
            else f"synth_{seed}"
        )
        _render_ecg_qc_tab(df_raw, df_disp, raw_fs_hz, ecg_data_source_key=ecg_src)


ECG_PRESETS: dict[str, dict[str, object]] = {
    "NeuroKit (default, HRV-friendly)": {
        "clean": "neurokit",
        "peak": "neurokit",
        "correct": True,
        "powerline": 50.0,
        "desc": (
            "Domyślny pipeline NK2: HP 0.5 Hz + powerline notch, detekcja R z NK2 "
            "(`neurokit`), korekcja artefaktów RR metodą Lipponen–Tarvainen 2019."
        ),
    },
    "Pan-Tompkins 1985 (klasyczna detekcja QRS)": {
        "clean": "pantompkins1985",
        "peak": "pantompkins1985",
        "correct": True,
        "powerline": 50.0,
        "desc": "BP 5–15 Hz, klasyczny detektor R z 1985 r. Korekcja RR włączona.",
    },
    "Hamilton 2002": {
        "clean": "hamilton2002",
        "peak": "hamilton2002",
        "correct": True,
        "powerline": 50.0,
        "desc": "Modyfikacja Pan-Tompkins z adaptacyjnymi progami.",
    },
    "Elgendi 2010 (szybki / mobilny)": {
        "clean": "elgendi2010",
        "peak": "elgendi2010",
        "correct": True,
        "powerline": 50.0,
        "desc": "Dwa kroczące okna; lekki obliczeniowo, dobry dla wearables.",
    },
}

ECG_DEFAULT_PRESET = "NeuroKit (default, HRV-friendly)"


def _apply_ecg_preset() -> None:
    """Callback: po zmianie presetu prefilluje wartości widgetów preprocessingu."""
    name = st.session_state.get("ecg_qc_preset", ECG_DEFAULT_PRESET)
    if name not in ECG_PRESETS:
        return
    p = ECG_PRESETS[name]
    st.session_state["ecg_qc_clean_method"] = str(p["clean"])
    st.session_state["ecg_qc_peak_method"] = str(p["peak"])
    st.session_state["ecg_qc_fix"] = bool(p["correct"])
    st.session_state["ecg_qc_powerline"] = float(p["powerline"])


def _render_ecg_qc_tab(
    df_raw: pd.DataFrame,
    df_disp: pd.DataFrame,
    raw_fs_hz: float,
    ecg_data_source_key: str = "demo",
) -> None:
    """Zakładka: preprocess + QC toru ECG (`ecg_mv`) z użyciem NeuroKit2."""
    st.subheader("QC / preprocessing — ECG (`ecg_mv`)")
    st.caption(
        "Tor w aplikacji jako **ecg_mv** (w BV często drugi pas oddechu — sprawdź nagłówek). "
        "Cały pipeline opiera się na **NeuroKit2** (czyszczenie, detekcja R, korekcja artefaktów "
        "Lipponen–Tarvainen 2019). Wyniki służą eksploracji i przygotowaniu danych — opis "
        "metody w pracy układaj wg Quigley 2024 / Laborde 2017."
    )

    if not NK_AVAILABLE:
        st.error(
            "NeuroKit2 nie jest zainstalowany — uruchom `py -3 -m pip install neurokit2`, "
            "a następnie odśwież aplikację. Bez NK2 detekcja R nie zadziała."
        )
    else:
        st.caption(f"NeuroKit2 v{NK_VERSION} · pip pakiet `neurokit2`.")

    st.caption(
        "Analiza zawsze działa na **surowym** sygnale (`df_raw`) — wersja przetworzona w innych "
        "zakładkach to decymacja do ~4 Hz, która jest poniżej Nyquista dla QRS i czyni detekcję "
        "R niemożliwą. Quigley 2024 zaleca **≥ 250 Hz** (najlepiej 1000 Hz)."
    )

    t_all, x_all = extract_ecg_series(df_raw)
    if t_all.size == 0:
        st.warning("Brak kolumny `ecg_mv` w danych.")
        return

    fs_est = estimate_fs_from_time(t_all)
    # Dla BV (Fs z nagłówka zwykle ≥250 Hz) zawsze ufamy nagłówkowi — błędna mediana kroków
    # w time_s (np. po imporcie) nie może zejść z 1000 Hz na 25 Hz i psuć całego NK2.
    if raw_fs_hz >= 250:
        fs_default = float(raw_fs_hz)
        if fs_est > 0 and abs(fs_est - raw_fs_hz) / max(raw_fs_hz, 1.0) > 0.15:
            st.warning(
                f"**Rozbieżność Fs:** z kolumny `time_s` wychodzi ok. **{fs_est:.0f} Hz**, "
                f"a nagłówek / meta nagrania to **{raw_fs_hz:.0f} Hz**. "
                "Pole **Fs** domyślnie ustawione jest na **wartość z nagłówka** (wymagane dla "
                "NeuroKit przy pełnej liczbie próbek). Jeśli ten plik jest świadomie przedecymowany "
                "do innej Fs, zmień wartość ręcznie."
            )
    else:
        fs_default = float(
            raw_fs_hz
            if abs(fs_est - raw_fs_hz) / max(raw_fs_hz, 0.1) < 0.2
            else fs_est
        )

    fs_use = float(
        st.number_input(
            "Fs użyte w obliczeniach (Hz)",
            min_value=0.5,
            max_value=5000.0,
            value=fs_default,
            step=0.5,
            help=(
                "Dla BrainVision domyślnie **Fs z nagłówka** (.vhdr). **Quigley et al. 2024** zaleca "
                "**1000 Hz** (±1 ms dokładności R). Ustawienie Fs **niższego** niż rzeczywiste "
                "próbkowanie psuje filtry i detekcję R w NeuroKit, a RR z osi `time_s` może "
                "wyglądać spójnie — nie daj się zwieść."
            ),
            key=f"ecg_qc_fs__{ecg_data_source_key}",
        )
    )

    for k, default in (
        ("ecg_qc_preset", ECG_DEFAULT_PRESET),
        ("ecg_qc_clean_method", str(ECG_PRESETS[ECG_DEFAULT_PRESET]["clean"])),
        ("ecg_qc_peak_method", str(ECG_PRESETS[ECG_DEFAULT_PRESET]["peak"])),
        ("ecg_qc_fix", bool(ECG_PRESETS[ECG_DEFAULT_PRESET]["correct"])),
        ("ecg_qc_powerline", float(ECG_PRESETS[ECG_DEFAULT_PRESET]["powerline"])),
    ):
        st.session_state.setdefault(k, default)

    st.selectbox(
        "Preset preprocessingu",
        options=list(ECG_PRESETS.keys()),
        key="ecg_qc_preset",
        on_change=_apply_ecg_preset,
        help=(
            "Wybór presetu **wpisuje** wartości w pola poniżej. Możesz potem zmienić "
            "dowolne pole ręcznie — preset nie blokuje nadpisywania."
        ),
    )
    preset_now = ECG_PRESETS[st.session_state["ecg_qc_preset"]]
    st.caption(f"**{st.session_state['ecg_qc_preset']}** — {preset_now['desc']}")

    with st.expander("Opcje preprocessingu (NeuroKit2)", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            st.selectbox(
                "Metoda czyszczenia (`nk.ecg_clean`)",
                options=list(CLEAN_METHODS),
                key="ecg_qc_clean_method",
                help=(
                    "`neurokit`: HP 0.5 Hz + powerline notch (default NK2). "
                    "`pantompkins1985`: BP 5–15 Hz. `hamilton2002`, `elgendi2010`: "
                    "warianty filtrów + integratorów. `biosppy`: pipeline BioSPPy. "
                    "`none`: brak czyszczenia."
                ),
            )
        with c2:
            st.selectbox(
                "Metoda detekcji R (`nk.ecg_peaks`)",
                options=list(PEAK_METHODS),
                key="ecg_qc_peak_method",
                help=(
                    "`neurokit` (default), klasyczne: `pantompkins1985`, `hamilton2002`, "
                    "`elgendi2010`, `engzeemod2012`, `kalidas2017` (deep-learning), "
                    "`rodrigues2021`. `promac` — agregat wielu detektorów."
                ),
            )

        c3, c4 = st.columns(2)
        with c3:
            st.checkbox(
                "Korekcja artefaktów RR (Lipponen–Tarvainen 2019, „Kubios” w NK2)",
                key="ecg_qc_fix",
                help=(
                    "Klasyfikator i poprawki RR: ectopic / missed / extra / longshort. "
                    "Zalecane przy nagraniach z ruchem; wyłącz, gdy chcesz „surowe” piki NK2."
                ),
            )
        with c4:
            st.radio(
                "Powerline notch (tylko `clean = neurokit`)",
                options=[50.0, 60.0],
                horizontal=True,
                key="ecg_qc_powerline",
                help="Częstotliwość sieci: 50 Hz w UE/PL, 60 Hz w US.",
            )

    with st.expander("Progi akceptacji RR i okien czasu"):
        c4, c5 = st.columns(2)
        with c4:
            rr_lo = st.number_input(
                "RR min (ms)", 200.0, 800.0, 300.0, 10.0,
                key="ecg_qc_rrlo",
                help="HR_max ≈ 200 bpm → 300 ms; w protokołach z wysiłkiem rozważ niższy próg.",
            )
            rr_hi = st.number_input(
                "RR max (ms)", 800.0, 3000.0, 2000.0, 50.0,
                key="ecg_qc_rrhi",
                help="HR_min ≈ 30 bpm → 2000 ms; w spoczynku zwykle <1500 ms.",
            )
        with c5:
            win_sec = st.number_input("Długość okna lokalnego (s)", 10.0, 300.0, 60.0, 10.0, key="ecg_qc_win")
            flat_rel = float(
                st.select_slider(
                    "Próg „płaskiego” sygnału (std / amplituda)",
                    options=[1e-8, 5e-8, 1e-7, 5e-7, 1e-6, 5e-6, 1e-5, 5e-5, 1e-4],
                    value=1e-5,
                    format_func=lambda v: f"{v:.0e}",
                    key="ecg_qc_flat",
                )
            )

    rr_a, rr_b = float(rr_lo), float(rr_hi)
    if rr_b <= rr_a:
        rr_b = rr_a + 50.0

    opt = EcgQcOptions(
        clean_method=str(st.session_state["ecg_qc_clean_method"]),
        peak_method=str(st.session_state["ecg_qc_peak_method"]),
        correct_artifacts=bool(st.session_state["ecg_qc_fix"]),
        powerline_hz=float(st.session_state["ecg_qc_powerline"]),
        rr_min_ms=rr_a,
        rr_max_ms=rr_b,
        window_sec=float(win_sec),
        flat_rel_std_max=float(flat_rel),
    )

    with st.spinner("NeuroKit2: czyszczenie ECG i detekcja R…"):
        x_filt = clean_ecg(x_all, fs_use, opt)
        rp = detect_r_peaks(x_filt, fs_use, opt)
        quality = signal_quality(x_filt, rp.peaks, fs_use)

    peaks_auto = rp.peaks

    if st.session_state.get("ecg_manual_clear_pending"):
        st.session_state["ecg_manual_add_text"] = ""
        st.session_state["ecg_manual_remove_text"] = ""
        st.session_state["ecg_manual_clear_pending"] = False

    with st.expander(
        "Ręczna edycja R-peaków (zaawansowane)",
        expanded=bool(
            st.session_state.get("ecg_manual_add_text")
            or st.session_state.get("ecg_manual_remove_text")
        ),
    ):
        st.caption(
            "Znajdź problematyczny pik w **„Podgląd sygnału i R-peaków”** poniżej (najedź "
            "kursorem — Plotly pokaże czas w sekundach), a następnie wpisz tu jego czas. "
            "Usuwanie snapuje do najbliższego wykrytego piku w tolerancji **±50 ms** "
            "(Quigley 2024 zaleca preferować wizualną korektę nad samym automatem)."
        )
        cma, cmb, cmc = st.columns([2, 2, 1])
        with cma:
            add_text = st.text_area(
                "Dodaj R przy (s)",
                placeholder="np. 12.34, 13.92, 18.01",
                key="ecg_manual_add_text",
                height=80,
            )
        with cmb:
            rm_text = st.text_area(
                "Usuń R przy (s)",
                placeholder="np. 7.50, 21.10",
                key="ecg_manual_remove_text",
                height=80,
            )
        with cmc:
            st.write(" ")
            if st.button("Wyczyść edycje", key="ecg_manual_clear"):
                st.session_state["ecg_manual_clear_pending"] = True
                st.rerun()

    add_times = parse_times_input(st.session_state.get("ecg_manual_add_text", ""))
    rm_times = parse_times_input(st.session_state.get("ecg_manual_remove_text", ""))
    peaks, n_added, n_removed = apply_manual_edits(
        peaks_auto, fs_use, add_times, rm_times
    )
    if n_added or n_removed:
        st.info(
            f"Ręczna edycja zastosowana: **+{n_added}** dodanych, "
            f"**-{n_removed}** usuniętych względem auto-detekcji."
        )

    rep = compute_ecg_qc_report(
        t_all,
        x_all,
        fs_use,
        opt,
        peaks_idx=peaks,
        n_corrections=rp.n_corrections,
        mean_quality=quality,
        n_manual_added=n_added,
        n_manual_removed=n_removed,
    )

    # Spodziewana rzędu wielkości liczba R przy HR 30–200 bpm (konservatywnie 25 bpm).
    min_peaks_expected = max(30, int(rep.duration_min * 25))
    if rep.n_peaks < min_peaks_expected:
        st.error(
            f"Wykryto tylko **{rep.n_peaks}** R-peaków w sesji **~{rep.duration_min:.1f} min** — "
            f"dla sensownego ECG oczekuje się co najmniej ok. **{min_peaks_expected}+** pików "
            f"(HR ok. 25–200 bpm). To **nie** jest problem domyślnego presetu NeuroKit, notch "
            "50/60 Hz ani progów RR 300–2000 ms: algorytm prawie nie widzi QRS w tym torze. "
            "Sprawdź w BrainVision, czy **`ecg_mv` to rzeczywiście ECG** (często drugi pas to oddech), "
            "ew. inny kanał w `.vhdr`."
        )

    st.markdown("---")
    st.markdown("#### Podsumowanie jakości")
    if rep.overall_label == "dobry":
        st.success("**Ocena ogólna: dobry** — większość kryteriów spełniona.")
    elif rep.overall_label == "slaby":
        st.error("**Ocena ogólna: słaby** — warto poprawić zapis lub parametry.")
    elif rep.overall_label == "ostroznie":
        st.warning(
            "**Ocena ogólna: ostrożnie** — sprawdź szczegóły i ewentualnie fragment ręcznie."
        )
    else:
        st.warning(f"**Ocena:** {rep.overall_label}")

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("Długość", f"{rep.duration_min:.1f} min")
    c_m2.metric("Próbek", f"{rep.n_samples:,}")
    c_m3.metric("NaN", f"{rep.nan_fraction * 100:.2f} %")
    c_m4.metric(
        "Okna OK",
        f"{rep.window_ok_fraction * 100:.0f} % ({rep.n_windows_ok}/{rep.n_windows})",
    )

    c_m5, c_m6, c_m7, c_m8 = st.columns(4)
    c_m5.metric("Wykryte R (po edycji)", str(rep.n_peaks))
    c_m6.metric(
        "Mediana RR (ms)",
        f"{rep.median_rr_ms:.0f}" if rep.median_rr_ms else "—",
    )
    c_m7.metric(
        "HR z med. RR (bpm)",
        f"{rep.mean_hr_bpm:.1f}" if rep.mean_hr_bpm else "—",
    )
    c_m8.metric("RR poza progiem", f"{rep.rr_outside_frac * 100:.1f} %")

    c_m9, c_m10, c_m11, c_m12 = st.columns(4)
    c_m9.metric(
        "Jakość NK2 (0–1)",
        f"{rep.mean_quality:.2f}" if rep.mean_quality is not None else "—",
        help="Średnia z `nk.ecg_quality` (1 = wzorzec szablonu QRS).",
    )
    c_m10.metric("Korekty RR (auto)", str(rp.n_corrections))
    c_m11.metric("Dodane ręcznie", str(rep.n_manual_added))
    c_m12.metric("Usunięte ręcznie", str(rep.n_manual_removed))

    st.caption(
        f"**Clipping:** {rep.clip_fraction * 100:.3f} % próbek blisko |max|. "
        f"**Płaski (całość):** {'tak' if rep.flat_signal else 'nie'}. "
        f"**Fs w raporcie:** {rep.fs_hz:.2f} Hz."
    )
    for note in rep.notes:
        st.caption(note)

    st.markdown("#### Jak dużo nagrania jest „do użycia” wg aplikacji?")
    st.caption(
        "**Ocena ogólna** i **RR poza progiem** patrzą na całą sesję naraz. **Okna OK** "
        "dzielą nagranie na kawałki po długości z pola „Długość okna lokalnego” i w każdym "
        "sprawdzają: NaN, czy sygnał nie jest płaski oraz czy RR w tym fragmencie nie wygląda "
        "jak lawina błędów. To bliższe odpowiedzi na pytanie „które minuty odrzucić”."
    )
    if rep.n_windows > 0:
        usable_pct = rep.window_ok_fraction * 100.0
        st.info(
            f"**Orientacyjnie możesz traktować jako „nadające się” ~{usable_pct:.0f} %** czasu "
            f"trwania nagrania (**{rep.n_windows_ok}** z **{rep.n_windows}** okien po "
            f"**{rep.window_sec:.0f} s**). Reszta — do ręcznej weryfikacji albo wykluczenia z HRV. "
            "To nie zastępuje decyzji metodologicznej w pracy."
        )
    else:
        st.warning(
            "Nie policzono mapy okien (sesja krótsza niż połowa wybranego okna albo brak danych). "
            "Skróć „Długość okna lokalnego” albo sprawdź Fs i kolumnę `ecg_mv`."
        )

    st.markdown("#### Mapa jakości w czasie")
    st.caption(
        "**Zielony** = okno spełnia kryteria lokalne. **Szary** = co najmniej jeden problem "
        "(powód w tabeli poniżej). Te same szare pasy są nałożone na **Podgląd sygnału** dalej."
    )
    if rep.window_segments:
        t_min = float(t_all[0])
        t_max = float(t_all[-1])
        fig_map = go.Figure()
        for ws in rep.window_segments:
            fillcolor = (
                "rgba(45, 180, 90, 0.38)" if ws.ok else "rgba(95, 95, 95, 0.48)"
            )
            fig_map.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=ws.t_start_s,
                x1=ws.t_end_s,
                y0=0.0,
                y1=1.0,
                fillcolor=fillcolor,
                line_width=0,
                layer="below",
            )
        fig_map.update_xaxes(
            title_text="Czas (s)",
            range=[t_min, t_max],
            constrain="range",
        )
        fig_map.update_yaxes(visible=False, range=[0, 1])
        fig_map.update_layout(
            height=110,
            margin=dict(l=50, r=20, t=28, b=45),
            title=dict(
                text="Zielone = OK · Szare = problem (szczegóły w tabeli)",
                font=dict(size=13),
            ),
            showlegend=False,
        )
        st.plotly_chart(fig_map, use_container_width=True, key="ecg_qc_fig_qmap")

        tbl = []
        for ws in rep.window_segments:
            tbl.append(
                {
                    "od (s)": round(ws.t_start_s, 1),
                    "do (s)": round(ws.t_end_s, 1),
                    "OK": "tak" if ws.ok else "nie",
                    "powód (heurystyka)": " · ".join(ws.reasons_pl)
                    if ws.reasons_pl
                    else "—",
                    "NaN %": round(ws.nan_fraction * 100, 2),
                    "płaski": "tak" if ws.flat_window else "nie",
                    "R w oknie": ws.n_peaks_in_window,
                    "RR poza % (w oknie)": (
                        "—"
                        if ws.rr_outside_frac_window is None
                        else f"{ws.rr_outside_frac_window * 100:.1f}"
                    ),
                }
            )
        st.dataframe(
            pd.DataFrame(tbl),
            use_container_width=True,
            height=min(420, 36 + 28 * len(tbl)),
        )
    else:
        st.caption("Brak segmentów do mapy — patrz wyżej.")

    if rp.n_corrections > 0 or rp.peaks_uncorrected.size != rp.peaks.size:
        with st.expander(
            f"Co zrobił automat (korekcja RR — Lipponen–Tarvainen 2019): "
            f"{rp.n_corrections} zmian",
            expanded=False,
        ):
            st.caption(
                "Typy korekt z algorytmu **Kubios / LT19**. Indeksy odnoszą się do "
                "**oryginalnej** (nieskorygowanej) listy R-peaków:\n"
                "- **ectopic** — uderzenie ektopowe: pozycja R została poprawiona,\n"
                "- **missed** — automat dodał brakujący R,\n"
                "- **extra** — automat usunął zbędny R,\n"
                "- **longshort** — para „długi-krótki” odstęp RR poprawiona."
            )
            n_ect = int(rp.fixes_ectopic.size)
            n_mis = int(rp.fixes_missed.size)
            n_xtr = int(rp.fixes_extra.size)
            n_ls = int(rp.fixes_longshort.size)
            cc1, cc2, cc3, cc4 = st.columns(4)
            cc1.metric("Ectopic", str(n_ect))
            cc2.metric("Missed", str(n_mis))
            cc3.metric("Extra", str(n_xtr))
            cc4.metric("Long-short", str(n_ls))

            rows: list[dict[str, object]] = []
            t_arr = np.asarray(t_all, dtype=float)
            uncorr = rp.peaks_uncorrected
            for label, arr in (
                ("ectopic", rp.fixes_ectopic),
                ("missed", rp.fixes_missed),
                ("extra", rp.fixes_extra),
                ("longshort", rp.fixes_longshort),
            ):
                for i in arr.tolist():
                    if 0 <= int(i) < uncorr.size:
                        s_idx = int(uncorr[int(i)])
                        if 0 <= s_idx < t_arr.size:
                            rows.append(
                                {
                                    "typ": label,
                                    "indeks R (uncorr.)": int(i),
                                    "czas (s)": float(t_arr[s_idx]),
                                }
                            )
            if rows:
                df_fix = pd.DataFrame(rows).sort_values("czas (s)").reset_index(
                    drop=True
                )
                df_fix["czas (s)"] = df_fix["czas (s)"].round(2)
                st.dataframe(df_fix, use_container_width=True, height=240)
                st.caption(
                    "Skopiuj „czas (s)” do pól ręcznej edycji powyżej, jeśli chcesz "
                    "ten konkretny pik dodatkowo skorygować lub usunąć."
                )

    with st.expander("Metryki HRV (NeuroKit2)", expanded=False):
        if not NK_AVAILABLE:
            st.info("NeuroKit2 niedostępny.")
        elif rep.n_peaks < 4:
            st.info("Za mało pików (< 4) — dodaj/popraw R i odśwież.")
        else:
            with st.spinner("Liczę HRV (czas + częstotliwość)…"):
                hrv_df = compute_hrv_metrics(peaks, fs_use)
            if hrv_df.empty:
                st.info("NK2 nie zwrócił metryk (nagranie zbyt krótkie?).")
            else:
                main_cols = [
                    c
                    for c in (
                        "HRV_MeanNN",
                        "HRV_SDNN",
                        "HRV_RMSSD",
                        "HRV_pNN50",
                        "HRV_LF",
                        "HRV_HF",
                        "HRV_LFHF",
                    )
                    if c in hrv_df.columns
                ]
                if main_cols:
                    st.dataframe(
                        hrv_df[main_cols].T.rename(columns={0: "wartość"}),
                        use_container_width=True,
                    )
                st.caption("Pełen zestaw (NK2 `hrv_time` + `hrv_frequency`):")
                st.dataframe(
                    hrv_df.T.rename(columns={0: "wartość"}),
                    use_container_width=True,
                )
                st.caption(
                    "Laborde 2017: do tonu wagalnego preferować **RMSSD** lub **HF**; "
                    "unikać `LF/HF` jako miary „balansu”. Pasma: HF 0.15–0.40 Hz, "
                    "LF 0.04–0.15 Hz, VLF 0.0033–0.04 Hz (Task Force 1996)."
                )

    st.markdown("#### Podgląd sygnału i R-peaków")
    duration_total = float(t_all[-1] - t_all[0]) if t_all.size > 1 else 1.0
    max_preview = min(120.0, max(3.0, duration_total))
    c_pa, c_pb = st.columns([2, 1])
    with c_pa:
        preview_s = float(
            st.slider(
                "Szerokość okna (s)",
                3.0,
                float(max_preview),
                float(min(15.0, max_preview)),
                1.0,
                key="ecg_qc_preview",
            )
        )
    with c_pb:
        start_s = float(
            st.number_input(
                "Start okna (s)",
                0.0,
                max(0.0, duration_total - 1.0),
                0.0,
                1.0,
                key="ecg_qc_preview_start",
                help="Przesuwaj okno po sesji, aby zlokalizować błędne R-peaki.",
            )
        )
    t0 = float(t_all[0]) + start_s
    t1 = t0 + preview_s
    m = (t_all >= t0) & (t_all <= t1)
    t_sub, raw_sub, filt_sub = t_all[m], x_all[m], x_filt[m]

    st.caption(
        "Górny panel — **surowy** sygnał z BrainVision (DC offset i drift widoczne). "
        f"Dolny panel — sygnał **po `nk.ecg_clean('{opt.clean_method}')`**: "
        "filtr i usunięcie wędrowania linii bazowej, dzięki czemu R-peaki są dobrze "
        "widoczne i szczyty mają porównywalną amplitudę. Detekcja R wykonywana jest "
        "na sygnale dolnego panelu."
    )

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Surowy ECG", f"Po NK2 ({opt.clean_method})"),
    )
    fig.add_trace(
        go.Scatter(
            x=t_sub,
            y=raw_sub,
            name="Surowy",
            line=dict(width=0.8, color="#888"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=t_sub,
            y=filt_sub,
            name="Po NK2",
            line=dict(width=1.1, color="#1f77b4"),
        ),
        row=2,
        col=1,
    )

    uncorr = rp.peaks_uncorrected
    if uncorr.size:
        m_unc = (t_all[uncorr] >= t0) & (t_all[uncorr] <= t1)
        pk_u = uncorr[m_unc]
        if pk_u.size:
            fig.add_trace(
                go.Scatter(
                    x=t_all[pk_u],
                    y=x_filt[pk_u],
                    mode="markers",
                    name=f"R przed korekcją ({opt.peak_method})",
                    marker=dict(size=7, color="#9aa", symbol="circle-open"),
                    hovertemplate="t=%{x:.3f}s<br>amp=%{y:.3g}<extra></extra>",
                ),
                row=2,
                col=1,
            )
    if peaks_auto.size and rp.n_corrections > 0:
        m_a = (t_all[peaks_auto] >= t0) & (t_all[peaks_auto] <= t1)
        pk_a = peaks_auto[m_a]
        if pk_a.size:
            fig.add_trace(
                go.Scatter(
                    x=t_all[pk_a],
                    y=x_filt[pk_a],
                    mode="markers",
                    name="R po korekcji RR (LT19)",
                    marker=dict(size=9, color="red", symbol="x"),
                    hovertemplate="t=%{x:.3f}s<br>amp=%{y:.3g}<extra></extra>",
                ),
                row=2,
                col=1,
            )
    elif peaks_auto.size:
        m_a = (t_all[peaks_auto] >= t0) & (t_all[peaks_auto] <= t1)
        pk_a = peaks_auto[m_a]
        if pk_a.size:
            fig.add_trace(
                go.Scatter(
                    x=t_all[pk_a],
                    y=x_filt[pk_a],
                    mode="markers",
                    name=f"R auto ({opt.peak_method})",
                    marker=dict(size=8, color="red", symbol="x"),
                    hovertemplate="t=%{x:.3f}s<br>amp=%{y:.3g}<extra></extra>",
                ),
                row=2,
                col=1,
            )
    if peaks.size and (n_added or n_removed):
        m_f = (t_all[peaks] >= t0) & (t_all[peaks] <= t1)
        pk_f = peaks[m_f]
        if pk_f.size:
            fig.add_trace(
                go.Scatter(
                    x=t_all[pk_f],
                    y=x_filt[pk_f],
                    mode="markers",
                    name="R po ręcznej edycji",
                    marker=dict(size=11, color="#2ca02c", symbol="circle-open"),
                    hovertemplate="t=%{x:.3f}s<br>amp=%{y:.3g}<extra></extra>",
                ),
                row=2,
                col=1,
            )

    for ws in rep.window_segments:
        if ws.ok:
            continue
        lo = max(ws.t_start_s, t0)
        hi = min(ws.t_end_s, t1)
        if hi <= lo:
            continue
        for row in (1, 2):
            fig.add_vrect(
                x0=lo,
                x1=hi,
                fillcolor="rgba(90, 90, 90, 0.18)",
                line_width=0,
                layer="below",
                row=row,
                col=1,
            )

    fig.update_layout(
        title=f"ECG · okno {start_s:.0f}–{start_s + preview_s:.0f} s",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(title_text="Surowy (jedn. z nagłówka)", row=1, col=1)
    fig.update_yaxes(title_text="Po NK2", row=2, col=1)
    fig.update_xaxes(title_text="Czas (s)", row=2, col=1)
    st.plotly_chart(fig, use_container_width=True, key="ecg_qc_fig_ts")
    if rep.window_segments and any(not ws.ok for ws in rep.window_segments):
        st.caption(
            "**Szare pasy** — fragmenty, które w mapie jakości (okna po "
            f"**{rep.window_sec:.0f} s**) dostały status „nie OK”; powody są w tabeli pod mapą."
        )

    if peaks.size >= 2:
        peak_times_s = t_all[peaks].astype(float)
        rr_ms = np.diff(peak_times_s) * 1000.0
        rr_times = peak_times_s[1:]
        out_mask = (rr_ms < opt.rr_min_ms) | (rr_ms > opt.rr_max_ms)

        st.markdown("#### Tachogram (RR w czasie)")
        st.caption(
            "Każdy punkt = jeden odstęp R-R w milisekundach, ustawiony w czasie "
            "**drugiego** uderzenia. Pomaga zlokalizować artefakty czasowo (nie tylko "
            "policzyć je): pojedyncze „skoki” to zwykle błędne piki, plateau → realna "
            "zmiana HR (np. mowa, ruch). Czerwone punkty leżą poza progami "
            f"{opt.rr_min_ms:g}–{opt.rr_max_ms:g} ms."
        )
        fig_t = go.Figure()
        fig_t.add_trace(
            go.Scatter(
                x=rr_times,
                y=rr_ms,
                mode="lines+markers",
                name="RR (ms)",
                line=dict(width=1, color="#1f77b4"),
                marker=dict(size=4),
                hovertemplate=(
                    "t=%{x:.2f}s<br>RR=%{y:.0f} ms<br>HR=%{customdata:.1f} bpm<extra></extra>"
                ),
                customdata=60000.0 / np.where(rr_ms > 0, rr_ms, np.nan),
            )
        )
        if out_mask.any():
            fig_t.add_trace(
                go.Scatter(
                    x=rr_times[out_mask],
                    y=rr_ms[out_mask],
                    mode="markers",
                    name="Poza progiem",
                    marker=dict(size=9, color="red", symbol="x"),
                    hovertemplate="t=%{x:.2f}s<br>RR=%{y:.0f} ms<extra></extra>",
                )
            )
        fig_t.add_hline(
            y=opt.rr_min_ms, line_dash="dash", line_color="gray", opacity=0.6
        )
        fig_t.add_hline(
            y=opt.rr_max_ms, line_dash="dash", line_color="gray", opacity=0.6
        )
        fig_t.update_yaxes(title_text="RR (ms)", tickformat="d")
        fig_t.update_xaxes(title_text="Czas (s)")
        fig_t.update_layout(height=320, legend=dict(orientation="h", y=1.02))
        st.plotly_chart(fig_t, use_container_width=True, key="ecg_qc_fig_tacho")

        with st.expander("Histogram odstępów RR (rozkład)", expanded=False):
            st.caption(
                "Rozkład wszystkich odstępów R-R. Oś **X w milisekundach** "
                "(typowo 600–1000 ms dla 60–100 bpm). Bimodalność / długi ogon "
                "sugeruje artefakty albo dwa różne stany HR w sesji."
            )
            fig_h = go.Figure()
            fig_h.add_trace(
                go.Histogram(
                    x=rr_ms, nbinsx=40, name="RR (ms)", marker_color="#1f77b4"
                )
            )
            fig_h.add_vline(x=opt.rr_min_ms, line_dash="dash", line_color="gray")
            fig_h.add_vline(x=opt.rr_max_ms, line_dash="dash", line_color="gray")
            fig_h.update_xaxes(title_text="RR (ms)", tickformat="d")
            fig_h.update_yaxes(title_text="Liczba odstępów", tickformat="d")
            fig_h.update_layout(height=300, bargap=0.05)
            st.plotly_chart(fig_h, use_container_width=True, key="ecg_qc_fig_rr")

    with st.expander("Referencje (cytuj w metodzie)"):
        st.markdown(
            """
- **Makowski, D. et al. (2021).** *NeuroKit2: A Python toolbox for neurophysiological signal processing.* Behav. Res. Methods, 53, 1689–1696.
  [doi:10.3758/s13428-020-01516-y](https://doi.org/10.3758/s13428-020-01516-y) — biblioteka użyta jako engine (`ecg_clean`, `ecg_peaks`, `ecg_quality`, `hrv_time`, `hrv_frequency`).
- **Lipponen, J. A., & Tarvainen, M. P. (2019).** *A robust algorithm for HRV time series artefact correction.* J. Med. Eng. Technol., 43(3), 173–181.
  [doi:10.1080/03091902.2019.1640306](https://doi.org/10.1080/03091902.2019.1640306) — algorytm korekcji artefaktów RR (`correct_artifacts=True`, NK2 method `Kubios`).
- **Pan, J., & Tompkins, W. J. (1985).** *A real-time QRS detection algorithm.* IEEE TBME, 32(3), 230–236.
  [doi:10.1109/TBME.1985.325532](https://doi.org/10.1109/TBME.1985.325532) — preset `pantompkins1985` (BP 5–15 Hz + refractory 200 ms).
- **Hamilton, P. (2002).** *Open Source ECG Analysis Software.* Comput. Cardiol., 29, 101–104. — preset `hamilton2002`.
- **Elgendi, M. et al. (2010).** *Frequency Bands Effects on QRS Detection.* BIOSIGNALS — preset `elgendi2010`.
- **Quigley, K. S. et al. (2024).** *Publication guidelines for human heart rate and HRV studies in psychophysiology — Part 1.* Psychophysiology, 61, e14604.
  [doi:10.1111/psyp.14604](https://doi.org/10.1111/psyp.14604) — Fs 1000 Hz, korekcja > usuwanie epok, wizualna inspekcja R.
- **Laborde, S., Mosley, E., & Thayer, J. F. (2017).** *HRV and Cardiac Vagal Tone in Psychophysiological Research — Recommendations.* Frontiers in Psychology, 8, 213.
  [doi:10.3389/fpsyg.2017.00213](https://doi.org/10.3389/fpsyg.2017.00213) — RMSSD/HF preferowane nad LF/HF; 500–1000 Hz; struktura *3R*.
- **Task Force ESC/NASPE (1996).** *Heart rate variability: standards of measurement.* Circulation, 93(5), 1043–1065.
  [doi:10.1161/01.CIR.93.5.1043](https://doi.org/10.1161/01.CIR.93.5.1043) — pasma VLF/LF/HF, 5-min standard.
            """
        )

    st.markdown("#### Interpretacja dla dalszej analizy")
    usable = rep.window_ok_fraction * 100 if rep.n_windows else 0.0
    if rep.overall_label == "dobry":
        st.info(
            f"~**{usable:.0f} %** czasu w oknach {rep.window_sec:.0f} s spełnia kryteria. "
            "Pipeline NK2 + LT19 jest gotowy do dalszej analizy HRV — przed publikacją "
            "udokumentuj wersję NK2, metody i % korekt RR (Quintana et al. 2016)."
        )
    elif rep.overall_label == "slaby":
        st.warning(
            "Znaczna część nagrania może wymagać odrzucenia lub innego preprocessingu. "
            "Sprawdź elektrody, ruch, saturację; rozważ inny preset (np. `pantompkins1985` "
            "przy silnym wędrowaniu) lub Fs."
        )
    elif rep.overall_label == "ostroznie":
        st.info(
            "Część nagrania nadaje się, część — nie. Skorzystaj z okna podglądu (powyżej) "
            "i pól ręcznej edycji, aby poprawić R-peaki w problematycznych fragmentach."
        )


def _render_session_tab(
    df_raw: pd.DataFrame,
    df_disp: pd.DataFrame,
    utterances: list[Utterance],
    geom: SessionGeom,
    nav_mode: str,
    window_s: float,
    segment_index: int,
    raw_fs_hz: float,
    data_note: str,
    loaded_bv: bool,
) -> None:
    stitles = (
        ("Oddech (Resp)", "HR (µV)", "Oddech B", "EDA (GSR)")
        if loaded_bv
        else None
    )
    if data_note and not loaded_bv:
        st.warning(data_note)
    if loaded_bv and data_note:
        st.caption(data_note)

    cursor_s = float(st.session_state.cursor_s)

    x_range = compute_view_x_range(nav_mode, cursor_s, window_s, segment_index, geom)

    st.subheader("Nakładka: 4 kanały (min–max)")
    st.caption(
        "Wszystkie sygnały na **jednej** osi Y po normalizacji min–max (porównanie kształtu). "
        "Czas ustawiasz **suwakiem** bezpośrednio pod wykresem."
    )
    overlay_src = st.radio(
        "Źródło nakładki",
        ["Przetworzone", "Surowe"],
        horizontal=True,
        key="overlay_top_src",
    )
    df_ov = df_disp if overlay_src == "Przetworzone" else df_raw
    if len(df_ov) > MAX_PLOT_POINTS_PER_TRACE:
        st.caption(
            f"Nakładka: do wykresu użyto **do {MAX_PLOT_POINTS_PER_TRACE:,} punktów na tor** (koperta min/max w czasie)."
        )
    suf = "przetworzone" if overlay_src == "Przetworzone" else "surowe"
    hz = DISPLAY_FS_HZ if overlay_src == "Przetworzone" else raw_fs_hz
    st.plotly_chart(
        fig_overlay_normalized(df_ov, cursor_s, f"{suf} (~{hz:.0f} Hz)", x_range, geom),
        use_container_width=True,
    )

    st.caption("**Kursor czasu** — przesuń suwak, aby przesunąć czerwoną linię na wykresach i w transkrypcie.")
    cursor_s = st.slider(
        "Kursor czasu (s)",
        min_value=0.0,
        max_value=float(geom.session_s),
        value=float(st.session_state.cursor_s),
        step=1.0,
        help="Pionowa linia na osi czasu i podświetlenie transkryptu.",
        key="cursor_slider_main",
        label_visibility="collapsed",
    )
    st.session_state.cursor_s = float(cursor_s)

    x_range = compute_view_x_range(nav_mode, cursor_s, window_s, segment_index, geom)
    seg_hint = ""
    if nav_mode == NAV_SEGMENT:
        lo, hi = geom.segment_bounds(segment_index)
        seg_hint = f"· **{geom.segment_label(segment_index)}** ({lo:.0f}–{hi:.0f} s)"
    st.info(
        f"**Widok osi X:** {nav_mode} · **zakres:** {x_range[0]:.0f}–{x_range[1]:.0f} s · kursor **{cursor_s:.1f} s** "
        + (f"· okno {window_s:.0f} s" if nav_mode == NAV_WINDOW else seg_hint)
    )

    if geom.uses_disjoint_windows:
        legenda_seg = "paski = **przedziały analizy CALM** z logu triggerów (pary kodów w `calm_triggers.py`). "
    else:
        legenda_seg = "paski = kolejne segmenty (z **.vmrk** / **New Segment** albo równe bloki). "
    st.markdown(
        f"**Legenda tła:** {legenda_seg}"
        "**Czerwona linia** = kursor czasu."
    )

    col_plots, col_tr = st.columns([1.65, 1.0], gap="large")

    with col_plots:
        tab_raw, tab_filt = st.tabs(
            [
                "Surowe próbki",
                "Widok przetworzony",
            ]
        )

        with tab_raw:
            st.caption(
                f"Surowe dane ~{raw_fs_hz:.0f} Hz — użyj zoomu w Plotly. "
                + ("Kanały z BrainVision." if loaded_bv else "Syntetyczne ECG.")
            )
            if len(df_raw) > MAX_PLOT_POINTS_PER_TRACE:
                st.caption(
                    f"Wykres pokazuje **{MAX_PLOT_POINTS_PER_TRACE:,} punktów na kanał** (decymacja kopertą min/max), "
                    "żeby nie przekroczyć limitu rozmiaru wiadomości Streamlit — kształt sesji zostaje czytelny."
                )
            st.plotly_chart(
                fig_stacked(
                    df_raw,
                    cursor_s,
                    "surowe (~{:.0f} Hz)".format(raw_fs_hz),
                    x_range,
                    geom,
                    subplot_titles=stitles,
                ),
                use_container_width=True,
            )

        with tab_filt:
            st.caption(
                f"Wygładzenie + decymacja ~{DISPLAY_FS_HZ:.0f} Hz — przegląd całości."
            )
            st.plotly_chart(
                fig_stacked(
                    df_disp,
                    cursor_s,
                    "przetworzone (średnia krocząca + decymacja)",
                    x_range,
                    geom,
                    subplot_titles=stitles,
                ),
                use_container_width=True,
            )

    cursor_s = float(st.session_state.cursor_s)

    with col_tr:
        st.subheader("Transkrypt")
        st.caption(
            f"Kursor: **{cursor_s:.1f} s** — poniżej podświetlona jest wypowiedź obejmująca ten moment; "
            "lista **przewija się** do niej przy zmianie kursora."
        )
        active = find_active_utterance(utterances, cursor_s)
        if active:
            st.info(f"**{active.start_s:.1f} – {active.end_s:.1f} s** · {active.text}")
        else:
            st.caption(
                "W tym momencie nie ma wypowiedzi (przerwa między frazami albo cisza w segmencie). "
                "Przesuń suwak pod wykresem nakładki, aby wybrać inny moment."
            )
        components.html(transcript_iframe_html(utterances, cursor_s), height=480, scrolling=False)

    with st.expander("Format plików transkryptu (uniwersalny)"):
        st.markdown(
            """
**JSON** (zalecany): pole `utterances` z listą obiektów `start_s`, `end_s`, `text`.  
Zobacz plik `data/transcript.example.json`.

**CSV** (nagłówki, przecinek):  
`start_s,end_s,text`

Możesz też wygenerować ten sam układ z narzędzi do annotacji (ELAN eksport, skrypt z Whisper z word timestamps — mapowanie do `start_s`/`end_s`).
            """
        )

    with st.expander("Pytania do promotorki — dane, wykresy, surowe vs przetworzone"):
        st.markdown(
            """
Zanim dopniemy kolejne moduły (EEG, kwestionariusze, grupy), warto ustalić odpowiedzi na poniższe — możesz je skopiować do notatek lub maila.

#### Dane i pliki
- Skąd jest „prawda czasowa”: rekorder BrainVision, osobny logger, trigger z PC? Czy wszystkie pliki (fizjologia, EEG później, audio) da się **zsynchronizować jednym offsetem** w sekundach, czy potrzebne są osobne korekty per urządzenie?  
- Jakie pliki są **ostatecznym archiwum** sesji (format, nazewnictwo), a co jest tylko pośrednie?  
- Transkrypt: czy czasy mają być względem **startu sesji**, nagrania audio, czy pierwszego bodźka?  
- Gdy dojdzie **EEG**: ta sama oś czasu i ten sam punkt „zero” co dla oddechu / EDA / HR?

#### Surowe vs przetworzone
- **Co jest „surowe” w rozumieniu pracy:** surowy eksport z urządzenia bez filtrów, czy już minimalnie oczyszczone (np. usunięcie skoku na początku)?  
- Czy w pracy / na obronie pokazujemy **surowe próbki** (np. do wiarygodności, artefaktów), **tylko przetworzone** (filtry, wygładzenie), czy **obie wersje** z jasnym opisem metody?  
- Dla każdego kanału (oddech, EDA, sygnał HR itd.): czy są u Was **ustalone filtry** (pasma, notch 50 Hz, rozdzielczość do wyświetlania)?  
- Czy „puls” ma być **bpm wyliczone z sygnału**, czy wystarczy **sygnał z urządzenia** (np. µV) z podpisem osi?  
- Czy **jednostka na osi czasu** w figurach ma być sekundy, minuty, czy „czas względem triggera”?

#### Wykresy — co i po co
- Jaki jest **główny przekaz** wizualizacji: jedna osoba w czasie (z mową), **porównanie warunków** w sesji, czy **średnie po grupie**?  
- Które typy wykresów są **obowiązkowe** w Twojej dziedzinie (szereg czasowy, epoki wokół zdarzenia, histogram, korelacja, spektrogram…), a które tylko pomocnicze?  
- Czy wykresy mają być pod **publikację** (statyczne, jedna oś, czytelna legenda), czy głównie **do eksploracji** (interakcja, zoom)?

#### Segmentacja i warunki
- Czy podział na **równe odcinki czasu** (np. 6 bloków) ma sens eksperymentalny, czy lepiej **markery / zadania** z pliku (np. `.vmrk`)?  
- Czy porównujemy **bloki ze sobą** (np. mowa vs cisza), czy tylko **wewnątrz bloku** zmienność w czasie?

#### Grupy i kwestionariusze
- Czy dashboard ma najpierw **jednego uczestnika**, a agregacja po grupach **później**, czy od razu **średnie po grupach** (np. wg stylów supresji z kwestionariusza)?  
- Jakie zmienne z CSV (kwestionariusze) mają **wejść na wykres** (kolor, facet, filtr podgrupy)?

#### Formalnie
- Czy promotorka oczekuje **gotowych figur do rozdziału** (rozdzielczość, font), czy wystarczy opis metody + wykresy z narzędzia?
            """
        )


def _render_gallery_tab(
    df_raw: pd.DataFrame,
    df_disp: pd.DataFrame,
    raw_fs_hz: float,
    geom: SessionGeom,
) -> None:
    """Różne typy wykresów na tych samych danych — do dyskusji z promotorką."""
    st.markdown(
        """
### Galeria podglądów (bieżące źródło danych z zakładki „Sesja”)

Poniżej **różne reprezentacje** typowe w psychofizjologii / psychofizjolingwistyce: podsumowania bloków, rozkłady, 
spektrogram, uproszczony podział EDA, zmienność pulsu itd.
        """
    )
    ga, gb, gc = st.tabs(
        [
            "Segmenty i porównania",
            "Rozkłady, korelacje, przestrzeń stanów",
            "Częstotliwość, EDA, zmienność",
        ]
    )

    with ga:
        seg_src = (
            "**przedziały CALM** (log `*triggers*.log`)"
            if geom.uses_disjoint_windows
            else "markery **New Segment** albo równe bloki"
        )
        st.caption(
            f"Agregaty po **{geom.n_seg}** segmentach ({seg_src}) — "
            "porównanie **warunków** między blokami."
        )
        st.plotly_chart(vg.fig_segment_bar_summary(df_disp, geom), use_container_width=True, key="gal_bar")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(
                vg.fig_box_by_segment(df_disp, "puls_bpm", geom),
                use_container_width=True,
                key="gal_box_hr",
            )
        with c2:
            st.plotly_chart(
                vg.fig_box_by_segment(df_disp, "eda_us", geom),
                use_container_width=True,
                key="gal_box_eda",
            )
        st.plotly_chart(vg.fig_radar_segment_profile(df_disp, geom), use_container_width=True, key="gal_radar")

    with gb:
        st.caption("Rozkłady wartości, zależności między kanałami, wielowymiarowy profil (po normalizacji).")
        st.plotly_chart(vg.fig_histograms_hr_eda(df_disp), use_container_width=True, key="gal_hist")
        st.plotly_chart(vg.fig_correlation_heatmap(df_disp), use_container_width=True, key="gal_corr")
        st.plotly_chart(vg.fig_scatter_hr_eda_timecolor(df_disp), use_container_width=True, key="gal_scatter")
        st.plotly_chart(vg.fig_parallel_coords(df_disp), use_container_width=True, key="gal_par")

    with gc:
        st.caption("Spektrogram wymaga **surowszego** próbkowania; EDA i zmienność pulsu — na danych wygładzonych.")
        st.plotly_chart(
            vg.fig_spectrogram(
                df_raw,
                "oddech",
                raw_fs_hz,
                "Spektrogram oddechu (STFT, dB) — wzorce okresowości w czasie",
            ),
            use_container_width=True,
            key="gal_spec_breath",
        )
        st.plotly_chart(
            vg.fig_spectrogram(
                df_raw,
                "ecg_mv",
                raw_fs_hz,
                "Spektrogram toru „ecg_mv” (u Ciebie: drugi oddech / tor B) — zależnie od fs",
            ),
            use_container_width=True,
            key="gal_spec_ecg",
        )
        st.plotly_chart(vg.fig_eda_tonic_phasic(df_disp), use_container_width=True, key="gal_eda")
        st.plotly_chart(vg.fig_rolling_hr_variability(df_disp), use_container_width=True, key="gal_hrv")


if __name__ == "__main__":
    main()
