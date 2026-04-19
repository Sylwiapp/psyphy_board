"""
PsyPhy Datalab — dashboard: oddech, puls, ECG, EDA + segmenty sesji + transkrypt z kursorem czasu.
Surowe vs przetworzone (jak w typowym pipeline: QC vs przegląd).

Uruchomienie: py -3 -m streamlit run app.py
"""

from __future__ import annotations

import html as html_module
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components

from transcript_io import Utterance, load_transcript_auto, load_transcript_json_bytes
from data_loader import BrainVisionMeta, find_vhdr_files, load_brainvision_auxiliary
from data_validation import format_report_body, validate_brainvision_dataframe, validate_transcript
from ecg_qc import (
    EcgQcOptions,
    compute_ecg_qc_report,
    detect_r_peaks,
    estimate_fs_from_time,
    extract_ecg_series,
    preprocess_visible,
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


@dataclass(frozen=True)
class SessionGeom:
    """Długość nagrania i podział na segmenty (np. 6 bloków po ~10 min)."""

    session_s: float
    n_seg: int = N_SEG_DEFAULT

    @property
    def seg_len_s(self) -> float:
        return self.session_s / max(self.n_seg, 1)

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
    sl = geom.seg_len_s
    for seg_i in range(geom.n_seg):
        base = seg_i * sl
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
        x0 = i * geom.seg_len_s
        x1 = min((i + 1) * geom.seg_len_s, geom.session_s)
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
        x0 = i * geom.seg_len_s
        x1 = min((i + 1) * geom.seg_len_s, geom.session_s)
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
    - Wybrany segment — stałe 10 min (segmenty 1–6)
    """
    if mode == NAV_FULL:
        return (0.0, float(geom.session_s))
    if mode == NAV_WINDOW:
        half = max(1.0, window_s / 2.0)
        return clamp_window(cursor_s, half, geom.session_s)
    if mode == NAV_SEGMENT:
        s = float(segment_index * geom.seg_len_s)
        return (s, min(s + geom.seg_len_s, geom.session_s))
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
    if src.startswith("Syntetyczne"):
        df_raw = make_physiology_raw(int(seed))
    else:
        vhdrs = find_vhdr_files(data_dir)
        if not vhdrs:
            st.sidebar.warning("Brak pliku `.vhdr` w `data`. Używam danych syntetycznych.")
            df_raw = make_physiology_raw(int(seed))
        else:
            pick = st.sidebar.selectbox("Nagranie", vhdrs, format_func=lambda p: p.name)
            df_bv, msg, meta = load_brainvision_auxiliary(pick)
            if df_bv is None:
                st.error(msg)
                if meta is not None:
                    eeg_path = pick.parent / meta.data_file
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
                st.success(msg)
                data_note = "BrainVision: kanały Ch65–68 → oddech, tor oddechu 2, GSR (EDA), HR (µV)."

    geom = SessionGeom(session_s=max(float(df_raw["time_s"].max()), 1.0), n_seg=N_SEG_DEFAULT)
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
    st.sidebar.markdown("**Nawigacja po czasie**")
    nav_mode = st.sidebar.radio(
        "Tryb widoku osi X",
        [NAV_FULL, NAV_WINDOW, NAV_SEGMENT],
        index=0,
        help=(
            "Pełna sesja — cała godzina na osi (zoom/pan w Plotly). "
            "Okno — oś pokazuje wycinek; środek okna = kursor (ustawiasz **klikając** punkt na krzywej w Plotly). "
            "Segment — stały blok 10 min (1–6)."
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
            format_func=lambda i: f"{i} (blok {i}/{geom.n_seg})",
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
        _render_gallery_tab(df_raw, df_disp, raw_fs_hz)

    with tab_ecg_qc:
        _render_ecg_qc_tab(df_raw, df_disp, raw_fs_hz)


def _render_ecg_qc_tab(
    df_raw: pd.DataFrame,
    df_disp: pd.DataFrame,
    raw_fs_hz: float,
) -> None:
    """Zakładka: preprocess + QC toru ECG (`ecg_mv`) dla naukowca."""
    st.subheader("QC / preprocessing — ECG (`ecg_mv`)")
    st.caption(
        "Tor w aplikacji jako **ecg_mv** (w BV często drugi pas oddechu — sprawdź nagłówek). "
        "Wyniki to **heurystyka** do eksploracji; nie zastępują opisu metody w pracy."
    )

    src = st.radio(
        "Źródło sygnału do analizy",
        ["Surowe (zalecane do QC)", "Przetworzone (decymacja ~4 Hz)"],
        horizontal=True,
        key="ecg_qc_source",
    )
    df_use = df_raw if src.startswith("Surowe") else df_disp
    fs_declared = float(raw_fs_hz) if src.startswith("Surowe") else float(DISPLAY_FS_HZ)

    t_all, x_all = extract_ecg_series(df_use)
    if t_all.size == 0:
        st.warning("Brak kolumny `ecg_mv` w danych.")
        return

    fs_est = estimate_fs_from_time(t_all)
    fs_use = float(st.number_input(
        "Fs użyte w obliczeniach (Hz)",
        min_value=0.5,
        max_value=5000.0,
        value=float(fs_declared if abs(fs_est - fs_declared) / max(fs_declared, 0.1) < 0.2 else fs_est),
        step=0.5,
        help="Domyślnie z nagłówka / trybu widoku; możesz nadpisać, jeśli znasz rzeczywiste Fs.",
        key="ecg_qc_fs",
    ))

    with st.expander("Opcje preprocessingu (przed detekcją R)", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            use_bp = st.checkbox("Pasmoprzepustowy (Butterworth)", value=True, key="ecg_qc_bp")
            detrend = st.checkbox("Detrend liniowy", value=True, key="ecg_qc_det")
        with c2:
            lo = st.slider("Dolne pasmo (Hz)", 0.5, 20.0, 5.0, 0.5, key="ecg_qc_lo")
            hi = st.slider("Górne pasmo (Hz)", 5.0, 80.0, 40.0, 1.0, key="ecg_qc_hi")
        with c3:
            prom = st.slider("Czułość szczytów (wsp. prominence)", 0.1, 1.0, 0.35, 0.05, key="ecg_qc_prom")
            dmin = st.slider("Min. odległość między R (s)", 0.2, 0.6, 0.28, 0.02, key="ecg_qc_dmin")

    with st.expander("Progi akceptacji RR i okien czasu"):
        c4, c5 = st.columns(2)
        with c4:
            rr_lo = st.number_input("RR min (ms)", 200.0, 800.0, 300.0, 10.0, key="ecg_qc_rrlo")
            rr_hi = st.number_input("RR max (ms)", 800.0, 3000.0, 2000.0, 50.0, key="ecg_qc_rrhi")
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

    lo_f, hi_f = float(lo), float(hi)
    if hi_f <= lo_f:
        hi_f = lo_f + 1.0
    rr_a, rr_b = float(rr_lo), float(rr_hi)
    if rr_b <= rr_a:
        rr_b = rr_a + 50.0

    opt = EcgQcOptions(
        use_bandpass=use_bp,
        bandpass_low_hz=lo_f,
        bandpass_high_hz=hi_f,
        detrend_linear=detrend,
        r_peak_min_distance_s=float(dmin),
        r_peak_prominence_factor=float(prom),
        rr_min_ms=rr_a,
        rr_max_ms=rr_b,
        window_sec=float(win_sec),
        flat_rel_std_max=float(flat_rel),
    )

    rep = compute_ecg_qc_report(t_all, x_all, fs_use, opt)
    x_filt = preprocess_visible(x_all, fs_use, opt)
    x_num = np.nan_to_num(
        x_all,
        nan=np.nanmedian(x_all[np.isfinite(x_all)]) if np.any(np.isfinite(x_all)) else 0.0,
    )
    peaks = detect_r_peaks(x_num, fs_use, opt) if t_all.size > 1 else np.array([], dtype=int)

    st.markdown("---")
    st.markdown("#### Podsumowanie jakości")
    if rep.overall_label == "dobry":
        st.success("**Ocena ogólna: dobry** — większość kryteriów spełniona.")
    elif rep.overall_label == "slaby":
        st.error("**Ocena ogólna: słaby** — warto poprawić zapis lub parametry.")
    elif rep.overall_label == "ostroznie":
        st.warning("**Ocena ogólna: ostrożnie** — sprawdź szczegóły i ewentualnie fragment ręcznie.")
    else:
        st.warning(f"**Ocena:** {rep.overall_label}")

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    c_m1.metric("Długość", f"{rep.duration_min:.1f} min")
    c_m2.metric("Probek", f"{rep.n_samples:,}")
    c_m3.metric("NaN", f"{rep.nan_fraction * 100:.2f} %")
    c_m4.metric("Okna OK", f"{rep.window_ok_fraction * 100:.0f} % ({rep.n_windows_ok}/{rep.n_windows})")

    c_m5, c_m6, c_m7, c_m8 = st.columns(4)
    c_m5.metric("Wykryte R (heuryst.)", str(rep.n_peaks))
    c_m6.metric("Mediana RR (ms)", f"{rep.median_rr_ms:.0f}" if rep.median_rr_ms else "—")
    c_m7.metric("HR z med. RR (bpm)", f"{rep.mean_hr_bpm:.1f}" if rep.mean_hr_bpm else "—")
    c_m8.metric("RR poza progiem", f"{rep.rr_outside_frac * 100:.1f} %")

    st.caption(
        f"**Clipping (heuryst.):** {rep.clip_fraction * 100:.3f} % próbek blisko |max|. "
        f"**Płaski (całość):** {'tak' if rep.flat_signal else 'nie'}. "
        f"**Fs w raporcie:** {rep.fs_hz:.2f} Hz."
    )
    for note in rep.notes:
        st.caption(note)

    preview_s = st.slider("Podgląd wykresu (początek sesji, s)", 3.0, 60.0, 15.0, 1.0, key="ecg_qc_preview")
    m = t_all <= (float(t_all[0]) + preview_s)
    t_sub, raw_sub, filt_sub = t_all[m], x_all[m], x_filt[m]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t_sub, y=raw_sub, name="Surowy / wybrany", line=dict(width=0.8), opacity=0.7))
    fig.add_trace(go.Scatter(x=t_sub, y=filt_sub, name="Po preprocessingu", line=dict(width=1.0)))
    if peaks.size:
        pk = peaks[t_all[peaks] <= (float(t_all[0]) + preview_s)]
        if pk.size:
            fig.add_trace(
                go.Scatter(
                    x=t_all[pk],
                    y=x_filt[pk],
                    mode="markers",
                    name="R (heuryst.)",
                    marker=dict(size=8, color="red", symbol="x"),
                )
            )
    fig.update_layout(
        title=f"ECG — pierwsze ~{preview_s:.0f} s",
        xaxis_title="Czas (s)",
        yaxis_title="ecg_mv (jednostka z naglowka)",
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True, key="ecg_qc_fig_ts")

    if rep.rr_ms_list:
        fig2 = go.Figure()
        fig2.add_trace(
            go.Histogram(x=rep.rr_ms_list, nbinsx=40, name="RR (ms)"),
        )
        fig2.add_vline(x=opt.rr_min_ms, line_dash="dash", line_color="gray")
        fig2.add_vline(x=opt.rr_max_ms, line_dash="dash", line_color="gray")
        fig2.update_layout(title="Rozkład odstępów RR", xaxis_title="RR (ms)", height=320)
        st.plotly_chart(fig2, use_container_width=True, key="ecg_qc_fig_rr")

    st.markdown("#### Interpretacja dla dalszej analizy")
    usable = rep.window_ok_fraction * 100 if rep.n_windows else 0.0
    if rep.overall_label == "dobry":
        st.info(
            f"Przy obecnych ustawieniach **~{usable:.0f} %** czasu w oknach {rep.window_sec:.0f} s wygląda na "
            "nadający się do dalszej pracy (niski udział NaN, sygnał niepłaski w oknie). "
            "Detekcja R służy orientacyjnie — przed HRV ustal ostateczny pipeline z promotorką."
        )
    elif rep.overall_label == "slaby":
        st.warning(
            "Znaczna część nagrania może wymagać odrzucenia lub innego preprocessingu. "
            "Sprawdź elektrody, ruch, saturację; rozważ inne Fs lub pasmo."
        )
    elif rep.overall_label == "ostroznie":
        st.info(
            "Część nagrania nadaje się do analizy, część — nie. Użyj wykresu i histogramu RR; "
            "możesz zawęzić progi RR lub okno lokalne i odświeżyć stronę."
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
    st.info(
        f"**Widok osi X:** {nav_mode} · **zakres:** {x_range[0]:.0f}–{x_range[1]:.0f} s · kursor **{cursor_s:.1f} s** "
        + (
            f"· okno {window_s:.0f} s"
            if nav_mode == NAV_WINDOW
            else (f"· segment {segment_index + 1}" if nav_mode == NAV_SEGMENT else "")
        )
    )

    st.markdown("**Legenda tła:** paski = kolejne segmenty sesji. **Czerwona linia** = kursor czasu.")

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


def _render_gallery_tab(df_raw: pd.DataFrame, df_disp: pd.DataFrame, raw_fs_hz: float) -> None:
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
        st.caption("Agregaty po 6 segmentach — przydatne przy różnych **warunkach** w blokach.")
        st.plotly_chart(vg.fig_segment_bar_summary(df_disp), use_container_width=True, key="gal_bar")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(vg.fig_box_by_segment(df_disp, "puls_bpm"), use_container_width=True, key="gal_box_hr")
        with c2:
            st.plotly_chart(vg.fig_box_by_segment(df_disp, "eda_us"), use_container_width=True, key="gal_box_eda")
        st.plotly_chart(vg.fig_radar_segment_profile(df_disp), use_container_width=True, key="gal_radar")

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
