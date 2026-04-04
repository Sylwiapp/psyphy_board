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

from transcript_io import Utterance, load_transcript_auto, load_transcript_json_bytes
from data_loader import find_vhdr_files, load_brainvision_auxiliary
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
    - Okno wokół kursora — przewijasz suwakiem (środek okna = kursor)
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
    for col, label, color in series_meta:
        fig.add_trace(
            go.Scatter(
                x=df["time_s"],
                y=min_max_norm(df[col]),
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
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["oddech"], name="oddech", line=dict(width=0.8)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["puls_bpm"], name="puls", line=dict(width=0.8)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["ecg_mv"], name="ECG", line=dict(width=0.6)),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=df["time_s"], y=df["eda_us"], name="EDA", line=dict(width=0.8)),
        row=4,
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


def sync_cursor_from_plotly(event: object | None, session_s: float) -> None:
    """Ustawia kursor czasu po kliknięciu / zaznaczeniu punktu na wykresie Plotly (Streamlit on_select)."""
    if event is None:
        return
    try:
        sel = getattr(event, "selection", None)
        if sel is None:
            return
        if isinstance(sel, dict):
            pts = sel.get("points") or []
        else:
            pts = getattr(sel, "points", None) or []
    except (TypeError, AttributeError):
        return
    if not pts:
        return
    p0 = pts[0]
    x = float(p0["x"]) if isinstance(p0, dict) else float(getattr(p0, "x", 0))
    st.session_state.cursor_s = float(np.clip(x, 0.0, session_s))


def find_active_utterance(utterances: list[Utterance], cursor_s: float) -> Utterance | None:
    """Przedziały domknięte-otwarte [start, end)."""
    for u in utterances:
        if u.start_s <= cursor_s < u.end_s:
            return u
    return None


def transcript_list_html(utterances: list[Utterance], cursor_s: float) -> str:
    """
    Lista wypowiedzi z podświetleniem — przez st.markdown (bez iframe),
    więc przy każdej zmianie kursora Streamlit faktycznie odświeża HTML.
    """
    parts: list[str] = []
    for i, u in enumerate(utterances):
        is_active = u.start_s <= cursor_s < u.end_s
        bg = "rgba(255,230,200,0.95)" if is_active else "rgba(255,255,255,0.92)"
        border = "2px solid #c44" if is_active else "1px solid #ccc"
        safe = html_module.escape(u.text)
        parts.append(
            f'<p id="utt-{i}" style="margin:6px 0;padding:8px;background:{bg};border:{border};'
            f'border-radius:6px;font-size:14px;line-height:1.35">'
            f'<span style="color:#666;font-size:12px">[{u.start_s:.1f} – {u.end_s:.1f} s]</span><br/>{safe}</p>'
        )
    body = "\n".join(parts) if parts else "<p>(brak wypowiedzi)</p>"
    return (
        '<div style="max-height:420px;overflow-y:auto;padding:10px;background:#fafafa;'
        'border:1px solid #ddd;border-radius:8px;font-family:system-ui,Segoe UI,sans-serif">'
        f"{body}</div>"
    )


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    st.title(APP_NAME)
    st.caption("Sesja z transkryptem i segmentami · prototyp open source")

    st.markdown(
        """
**Dane:** tryb syntetyczny albo **BrainVision** (`.vhdr` + `.eeg` w folderze `data`) — kanały pomocnicze Resp / GSR / HR.  
**Segmenty:** domyślnie **6** bloków o równej długości (długość sesji zależy od nagrania).  
**Nawigacja:** pełna oś · **okno** wokół kursora · **jeden segment**.  
**Nakładka (4 krzywe, min–max)** jest na górze zakładki Sesja; **klik** w punkt krzywej w Plotly ustawia kursor przy **kolejnym** odświeżeniu (standard Streamlit).  
**Galeria:** histogramy, spektrogramy, korelacje, boxploty itd.  
**Transkrypt:** JSON/CSV — podświetlenie wg czasu; przygotujemy import **kwestionariuszy** (CSV) osobno.

**Surowe vs przetworzone:** przegląd vs QC — oba widoki są w zakładkach obok siebie.
        """
    )

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
                st.warning(msg)
                df_raw = make_physiology_raw(int(seed))
                data_note = msg
            else:
                df_raw = df_bv
                loaded_bv = True
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
    if use_upload:
        up = st.sidebar.file_uploader("Plik JSON lub CSV", type=["json", "csv"])
        if up is not None:
            utterances = load_transcript_auto(up, up.name)
        else:
            utterances = make_synthetic_utterances(geom)
            st.sidebar.info("Brak pliku — używam syntetycznego transkryptu.")
    else:
        utterances = make_synthetic_utterances(geom)

    if default_path.exists() and st.sidebar.checkbox("Dopisz przykład z `data/transcript.example.json`", value=False):
        raw = default_path.read_bytes()
        extra = load_transcript_json_bytes(raw)
        utterances = sorted(utterances + extra, key=lambda u: u.start_s)

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Nawigacja po czasie**")
    nav_mode = st.sidebar.radio(
        "Tryb widoku osi X",
        [NAV_FULL, NAV_WINDOW, NAV_SEGMENT],
        index=0,
        help=(
            "Pełna sesja — cała godzina na osi (zoom/pan w Plotly). "
            "Okno — oś pokazuje wycinek; środek okna = kursor (przesuwasz suwak = „przewijasz” dane). "
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
        st.sidebar.caption("Przesuwaj **kursor czasu** — okno jest wyśrodkowane na kursorze.")
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

    tab_session, tab_gallery = st.tabs(["Sesja i transkrypt", "Galeria wizualizacji (warianty)"])

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

    cursor_s = st.slider(
        "Kursor czasu (s)",
        min_value=0.0,
        max_value=float(geom.session_s),
        value=float(st.session_state.cursor_s),
        step=1.0,
        help="Transkrypt i pionowa linia. Klik w wykresie Plotly (tryb zaznaczania punktów) ustawia czas przy następnym odświeżeniu.",
        key="cursor_slider_main",
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

    st.subheader("Nakładka: 4 kanały (min–max)")
    st.caption(
        "Wszystkie sygnały na **jednej** osi Y po normalizacji min–max (porównanie kształtu). "
        "Wybierz źródło, potem **kliknij punkt na krzywej** (Plotly), aby ustawić kursor."
    )
    overlay_src = st.radio(
        "Źródło nakładki",
        ["Przetworzone", "Surowe"],
        horizontal=True,
        key="overlay_top_src",
    )
    df_ov = df_disp if overlay_src == "Przetworzone" else df_raw
    suf = "przetworzone" if overlay_src == "Przetworzone" else "surowe"
    hz = DISPLAY_FS_HZ if overlay_src == "Przetworzone" else raw_fs_hz
    ev_ov = st.plotly_chart(
        fig_overlay_normalized(df_ov, cursor_s, f"{suf} (~{hz:.0f} Hz)", x_range, geom),
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="overlay_main_pick",
    )
    sync_cursor_from_plotly(ev_ov, geom.session_s)

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
            ev_r = st.plotly_chart(
                fig_stacked(
                    df_raw,
                    cursor_s,
                    "surowe (~{:.0f} Hz)".format(raw_fs_hz),
                    x_range,
                    geom,
                    subplot_titles=stitles,
                ),
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="plot_raw_click",
            )
            sync_cursor_from_plotly(ev_r, geom.session_s)

        with tab_filt:
            st.caption(
                f"Wygładzenie + decymacja ~{DISPLAY_FS_HZ:.0f} Hz — przegląd całości."
            )
            ev_f = st.plotly_chart(
                fig_stacked(
                    df_disp,
                    cursor_s,
                    "przetworzone (średnia krocząca + decymacja)",
                    x_range,
                    geom,
                    subplot_titles=stitles,
                ),
                use_container_width=True,
                on_select="rerun",
                selection_mode="points",
                key="plot_disp_click",
            )
            sync_cursor_from_plotly(ev_f, geom.session_s)

    cursor_s = float(st.session_state.cursor_s)

    with col_tr:
        st.subheader("Transkrypt")
        st.caption(f"Kursor: **{cursor_s:.1f} s** — poniżej podświetlona jest wypowiedź obejmująca ten moment.")
        active = find_active_utterance(utterances, cursor_s)
        if active:
            st.info(f"**{active.start_s:.1f} – {active.end_s:.1f} s** · {active.text}")
        else:
            st.caption(
                "W tym momencie nie ma wypowiedzi (przerwa między frazami albo cisza w segmencie). "
                "Przesuń kursor lub kliknij wykres."
            )
        st.markdown(transcript_list_html(utterances, cursor_s), unsafe_allow_html=True)

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
