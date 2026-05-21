"""
Wczytywanie danych sesji BrainVision (.vhdr + .eeg) — kanały pomocnicze (oddech, EDA, HR, PPG).

Mapowanie kanałów jest **po nazwach** z sekcji `[Channel Infos]` w `.vhdr`
(Resp01T, Resp02B, GSR, HR, PPG itd.); fallback pozycyjny dla starszych zapisów.

Kolumny wyjściowe DataFrame:
  - `time_s`         — oś czasu w sekundach
  - `resp_t`         — pas oddechowy klatki (Resp01T)
  - `resp_b`         — pas oddechowy brzucha (Resp02B)
  - `eda_us`         — przewodnictwo skóry (GSR), µS
  - `hr_aux_uv`      — sygnał HR z rekordera (kanał HR), µV
  - `ppg`            — PPG z palca (jeśli kanał obecny), w jednostkach urządzenia

Aliasy wsteczne (dla istniejącego kodu, który zna stare nazwy):
  - `oddech`   = `resp_t`
  - `puls_bpm` = `hr_aux_uv`  (UWAGA: to nie jest BPM tylko sygnał µV)
  - `ecg_mv`   = `resp_b`     (UWAGA: to nie jest ECG, tylko drugi pas oddechu)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BrainVisionMeta:
    sampling_hz: float
    n_channels: int
    data_file: str
    marker_file: str | None
    resolutions: dict[int, float]
    names: dict[int, str]


# Mapa: kanonical column name -> lista możliwych nazw kanału (case-insensitive)
# Pierwsza znaleziona nazwa wygrywa; kolejność w liście to priorytet.
_CHANNEL_ALIASES: dict[str, tuple[str, ...]] = {
    "resp_t": ("resp01t", "resp_t", "resp1", "resp", "respiration_thoracic"),
    "resp_b": ("resp02b", "resp_b", "resp2", "respiration_abdominal"),
    "eda_us": ("gsr", "eda", "eda_us"),
    "hr_aux_uv": ("hr", "heartrate"),
    "ppg": ("ppg", "pleth", "pulse_oximeter"),
}

# Domyślne rozdzielczości (mnożniki INT16 → jednostka fizyczna) jeśli .vhdr ich nie poda
_DEFAULT_RESOLUTIONS: dict[str, float] = {
    "resp_t": 0.1526,
    "resp_b": 0.1526,
    "eda_us": 0.6104,
    "hr_aux_uv": 0.1,
    "ppg": 0.6104,
}


def _parse_vhdr(path: Path) -> BrainVisionMeta:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    data_file = "recording.eeg"
    marker_file: str | None = None
    n_channels = 1
    sampling_hz = 1000.0

    m = re.search(r"DataFile=(.+)", text)
    if m:
        data_file = m.group(1).strip()
    m = re.search(r"MarkerFile=(.+)", text)
    if m:
        marker_file = m.group(1).strip()
    m = re.search(r"NumberOfChannels=(\d+)", text)
    if m:
        n_channels = int(m.group(1))
    m = re.search(r"SamplingInterval=(\d+)", text)
    if m:
        sampling_hz = 1e6 / float(m.group(1))

    resolutions: dict[int, float] = {}
    names: dict[int, str] = {}
    # Tylko [Channel Infos] — w [Coordinates] są linie ChN=0,0,0 które inaczej nadpisują Resolution=0.
    in_channel_infos = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_channel_infos = stripped.casefold() == "[channel infos]"
            continue
        if not in_channel_infos:
            continue
        if line.startswith("Ch") and "=" in line and not line.startswith("Channels"):
            mm = re.match(r"Ch(\d+)=(.+)", line)
            if not mm:
                continue
            ch = int(mm.group(1))
            rest = mm.group(2).split(",")
            names[ch] = rest[0].strip() if rest else f"Ch{ch}"
            res = 0.1
            if len(rest) >= 3 and rest[2].strip():
                try:
                    res = float(rest[2].strip())
                except ValueError:
                    res = 0.1
            resolutions[ch] = res

    return BrainVisionMeta(
        sampling_hz=sampling_hz,
        n_channels=n_channels,
        data_file=data_file,
        marker_file=marker_file,
        resolutions=resolutions,
        names=names,
    )


def _read_multiplexed_int16(eeg_path: Path, n_channels: int) -> np.ndarray:
    raw = np.fromfile(eeg_path, dtype="<i2")
    n_complete = (len(raw) // n_channels) * n_channels
    if n_complete <= 0:
        return np.zeros((0, n_channels))
    raw = raw[:n_complete]
    return raw.reshape(-1, n_channels)


def _resolve_channels(meta: BrainVisionMeta) -> dict[str, int]:
    """Mapuj kanoniczne nazwy kolumn (resp_t, resp_b, eda_us, hr_aux_uv, ppg)
    na 1-based indeksy kanałów w `meta.names`. Wyszukiwanie po nazwach (case-insensitive).

    Jeśli nazwa nie znaleziona — klucz nie pojawia się w wyniku (kolumna będzie pominięta).
    """
    out: dict[str, int] = {}
    name_to_ch: dict[str, int] = {}
    for ch, raw_name in meta.names.items():
        key = raw_name.strip().casefold()
        # nadpisuj tylko jeśli nowy ch < poprzedni (preferuj niższy numer dla zduplikowanych nazw)
        if key not in name_to_ch or ch < name_to_ch[key]:
            name_to_ch[key] = ch
    for canon, aliases in _CHANNEL_ALIASES.items():
        for alias in aliases:
            if alias in name_to_ch:
                out[canon] = name_to_ch[alias]
                break
    return out


def _fallback_positional_64plus(meta: BrainVisionMeta) -> dict[str, int]:
    """Awaryjne mapowanie pozycyjne dla starszych zapisów 64+4 bez czytelnych nazw."""
    n = meta.n_channels
    out: dict[str, int] = {}
    if n >= 65:
        out["resp_t"] = 65
    if n >= 66:
        out["resp_b"] = 66
    if n >= 67:
        out["eda_us"] = 67
    if n >= 68:
        out["hr_aux_uv"] = 68
    return out


def load_brainvision_auxiliary(
    vhdr_path: Path,
) -> tuple[pd.DataFrame | None, str, BrainVisionMeta | None]:
    """Wczytuje kanały pomocnicze z BrainVision.

    Mapuje kolumny po nazwach z `.vhdr`; gdy nazw brak (lub są nieczytelne) — fallback
    pozycyjny dla typowego setupu 64+4. Zachowuje aliasy wsteczne (`oddech`, `ecg_mv`,
    `puls_bpm`) na potrzeby istniejącego kodu.

    Zwraca: (df | None, komunikat, meta | None)
    """
    vhdr_path = vhdr_path.resolve()
    if not vhdr_path.suffix.lower() == ".vhdr":
        return None, "Wybierz plik .vhdr", None

    meta = _parse_vhdr(vhdr_path)
    eeg_path = vhdr_path.parent / meta.data_file
    if not eeg_path.is_file():
        return (
            None,
            f"Brak pliku z próbkami: `{eeg_path.name}` (oczekiwany obok `{vhdr_path.name}`). "
            "Skopiuj plik .eeg do folderu `data`.",
            meta,
        )

    nbytes = eeg_path.stat().st_size
    if nbytes == 0:
        return (
            None,
            f"Plik `{eeg_path.name}` ma rozmiar 0 B — brak danych próbek. "
            f"Sprawdź kopię z rekordera (pełna ścieżka: {eeg_path}).",
            meta,
        )

    bytes_per_frame = 2 * meta.n_channels
    if nbytes < bytes_per_frame:
        return (
            None,
            f"Plik `{eeg_path.name}` jest za krótki ({nbytes} B) na jedną ramkę multiplex "
            f"({bytes_per_frame} B dla {meta.n_channels} kanałów × INT16).",
            meta,
        )
    if nbytes % bytes_per_frame != 0:
        return (
            None,
            f"Rozmiar `{eeg_path.name}` ({nbytes} B) nie jest wielokrotnością "
            f"{bytes_per_frame} B (nagłówek mówi o {meta.n_channels} kanałach). "
            "Plik może być uszkodzony lub niekompletny.",
            meta,
        )

    data = _read_multiplexed_int16(eeg_path, meta.n_channels)
    if data.size == 0:
        return (
            None,
            f"Po odczycie `{eeg_path.name}` nie uzyskano żadnej pełnej próbki "
            f"(plik: {nbytes} B). Sprawdź zgodność `NumberOfChannels` w `.vhdr` z `.eeg`.",
            meta,
        )

    channels = _resolve_channels(meta)
    fallback_used = False
    if "resp_t" not in channels and "eda_us" not in channels:
        channels = _fallback_positional_64plus(meta)
        fallback_used = True

    if not channels:
        return (
            None,
            f"Nie udało się zmapować żadnego kanału pomocniczego z `{vhdr_path.name}` "
            f"(NumberOfChannels={meta.n_channels}).",
            meta,
        )

    n = data.shape[0]
    t = np.arange(n) / meta.sampling_hz
    df_cols: dict[str, np.ndarray] = {"time_s": t}
    mapping_lines: list[str] = []
    for canon, ch in channels.items():
        col_idx = ch - 1  # 1-based -> 0-based
        if col_idx < 0 or col_idx >= data.shape[1]:
            continue
        res = meta.resolutions.get(ch, _DEFAULT_RESOLUTIONS.get(canon, 0.1))
        df_cols[canon] = data[:, col_idx].astype(np.float64) * float(res)
        ch_name = meta.names.get(ch, f"Ch{ch}")
        mapping_lines.append(f"- `{canon}` ← **Ch{ch}** = *{ch_name}* (res={res})")

    # Aliasy wsteczne — żeby istniejący kod (np. ecg_qc.py, viz_gallery, app.py) wciąż działał.
    aliases_pl: list[str] = []
    if "resp_t" in df_cols:
        df_cols["oddech"] = df_cols["resp_t"]
        aliases_pl.append("`oddech` = `resp_t`")
    if "resp_b" in df_cols:
        df_cols["ecg_mv"] = df_cols["resp_b"]
        aliases_pl.append("`ecg_mv` = `resp_b` — to **NIE jest ECG**, tylko drugi pas oddechu")
    if "hr_aux_uv" in df_cols:
        df_cols["puls_bpm"] = df_cols["hr_aux_uv"]
        aliases_pl.append("`puls_bpm` = `hr_aux_uv` — sygnał µV z rekordera, nie BPM")

    df = pd.DataFrame(df_cols)

    msg_lines = [
        f"Wczytano BrainVision: {n} próbek × {meta.sampling_hz:.0f} Hz, długość {t[-1] / 60:.1f} min.",
    ]
    if fallback_used:
        msg_lines.append(
            "_Uwaga:_ użyto **mapowania pozycyjnego** (ostatnie 4 kanały z 68) — w `.vhdr` "
            "nie znaleziono czytelnych nazw kanałów pomocniczych."
        )
    msg_lines.append("\n**Mapowanie kolumn:**\n" + "\n".join(mapping_lines))
    if aliases_pl:
        msg_lines.append("\n**Aliasy wsteczne (do starego kodu):**\n- " + "\n- ".join(aliases_pl))
    if "ecg_mv" in df_cols and not _looks_like_real_ecg(meta):
        msg_lines.append(
            "\n⚠️ **Zakładka „QC / preprocessing — ECG” pracuje na kolumnie `ecg_mv`, "
            "która w tym pliku to drugi pas oddechu (Resp02B), nie ECG.** "
            "Detekcja R, RR, HRV — wyniki są nieprawidłowe fizjologicznie."
        )
    msg = "\n\n".join(msg_lines)
    return df, msg, meta


def _looks_like_real_ecg(meta: BrainVisionMeta) -> bool:
    """Heurystyka: czy w pliku jest kanał o nazwie sugerującej prawdziwe ECG."""
    for raw_name in meta.names.values():
        if "ecg" in raw_name.casefold() and "resp" not in raw_name.casefold():
            return True
    return False


def find_vhdr_files(data_dir: Path) -> list[Path]:
    """Wszystkie `.vhdr` w `data/` i podfolderach."""
    return sorted(data_dir.glob("**/*.vhdr"))
