"""
Wczytywanie danych sesji: BrainVision (.vhdr + .eeg) — kanały pomocnicze (oddech, GSR, HR).
Pliki samego .vhdr bez .eeg nie zawierają próbek — wtedy zwracamy None i komunikat.
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
    resolutions: dict[int, float]  # 1-based channel -> multiplier
    names: dict[int, str]


def _parse_vhdr(path: Path) -> BrainVisionMeta:
    text = path.read_text(encoding="utf-8-sig", errors="replace")
    data_file = "recording.eeg"
    n_channels = 1
    sampling_hz = 1000.0

    m = re.search(r"DataFile=(.+)", text)
    if m:
        data_file = m.group(1).strip()
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
            # Format: Name,Ref,Resolution,Unit — rozdzielczość zwykle 3. pole
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


def load_brainvision_auxiliary(vhdr_path: Path) -> tuple[pd.DataFrame | None, str, BrainVisionMeta | None]:
    """
    Wczytuje kanały pomocnicze z nagrania BrainVision (ostatnie 4: Resp, Resp, GSR, HR w typowym setupie 64+4).

    Zwraca: (df | None, komunikat, meta | None)
    Kolumny: time_s, oddech, puls_bpm, ecg_mv, eda_us
    (HR i drugi oddech są w jednostkach z nagłówka — oś może wymagać podpisu „sygnał”, nie bpm).
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
            f"Plik `{eeg_path.name}` jest za krótki ({nbytes} B) na jedną ramkę multiplex ({bytes_per_frame} B dla "
            f"{meta.n_channels} kanałów × INT16).",
            meta,
        )
    if nbytes % bytes_per_frame != 0:
        return (
            None,
            f"Rozmiar `{eeg_path.name}` ({nbytes} B) nie jest wielokrotnością {bytes_per_frame} B "
            f"(nagłówek mówi o {meta.n_channels} kanałach). Plik może być uszkodzony lub niekompletny.",
            meta,
        )

    data = _read_multiplexed_int16(eeg_path, meta.n_channels)
    if data.size == 0:
        return (
            None,
            f"Po odczycie `{eeg_path.name}` nie uzyskano żadnej pełnej próbki (plik: {nbytes} B). "
            "Sprawdź zgodność `NumberOfChannels` w `.vhdr` z rzeczywistym plikiem `.eeg`.",
            meta,
        )

    n = data.shape[0]
    t = np.arange(n) / meta.sampling_hz

    # Indeksy 0-based dla Ch65–Ch68 (ostatnie 4 w typowym zapisie)
    if meta.n_channels < 68:
        return (
            None,
            f"Oczekiwano ≥68 kanałów (Resp/GSR/HR na końcu), jest {meta.n_channels}.",
            meta,
        )

    i0, i1, i2, i3 = 64, 65, 66, 67
    r0 = meta.resolutions.get(65, 0.1526)
    r1 = meta.resolutions.get(66, 0.1526)
    r2 = meta.resolutions.get(67, 0.006104)
    r3 = meta.resolutions.get(68, 0.1)

    df = pd.DataFrame(
        {
            "time_s": t,
            "oddech": data[:, i0].astype(np.float64) * r0,
            "ecg_mv": data[:, i1].astype(np.float64) * r1,
            "eda_us": data[:, i2].astype(np.float64) * r2,
            "puls_bpm": data[:, i3].astype(np.float64) * r3,
        }
    )
    msg = (
        f"Wczytano BrainVision: {n} próbek × {meta.sampling_hz:.0f} Hz, długość {t[-1]/60:.1f} min. "
        "Kolumna `puls_bpm` to sygnał HR z nagłówka (µV) — nazwa historyczna w dashboardzie."
    )
    return df, msg, meta


def find_vhdr_files(data_dir: Path) -> list[Path]:
    """Wszystkie `.vhdr` w `data/` i podfolderach (np. `validation_samples/`)."""
    return sorted(data_dir.glob("**/*.vhdr"))
