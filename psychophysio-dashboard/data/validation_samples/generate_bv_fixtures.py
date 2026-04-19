"""
Jednorazowo buduje małe pliki .vhdr + .eeg do testów walidacji w PsyPhy Datalab.

Uruchom z katalogu projektu:
  py data/validation_samples/generate_bv_fixtures.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FS = 1000.0
N_SAMPLES = 8000  # 8 s
N_CH = 68


def _write_eeg(path: Path, data_int16: np.ndarray) -> None:
    assert data_int16.shape == (N_SAMPLES, N_CH)
    data_int16.astype("<i2").tofile(path)


def _vhdr(name: str, data_file: str, sampling_interval_us: int) -> str:
    # Minimalny nagłówek zgodny z data_loader._parse_vhdr (Ch1..Ch68, rozdzielczości jak w MK_0123)
    lines = [
        "BrainVision Data Exchange Header File Version 1.0",
        "",
        "[Common Infos]",
        "DataFile=" + data_file,
        "DataFormat=BINARY",
        "DataOrientation=MULTIPLEXED",
        f"NumberOfChannels={N_CH}",
        f"SamplingInterval={sampling_interval_us}",
        "",
        "[Binary Infos]",
        "BinaryFormat=INT_16",
        "",
        "[Channel Infos]",
    ]
    for i in range(1, 65):
        lines.append(f"Ch{i}=EEG{i},,0.1,µV")
    lines.extend(
        [
            "Ch65=Resp01T,,0.1526,ARU",
            "Ch66=Resp02B,,0.1526,ARU",
            "Ch67=GSR,,0.006104,µS",
            "Ch68=HR,,0.1,µV",
        ]
    )
    return "\n".join(lines) + "\n"


def build_val_ok() -> None:
    """Zsynchronizowane sygnały na ostatnich 4 kanałach — walidacja OK."""
    t = np.arange(N_SAMPLES) / FS
    raw = np.zeros((N_SAMPLES, N_CH), dtype=np.int16)
    raw[:, 64] = (500 * np.sin(2 * np.pi * 0.25 * t)).astype(np.int16)
    raw[:, 65] = (300 * np.sin(2 * np.pi * 0.3 * t + 0.5)).astype(np.int16)
    raw[:, 66] = (200 + 50 * np.sin(2 * np.pi * 0.1 * t)).astype(np.int16)
    raw[:, 67] = (400 + 80 * np.sin(2 * np.pi * 0.05 * t)).astype(np.int16)
    base = HERE / "bv_val_ok"
    _write_eeg(base.with_suffix(".eeg"), raw)
    base.with_suffix(".vhdr").write_text(_vhdr("val_ok", base.name + ".eeg", 1000), encoding="utf-8")


def build_val_flat() -> None:
    """Ostatnie 4 kanały stale 0 — ostrzeżenie „płaski sygnał”."""
    raw = np.zeros((N_SAMPLES, N_CH), dtype=np.int16)
    base = HERE / "bv_val_flat"
    _write_eeg(base.with_suffix(".eeg"), raw)
    base.with_suffix(".vhdr").write_text(_vhdr("val_flat", base.name + ".eeg", 1000), encoding="utf-8")


def build_val_warn_fs() -> None:
    """Fs = 40 Hz (SamplingInterval=25000 µs) — ostrzeżenie nietypowej Fs."""
    t = np.arange(N_SAMPLES) / 40.0
    raw = np.zeros((N_SAMPLES, N_CH), dtype=np.int16)
    raw[:, 64] = (400 * np.sin(2 * np.pi * 0.2 * t)).astype(np.int16)
    raw[:, 65] = (400 * np.sin(2 * np.pi * 0.2 * t + 0.3)).astype(np.int16)
    raw[:, 66] = (200 * np.ones(N_SAMPLES)).astype(np.int16)
    raw[:, 67] = (300 * np.sin(2 * np.pi * 0.1 * t)).astype(np.int16)
    base = HERE / "bv_val_warn_fs"
    _write_eeg(base.with_suffix(".eeg"), raw)
    base.with_suffix(".vhdr").write_text(_vhdr("val_warn_fs", base.name + ".eeg", 25000), encoding="utf-8")


def main() -> None:
    build_val_ok()
    build_val_flat()
    build_val_warn_fs()
    print("Zapisano w:", HERE)
    print("  bv_val_ok.vhdr + .eeg")
    print("  bv_val_flat.vhdr + .eeg")
    print("  bv_val_warn_fs.vhdr + .eeg")


if __name__ == "__main__":
    main()
