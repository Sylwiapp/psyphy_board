# -*- coding: utf-8 -*-
"""Loader of the CASE dataset (Sharma et al. 2019, Scientific Data, DOI 10.1038/s41597-019-0209-0).

We use the *interpolated* version (1000 Hz, aligned time axis). CASE signals are mapped onto the
column schema used across the app:

    time_s    <- daqtime / 1000     (daqtime is in ms)
    ecg_mv    <- ecg                 (REAL ECG, in volts)
    oddech    <- rsp                 (respiration belt)
    eda_us    <- gsr                 (skin conductance)
    puls_bpm  <- bvp                 (BVP/PPG waveform, NOT a BPM value)

The `video` column in CASE files encodes the watched stimulus (ID). Contiguous runs of the same
ID define segments (= emotional conditions), which map onto the app's `SessionGeom`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CASE_FS_HZ = 1000.0

# Global video-ID -> label mapping (confirmed empirically, stable across the whole dataset).
CASE_VIDEO_LABELS: dict[int, str] = {
    1: "amusing-1",
    2: "amusing-2",
    3: "boring-1",
    4: "boring-2",
    5: "relaxed-1",
    6: "relaxed-2",
    7: "scary-1",
    8: "scary-2",
    10: "startVid",
    11: "bluVid",
    12: "endVid",
}

# Emotional videos (the unit of analysis for per-condition HRV).
CASE_EMOTION_IDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)

# Columns read from CSV (skip emg_*/skt, unused in the app; saves memory/time).
_USECOLS = ["daqtime", "ecg", "bvp", "gsr", "rsp", "video"]
_DTYPES = {
    "daqtime": np.int64,
    "ecg": np.float32,
    "bvp": np.float32,
    "gsr": np.float32,
    "rsp": np.float32,
    "video": np.int16,
}


@dataclass(frozen=True)
class CaseSegment:
    """A single stimulus (video) segment within a CASE session."""

    index: int
    video_id: int
    label: str
    t_start_s: float
    t_end_s: float

    @property
    def duration_s(self) -> float:
        return self.t_end_s - self.t_start_s

    @property
    def is_emotional(self) -> bool:
        return self.video_id in CASE_EMOTION_IDS


def video_label(video_id: int) -> str:
    return CASE_VIDEO_LABELS.get(int(video_id), f"video-{int(video_id)}")


def _phys_dir(case_root: Path) -> Path:
    """Physiological folder (handles extraction with or without a parent `data/`)."""
    p = case_root / "data" / "interpolated" / "physiological"
    if p.is_dir():
        return p
    return case_root / "interpolated" / "physiological"


def find_case_root(data_dir: Path) -> Path | None:
    """Find the CASE root (containing `.../interpolated/physiological`) inside `data_dir`."""
    candidates = [data_dir, *(p for p in data_dir.glob("*") if p.is_dir())]
    for c in candidates:
        if _phys_dir(c).is_dir():
            return c
    return None


def find_case_subjects(case_root: Path) -> list[int]:
    """List of subject IDs (from `sub_<n>.csv`), sorted ascending."""
    out: list[int] = []
    phys = _phys_dir(case_root)
    if not phys.is_dir():
        return out
    for f in phys.glob("sub_*.csv"):
        try:
            out.append(int(f.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(out)


def _segments_from_video(daqtime_ms: np.ndarray, video: np.ndarray) -> list[CaseSegment]:
    """Build segments from contiguous runs of constant `video`."""
    if video.size == 0:
        return []
    change = np.flatnonzero(np.diff(video) != 0) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [video.size]))
    segs: list[CaseSegment] = []
    for i, (s, e) in enumerate(zip(starts, ends)):
        vid = int(video[s])
        segs.append(
            CaseSegment(
                index=i,
                video_id=vid,
                label=video_label(vid),
                t_start_s=float(daqtime_ms[s]) / 1000.0,
                t_end_s=float(daqtime_ms[e - 1]) / 1000.0,
            )
        )
    return segs


def _subject_path(case_root: Path, subject_id: int) -> Path:
    return _phys_dir(case_root) / f"sub_{int(subject_id)}.csv"


def load_case_subject_full(
    case_root: Path, subject_id: int
) -> tuple[pd.DataFrame, list[CaseSegment]]:
    """Load a subject's full session in the app schema.

    Returns `(df, segments)` where `df` has app columns with `time_s` starting at 0, and
    `segments` is the list of video runs (to build a `SessionGeom` or to pick a fragment).
    """
    path = _subject_path(case_root, subject_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing CASE file: {path}")
    raw = pd.read_csv(path, usecols=_USECOLS, dtype=_DTYPES)
    daqtime = raw["daqtime"].to_numpy()
    video = raw["video"].to_numpy()
    segments = _segments_from_video(daqtime, video)

    df = pd.DataFrame(
        {
            "time_s": daqtime.astype(np.float64) / 1000.0,
            "oddech": raw["rsp"].to_numpy(dtype=np.float64),
            "puls_bpm": raw["bvp"].to_numpy(dtype=np.float64),  # BVP/PPG, not BPM
            "ecg_mv": raw["ecg"].to_numpy(dtype=np.float64),  # REAL ECG
            "eda_us": raw["gsr"].to_numpy(dtype=np.float64),
        }
    )
    return df, segments


def slice_segment(df_full: pd.DataFrame, seg: CaseSegment) -> pd.DataFrame:
    """Slice one segment out of the full session and reset `time_s` to 0."""
    m = (df_full["time_s"] >= seg.t_start_s) & (df_full["time_s"] <= seg.t_end_s)
    out = df_full.loc[m].reset_index(drop=True)
    if not out.empty:
        out["time_s"] = out["time_s"] - float(out["time_s"].iloc[0])
    return out


def default_segment_index(segments: list[CaseSegment]) -> int:
    """Index of the first emotional segment (default pick); 0 if none."""
    for s in segments:
        if s.is_emotional:
            return s.index
    return 0
