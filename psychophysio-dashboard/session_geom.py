# -*- coding: utf-8 -*-
"""Geometria sesji: dlugosc nagrania i krawedzie segmentow (rowny podzial albo markery BrainVision)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SessionGeom:
    """
    Dwa tryby segmentacji:
    - **Partycja** (domyślna): `segment_edges` — segment i = [edges[i], edges[i+1]).
    - **Przedziały analizy** (np. CALM): `analysis_windows` — segment i = `analysis_windows[i]` (rozłączne
      okna na osi czasu); `segment_edges` wtedy tylko (0, session_s) dla kompatybilności.
    """

    session_s: float
    segment_edges: tuple[float, ...]
    analysis_windows: tuple[tuple[float, float], ...] = ()
    segment_labels: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(self.segment_edges) < 2:
            raise ValueError("segment_edges: potrzeba co najmniej 2 punktow (start i koniec sesji).")
        if self.analysis_windows:
            if abs(float(self.segment_edges[0])) > 1e-9 or abs(float(self.segment_edges[-1]) - float(self.session_s)) > 1e-6:
                raise ValueError("Przy analysis_windows oczekiwane segment_edges = (0.0, session_s).")
            for lo, hi in self.analysis_windows:
                if hi <= lo + 1e-9:
                    raise ValueError("analysis_windows: kazdy przedzial musi miec hi > lo.")
            if self.segment_labels and len(self.segment_labels) != len(self.analysis_windows):
                raise ValueError("segment_labels musi miec tyle elementow co analysis_windows (albo puste).")
        else:
            for a, b in zip(self.segment_edges, self.segment_edges[1:]):
                if b <= a + 1e-15:
                    raise ValueError("segment_edges musi byc scisle rosnacy.")

    @property
    def uses_disjoint_windows(self) -> bool:
        return len(self.analysis_windows) > 0

    @property
    def n_seg(self) -> int:
        if self.analysis_windows:
            return len(self.analysis_windows)
        return len(self.segment_edges) - 1

    def segment_bounds(self, i: int) -> tuple[float, float]:
        if i < 0 or i >= self.n_seg:
            raise IndexError(f"segment {i} poza zakresem 0..{self.n_seg - 1}")
        if self.analysis_windows:
            return (float(self.analysis_windows[i][0]), float(self.analysis_windows[i][1]))
        return (self.segment_edges[i], self.segment_edges[i + 1])

    def segment_label(self, i: int) -> str:
        if self.segment_labels and 0 <= i < len(self.segment_labels):
            return self.segment_labels[i]
        return f"Seg. {i + 1}"

    def segment_index_at_time(self, t: float) -> int:
        """Indeks segmentu zawierającego czas t, lub -1 (poza przedziałami w trybie okien analizy)."""
        t = float(t)
        if self.analysis_windows:
            for i, (lo, hi) in enumerate(self.analysis_windows):
                if lo <= t <= hi:
                    return i
            return -1
        edges = self.segment_edges
        if t < edges[0] or t > edges[-1]:
            return -1
        for i in range(self.n_seg):
            lo, hi = edges[i], edges[i + 1]
            if i < self.n_seg - 1:
                if lo <= t < hi:
                    return i
            else:
                if lo <= t <= hi:
                    return i
        return self.n_seg - 1


def session_geom_equal_split(session_s: float, n_seg: int) -> SessionGeom:
    """N rownych segmentow czasu (syntetyczne / brak markerow New Segment)."""
    n = max(1, int(n_seg))
    edges = [session_s * i / n for i in range(n + 1)]
    edges[-1] = float(session_s)
    return SessionGeom(session_s=float(session_s), segment_edges=tuple(edges))


def session_geom_from_marker_starts(
    session_s: float,
    marker_start_times_s: list[float],
) -> SessionGeom:
    """Przedzialy miedzy kolejnymi czasami markerow (np. New Segment); koncowka do konca sesji."""
    session_s = float(session_s)
    edges: list[float] = [0.0]
    for t in sorted(marker_start_times_s):
        t = float(t)
        if t <= edges[-1] + 1e-9:
            continue
        if t >= session_s - 1e-9:
            break
        edges.append(t)
    if edges[-1] < session_s - 1e-9:
        edges.append(session_s)
    else:
        edges[-1] = session_s
    return SessionGeom(session_s=session_s, segment_edges=tuple(edges))
