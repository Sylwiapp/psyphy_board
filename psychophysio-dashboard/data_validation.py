"""
Walidacja „czy plik nadaje się do dalszej pracy” — heurystyki QC, nie formalny certyfikat pomiaru.

Używane po wczytaniu BrainVision oraz transkryptu użytkownika.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from data_loader import BrainVisionMeta
from transcript_io import Utterance

Severity = Literal["ok", "warn", "error"]


@dataclass
class CheckResult:
    severity: Severity
    title: str
    detail: str


@dataclass
class ValidationReport:
    """Zbiór pojedynczych sprawdzeń + krótkie podsumowanie."""

    source: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def has_error(self) -> bool:
        return any(c.severity == "error" for c in self.checks)

    @property
    def has_warn(self) -> bool:
        return any(c.severity == "warn" for c in self.checks)

    def summary_line(self) -> str:
        if self.has_error:
            return "Są **problemy**, które warto rozwiązać przed analizą."
        if self.has_warn:
            return "Plik wczytano — są **ostrzeżenia** (sprawdź listę poniżej)."
        return "Podstawowe kryteria **OK** — nadal zalecam weryfikację merytoryczną w labie."


PHYS_COLS = ("oddech", "puls_bpm", "ecg_mv", "eda_us")


def validate_brainvision_dataframe(
    df: pd.DataFrame,
    meta: BrainVisionMeta,
    *,
    session_s: float,
) -> ValidationReport:
    """Heurystyki po wczytaniu BV (kolumny już przeskalowane)."""
    report = ValidationReport(source="BrainVision (.vhdr + .eeg)")
    n = len(df)
    fs = float(meta.sampling_hz)

    if n == 0:
        report.checks.append(
            CheckResult("error", "Brak próbek", "DataFrame jest pusty — nie ma czego analizować.")
        )
        return report

    if session_s < 0.5:
        report.checks.append(
            CheckResult(
                "error",
                "Zbyt krótka sesja",
                f"Szacowana długość ~{session_s:.2f} s — poniżej sensownego progu.",
            )
        )

    if fs < 50 or fs > 20000:
        report.checks.append(
            CheckResult(
                "warn",
                "Częstotliwość próbkowania",
                f"Fs = {fs:.1f} Hz — nietypowo; typowe BV to często setki–kilka kHz. Sprawdź nagłówek.",
            )
        )
    else:
        report.checks.append(CheckResult("ok", "Częstotliwość próbkowania", f"Fs = {fs:.1f} Hz."))

    missing = [c for c in PHYS_COLS if c not in df.columns]
    if missing:
        report.checks.append(
            CheckResult(
                "error",
                "Brakujące kolumny",
                f"Brakuje: {', '.join(missing)} — oczekiwany układ Ch65–68 nie został spełniony.",
            )
        )
        return report

    nan_bad: list[str] = []
    nan_warn: list[str] = []
    flat_zero: list[str] = []
    flat_const: list[str] = []
    ok_sig: list[str] = []

    for col in PHYS_COLS:
        s = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        nan_ratio = float(np.mean(~np.isfinite(s)))
        if nan_ratio > 0.5:
            nan_bad.append(f"{col} (~{nan_ratio * 100:.0f}% braków)")
        elif nan_ratio > 0.02:
            nan_warn.append(f"{col} (~{nan_ratio * 100:.1f}%)")

        s_clean = s[np.isfinite(s)]
        if s_clean.size == 0:
            nan_bad.append(f"{col} (wszystko NaN)")
            continue

        stdev = float(np.std(s_clean))
        absmax = float(np.max(np.abs(s_clean)))
        if stdev < 1e-12:
            if absmax < 1e-15:
                flat_zero.append(col)
            else:
                flat_const.append(col)
        else:
            ok_sig.append(f"{col} (σ≈{stdev:.4g})")

    if nan_bad:
        report.checks.append(
            CheckResult(
                "error",
                "Poważne braki w sygnale",
                "; ".join(nan_bad),
            )
        )
    elif nan_warn:
        report.checks.append(
            CheckResult(
                "warn",
                "Częściowe braki (NaN/inf)",
                "Kanały: " + "; ".join(nan_warn) + " — rozważ naprawę lub wykluczenie odcinków.",
            )
        )
    else:
        report.checks.append(CheckResult("ok", "Kompletność próbek", "Brak masowych NaN/inf na torach fizjologicznych."))

    if flat_zero:
        report.checks.append(
            CheckResult(
                "warn",
                "Stałe zera na torach dodatkowych",
                "Kanały: "
                + ", ".join(flat_zero)
                + " — w pliku `.eeg` te próbki są **zerowe** (Ch65–68). Często: **brak podłączenia** Resp/GSR/HR, "
                "inny montaż kanałów albo **nie ten plik** próbek. EEG może być OK; sprawdź nagranie w Analyzerze.",
            )
        )
    if flat_const:
        report.checks.append(
            CheckResult(
                "warn",
                "„Płaski” sygnał (stała wartość ≠ 0)",
                "Kanały: "
                + ", ".join(flat_const)
                + " — brak zmienności (σ≈0); możliwy zepsuty tor, zła skala w `.vhdr` lub saturacja.",
            )
        )
    elif ok_sig:
        report.checks.append(
            CheckResult("ok", "Zmienność sygnału", "Kanały wykazują zmienność: " + "; ".join(ok_sig))
        )

    return report


def validate_transcript(
    utterances: list[Utterance],
    *,
    session_s: float,
) -> ValidationReport:
    """Spójność transkryptu z długością sesji i poprawność przedziałów."""
    report = ValidationReport(source="Transkrypt (JSON / CSV)")

    if not utterances:
        report.checks.append(
            CheckResult("error", "Pusta lista wypowiedzi", "Nie ma żadnej wypowiedzi — sprawdź plik i kodowanie.")
        )
        return report

    report.checks.append(CheckResult("ok", "Liczba wypowiedzi", f"{len(utterances)} segmentów tekstu."))

    bad_order = 0
    empty_text = 0
    neg_start = 0
    for u in utterances:
        if u.end_s <= u.start_s:
            bad_order += 1
        if not (u.text or "").strip():
            empty_text += 1
        if u.start_s < -1e-6:
            neg_start += 1

    if bad_order:
        report.checks.append(
            CheckResult(
                "error",
                "Niepoprawne przedziały czasu",
                f"{bad_order} wypowiedzi ma end_s ≤ start_s — popraw timestamps.",
            )
        )
    else:
        report.checks.append(CheckResult("ok", "Przedziały czasu", "Dla każdej wypowiedzi: start < koniec."))

    if neg_start:
        report.checks.append(
            CheckResult(
                "warn",
                "Ujemny start czasu",
                f"{neg_start} wypowiedzi ma start_s < 0 — upewnij się co do punktu odniesienia czasu.",
            )
        )

    max_end = max(u.end_s for u in utterances)
    if max_end > session_s * 1.02:
        report.checks.append(
            CheckResult(
                "warn",
                "Transkrypt dłuższy niż sesja",
                f"Ostatnia wypowiedź kończy się ~{max_end:.1f} s, sesja ~{session_s:.1f} s — "
                "sprawdź synchronizację lub skalowanie czasu.",
            )
        )
    elif max_end > session_s * 0.98:
        report.checks.append(
            CheckResult(
                "ok",
                "Zakres czasu vs sesja",
                f"Transkrypt mieści się w długości sesji (~{session_s:.0f} s).",
            )
        )
    else:
        report.checks.append(
            CheckResult(
                "ok",
                "Zakres czasu vs sesja",
                f"Ostatnia wypowiedź ~{max_end:.0f} s (sesja ~{session_s:.0f} s) — zapas czasu na końcu nagrania.",
            )
        )

    if empty_text:
        report.checks.append(
            CheckResult(
                "warn",
                "Puste teksty",
                f"{empty_text} wypowiedzi bez treści — może być OK (markery), ale sprawdź intencję.",
            )
        )

    overlaps = 0
    u_sorted = sorted(utterances, key=lambda x: x.start_s)
    for a, b in zip(u_sorted, u_sorted[1:]):
        if a.end_s > b.start_s + 1e-9:
            overlaps += 1
    if overlaps:
        report.checks.append(
            CheckResult(
                "warn",
                "Nakładające się wypowiedzi",
                f"Wykryto ~{overlaps} par z nakładaniem — możliwe u ELAN/Whisper; upewnij się, czy to zamierzone.",
            )
        )

    return report


def format_report_body(rep: ValidationReport) -> str:
    """Szczegóły listy sprawdzeń — do `st.markdown` pod komunikatem podsumowującym."""
    icon = {"ok": "✓", "warn": "⚠", "error": "✗"}
    lines = [f"**{rep.source}**", ""]
    for c in rep.checks:
        lines.append(f"- {icon[c.severity]} **{c.title}** — {c.detail}")
    return "\n".join(lines)
