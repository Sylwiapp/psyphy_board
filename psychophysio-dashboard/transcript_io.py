"""
Uniwersalne wczytywanie transkryptu z timestamami (PsyPhy Datalab / psychofizjolingwistyka).

Obsługiwane formaty:
1) JSON — zalecany do metadanych i walidacji
2) CSV — prosty eksport z ELAN / Praat / skryptu

Schemat JSON (minimalny):
{
  "version": 1,
  "time_reference": "seconds_from_session_start",
  "utterances": [ { "start_s": float, "end_s": float, "text": str }, ... ]
}

CSV (nagłówki):
  start_s,end_s,text
  (separator przecinek; tekst w cudzysłowie jeśli zawiera przecinek)
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from typing import BinaryIO, List


@dataclass(frozen=True)
class Utterance:
    start_s: float
    end_s: float
    text: str


def load_transcript_json_bytes(raw: bytes) -> List[Utterance]:
    data = json.loads(raw.decode("utf-8-sig"))
    out: List[Utterance] = []
    for u in data.get("utterances", []):
        out.append(
            Utterance(
                start_s=float(u["start_s"]),
                end_s=float(u["end_s"]),
                text=str(u.get("text", "")).strip(),
            )
        )
    out.sort(key=lambda x: x.start_s)
    return out


def load_transcript_csv_bytes(raw: bytes) -> List[Utterance]:
    text = raw.decode("utf-8-sig")
    r = csv.DictReader(io.StringIO(text))
    if not r.fieldnames:
        return []
    # akceptuj warianty nazw kolumn
    def pick(row: dict, *names: str) -> str | None:
        lower = {k.lower().strip(): v for k, v in row.items() if k}
        for n in names:
            if n.lower() in lower:
                return lower[n.lower()]
        return None

    out: List[Utterance] = []
    for row in r:
        s = pick(row, "start_s", "t0", "start", "begin_s")
        e = pick(row, "end_s", "t1", "end")
        t = pick(row, "text", "transcript", "utterance")
        if s is None or e is None:
            continue
        out.append(Utterance(start_s=float(s), end_s=float(e), text=(t or "").strip()))
    out.sort(key=lambda x: x.start_s)
    return out


def load_transcript_auto(fileobj: BinaryIO, name: str) -> List[Utterance]:
    raw = fileobj.read()
    lower = name.lower()
    if lower.endswith(".json"):
        return load_transcript_json_bytes(raw)
    if lower.endswith(".csv"):
        return load_transcript_csv_bytes(raw)
    try:
        return load_transcript_json_bytes(raw)
    except json.JSONDecodeError:
        return load_transcript_csv_bytes(raw)
