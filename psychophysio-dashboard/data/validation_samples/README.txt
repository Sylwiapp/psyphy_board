Pliki testowe walidacji (PsyPhy Datalab)
=====================================

BrainVision (.vhdr + .eeg)
--------------------------
Wygenerowane skryptem generate_bv_fixtures.py (możesz odtworzyć: py data/validation_samples/generate_bv_fixtures.py).

  bv_val_ok       — sygnały na Ch65–68 OK (oczekuj: zielone podsumowanie).
  bv_val_flat     — ostatnie 4 kanały = 0 (oczekuj: ostrzeżenie „płaski sygnał”).
  bv_val_warn_fs  — Fs = 40 Hz (oczekuj: ostrzeżenie „nietypowa Fs”).

W aplikacji: Źródło sygnałów → BrainVision → wybierz plik z listy.

Transkrypt (.json)
------------------
W sidebarze: zaznacz wczytywanie transkryptu z pliku i wskaż plik.

  transcript_ok_short_session.json   — OK przy krótkim nagraniu (~8 s), np. bv_val_ok.
  transcript_ok_hour_demo.json       — OK przy trybie syntetycznym 60 min.

  transcript_bad_intervals.json      — błąd: end ≤ start.
  transcript_bad_empty.json          — błąd: pusta lista wypowiedzi.

  transcript_warn_beyond_session.json — ostrzeżenie: koniec > długość sesji.
  transcript_warn_overlap.json       — ostrzeżenie: nakładające się przedziały.
  transcript_warn_empty_text.json    — ostrzeżenie: puste teksty.
  transcript_warn_negative_start.json — ostrzeżenie: ujemny start_s.

Uwaga: przy „Dopisz przykład z transcript.example.json” wynik walidacji dotyczy połączonej listy.
