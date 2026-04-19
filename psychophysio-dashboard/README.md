# PsyPhy Datalab

Dashboard w **Streamlit** do przeglądania w czasie sygnałów psychofizjologicznych (oddech, EDA, tor HR itd.) wraz z **transkryptem** — pod eksplorację pojedynczej sesji i przygotowanie pod dalszą analizę (m.in. psychofizjolingwistyka, cognitive science).

To **prototyp**: kod i interfejs mogą się zmieniać; wykresy i heurystyki QC nie zastępują opisu metody w publikacji ani procedur w laboratorium.

## Funkcje

- **Źródło danych:** tryb **syntetyczny** (demo) albo **BrainVision** — pliki `.vhdr` + `.eeg` w folderze `data/` (rekurencyjnie, także podfoldery). Ostatnie cztery kanały w typowym zapisie 68-kanałowym są mapowane na: oddech (Resp01T), drugi tor oddechu / „ECG” w UI (Resp02B), EDA (GSR), sygnał HR z nagłówka (µV; kolumna w kodzie historycznie `puls_bpm`).
- **Sesja i transkrypt:** cztery szeregi czasowe (surowe i wersja wygładzona / rzadsza), nakładka po normalizacji min–max, **nawigacja** po osi czasu (cała sesja / okno wokół kursora / segmenty), transkrypt w iframe z podświetleniem wg kursora.
- **Walidacja po wczytaniu** (expander): heurystyki dla BV (Fs, NaN, zmienność torów, stałe zera na dodatkowych kanałach) oraz dla transkryptu (przedziały czasu, długość vs sesja, nakładania itd.).
- **Galeria:** histogramy, korelacje, wykresy po segmentach, spektrogram (z decymacją przy bardzo długich nagraniach), uproszczony podział EDA tonic/phasic, zmienność pulsu w oknach.
- **QC / preprocessing — ECG:** tor `ecg_mv`, filtracja, detekcja R (heurystyka), histogram RR, krótkie podsumowanie jakości (parametryzowalne progi).

**Uwagi techniczne:** wykresy Plotly używają **decymacji kopertą min/max**, żeby nie przekraczać limitu rozmiaru wiadomości Streamlit przy długich nagraniach wysokiej częstotliwości. Parser `.vhdr` czyta **nazwy i rozdzielczości kanałów tylko z sekcji `[Channel Infos]`** — linie `ChN=…` z sekcji `[Coordinates]` nie nadpisują już skali.

## Wymagania

- Python 3.10+ (testowane m.in. na 3.14)
- Zależności w `requirements.txt` (Streamlit, Plotly, pandas, NumPy, SciPy)

## Instalacja i uruchomienie

1. **Sklonuj repozytorium** (lub rozpakuj archiwum) i przejdź do folderu projektu:
   ```bash
   cd psychophysio-dashboard
   ```
2. **Zainstaluj zależności:**
   ```bash
   python -m pip install -r requirements.txt
   ```
   Na Windowsie, jeśli `python` nie jest w PATH:
   ```bash
   py -3 -m pip install -r requirements.txt
   ```
3. **Dane BrainVision** (często **poza Gitem**: duży `.eeg`, wrażliwość danych): skopiuj lokalnie do `data/`:
   - plik **`*.vhdr`**,
   - **`*.eeg`** o nazwie z pola `DataFile=` w `[Common Infos]` (np. `DataFile=MK_0123.eeg` → `data/MK_0123.eeg` obok tego samego nagłówka),
   - opcjonalnie `*.vmrk` i inne pliki z sesji rekordera.
   Bez **niepustego** `.eeg` wczytanie próbek się nie uda — aplikacja pokaże komunikat z rozmiarem ścieżki.
4. **Transkrypt (opcjonalnie):** JSON lub CSV wg formatu z `data/transcript.example.json` / opisu w UI.
5. **Uruchom:**
   ```bash
   python -m streamlit run app.py
   ```
   (lub `py -3 -m streamlit run app.py` na Windowsie.)

Domyślnie: `http://localhost:8501`. W sidebarze: źródło danych, wybór `.vhdr`, ewentualnie plik transkryptu.

## Struktura projektu

| Plik / folder | Opis |
|---------------|------|
| `app.py` | Aplikacja Streamlit (zakładki: sesja, galeria, QC ECG) |
| `data_loader.py` | BrainVision: nagłówek `.vhdr`, multipleks INT16 z `.eeg` |
| `data_validation.py` | Raporty walidacji BV i transkryptu |
| `ecg_qc.py` | Preprocessing i heurystyczny QC toru `ecg_mv` |
| `transcript_io.py` | Wczytywanie transkryptu JSON / CSV |
| `viz_gallery.py` | Wykresy galerii |
| `data/` | Dane sesji, przykłady transkryptów; opcjonalnie `validation_samples/` (małe pliki BV do testów QC) |
| `.gitignore` | m.in. `__pycache__/` |

## Dane w `data/`

- **BrainVision:** spójny zestaw `.vhdr` + `.eeg` (i ewentualnie `.vmrk`). Pliki `Data-*.txt` to **logi impedancji**, nie szereg czasowy sygnału.
- **Repozytorium:** rozważ `.gitignore` na `*.eeg` / duże surowe pliki, jeśli nie mają trafiać na zdalne repo; **RODO / etyka badań** przy udostępnianiu danych.

## Zagadnienia do rozważenia

- **Synchronizacja czasu** między rekorderem BV, audio, transkryptem (wspólna oś w sekundach vs osobne korekty).
- **Publikacja vs eksploracja:** które wersje sygnału (surowy / filtrowany / decymowany) idą do rozdziałów metody i figur.
- **Segmentacja:** równe bloki czasu w UI vs **markery** z `.vmrk` lub zewnętrznego harmonogramu zadania.
- **Jednostki i nazewnictwo:** oś „puls” jako sygnał z czujnika (µV) vs wyliczone BPM; spójność podpisów osi z metodą.
- **QC i HRV:** obecna detekcja R to orientacyjny pipeline — przed analizą HRV warto ustalić finalny łańcuch z narzędziami klinicznymi lub zespołową procedurą.

## Dalsze kroki rozwoju (skrót)

- Wczytywanie **markerów** (np. z `.vmrk`) na oś czasu i warunki eksperymentalne.
- **Kwestionariusze** (CSV) i łączenie z `subject_id` / sesją na wykresach.
- Moduł **EEG** (osobna ścieżka czasu, epoki, ewentualnie MNE).
- Eksport figur (rozdzielczość, fonty) i ewentualnie testy automatyczne dla `data_loader` / walidacji.

## Licencja i cytowanie

Dodaj `LICENSE` i zasady cytowania, jeśli projekt ma być publiczny.

## Autorstwo

Uzupełnij autorów / kontakt według potrzeb pracy lub zespołu.
