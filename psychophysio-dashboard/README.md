# PsyPhy Datalab

Dashboard w **Streamlit** do przeglądania w czasie sygnałów psychofizjologicznych (oddech, EDA, tor HR itd.) wraz z **transkryptem** — pod eksplorację pojedynczej sesji i przygotowanie pod dalszą analizę (m.in. psychofizjolingwistyka, cognitive science).

To **prototyp**: kod i interfejs mogą się zmieniać; wykresy i heurystyki QC nie zastępują opisu metody w publikacji ani procedur w laboratorium.

## Funkcje

- **Źródło danych:** tryb **syntetyczny** (demo), **BrainVision** — pliki `.vhdr` + `.eeg` w folderze `data/` (rekurencyjnie, także podfoldery), albo **CASE** (Sharma et al. 2019) — otwarty zbiór z **prawdziwym ECG** (patrz sekcja „Dane CASE” niżej). Ostatnie cztery kanały w typowym zapisie BrainVision 68-kanałowym są mapowane na: oddech (Resp01T), drugi tor oddechu / „ECG” w UI (Resp02B), EDA (GSR), sygnał HR z nagłówka (µV; kolumna w kodzie historycznie `puls_bpm`).
- **Sesja i transkrypt:** cztery szeregi czasowe (surowe i wersja wygładzona / rzadsza), nakładka po normalizacji min–max, **nawigacja** po osi czasu (cała sesja / okno wokół kursora / **segmenty**), transkrypt w iframe z podświetleniem wg kursora. **Segmenty** przy BrainVision: z **`.vmrk`** — granice z markerów **`New Segment`** (czas z pozycji próbek i Fs z nagłówka). Gdy markery dają **tylko jeden** segment (np. jeden `New Segment` na początku pliku), używany jest **równy podział na 6 bloków** jak przy danych syntetycznych; bez markerów `New Segment` — tak samo **6 równych** bloków.
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
3. **Dodaj własne dane do folderu `data/`** (w katalogu projektu, obok `app.py` — folder nazywa się dokładnie **`data`**). Tam kładziesz pliki sesji, które chcesz przeglądać; aplikacja **skanuje `data/` rekurencyjnie** (możesz też użyć podfolderów). Surowe nagrania zwykle **nie są commitowane** do Gita — patrz `.gitignore`.
   - **BrainVision:** skopiuj **`*.vhdr`** oraz **`*.eeg`** o nazwie z pola `DataFile=` w `[Common Infos]` nagłówka (np. `DataFile=MK_0123.eeg` → plik `data/MK_0123.eeg` **obok** tego samego `.vhdr`), opcjonalnie **`*.vmrk`** i inne pliki z rekordera. Bez **niepustego** `.eeg` wczytanie próbek się nie uda — w UI pojawi się komunikat z rozmiarem ścieżki.
   - **Transkrypt (opcjonalnie):** JSON lub CSV wg formatu z `data/transcript.example.json` / opisu w aplikacji — możesz trzymać go w `data/` i wskazać w sidebarze.
4. **Uruchom:**
   ```bash
   python -m streamlit run app.py
   ```
   (lub `py -3 -m streamlit run app.py` na Windowsie.)

Domyślnie: `http://localhost:8501`. W sidebarze: źródło danych, wybór `.vhdr` z listy (pliki z `data/`), ewentualnie plik transkryptu.

## Struktura projektu

| Plik / folder | Opis |
|---------------|------|
| `app.py` | Aplikacja Streamlit (zakładki: sesja, galeria, QC ECG) |
| `data_loader.py` | BrainVision: nagłówek `.vhdr`, multipleks INT16 z `.eeg` |
| `case_loader.py` | CASE: wczytanie `interpolated/physiological/sub_*.csv`, segmenty z kolumny `video` |
| `bv_markers.py` | Parsowanie `.vmrk`, czasy markerów **New Segment** |
| `session_geom.py` | Krawędzie segmentów (równy podział albo z markerów) |
| `data_validation.py` | Raporty walidacji BV i transkryptu |
| `ecg_qc.py` | Preprocessing i heurystyczny QC toru `ecg_mv` |
| `transcript_io.py` | Wczytywanie transkryptu JSON / CSV |
| `viz_gallery.py` | Wykresy galerii |
| `data/` | Dane sesji, przykłady transkryptów; opcjonalnie `validation_samples/` (małe pliki BV do testów QC) |
| `.gitignore` | m.in. `__pycache__/` |

## Dane w `data/`

- **BrainVision:** spójny zestaw `.vhdr` + `.eeg` (i ewentualnie `.vmrk`). Pliki `Data-*.txt` to **logi impedancji**, nie szereg czasowy sygnału.
- **Repozytorium:** w `.gitignore` są wzorce na surowe pliki sesji w `data/` (`.eeg`, `.vhdr`, `.vmrk`, logi impedancji, transkrypt JSON poza przykładem, folder `articles/`), a także zbiór CASE (`data/CASE_full/`). **Małe fixture’y** zostają w `data/validation_samples/`. **RODO / etyka** przy udostępnianiu danych poza Gitem.

## Dane CASE (Sharma et al. 2019)

CASE to otwarty zbiór ciągłych sygnałów fizjologicznych nagranych podczas oglądania filmów emocjonalnych (30 osób). Zawiera **prawdziwe ECG**, dlatego używamy go do zakładki QC/preprocessing ECG i analiz HRV.

- **Źródło:** Sharma, K., Castellini, C., van den Broek, E. L., Albu-Schaeffer, A., Schwenker, F. (2019). *A dataset of continuous affect annotations and physiological signals for emotion analysis.* Scientific Data 6, 196. DOI **10.1038/s41597-019-0209-0**. Pełna paczka danych jest hostowana na **figshare** (link w artykule; pobierasz archiwum „full”).
- **Dane są poza Gitem** (`.gitignore`) — po sklonowaniu repo musisz pobrać CASE samodzielnie.
- **Rozpakowanie:** wypakuj tak, aby istniała ścieżka
  `data/CASE_full/data/interpolated/physiological/sub_*.csv`
  (loader szuka katalogu zawierającego `.../interpolated/physiological`, więc dokładna nazwa folderu nadrzędnego nie jest krytyczna).
- **Wariant:** korzystamy z **`interpolated`** (1000 Hz, równa siatka czasu). To jest **sygnał surowy** — CASE **nie filtruje** fizjologii (dodaje tylko video-ID, konwersję jednostek części kanałów i interpolację liniową). Filtracja (Stage 2: HP/LP/notch) jest **po Twojej stronie** w aplikacji.
- **Mapowanie kanałów** (w `case_loader.py`):

| CASE | kolumna aplikacji | uwaga |
|------|-------------------|-------|
| `ecg` | `ecg_mv` | **prawdziwe ECG** (w woltach) |
| `rsp` | `oddech` | pas oddechowy |
| `gsr` | `eda_us` | przewodność skóry |
| `bvp` | `puls_bpm` | BVP/PPG — **przebieg pulsu, nie BPM** |
| `daqtime/1000` | `time_s` | `daqtime` jest w ms |

- **Segmenty = wideo:** kolumna `video` koduje bodziec; loader buduje z niej segmenty (8 wideo emocjonalnych + `bluVid`/`startVid`/`endVid`). W sidebarze wybierasz **podmiot** i **segment** — domyślnie pierwsze wideo emocjonalne (~2–3 min, szybki pipeline). Opcja **„Cała sesja”** ładuje ~2,5 mln próbek (wolne).

## Zagadnienia do rozważenia

- **Synchronizacja czasu** między rekorderem BV, audio, transkryptem (wspólna oś w sekundach vs osobne korekty).
- **Publikacja vs eksploracja:** które wersje sygnału (surowy / filtrowany / decymowany) idą do rozdziałów metody i figur.
- **Segmentacja:** inne typy markerów BV (Stimulus, Response…) na osi czasu i w etykietach bloków — obecnie tylko **New Segment** buduje podział.
- **Jednostki i nazewnictwo:** oś „puls” jako sygnał z czujnika (µV) vs wyliczone BPM; spójność podpisów osi z metodą.
- **QC i HRV:** obecna detekcja R to orientacyjny pipeline — przed analizą HRV warto ustalić finalny łańcuch z narzędziami klinicznymi lub zespołową procedurą.

## Dalsze kroki rozwoju (skrót)

- Podgląd **wszystkich** markerów `.vmrk` na osi czasu i wybór typu do segmentacji (nie tylko **New Segment**).
- **Kwestionariusze** (CSV) i łączenie z `subject_id` / sesją na wykresach.
- Moduł **EEG** (osobna ścieżka czasu, epoki, ewentualnie MNE).
- Eksport figur (rozdzielczość, fonty) i ewentualnie testy automatyczne dla `data_loader` / walidacji.

## Licencja i cytowanie

Dodaj `LICENSE` i zasady cytowania, jeśli projekt ma być publiczny.

## Autorstwo

Uzupełnij autorów / kontakt według potrzeb pracy lub zespołu.
