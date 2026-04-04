# PsyPhy Datalab

Dashboard w **Streamlit** do przeglądania zsynchronizowanych w czasie sygnałów psychofizjologicznych (oddech, EDA, sygnał HR itd.) oraz transkryptu — w kontekście badań z psychofizjolingwistyki i cognitive science.

Stan repozytorium odpowiada prototypowi narzędzia do eksploracji sesji, rozmów z promotorką o wizualizacjach oraz przygotowania pod dalsze moduły (EEG, kwestionariusze, agregaty grupowe).

## Funkcje (skrót)

- **Źródło danych:** tryb syntetyczny (demo) albo pliki **BrainVision** (`.vhdr` + `.eeg` w folderze `data`) — kanały pomocnicze typu Resp / GSR / HR z nagłówka nagrania.
- **Wykresy:** szeregi czasowe (subploty), nakładka czterech kanałów po normalizacji min–max, nawigacja po osi czasu (pełna sesja / okno wokół kursora / segment).
- **Transkrypt:** JSON lub CSV (`start_s`, `end_s`, `text`) — podświetlanie wypowiedzi wg kursora czasu; klik w wykres Plotly ustawia kursor przy kolejnym odświeżeniu (ograniczenie Streamlit).
- **Galeria wizualizacji:** histogramy, korelacje, boxploty po segmentach, spektrogram (SciPy), uproszczony podział EDA tonic/phasic, itd.

## Wymagania

- Python 3.10+ (testowane m.in. na 3.14)
- Zależności w `requirements.txt`

## Instalacja i uruchomienie

```bash
cd psychophysio-dashboard
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Na Windowsie, jeśli `python` nie jest w PATH:

```bash
py -3 -m pip install -r requirements.txt
py -3 -m streamlit run app.py
```

Aplikacja otworzy się w przeglądarce (domyślnie `http://localhost:8501`).

## Struktura projektu

| Plik / folder | Opis |
|---------------|------|
| `app.py` | Główna aplikacja Streamlit |
| `data_loader.py` | Wczytywanie BrainVision (nagłówek + binarny `.eeg`) |
| `transcript_io.py` | Wczytywanie transkryptu JSON / CSV |
| `viz_gallery.py` | Dodatkowe typy wykresów (galeria) |
| `data/` | Przykładowe metadane / nagłówki; **pełne nagrania binarne zwykle nie są commitowane** (duży rozmiar, dane osobowe) |

## Dane w `data/`

- **BrainVision:** obok `*.vhdr` musi leżeć plik `*.eeg` wskazany w nagłówku (np. `DataFile=MK_0123.eeg`). Bez `.eeg` wczytanie próbek nie jest możliwe.
- Pliki `Data-*.txt` z logów impedancji **nie** są szeregami czasowymi sygnałów.
- Transkrypt: zobacz `data/transcript.example.json` i opis w aplikacji (expander „Format plików transkryptu”).

Przed publikacją repozytorium na GitHubie rozważ:

- dodanie `*.eeg` (i ewentualnie dużych surowych plików) do `.gitignore`;
- nie umieszczanie danych osobowych bez zgody i zgodnie z regulacjami (RODO / etyka badań).

## Licencja i cytowanie

Dopisz plik `LICENSE` oraz ewentualnie sposób cytowania, jeśli projekt ma być udostępniany publicznie.

## Autorstwo

Uzupełnij sekcję autorów / kontaktu według potrzeb zespołu lub pracy dyplomowej.
