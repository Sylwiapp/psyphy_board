# Notatki robocze do magisterki (HRV / preprocessing)

> Plik roboczy — przypomnienia „do opracowania”, nie finalny tekst pracy.
> Skróty źródeł: Task Force (1996); Quigley et al. (2024, *Psychophysiology* 61:e14604);
> Laborde et al. (2017); Shaffer & Ginsberg (2017); Grossman & Taylor (2007); Billman (2013); Sacha (2014).

## DO OPRACOWANIA — Metoda / Ograniczenia HRV

Analiza opiera się na segmentach ~2–3 min (pojedyncze bodźce wideo CASE) = zakres **krótkoczasowego HRV**.
Wynikają z tego ograniczenia, które muszę uwzględnić i opisać:

1. **Dobór metryk wg długości okna.**
   - Wiarygodne w moich oknach: **RMSSD, SDNN, pNN50** (≥ ~60 s) oraz **HF** (≥ ~60 s).
   - Na granicy: **LF, LF/HF, Total Power** (≥ ~120 s) — raportować z ostrożnością lub pominąć.
   - **Nie używam** metryk dla nagrań długich/dobowych: VLF, ULF, SDANN, indeks trójkątny (HTI), TINN, DFA α1/α2, SD2.
2. **RMSSD jako główna miara wagalna** — preferowana nad HF, bo mniej wrażliwa na oddech i artefakty (Laborde 2017).
3. **Konfund oddechowy HF.** HF = rytm zatokowo-oddechowy (RSA), zależy od częstości/głębokości oddechu. Muszę:
   (a) policzyć częstość oddechu z kanału `rsp` (CASE go ma),
   (b) sprawdzić, czy mieści się w 9–24/min,
   (c) sprawdzić, czy **nie różni się między warunkami emocjonalnymi** — inaczej różnice HF mogą wynikać z oddechu, a nie z autonomii (Grossman & Taylor 2007).
4. **SDNN zależy od długości nagrania** — porównuję go tylko między oknami o **tej samej** długości (Task Force 1996).
5. **LF/HF** nie interpretuję jako „balansu współczulno-wagalnego” (Billman 2013).
6. **HRV a średnie HR** — sprawdzić, czy różnice HRV nie wynikają z różnic HR między warunkami;
   rozważyć raportowanie HR obok HRV lub korektę wg Sacha (2014). *(decyzja później)*
7. **Narzędzie (dashboard)** to dodatek **dydaktyczny** pokazujący wpływ preprocessingu na sygnał i metryki
   krótkoczasowe — **nie** kliniczny pipeline HRV. Zaznaczyć jako ograniczenie.

## Tabela: metryka → minimalna długość → status dla okna ~120–180 s

| Metryka        | Min. zalecana długość | Status dla ~2–3 min |
|----------------|-----------------------|---------------------|
| MeanNN         | ~10 s                 | ✅ wiarygodne       |
| RMSSD          | ~60 s                 | ✅ wiarygodne       |
| SDNN           | ~60 s                 | ✅ (nie porównuj między różnymi długościami) |
| pNN50          | ~60 s                 | ✅ wiarygodne       |
| HF             | ~60 s                 | ✅ (ale konfund oddechu) |
| LF             | ~120 s                | ⚠️ ostrożnie        |
| LF/HF, TP      | ~120 s                | ⚠️ ostrożnie / unikać LF/HF |
| VLF, SDANN, DFA-α1, SD2 | ~300 s       | ❌ niewiarygodne    |
| HTI, TINN, DFA-α2 | ~20 min            | ❌ niewiarygodne    |
| ULF            | ~24 h                 | ❌ niewiarygodne    |

## Źródła do zacytowania
- Task Force ESC/NASPE (1996). *Heart rate variability: standards of measurement.* Circulation 93(5):1043–1065.
- Quigley, K. S. et al. (2024). *Publication guidelines for human heart rate and HRV studies — Part 1.* Psychophysiology 61:e14604.
- Laborde, S., Mosley, E., & Thayer, J. F. (2017). *HRV and Cardiac Vagal Tone — Recommendations.* Frontiers in Psychology 8:213.
- Shaffer, F., & Ginsberg, J. P. (2017). *An Overview of Heart Rate Variability Metrics and Norms.* Frontiers in Public Health 5:258.
- Grossman, P., & Taylor, E. W. (2007). *Toward understanding respiratory sinus arrhythmia.* Biological Psychology 74(2):263–285.
- Billman, G. E. (2013). *The LF/HF ratio does not accurately measure cardiac sympatho-vagal balance.* Frontiers in Physiology 4:26.
- Sacha, J. (2014). *Interaction between heart rate and heart rate variability.* Ann. Noninvasive Electrocardiol. 19(3):207–216.
