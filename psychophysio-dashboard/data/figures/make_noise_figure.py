# -*- coding: utf-8 -*-
"""Generate an illustrative figure of common ECG noise types.

Produces one stacked-panel PNG showing a clean ECG and the same trace
contaminated by: baseline wander, 50 Hz powerline interference,
broadband EMG noise, and exaggerated (spectrally overlapping) T-waves.
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import neurokit2 as nk

rng = np.random.default_rng(42)

FS = 500          # sampling rate [Hz]
DURATION = 5      # seconds
HR = 70           # heart rate [bpm]

# --- clean ECG ---------------------------------------------------------
ecg = nk.ecg_simulate(duration=DURATION, sampling_rate=FS, heart_rate=HR,
                      method="ecgsyn", noise=0, random_state=42)
ecg = np.asarray(ecg, dtype=float)
ecg = ecg / np.max(np.abs(ecg))          # normalise so R-peak ~ 1
t = np.arange(len(ecg)) / FS

# show a 4 s window for readability
mask = t <= 4.0
t = t[mask]
ecg = ecg[mask]

# --- 1) baseline wander (very low frequency, < 0.5 Hz) -----------------
baseline = 0.45 * np.sin(2 * np.pi * 0.3 * t) + 0.20 * np.sin(2 * np.pi * 0.12 * t)
ecg_bw = ecg + baseline

# --- 2) powerline interference (narrowband, exactly 50 Hz) -------------
powerline = 0.15 * np.sin(2 * np.pi * 50 * t)
ecg_pl = ecg + powerline

# --- 3) EMG / muscle noise (broadband, high-frequency) -----------------
emg = rng.standard_normal(len(t))
# crude high-pass so it sits above ~20 Hz (broadband, overlaps QRS band)
emg = emg - np.convolve(emg, np.ones(25) / 25, mode="same")
emg = 0.12 * emg / np.max(np.abs(emg)) * 3
ecg_emg = ecg + emg

# --- 4) tall T-waves (real P/T waves overlapping the QRS band) ---------
# add a Gaussian bump ~250 ms after each R-peak to exaggerate the T-wave
try:
    _, info = nk.ecg_peaks(ecg, sampling_rate=FS)
    rpeaks = info["ECG_R_Peaks"]
except Exception:
    rpeaks = np.where(ecg > 0.6)[0]
    rpeaks = rpeaks[np.insert(np.diff(rpeaks) > int(0.3 * FS), 0, True)]

tall_t = np.zeros_like(ecg)
width = int(0.05 * FS)
offset = int(0.25 * FS)
for r in rpeaks:
    c = r + offset
    if c < len(ecg):
        idx = np.arange(len(ecg))
        tall_t += 0.55 * np.exp(-0.5 * ((idx - c) / width) ** 2)
ecg_t = ecg + tall_t

# --- plot --------------------------------------------------------------
panels = [
    (ecg,     "Clean ECG  (P-QRS-T visible)",                         "#1f3b73"),
    (ecg_bw,  "+ Baseline wander  (low-frequency drift, < 0.5 Hz)",   "#1f3b73"),
    (ecg_pl,  "+ Powerline interference  (narrowband 50 Hz hum)",     "#1f3b73"),
    (ecg_emg, "+ Muscle / EMG noise  (broadband, high-frequency)",    "#1f3b73"),
    (ecg_t,   "Tall T-waves  (P/T waves overlap the QRS band)",       "#1f3b73"),
]

fig, axes = plt.subplots(len(panels), 1, figsize=(11, 11), sharex=True)
for ax, (sig, title, color) in zip(axes, panels):
    ax.plot(t, sig, color=color, lw=1.0)
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold")
    ax.set_yticks([])
    ax.margins(x=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)

# annotate P, QRS, T on the clean panel
ax0 = axes[0]
ax0.set_ylim(-0.65, 1.65)
if len(rpeaks) > 0:
    r0 = rpeaks[0] if rpeaks[0] < len(t) else 0
    tr = t[r0]
    ax0.annotate("QRS", xy=(tr, ecg[r0]), xytext=(tr, 1.35),
                 ha="center", fontsize=9, color="#b00020",
                 arrowprops=dict(arrowstyle="->", color="#b00020"))
    ax0.annotate("T", xy=(tr + 0.25, 0.15), xytext=(tr + 0.25, 0.6),
                 ha="center", fontsize=9, color="#0a7d2c",
                 arrowprops=dict(arrowstyle="->", color="#0a7d2c"))
    ax0.annotate("P", xy=(tr - 0.16, 0.1), xytext=(tr - 0.16, 0.55),
                 ha="center", fontsize=9, color="#8a5a00",
                 arrowprops=dict(arrowstyle="->", color="#8a5a00"))

axes[-1].set_xlabel("Time [s]", fontsize=10)
fig.suptitle("Common noise types on the ECG signal", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.98])

out = os.path.join(os.path.dirname(__file__), "ecg_noise_types.png")
fig.savefig(out, dpi=140)
print("saved:", out)
