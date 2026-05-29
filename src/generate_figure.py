import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
from pathlib import Path

DATA = Path("data/raw_imu_samples.csv")
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(DATA)

# Pick example activities
activities = ["using_utensil", "pouring", "assisting"]
channel = "wrist_acc_x"

# ---------- Figure 1: Time-domain signals ----------
plt.figure(figsize=(7, 4))

for act in activities:
    sample = df[df["atomic_activity"].eq(act) | df["interaction_type"].eq(act)]
    if len(sample) == 0:
        continue
    sample = sample.sort_values(["session_id", "user_id", "sample_index"]).head(250)
    t = np.arange(len(sample)) / 50.0
    plt.plot(t, sample[channel].values, label=act)

plt.xlabel("Time (s)")
plt.ylabel("Acceleration (m/s$^2$)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "imu_signal_comparison.png", dpi=300)
plt.savefig(OUT / "imu_signal_comparison.pdf")
plt.close()

# ---------- Figure 2: Frequency-domain comparison ----------
plt.figure(figsize=(7, 4))

for act in activities:
    sample = df[df["atomic_activity"].eq(act) | df["interaction_type"].eq(act)]
    if len(sample) == 0:
        continue
    x = sample.sort_values(["session_id", "user_id", "sample_index"])[channel].values[:512]
    x = x - np.mean(x)
    freqs = np.fft.rfftfreq(len(x), d=1/50.0)
    amp = np.abs(np.fft.rfft(x))
    plt.plot(freqs, amp, label=act)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Spectral Amplitude")
plt.xlim(0, 15)
plt.legend()
plt.tight_layout()
plt.savefig(OUT / "frequency_domain_comparison.png", dpi=300)
plt.savefig(OUT / "frequency_domain_comparison.pdf")
plt.close()

# ---------- Figure 3: Spectrogram ----------
sample = df[df["atomic_activity"].eq("using_utensil")]
sample = sample.sort_values(["session_id", "user_id", "sample_index"])
x = sample[channel].values[:1000]
x = x - np.mean(x)

f, t, Sxx = spectrogram(x, fs=50, nperseg=128, noverlap=64)

plt.figure(figsize=(7, 4))
plt.pcolormesh(t, f, Sxx, shading="gouraud")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power")
plt.tight_layout()
plt.savefig(OUT / "using_utensil_spectrogram.png", dpi=300)
plt.savefig(OUT / "using_utensil_spectrogram.pdf")
plt.close()

print("Figures saved in:", OUT)
