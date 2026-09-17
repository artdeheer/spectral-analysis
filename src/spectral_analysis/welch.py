import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, windows

# 1. Load the instrument signal
instrument_file = 'samples/sample8.mp3'
y, fs = librosa.load(instrument_file, sr=None)

# --- MANUAL WELCH'S METHOD ---
nperseg = 1024
noverlap = 512
step = nperseg - noverlap

# We use sym=False to perfectly match SciPy's default spectral window behavior
w = windows.hann(nperseg, sym=False) 
scale_factor = 1.0 / (fs * np.sum(w**2))

# Determine how many full segments we can extract
num_segments = (len(y) - noverlap) // step
Pxx_manual = np.zeros(nperseg // 2 + 1)

for i in range(num_segments):
    start = i * step
    segment = y[start : start + nperseg]
    
    windowed_segment = segment * w
    
    # rfft automatically computes only the positive frequencies (single-sided)
    Y = np.fft.rfft(windowed_segment)
    P = scale_factor * np.abs(Y)**2
    
    # Single-sided scaling: multiply by 2 (except DC and Nyquist frequencies)
    P[1:-1] *= 2.0 
    
    Pxx_manual += P

# Average all the periodograms
Pxx_manual /= num_segments
freqs_manual = np.fft.rfftfreq(nperseg, 1/fs)

# --- BUILT-IN WELCH'S METHOD (For Comparison) ---
freqs_builtin, Pxx_builtin = welch(y, fs=fs, window='hann', nperseg=1024, noverlap=512, scaling='density')

# --- REPEAT WITH SHORT AND LONG SEGMENTS ---
f_short, Pxx_short = welch(y, fs=fs, window='hann', nperseg=256, noverlap=128)
f_long, Pxx_long = welch(y, fs=fs, window='hann', nperseg=4096, noverlap=2048)

# --- PLOTTING ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot 1: Manual vs Built-in
ax1.plot(freqs_builtin, Pxx_builtin, label='SciPy Built-in', linewidth=3, alpha=0.6, color='blue')
ax1.plot(freqs_manual, Pxx_manual, label='Manual Implementation', linewidth=1, linestyle='--', color='red')
ax1.set_title("Welch's Method: Manual vs Built-in (1024 samples)")
ax1.set_ylabel("PSD")
ax1.set_xlim(0, 3000)
ax1.legend()
ax1.grid(True)

# Plot 2: Short vs Long Segments
ax2.plot(f_short, Pxx_short, label='Short Window (256)', alpha=0.8)
ax2.plot(f_long, Pxx_long, label='Long Window (4096)', alpha=0.8)
ax2.set_title("Welch's Method: Segment Length Comparison")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("PSD")
ax2.set_xlim(0, 3000)
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# --- NEW: Save the plot to a folder ---
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/welchs_method.png', bbox_inches='tight')
# --------------------------------------

plt.show()