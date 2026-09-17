import os
import librosa
import matplotlib.pyplot as plt

# 1. Load the instrument signal
instrument_file = 'samples/sample8.mp3'
y, fs = librosa.load(instrument_file, sr=None)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Spectrogram 1: Short Window (256 samples)
# NFFT sets the window length. We use a 50% overlap.
ax1.specgram(y, NFFT=256, Fs=fs, noverlap=128, cmap='inferno')
ax1.set_title("Spectrogram: Short Window (NFFT = 256)")
ax1.set_ylabel("Frequency (Hz)")
ax1.set_ylim(0, 5000) # Zoom in to the musical range

# Spectrogram 2: Long Window (4096 samples)
ax2.specgram(y, NFFT=4096, Fs=fs, noverlap=2048, cmap='inferno')
ax2.set_title("Spectrogram: Long Window (NFFT = 4096)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Frequency (Hz)")
ax2.set_ylim(0, 5000)

plt.tight_layout()

# --- NEW: Save the plot to a folder ---
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/spectrograms.png', bbox_inches='tight')
# --------------------------------------

plt.show()