import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# 1. Load the instrument signal
instrument_file = 'samples/sample8.mp3'
y_audio, fs_audio = librosa.load(instrument_file, sr=None)
N = len(y_audio)

# 2. Compute the discrete Fourier transform (FFT)
Y = np.fft.fft(y_audio)
freqs = np.fft.fftfreq(N, 1/fs_audio)

# 3. Double-Sided Amplitude Spectrum
# Shift zero frequency to the center for the double-sided plot
Y_shifted = np.fft.fftshift(Y)
freqs_shifted = np.fft.fftshift(freqs)

# Amplitude for double-sided is simply |Y| / N
amp_double = np.abs(Y_shifted) / N

plt.figure(figsize=(10, 4))
plt.plot(freqs_shifted, amp_double)
plt.title("Double-Sided Amplitude Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.xlim(-5000, 5000) # Zooming in to standard musical range limits
plt.grid(True)

# --- NEW: Save the plot to a folder ---
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/double_sided_spectrum.png', bbox_inches='tight')
plt.show() # You can remove this if you only want to save it without popping up a window
# --------------------------------------

# 4. Extract frequencies and map to notes
amp_single = 2.0 / N * np.abs(Y[:N//2])
freqs_single = freqs[:N//2]

# Ignore frequencies below 50 Hz by setting their amplitudes to 0
amp_single[freqs_single < 50] = 0

# Find peaks (setting a threshold at 10% of the max amplitude to ignore quiet noise)
threshold = np.max(amp_single) * 0.10
peaks, _ = find_peaks(amp_single, height=threshold, distance=50)

def hz_to_note(f):
    if f <= 0: return "N/A"
    # Using the standard MIDI note formula where A4 = 440 Hz is MIDI note 69
    midi = int(round(12 * np.log2(f / 440.0) + 69))
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    name = notes[midi % 12]
    octave = (midi // 12) - 1
    return f"{name}{octave}"

print("--- INSTRUMENT FREQUENCIES ---")
if len(peaks) > 0:
    fundamental = freqs_single[peaks[0]]
    for i, p in enumerate(peaks):
        f = freqs_single[p]
        amp = amp_single[p]
        note = hz_to_note(f)
        multiple = f / fundamental
        print(f"Peak {i+1}: {f:.2f} Hz | Note: {note} | Amp: {amp:.4f} | Ratio to Fundamental: {multiple:.2f}x")
else:
    print("No prominent frequencies found.")