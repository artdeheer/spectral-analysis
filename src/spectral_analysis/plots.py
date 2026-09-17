import os
import numpy as np
import matplotlib.pyplot as plt
import librosa

# 1. Load the signal
test_signal = 'samples/test_440hz1.0_1000hz0.5.mp3'
signal = 'samples/sample8.mp3'
y_test, fs_test = librosa.load(signal)

N = len(y_test)
freqs = np.fft.fftfreq(N, 1/fs_test)[:N//2]

# 2. Apply the Hann Window
w_hann = np.hanning(N)
y_windowed = y_test * w_hann

# 3. Compute FFT of the windowed signal
Y_windowed = np.fft.fft(y_windowed)
Y_single = Y_windowed[:N//2]

# 4. Power Spectrum (PS) Calculation
# Coherent gain correction for amplitude loss: sum(w) / N (which is the mean of the window)
# We divide the amplitude by this mean, and multiply by 2 for the single-sided spectrum.
mean_w = np.mean(w_hann) 
amp_corrected = (2.0 / N) * (np.abs(Y_single) / mean_w)
power_spectrum = amp_corrected**2

# 5. Power Spectral Density (PSD) Calculation
# Incoherent gain correction for energy loss: sum(w^2)
# Formula scales by 2 (single-sided) and divides by (fs * sum(w^2))
psd_correction = 2.0 / (fs_test * np.sum(w_hann**2))
power_spectral_density = psd_correction * np.abs(Y_single)**2

# --- Verify Test Signal Amplitudes ---
print("--- SIGNAL PS VERIFICATION ---")
peak_indices = np.argsort(power_spectrum)[-2:]
for idx in peak_indices:
    f = freqs[idx]
    # Square root of Power Spectrum = Peak Amplitude
    recovered_amp = np.sqrt(power_spectrum[idx])
    print(f"Frequency: {f:.1f} Hz | Recovered Amplitude: {recovered_amp:.3f}")

# --- Plotting ---
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

ax1.plot(freqs, power_spectrum)
ax1.set_title("Single-Sided Power Spectrum (PS) with Hann Window")
ax1.set_ylabel("Power (Amplitude^2)")
ax1.set_xlim(0, 3000)
ax1.grid(True)

ax2.plot(freqs, power_spectral_density)
ax2.set_title("Single-Sided Power Spectral Density (PSD) with Hann Window")
ax2.set_xlabel("Frequency (Hz)")
ax2.set_ylabel("Power / Hz")
ax2.set_xlim(0, 3000)
ax2.grid(True)

plt.tight_layout()

# --- NEW: Save the plot to a folder ---
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/power_spectra.png', bbox_inches='tight')
# --------------------------------------

plt.show()