import librosa
import numpy as np
from scipy.signal import find_peaks

# --- 1. VERIFY TEST SIGNAL ---
test_file = 'samples/test_440hz1.0_1000hz0.5.mp3'
y_test, fs_test = librosa.load(test_file, sr=None)

N_test = len(y_test)
Y_test = np.fft.fft(y_test)
freqs_test = np.fft.fftfreq(N_test, 1/fs_test)

amp_test = 2.0 / N_test * np.abs(Y_test[:N_test//2])
freqs_test = freqs_test[:N_test//2]

# Find peaks (lowered threshold slightly to ensure we catch the 0.318 peak)
peaks, _ = find_peaks(amp_test, height=0.1)

print("--- TEST SIGNAL VERIFICATION ---")
if len(peaks) >= 2:
    # Extract the two largest peaks and sort them by frequency
    top_peaks_idx = peaks[np.argsort(amp_test[peaks])[-2:]]
    sorted_peaks_idx = sorted(top_peaks_idx, key=lambda idx: freqs_test[idx])
    
    f1, a1 = freqs_test[sorted_peaks_idx[0]], amp_test[sorted_peaks_idx[0]]
    f2, a2 = freqs_test[sorted_peaks_idx[1]], amp_test[sorted_peaks_idx[1]]
    
    ratio = a1 / a2
    
    print(f"Peak 1: {f1:.1f} Hz, Amplitude: {a1:.3f}")
    print(f"Peak 2: {f2:.1f} Hz, Amplitude: {a2:.3f}")
    print(f"Amplitude Ratio (Peak 1 / Peak 2): {ratio:.3f}")
    
    # Check if frequencies match and the ratio is approximately 2.0 (1.0 / 0.5)
    freqs_match = (435 < f1 < 445) and (995 < f2 < 1005)
    ratio_match = (1.9 < ratio < 2.1) 
    
    if freqs_match and ratio_match:
        print("SUCCESS: Frequencies match and amplitude ratio is perfectly 2:1!\n")
    else:
        print("WARNING: Test signal did not match expected inputs or ratio.\n")
else:
    print("WARNING: Could not find at least 2 peaks.\n")


# --- 2. PROCESS INSTRUMENT SIGNAL ---
instrument_file = 'samples/sample8.mp3'
print("--- INSTRUMENT SIGNAL PROPERTIES ---")
y_audio, fs_audio = librosa.load(instrument_file, sr=None) 

# Calculate the required properties
nyquist_freq = fs_audio / 2
sample_length = len(y_audio)
duration_sec = sample_length / fs_audio

# Print the results
print(f"File loaded: {instrument_file}")
print(f"Sampling Frequency (fs): {fs_audio} Hz")
print(f"Nyquist Frequency: {nyquist_freq} Hz")
print(f"Sample Length: {sample_length} samples")
print(f"Duration: {duration_sec:.2f} seconds")