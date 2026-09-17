import os
import librosa
import numpy as np
from pydub import AudioSegment

# 1. Load original flute recording
y, fs = librosa.load('samples/sample8.mp3', sr=None)

# 2. Compute the FFT and filter out the noise
Y = np.fft.fft(y)
freqs = np.fft.fftfreq(len(y), 1/fs)

# Zero out frequencies between -50 Hz and 50 Hz
Y[(freqs > -50) & (freqs < 50)] = 0

# 3. Reconstruct the audio using Inverse FFT (IFFT)
y_filtered = np.fft.ifft(Y).real

# 4. Prepare audio for export (Normalize to 16-bit PCM)
y_filtered_norm = y_filtered / np.max(np.abs(y_filtered))
y_int16 = np.int16(y_filtered_norm * 32767)

# 5. Create the output directory
os.makedirs('ifft', exist_ok=True)

# 6. Convert the numpy array to a pydub AudioSegment and export as MP3
audio_segment = AudioSegment(
    y_int16.tobytes(),
    frame_rate=fs,
    sample_width=2, # 2 bytes for 16-bit audio
    channels=1      # Mono audio
)

filename = 'ifft/flute_noise_reduced.mp3'
audio_segment.export(filename, format="mp3")

print(f"Success! Saved filtered audio as {filename}")