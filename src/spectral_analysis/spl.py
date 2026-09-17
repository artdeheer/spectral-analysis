import os
import librosa
import numpy as np
import matplotlib.pyplot as plt

# 1. Load the instrument signal
instrument_file = 'samples/sample8.mp3'
y, fs = librosa.load(instrument_file, sr=None)

# 2. Create a time array for the raw waveform
time = np.linspace(0, len(y) / fs, num=len(y))

# 3. Calculate RMS energy (proxy for SPL) and convert to Decibels (dB)
# frame_length of 2048 groups samples to calculate the average energy over small windows
rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
time_rms = librosa.frames_to_time(np.arange(len(rms)), sr=fs, hop_length=512)
db_level = librosa.amplitude_to_db(rms, ref=np.max)

# 4. Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Plot 1: Raw Waveform
ax1.plot(time, y, color='blue', alpha=0.7)
ax1.set_title("Time-Domain Waveform")
ax1.set_ylabel("Amplitude")
ax1.grid(True)

# Plot 2: Relative Sound Pressure Level (RMS in dBFS)
ax2.plot(time_rms, db_level, color='red')
ax2.set_title("Relative Sound Pressure Level Envelope (dBFS)")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Level (dB)")
ax2.set_ylim(-60, 5) # Standard dB scale limit
ax2.grid(True)

plt.tight_layout()

# Save the plot
os.makedirs('plots', exist_ok=True)
plt.savefig('plots/waveform_spl.png', bbox_inches='tight')

plt.show()