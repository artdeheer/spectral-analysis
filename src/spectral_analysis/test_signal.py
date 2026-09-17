import numpy as np
from pydub import AudioSegment

# 1. Define the parameters
fs = 44100              
t = np.arange(0, 1.0, 1/fs) 
f1 = 440.0              
A1 = 1.0                
f2 = 1000.0             
A2 = 0.5    

FILE_LOCATION = 'samples/'

# 2. Create the audio array
y_test = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)

# 3. Convert the float array to 16-bit PCM (required for pydub)
y_test_norm = y_test / np.max(np.abs(y_test))
y_test_int = np.int16(y_test_norm * 32767)

# 4. Create an AudioSegment and export as MP3
audio = AudioSegment(
    y_test_int.tobytes(), 
    frame_rate=fs,
    sample_width=2, 
    channels=1
)

filename = f"test_{int(f1)}hz{A1}_{int(f2)}hz{A2}.mp3"
audio.export(FILE_LOCATION + filename, format="mp3")
print(f"Successfully saved as {filename}")