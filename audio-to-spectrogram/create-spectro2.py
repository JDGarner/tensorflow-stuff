import matplotlib.pyplot as plt
import scipy.signal
from scipy.signal import spectrogram
from scipy.io import wavfile

sample_rate, samples = wavfile.read('output.wav')
frequencies, times, spectogram = signal.spectrogram(samples, sample_rate)

plt.imshow(spectogram)
plt.pcolormesh(times, frequencies, spectogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()