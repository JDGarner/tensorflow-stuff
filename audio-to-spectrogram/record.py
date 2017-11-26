import pyaudio
import struct
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


THRESHOLD = 40 # dB
RATE = 44100
INPUT_BLOCK_TIME = 0.03 # 30 ms
INPUT_FRAMES_PER_BLOCK = int(RATE * INPUT_BLOCK_TIME)

def get_rms(block):
    return np.sqrt(np.mean(np.square(block)))

class AudioHandler(object):
    def __init__(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.open_mic_stream()
        self.threshold = THRESHOLD
        self.plot_counter = 0

    def stop(self):
        self.stream.close()

    def find_input_device(self):
        device_index = None
        for i in range( self.pa.get_device_count() ):
            devinfo = self.pa.get_device_info_by_index(i)
            print('Device %{}: %{}'.format(i, devinfo['name']))

            for keyword in ['mic','input']:
                if keyword in devinfo['name'].lower():
                    print('Found an input: device {} - {}'.format(i, devinfo['name']))
                    device_index = i
                    return device_index