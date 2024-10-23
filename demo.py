import numpy as np
import pyaudio
from model import *
import time
import keras
import os
import urllib.request
import tensorflow as tf
import ssl

ssl._create_default_https_context = ssl._create_unverified_context


def predict(audio: np.ndarray, model: keras.Sequential) -> str:

    spect = get_spectrogram(audio, 512)
    spect = np.expand_dims(spect, axis=0)
    predictions = model.predict(spect, verbose=0)
    classes = np.argmax(predictions, axis=1)

    print('Confidence :', predictions[0][classes[0]])
    return inv_label_map[classes[0]]

    # Preprocess the dataset into mel spectrograms


class EmotionClassifier(object):
    def __init__(self, model: keras.Sequential):
        self.FORMAT = pyaudio.paFloat32
        self.CHANNELS = 1
        self.RATE = 16000
        self.CHUNK = 64 * 512
        self.p = None
        self.stream = None
        self.model = model

    def start(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=self.FORMAT,
                                  channels=self.CHANNELS,
                                  rate=self.RATE,
                                  input=True,
                                  output=False,
                                  stream_callback=self.callback,
                                  frames_per_buffer=self.CHUNK)

    def stop(self):
        self.stream.close()
        self.p.terminate()

    def callback(self, in_data, frame_count, time_info, flag):
        numpy_array = np.frombuffer(in_data, dtype=np.float32)
        print(predict(numpy_array, self.model))
        return None, pyaudio.paContinue

    def mainloop(self):
        while (self.stream.is_active()):
            time.sleep(2.0)


model_name = 'base1-e50.keras'
model = keras.models.load_model(
    os.path.join('models', model_name))
audio = EmotionClassifier(model)
audio.start()     # open the the stream
audio.mainloop()  # main operations with librosa
audio.stop()
