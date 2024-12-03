import model as m
import numpy as np
import librosa as lr
import keras
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import math

audio_dir = 'test-audio'
SAMPLE_RATE = 16000
CHUNK_SIZE = 64 * 512
CHUNK_RATE=(SAMPLE_RATE/CHUNK_SIZE) #There are this many chunks per second
CHUNK_RATE_MULTIPLIER=32 # the number of chunks to have every second
CHUNK_PER_SECOND=CHUNK_RATE_MULTIPLIER/CHUNK_RATE # This is how many chunks we need to sample
WINDOW_WIDTH_SECONDS=1/4 # How many seconds the moving average moves over
WINDOW_WIDTH = math.floor(CHUNK_RATE_MULTIPLIER*WINDOW_WIDTH_SECONDS) # Length of the window in predicted chunks
CHANNELS = 1
MODEL_NAME = 'base1-e200.keras'

def predict(audio: np.ndarray, model: keras.Sequential) -> str:

    # Preprocess the dataset into mel spectrograms
    spect = m.get_spectrogram(audio, 512)

    spect = np.expand_dims(spect, axis=0)
    
    # Predict the class of the audio
    predictions = model.predict(spect, verbose=0)
    classes = np.argmax(predictions, axis=1)

    # I need to figure out how the return values are stored.
    # print('Confidence :', predictions[0][classes[0]])
    # print(f'Predicted class: {classes[0]}\nPredictions: {predictions}\nClasses: {classes}')
    return predictions[0] # TODO: This should return the array of confidence values for each class.

def process_audio_file(file_path: str, model: keras.Sequential, chunk_size: int = CHUNK_SIZE, sample_rate: int = SAMPLE_RATE):
    """Processes a single .wav file and classifies it over time."""
    # Load audio file
    audio, _ = lr.load(file_path, sr=sample_rate) # resamples the audio to the given sample rate

    classifications = []
    timestamps = []


    # print(CHUNK_RATE)
    # print(CHUNK_PER_SECOND)
    num_seconds = len(audio) // sample_rate
    for i in range(math.floor(num_seconds*CHUNK_PER_SECOND)):
        start = math.floor(i*chunk_size / CHUNK_PER_SECOND)
        end = start + chunk_size

        chunk = audio[start:end]
        if len(chunk) == chunk_size: # Ensure the chunk is the correct size before processing
            print(f'Progress {i/(num_seconds*CHUNK_PER_SECOND)*100:2.2f}%', end ='\r')
            classifications.append(predict(chunk, model))
            timestamps.append(i * (chunk_size / sample_rate)/CHUNK_PER_SECOND)  # Time in seconds

    return classifications, timestamps

def plot_classifications(labels, confidence, timestamps, name, colors):
    """
    Plots the classification confidence levels over time for multiple labels.

    Parameters:
        labels (list of str): The 6 classification labels.
        confidence (list of list of float): A list where each element is a list of 6 confidence values.
        timestamps (list of float): A list of timestamps corresponding to the confidence values.
        name (str): The name of the sample being classified.

    Returns:
        None
    """
    # Ensure the data dimensions match
    if len(confidence) != len(timestamps):
        raise ValueError("The length of confidence and timestamps must match.")

    if any(len(c) != len(labels) for c in confidence):
        raise ValueError("Each confidence entry must have the same length as the labels list.")


    # Convert confidence to a numpy array for easier slicing
    confidence = np.array(confidence)

    ma_confidence = np.zeros_like(confidence, dtype=float)
    
    # Pad the array with zeros
    padded_confidence = np.pad(confidence, ((WINDOW_WIDTH - 1, 0), (0, 0)), mode='constant', constant_values=0)

    for i in range(confidence.shape[0]):
        ma_confidence[i] = np.mean(padded_confidence[i:i + WINDOW_WIDTH], axis=0)

    # Create the plot
    plt.figure(figsize=(10, 6))

    for i, label in enumerate(labels):
        plt.plot(timestamps, ma_confidence[:, i], label=label, marker=' ', linestyle='-', alpha=0.7, c = colors[i])

    # Add title, labels, and legend
    plt.title(f"Classification Confidence Over Time for '{name}'", fontsize=16)
    plt.xlabel("Timestamps (seconds)", fontsize=14)
    plt.ylabel("Confidence", fontsize=14)
    plt.ylim(0, 1)  # Confidence values are expected to be between 0 and 1
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(title="Labels", loc='upper left', fontsize=10)

    # Show the plot
    plt.tight_layout()
    plt.show()


def main():
    model = keras.models.load_model(os.path.join('models', MODEL_NAME))
    
    labels = [
        'Anger',
        'Disgust',
        'Fear',
        'Happiness',
        'Neutral',
        'Sadness'
    ]

    colors = [
        'r', # Red for Anger
        'g', # Green for Disgust
        'p', # Purple for Fear
        'y', # Yellow for Happiness
        'k', # Black for Neutral
        'c', # Cyan for Sadness
    ]

    # Grab an audio file from the local audio clips directory
    for file in os.listdir(audio_dir):
        if file.endswith('.wav'):
            print(f"Processing {file}...")
            file_path = os.path.join(audio_dir, file)

            classifications, timestamps = process_audio_file(file_path, model, CHUNK_SIZE, SAMPLE_RATE)
            plot_classifications(labels, classifications, timestamps, file, colors)


if __name__ == '__main__':
    main()
