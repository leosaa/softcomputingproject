from typing import Tuple
import tensorflow as tf
from tensorflow._api.v2.data import Dataset
import tensorflow_io as tfio

import os
import numpy as np
import keras

import gc
from keras import backend as k


# While training adam doesn't do the best job of cleaning up its mess with
# our dataset. So we have to tell the bitch to clean up after each epoch
class ClearMemory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()


def get_spectrogram(waveform):
    print(waveform.shape)
    # Convert the waveform to a spectrogram via a STFT
    spectrogram = tfio.audio.spectrogram(
        waveform, nfft=2*512, window=2*512, stride=2*512)

    mel_spectrogram = tfio.audio.melscale(
        spectrogram, rate=16000, mels=64, fmin=0, fmax=8000)
# Convert to db scale mel-spectrogram
    dbscale_mel_spectrogram = tfio.audio.dbscale(
        mel_spectrogram, top_db=80)
    # Obtain the magnitude of the STFT.
    # dbscale_mel_spectrogram = tf.abs(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    dbscale_mel_spectrogram = dbscale_mel_spectrogram[..., tf.newaxis]
    return dbscale_mel_spectrogram


def make_spec_ds(ds: Dataset) -> Dataset:
    return ds.map(map_func=lambda audio, label: (get_spectrogram(audio), label), num_parallel_calls=tf.data.AUTOTUNE)


# Remove the extra dimension thats used for audio channels.
def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


# Convert the emotion labels to integers for training
label_map = {
    'ANG': 0,  # Anger
    'DIS': 1,  # Disgust
    'FEA': 2,  # Fear
    'HAP': 3,  # Happiness
    'NEU': 4,  # Neutral
    'SAD': 5  # Sadness
}


def label_to_int(label):
    return label_map[label]


def create_dataset(batch_size: int = 32) -> tuple[Dataset, Dataset, Dataset]:
    labels: list[int] = []
    # Extract features for each audio file in the dataset
    for _, _, filenames in os.walk("tmp"):
        for filename in filenames:
            if (not filename.endswith(".wav")):  # Skip non-wav files (this shouldn't happen)
                print(
                    f"Skipping {filename} as it is not a .wav file (how did this happen?)")
                continue

            # Get the label from the filename
            # Assuming the label is the third part of the filename
            labels.append(label_to_int(filename.split('_')[2]))

    augment = keras.Sequential([
        keras.layers.RandomFlip("horizontal", data_format="channels_last"),
        keras.layers.RandomRotation(0.1, data_format="channels_last"),
        keras.layers.RandomTranslation(
            height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), data_format="channels_last"),
    ])

    train_ds, val_ds = keras.utils.audio_dataset_from_directory(
        directory="tmp", labels=labels, label_mode="int", batch_size=batch_size, validation_split=0.3, seed=0, subset="both",  output_sequence_length=64*1024)

    # Drop unecessary channels
    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    # Preprocess the dataset
    train_ds = make_spec_ds(train_ds)
    val_ds = make_spec_ds(val_ds)

    # split the validation and test datasets
    val_ds = val_ds.shard(num_shards=2, index=1)
    test_ds = val_ds.shard(num_shards=2, index=0)

    # Augment the dataset with more images for more training data
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y)
    )

    # 2**(num of layers in augment)
    train_ds = train_ds.repeat().shuffle(
        2**3 * batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = val_ds.repeat()
    val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    print(train_ds.element_spec)

    return train_ds, val_ds, test_ds


def build_model(input_shape, num_classes) -> keras.Sequential:
    model: keras.Sequential = keras.Sequential()

    model.add(keras.layers.Input(shape=input_shape))
    model.add(keras.layers.BatchNormalization())
    for _ in range(8):
        model.add(keras.layers.Conv2D(filters=4, kernel_size=5,
                                      activation="relu", padding="same"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    for _ in range(16):
        model.add(keras.layers.Conv2D(filters=8, kernel_size=5,
                                      activation="relu", padding="same"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    for _ in range(32):
        model.add(keras.layers.Conv2D(filters=16, kernel_size=5,
                                      activation="relu", padding="same"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    for _ in range(64):
        model.add(keras.layers.Conv2D(filters=32, kernel_size=5,
                                      activation="relu", padding="same"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    for _ in range(128):
        model.add(keras.layers.Conv2D(filters=64, kernel_size=5,
                                      activation="relu", padding="same"))
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dropout(0.2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(1024))
    model.add(keras.layers.BatchNormalization())

    model.add(keras.layers.Dense(256))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(num_classes, activation="softmax"))
# Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


if __name__ == '__main__':

    batch_size = 32
    train_ds, val_ds, test_ds = create_dataset(batch_size=batch_size)

    shape = ()

    for example_audio, example_labels in train_ds.take(1):
        shape: Tuple = example_audio[0].shape
        print(shape)

    model = build_model(input_shape=shape, num_classes=6)
    model.summary()
    model.fit(train_ds, validation_data=val_ds, epochs=20, steps_per_epoch=2**3*int(5210/batch_size),
              validation_steps=int(2232/batch_size), callbacks=ClearMemory())
    model.evaluate(test_ds)
