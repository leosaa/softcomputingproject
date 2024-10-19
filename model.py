from typing import Tuple
import tensorflow as tf
from tensorflow._api.v2.data import Dataset
import tensorflow_io as tfio

import os
import numpy as np
import keras


def get_spectrogram(waveform):
    print(waveform.shape)
    # Convert the waveform to a spectrogram via a STFT
    spectrogram = tfio.audio.spectrogram(
        waveform, nfft=512, window=512, stride=256)

    mel_spectrogram = tfio.audio.melscale(
        spectrogram, rate=16000, mels=128, fmin=0, fmax=8000)
# Convert to db scale mel-spectrogram
    dbscale_mel_spectrogram = tfio.audio.dbscale(
        mel_spectrogram, top_db=80)
    # Obtain the magnitude of the STFT.
    dbscale_mel_spectrogram = tf.abs(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    dbscale_mel_spectrogram = dbscale_mel_spectrogram[..., tf.newaxis]
    return dbscale_mel_spectrogram


def make_spec_ds(ds):
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


def create_dataset() -> tuple[Dataset, Dataset, Dataset]:
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

    train_ds, val_ds = keras.utils.audio_dataset_from_directory(
        directory="tmp", labels=labels, label_mode="int", batch_size=16, validation_split=0.2, seed=0, subset="both",  output_sequence_length=10*16000)

    # Drop unecessary channels
    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    # Rebatching to drop remainder so training works
    train_ds = train_ds.rebatch(16, drop_remainder=True)

    train_ds = make_spec_ds(train_ds)
    val_ds = make_spec_ds(val_ds)

    print(train_ds.element_spec)

    val_ds = val_ds.shard(num_shards=2, index=1)
    test_ds = val_ds.shard(num_shards=2, index=0)

    val_ds = val_ds.rebatch(16, drop_remainder=True)
    test_ds = test_ds.rebatch(16, drop_remainder=True)

    return train_ds.repeat(), val_ds.repeat(), test_ds.repeat()


def build_model(input_shape, num_classes) -> keras.Sequential:
    model: keras.Sequential = keras.Sequential(layers=[
        keras.layers.Input(shape=input_shape),
        # keras.layers.Flatten(),
        keras.layers.Conv2D(filters=9, kernel_size=3, padding="same"),
        keras.layers.MaxPool2D(pool_size=(3, 3)),
        # Hidden layer 1
        keras.layers.Flatten(),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(128),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(num_classes)
    ])

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )

    return model


if __name__ == '__main__':

    train_ds, val_ds, test_ds = create_dataset()

    for example_audio, example_labels in train_ds.take(1):
        shape: Tuple = example_audio[0].shape
        print(shape)

    model = build_model(input_shape=shape, num_classes=6)
    model.fit(train_ds, epochs=4, validation_data=val_ds)
    model.evaluate(test_ds)
