from typing import Tuple
import tensorflow as tf
from tensorflow._api.v2.data import Dataset
import tensorflow_io as tfio
import math

import os
import numpy as np
import keras

import matplotlib.pyplot as plt


def get_spectrogram(waveform, window_size: int):
    # Convert the waveform to a spectrogram via a STFT
    spectrogram = tfio.audio.spectrogram(
        waveform, nfft=window_size, window=window_size, stride=window_size
    )

    mel_spectrogram = tfio.audio.melscale(
        spectrogram, rate=16000, mels=64, fmin=0, fmax=8000
    )
    # Convert to db scale mel-spectrogram
    dbscale_mel_spectrogram = tfio.audio.dbscale(mel_spectrogram, top_db=80)
    # Obtain the magnitude of the STFT.
    # dbscale_mel_spectrogram = tf.abs(spectrogram)

    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    dbscale_mel_spectrogram = dbscale_mel_spectrogram[..., tf.newaxis]
    return dbscale_mel_spectrogram


def make_spec_ds(ds: Dataset, window_size: int) -> Dataset:
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio, window_size), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


# Remove the extra dimension thats used for audio channels.
def squeeze(audio, labels):
    audio = tf.squeeze(audio, axis=-1)
    return audio, labels


# Convert the emotion labels to integers for training
label_map = {
    "ANG": 0,  # Anger
    "DIS": 1,  # Disgust
    "FEA": 2,  # Fear
    "HAP": 3,  # Happiness
    "NEU": 4,  # Neutral
    "SAD": 5,  # Sadness
}

inv_label_map = {
    0: "Anger",  # Anger
    1: "Disgust",  # Disgust
    2: "Fear",  # Fear
    3: "Happiness",  # Happiness
    4: "Neutral",  # Neutral
    5: "SAD",  # Sadness
}


def label_to_int(label):
    return label_map[label]


def create_dataset(batch_size: int = 64, window_size: int = 512, shuffle_size=8):
    labels: list[int] = []
    # Extract features for each audio file in the dataset
    for _, _, filenames in os.walk("tmp"):
        for filename in filenames:
            if not filename.endswith(
                ".wav"
            ):  # Skip non-wav files (this shouldn't happen)
                print(
                    f"Skipping {filename} as it is not a .wav file (how did this happen?)"
                )
                continue

            # Get the label from the filename
            # Assuming the label is the third part of the filename
            labels.append(label_to_int(filename.split("_")[2]))

    augment = keras.Sequential(
        [
            keras.layers.RandomTranslation(
                height_factor=(-0.2, 0.2),
                width_factor=(-0.2, 0.2),
                data_format="channels_last",
            ),
            keras.layers.RandomZoom(
                height_factor=(-0.2, 0.2),
                width_factor=(-0.2, 0.2),
                data_format="channels_last",
            ),
            keras.layers.RandomFlip("horizontal", data_format="channels_last"),
        ]
    )

    # A lot is going on here, try and set window size to roughly sample_rate*time/64
    train_ds, val_ds = keras.utils.audio_dataset_from_directory(
        directory="tmp",
        labels=labels,
        label_mode="int",
        batch_size=None,
        shuffle=True,
        output_sequence_length=64 * window_size,
        validation_split=0.2,
        seed=0,
        subset="both",
    )

    # Drop unnecessary channels
    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    # Preprocess the dataset into mel spectrograms
    train_ds = make_spec_ds(train_ds, window_size=window_size)
    val_ds = make_spec_ds(val_ds, window_size=window_size)

    # spl;it the validation set into test and validation pairs
    test_ds = val_ds.take(math.floor(val_ds.cardinality() / 2))
    val_ds = val_ds.skip(math.ceil((val_ds.cardinality() + 1) / 2))

    train_ds = train_ds.shuffle(shuffle_size * batch_size).batch(batch_size=batch_size)
    val_ds = val_ds.shuffle(shuffle_size * batch_size).batch(batch_size=batch_size)
    test_ds = test_ds.shuffle(shuffle_size * batch_size).batch(batch_size=batch_size)

    # Augment the dataset with more images for more training data
    train_ds = train_ds.map(
        lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE
    )

    print(train_ds.element_spec)

    return (
        train_ds.prefetch(tf.data.AUTOTUNE),
        val_ds.prefetch(tf.data.AUTOTUNE),
        test_ds.prefetch(tf.data.AUTOTUNE),
    )


def build_model(input_shape, num_classes) -> keras.Sequential:
    model: keras.Sequential = keras.Sequential()

    # This is model is based on the arxiv paper posted in the models-datasets nots file
    model.add(keras.layers.Input(shape=input_shape))
    model.add(
        keras.layers.Conv2D(filters=8, kernel_size=5, activation="relu", padding="same")
    )
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(
        keras.layers.Conv2D(
            filters=16, kernel_size=5, activation="relu", padding="same"
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(
        keras.layers.Conv2D(
            filters=100, kernel_size=5, activation="relu", padding="same"
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(
        keras.layers.Conv2D(
            filters=200, kernel_size=5, activation="relu", padding="same"
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(0.2))

    model.add(
        keras.layers.Conv2D(
            filters=200, kernel_size=4, activation="relu", padding="same"
        )
    )

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(4 * 4 * 200))

    model.add(keras.layers.Dense(4 * 200))

    model.add(keras.layers.Dense(200))

    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(use_ema=True),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model


def train_and_test(
    epochs: int = 100,
    graphs: bool = False,
    save_model: bool = False,
    model_file: str = None,
    ckpt_rate: int = 10,
):
    """
    Train and test a model
    @param datasets the training validation and testing datasets
    @param epochs number of epochs to train for
    @param graphs whether or not to output graphs tracking loss and accuracy per epoch
    @param save_weights Boolean on whether or not to save a the model after training
    @param weight_file name of the model to save, must be set if save_weights is true
    """

    batch_size = 64

    models_dir = f"models/{model_file}/"
    checkpoint_path = models_dir + "{epoch:04d}.weights.h5"

    train_ds, val_ds, test_ds = create_dataset(batch_size=batch_size)

    if save_model and not model_file:
        raise ValueError("model_file must have a name if save_model is set TRUE")
    # saves every 10 epochs
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq=ckpt_rate * len(train_ds),
    )

    callbacks = []

    if save_model:
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        callbacks.append(cp_callback)

    for example_audio, example_labels in train_ds.take(1):
        shape: Tuple = example_audio[0].shape
        print(shape)

    model = build_model(input_shape=shape, num_classes=6)

    model.summary()

    history = model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks
    )

    # summarize history for accuracy
    if graphs:
        plt.plot(history.history["accuracy"])
        plt.plot(history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

        # summarize history for loss
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "test"], loc="upper left")
        plt.show()

    model.evaluate(test_ds)
    if save_model:
        model.save(f"models/{model_file}.keras")
        np.save(os.path.join(models_dir, "history.npy"), history.history)
        # To load the history
        # history=np.load(os.path.join(models_dir,'history.npy'),allow_pickle='TRUE').item()

    return history, model


def main():
    epochs = 50
    name = "base1"
    train_and_test(
        epochs=epochs, graphs=True, save_model=True, model_file=f"{name}-e{epochs}"
    )


if __name__ == "__main__":
    main()
