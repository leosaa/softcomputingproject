from typing import Tuple
import tensorflow as tf
from tensorflow._api.v2.data import Dataset
import tensorflow_io as tfio
import math
import keras_tuner as kt
import os
import numpy as np
import keras
import urllib.request as request
import ssl # import the ssl module to ignore SSL certificate errors
import re # import the re module for regular expressions
import matplotlib.pyplot as plt

RAND_SEED = 0 # Random seed for reproducibility
EPOCHS = 50 # Number of epochs to train for
DOWNLOAD_FILES = False # Whether or not to download the audio files if they don't exist in the local directory
DATASET_URL = 'https://www.cs.nmt.edu/~leo/CREMA-D/AudioWAV' # Path to the dataset
LOCAL_DIR = './tmp' # Local directory to store the dataset

# Download the audio files from the dataset URL
def download_files(url : str = DATASET_URL, dir : str = LOCAL_DIR) -> None:
    # Load the file list from the dataset URL
    filelist = []
    # Ignore SSL certificate errors (Ty CS Dept)
    with request.urlopen(url, context=ssl._create_unverified_context()) as response:
        html = response.read().decode()

    # Get the text located between the <a href="..."> NEEDED_VALUE </a> tag
    # This is a simple way to extract the filenames from the HTML response
    filename_regex = re.compile(r'<a href="([^"]+)">([^<]+)</a>')

    # Build a list of all the files in the dataset
    for line in html.splitlines():
        if '.wav' in line: # This line contains a .wav file, so extract the filename
            match = filename_regex.search(line)
            if match:
                filelist.append(match.group(1)) # Append the filename to the list
    
    # Create the local directory if it doesn't exist
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Download the files
    total_files = len(filelist)
    files_downloaded = len([f for f in os.listdir(dir) if f.endswith('.wav')])
    for file in filelist:
        # If the file exists in the directory, skip the download
        if os.path.exists(f'{dir}/{file}'):
            print(f'{file} already exists in {dir}')
            return False

        # Download the file from the URL
        with request.urlopen(f'{url}/{file}', context=ssl._create_unverified_context()) as response:
            with open(f'{dir}/{file}', 'wb') as file:
                file.write(response.read())
        files_downloaded += 1
        print(f'Downloaded {file} ({files_downloaded}/{total_files} files).')
        # Print a progress bar
        print(
                ('*' * int(50 * (files_downloaded / total_files)))
            + ('-' * int(50 * (1 - files_downloaded / total_files)))
            )

# Returns a tensor of the spectrogram of the audio file
def get_spectrogram(waveform, window_size: int) -> tf.Tensor:
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


# Convert the audio files to spectrograms
def make_spec_ds(ds: Dataset, window_size: int) -> Dataset:
    return ds.map(
        map_func=lambda audio, label: (get_spectrogram(audio, window_size), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


# Remove the extra dimension thats used for audio channels.
def squeeze(audio, labels) -> Tuple[tf.Tensor, tf.Tensor]:
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

# Convert the emotion labels to integers for training
def label_to_int(label) -> int:
    return label_map[label]

# Create the dataset from the audio files
def create_dataset(batch_size: int = 64,
                   window_size: int = 512,
                   shuffle_size=8) -> Tuple[Dataset, Dataset, Dataset]:
    
    # Get the labels from the filenames
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

    # Augment the dataset with more images for more training data by translating, zooming, and flipping the images
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
        seed=RAND_SEED,
        subset="both",
    )

    # Drop unnecessary channels
    train_ds = train_ds.map(squeeze, tf.data.AUTOTUNE)
    val_ds = val_ds.map(squeeze, tf.data.AUTOTUNE)

    # Preprocess the dataset into mel spectrograms
    train_ds = make_spec_ds(train_ds, window_size=window_size)
    val_ds = make_spec_ds(val_ds, window_size=window_size)

    # split the validation set into test and validation pairs
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

# Build the model
def build_model(hp: kt.HyperParameters,
                input_shape: Tuple[int, int, int],
                num_classes: int
                ) -> keras.Sequential:
    model: keras.Sequential = keras.Sequential()

    # This is model is based on the arxiv paper posted in the models-datasets nots file

    # Input layer
    model.add(keras.layers.Input(shape=input_shape))

    # Convolutional layers

    # Convolutional layer 1
    model.add(
        keras.layers.Conv2D(
            filters=hp.Int("conv_1_filters", min_value=8, max_value=64, step=8), # Hyperparameter tuned number of filters
            kernel_size=hp.Choice('conv_1_kernel', values=[3, 5]), # Hyperparameter tuned kernel size between 3 and 5
            activation="relu",
            padding="same"
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2))) # Max pooling layer to reduce the size of the image (Shrinks the image by 2)
    model.add(keras.layers.Dropout(hp.Float("dropout_1", min_value=0.1, max_value=0.5, step=0.1))) # Dropout layer to prevent overfitting (hyperparameter tuned)

    # Convolutional layer 2
    model.add(
        keras.layers.Conv2D(
            filters = hp.Int('conv_2_filters', min_value=16, max_value=128, step=16),
            kernel_size = hp.Choice('conv_2_kernel', values=[3, 5]),
            activation="relu",
            padding="same"
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2))) # Max pooling layer to reduce the size of the image
    model.add(keras.layers.Dropout(hp.Float("dropout_2", min_value=0.1, max_value=0.5, step=0.1))) # Dropout layer

    # Convolutional layer 3
    model.add(
        keras.layers.Conv2D(
            filters=hp.Int('conv_3_filters', min_value=32, max_value=256, step=32),
            kernel_size=hp.Choice('conv_3_kernel', values=[3, 5]),
            activation="relu",
            padding="same"
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2))) # Max pooling layer
    model.add(keras.layers.Dropout(hp.Float("dropout_3", min_value=0.1, max_value=0.5, step=0.1))) # Dropout layer

    # Convolutional layer 4
    model.add(
        keras.layers.Conv2D(
            filters=hp.Int('conv_4_filters', min_value=64, max_value=512, step=32),
            kernel_size=hp.Choice('conv_4_kernel', values=[3, 5]),
            activation="relu",
            padding="same"
        )
    )
    model.add(keras.layers.MaxPool2D(pool_size=(2, 2))) # Max pooling layer
    model.add(keras.layers.Dropout(hp.Float("dropout_4", min_value=0.1, max_value=0.5, step=0.1))) # Dropout layer

    # Convolutional layer 5 (Final convolutional layer)
    model.add(
        keras.layers.Conv2D(
            filters=hp.Int('conv_5_filters', min_value=64, max_value=1024, step=32),
            kernel_size=hp.Choice('conv_5_kernel', values=range(3, 7)),
            activation="relu",
            padding="same"
        )
    )

    # Flatten the output of the convolutional layers
    model.add(keras.layers.Flatten())

    # Dense layers
    # Dense layer 1 (First fully connected layer)
    model.add(keras.layers.Dense(
        hp.Int("dense_1", min_value=64, max_value=1024, step=64), # Hyperparameter tuned number of neurons
        activation="relu"
    )) 

    # Dense layer 2 (Second fully connected layer)
    model.add(keras.layers.Dense(
        hp.Int("dense_2", min_value=32, max_value=512, step=32),
        activation="relu"
    ))

    # Dense layer 3 (Third fully connected layer)
    model.add(keras.layers.Dense(
        hp.Int("dense_3", min_value=16, max_value=256, step=16),
        activation="relu"
    ))

    # Output layer
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.Adam(use_ema=True, learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )

    return model


# Get the hyperparameter tuner
def getHyperTuner(shape, classes) -> kt.Hyperband:
    tuner = kt.Hyperband(
        lambda hp: build_model(hp, input_shape=shape, num_classes=classes), # Function to build the model
        objective="val_accuracy", # Objective to optimize
        max_epochs=50, # Maximum number of epochs to train for
        factor=3, # Reduction factor for the number of epochs and number of models
        hyperband_iterations=2, # Number of times to iterate over the hyperband algorithm
        directory="models", # Directory to save the models
        project_name="hyperband", # Name of the project
    )

    return tuner

def train_and_test(
                   epochs: int = EPOCHS,
                   graphs: bool = False, 
                   save_model: bool = False, 
                   model_file: str = None,
                   ckpt_rate: int = 10,
                   tuning_epochs: int = 10
                ) -> Tuple[dict, keras.Sequential]:
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

    # Save the model after training
    if save_model:
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        callbacks.append(cp_callback)

    for example_audio, example_labels in train_ds.take(1):
        shape: Tuple = example_audio[0].shape
        print(shape)

    # Early stopping callback to stop training if the model is not improving after 5 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=5, restore_best_weights=True
    )
    callbacks.append(early_stopping)

    # Get the hyperparameter tuner
    tuner = getHyperTuner(shape=shape, classes=6)

    # Search for the best hyperparameters
    tuner.search(train_ds, validation_data=val_ds, epochs=tuning_epochs)
    
    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    # Build the model with the best hyperparameters
    model = tuner.hypermodel.build(best_hps)

    print(f"Best hyperparameters: {best_hps}")
    model.summary()

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks
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

    # Save the model
    if save_model:
        model.save(f"models/{model_file}.keras")
        np.save(os.path.join(models_dir, "history.npy"), history.history)
        # To load the history
        # history=np.load(os.path.join(models_dir,'history.npy'),allow_pickle='TRUE').item()

    return history, model


def main():
    if DOWNLOAD_FILES:
        download_files()
    name = "base1"
    train_and_test(
        graphs=True, save_model=True, model_file=f"{name}-e{EPOCHS}"
    )


if __name__ == "__main__":
    main()
