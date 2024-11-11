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

from time import sleep # import the sleep function from the time module because this goes too fast for me to read my print statements


RAND_SEED = 0 # Random seed for reproducibility
EPOCHS = 100 # Number of epochs to train for
BATCH_SIZE = 128 # Batch size for training
REPEAT_COUNT = 4 # How many times to repeat the training data (fluff out the data for faster training on GPU)
LOCAL_DIR = './tmp' # Local directory of the dataset. Should be split into subdirectories corresponding to the labels

SLEEP_TIME = 1 # Time to sleep between print statements

HYPERTUNING = True # Whether or not to use hyperparameter tuning


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
    
    # A lot is going on here, try and set window size to roughly sample_rate*time/64
    train_ds, val_ds = keras.utils.audio_dataset_from_directory(
        directory="tmp",
        labels="inferred",
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



    print(train_ds.element_spec)

    return (
        train_ds.repeat(REPEAT_COUNT).prefetch(tf.data.AUTOTUNE),
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
    # If its nots a choice keras doesn't seem to do the if statements soooo
    # Sorry for the cursed shit
    cnn_layers = hp.Int("cnn_layers", min_value=1,max_value=3)
    dense_layers = hp.Int("dense_layers",  min_value=1,max_value=3)

    # Augment layer
    # Augment the dataset with more images for more training data by translating, zooming, and flipping the images
    hp_translation= hp.Float("translation", min_value=0.1, max_value=0.5, step=0.1)
    hp_zoom= hp.Float("zoom", min_value=0.1, max_value=0.5, step=0.1)
    augment = keras.Sequential(
        [
            keras.layers.RandomTranslation(
                height_factor=(-hp_translation,hp_translation),
                width_factor= (-hp_translation,hp_translation),
                data_format="channels_last",
            ),
            keras.layers.RandomZoom(
                height_factor=(-hp_zoom,hp_zoom),
                width_factor=(-hp_zoom,hp_zoom),
                data_format="channels_last",
            ),
            keras.layers.RandomFlip("horizontal", data_format="channels_last"),
        ]
    )
    model.add(
        augment
    )

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
    model.add(keras.layers.SpatialDropout2D(hp.Float("dropout_1", min_value=0.1, max_value=0.5, step=0.1))) # Dropout layer to prevent overfitting (hyperparameter tuned)
    model.add(keras.layers.BatchNormalization()) # Normalization technique 

    # Convolutional layer 2
    with hp.conditional_scope('cnn_layers', [2,3]):
        if cnn_layers >=2:
            model.add(
                keras.layers.Conv2D(
                    filters = hp.Int('conv_2_filters', min_value=16, max_value=128, step=16),
                    kernel_size = hp.Choice('conv_2_kernel', values=[3, 5]),
                    activation="relu",
                    padding="same"
                )
            )
            model.add(keras.layers.MaxPool2D(pool_size=(2, 2))) # Max pooling layer to reduce the size of the image
            model.add(keras.layers.SpatialDropout2D(hp.Float("dropout_2", min_value=0.1, max_value=0.5, step=0.1))) # Dropout layer
            model.add(keras.layers.BatchNormalization()) # Normalization technique 

        # Convolutional layer 3
    with hp.conditional_scope('cnn_layers', [3]):
        if cnn_layers >=3:
            model.add(
                keras.layers.Conv2D(
                    filters=hp.Int('conv_3_filters', min_value=64, max_value=512, step=32),
                    kernel_size=hp.Choice('conv_3_kernel', values=[3, 5]),
                    activation="relu",
                    padding="same"
                )
            )
            model.add(keras.layers.MaxPool2D(pool_size=(2, 2))) # Max pooling layer
            model.add(keras.layers.SpatialDropout2D(hp.Float("dropout_3", min_value=0.1, max_value=0.5, step=0.1))) # Dropout layer
            model.add(keras.layers.BatchNormalization()) # Normalization technique 

    model.add(
        keras.layers.ConvLSTM1D(
            filters = hp.Int('lstm_filters', min_value=8, max_value=128, step=8),
            kernel_size = hp.Choice('lstm_kernel', values=[3, 5]),
            dropout= hp.Float("lstm_dropout", min_value=0.0, max_value=0.5, step=0.1),
            recurrent_dropout=hp.Float("lstm_rec_dropout", min_value=0.0, max_value=0.5, step=0.1),
            padding="same"
        )
    )
    model.add(keras.layers.BatchNormalization()) # Normalization technique 
    
    # Flatten the output of the convolutional layers
    model.add(keras.layers.Flatten())

    # Dense layers
    # Dense layer 3 (First fully connected layer)
    with hp.conditional_scope('dense_layers', [3]):
        if dense_layers >=3:
            model.add(keras.layers.Dense(
                hp.Int("dense_3", min_value=64, max_value=1024, step=64), # Hyperparameter tuned number of neurons
                activation="relu"
            )) 
            model.add(keras.layers.BatchNormalization())

    # Dense layer 2 (Second fully connected layer)
    with hp.conditional_scope('dense_layers', [2,3]):
        if dense_layers >=2:
            model.add(keras.layers.Dense(
                hp.Int("dense_2", min_value=32, max_value=512, step=32),
                activation="relu"
            ))
            model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.BatchNormalization())

    # Dense layer 1 (Third fully connected layer)
    model.add(keras.layers.Dense(
        hp.Int("dense_1", min_value=16, max_value=256, step=16),
        activation="relu"
    ))
    model.add(keras.layers.BatchNormalization())
    
    # Output layer
    model.add(keras.layers.Dense(num_classes, activation="softmax"))

    # Compile the model
    model.compile(
        optimizer=keras.optimizers.AdamW(use_ema=True, learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])),
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

    if HYPERTUNING is True, the model will be hyperparameter tuned before training
    otherwise the model will be loaded from models/{model_file} and trained

    @param epochs number of epochs to train for
    @param graphs whether or not to output graphs tracking loss and accuracy per epoch
    @param save_model whether or not to save the model after training is complete
        - if HYPERTUNING is True, the best model will be saved as {model_file}-HYPERTUNED.keras regardless of this value
    @param model_file the name of the model file to load and save to (this will append an int to the end of the file name to prevent overwriting)
    @param ckpt_rate the rate at which to save checkpoints
    @param tuning_epochs the number of epochs to train for during hyperparameter tuning
    """
    models_dir = f"models/{model_file}/"
    checkpoint_path = models_dir + "{epoch:04d}.weights.h5"

    train_ds, val_ds, test_ds = create_dataset(batch_size=BATCH_SIZE)

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

    if not os.path.exists("models"):
        os.makedirs("models")
    # Save the model after training
    if save_model:
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)

        callbacks.append(cp_callback)

    for example_audio, example_labels in train_ds.take(1):
        shape: Tuple = example_audio[0].shape
        print(shape)

    # Early stopping callback to stop training if the model is not improving after 5 epochs
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True
    )
    callbacks.append(early_stopping)

    # Tune before loading the model
    if HYPERTUNING:
        print("Hypertuning is enabled. Training the model with the best hyperparameters.")
        sleep(SLEEP_TIME)

        # Get the hyperparameter tuner
        tuner = getHyperTuner(shape=shape, classes=6)

        # Search for the best hyperparameters
        tuner.search(train_ds, validation_data=val_ds, epochs=tuning_epochs)

        # Get the best hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Build the model with the best hyperparameters
        model = tuner.hypermodel.build(best_hps)

        # Save the model
        if model_file.endswith(".keras"):
            model_file = model_file[:-6]
        filename = f"{model_file}-HYPERTUNED"
        i = 1
        while os.path.exists(f"models/{filename}.keras"):
            filename = f"{model_file}-HYPERTUNED-{i}"
            i += 1
        model.save(f"models/{filename}.keras")
        print(f"Model saved as {filename}.keras")

    # Load the model
    else: 
        # Build the model from models/{model_file}.keras
        if not model_file.endswith(".keras"):
            model_file += ".keras"
        print(f"Hypertuning is disabled. Loading the model from {model_file}.")

        model = keras.models.load_model(f"models/{model_file}")

        # Train the model
        model.compile(
            optimizer=keras.optimizers.AdamW(use_ema=True, learning_rate=1e-3),
            loss=keras.losses.SparseCategoricalCrossentropy(),
            metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
        )

    print("Model has been loaded and compiled.")
    # Print the model summary
    model.summary()
    sleep(SLEEP_TIME)

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
        if model_file.endswith(".keras"):
            model_file = model_file[:-6]
        filename = f"{model_file}"
        i = 1
        while os.path.exists(f"models/{filename}.keras"):
            filename = f"{model_file}-{i}"
            i += 1
        model.save(f"models/{filename}.keras")
        np.save(os.path.join(models_dir, "history.npy"), history.history)
        # To load the history
        # history=np.load(os.path.join(models_dir,'history.npy'),allow_pickle='TRUE').item()

    return history, model


def main():
    name = "base1"
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    train_and_test(
        graphs=True, save_model=True, model_file=f"{name}-e{EPOCHS}"
    )


if __name__ == "__main__":
    main()
