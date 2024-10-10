# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: hydrogen
#       format_version: '1.3'
#       jupytext_version: 1.16.4
#   kernelspec:
#     display_name: .env.venv
#     language: python
#     name: python3
# ---

# %%
import audio_processor.ipynb # import the audio_processor module (Trust that groupmates have implemented the module)
import model_trainer.ipynb # import the model_trainer module (Trust that groupmates have implemented the module)
import tensorflow as tf # import tensorflow for model training

import os

RAND_SEED = 42
SPLIT_RATIO = 0.8


# %%
# For every audio file in the dataset, extract the features and store them in a list
def prepare_dataset(dataset_path):
    features = []

    # Extract features for each audio file in the dataset
    for audio_filename in os.listdir(dataset_path):
        if(not audio_filename.endswith(".wav")): # Skip non-wav files
            continue
        features.append(audio_processor.extract_features(audio_filename))    

    return features

def get_labels(label_path):
    labels = []
    with open(label_path, 'r') as f:
        for line in f:
            labels.append(line.strip())
    return labels


# %%
# From our extracted features, build our tensors for training
def build_tensors(features, labels):
    # Convert the features and labels to TensorFlow tensors
    # Assuming each extracted feature is a numpy array or list
    features_tensor = tf.convert_to_tensor(features, dtype=tf.float32)  # Convert features to tensor
    
    # Convert labels to one-hot encoded vectors, assuming labels are integers (e.g., [0, 1, 2,...])
    num_classes = len(set(labels))  # Get the number of unique emotion labels
    labels_tensor = tf.one_hot(labels, depth=num_classes, dtype=tf.float32)  # One-hot encode labels
    
    return features_tensor, labels_tensor


# %%
# From our dataset, pseudo-randomly shuffle the data & split it into training and testing sets
def split_dataset(features, labels, ratio=SPLIT_RATIO, seed=RAND_SEED):
    # Where features and labels are TensorFlow tensors
    # Split the dataset into training and testing sets

    size = tf.shape(features)[0]  # Get the size of the dataset

    # Shuffle the dataset by generating a random permutation of indices
    indices = tf.range(size)  # Generate a range of indices
    shuffled_indices = tf.random.shuffle(indices, seed=seed)  # Shuffle the indices

    # Split the dataset into training and testing sets
    train_size = int(size * ratio)  # Calculate the size of the training set
    train_indices = shuffled_indices[:train_size]  # Get the training set indices
    test_indices = shuffled_indices[train_size:]  # Get the testing set indices

    # Extract the training and testing features and labels
    train_features = tf.gather(features, train_indices)  # Extract training features
    train_labels = tf.gather(labels, train_indices)  # Extract training labels
    test_features = tf.gather(features, test_indices)  # Extract testing features
    test_labels = tf.gather(labels, test_indices)  # Extract testing labels

    return train_features, train_labels, test_features, test_labels


# %%
# Train the model using the extracted features
def train_model(features, labels):
    return model_trainer.train_model(features, labels)


# %%
PATH = './dataset' # Path to the dataset

# Build the dataset
features = prepare_dataset(PATH)
labels = get_labels('{PATH}/labels.txt') # Assuming labels are stored in a text file
# Build the tensors
features_tensor, labels_tensor = build_tensors(features, labels)

# Split the dataset
train_features, train_labels, test_features, test_labels = split_dataset(features_tensor, labels_tensor)

# Train the model
model = train_model(train_features, train_labels)

# Test the model
model.test_model(test_features, test_labels)
