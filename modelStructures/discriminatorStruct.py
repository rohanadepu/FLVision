#########################################################
#    Imports / Env setup                                #
#########################################################

import os
import random
import time
from datetime import datetime
import argparse

if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']

import flwr as fl

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LSTM, Conv1D, MaxPooling1D, GRU, LeakyReLU, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import SpectralNormalization
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# Function for creating the discriminator model
def create_discriminator(input_dim):
    # Discriminator is designed to classify three classes:
    # - Normal (Benign) traffic
    # - Intrusive (Malicious) traffic
    # - Generated (Fake) traffic from the generator
    discriminator = tf.keras.Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax')  # 3 classes: Normal, Intrusive, Fake
    ])  # see if sigmoid is the issue in transfer
    return discriminator


def create_discriminator_binary(input_dim):
    # Discriminator to classify two classes:
    # - Real (Benign & Malicious) traffic
    # - Generated (Fake) traffic from the generator
    discriminator = tf.keras.Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(1, activation='sigmoid')  # 2 classes: Real, Fake
    ])
    return discriminator


def create_discriminator_binary_optimized_spectral(input_dim):
    # Discriminator to classify two classes: Real (Benign & Malicious) traffic vs. Fake traffic
    discriminator = Sequential([
        # First Layer
        SpectralNormalization(Dense(512, use_bias=False, input_shape=(input_dim,))),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        # Second Layer
        SpectralNormalization(Dense(256, use_bias=False)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        # Third Layer
        SpectralNormalization(Dense(128, use_bias=False)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        # Final Layer (Output)
        Dense(1, activation='sigmoid')  # 2 classes: Real, Fake
    ])

    return discriminator


def create_discriminator_binary_optimized(input_dim):
    discriminator = Sequential([
        # Input Layer
        Dense(512, use_bias=False, input_shape=(input_dim,)),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        # Hidden Layers
        Dense(256, use_bias=False),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        Dense(128, use_bias=False),
        BatchNormalization(),
        LeakyReLU(alpha=0.2),

        # Final Layer (Binary Classification)
        Dense(1, activation='sigmoid')  # 2 classes: Real, Fake
    ])
    return discriminator




def create_discriminator_realtime_GRU(input_dim):
    # Discriminator to classify three classes: Normal, Intrusive, Fake
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),  # For real-time step input

        # GRU to capture temporal dependencies
        GRU(50, return_sequences=False, activation='tanh'),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Output layer for three classes
        Dense(3, activation='softmax')  # 3 classes: Normal, Intrusive, Fake
    ])
    return discriminator

def create_discriminator_realtime_GRU_binary(input_dim):
    # Discriminator to classify two classes: Real, Fake
    discriminator = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(1, input_dim)),  # For real-time step input

        # GRU to capture temporal dependencies
        GRU(50, return_sequences=False, activation='tanh'),

        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),

        # Output layer for two classes
        Dense(1, activation='sigmoid')  # 2 classes: Real, Fake
    ])
    return discriminator
