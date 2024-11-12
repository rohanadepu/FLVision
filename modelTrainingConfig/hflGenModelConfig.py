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
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import expand_dims

# import math
# import glob

# from tqdm import tqdm

# import seaborn as sns

# import pickle
# import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle


################################################################################################################
#                                       GAN Model Setup (Generator Training)                                      #
################################################################################################################

# Function for creating the generator model
def create_generator(input_dim, noise_dim):
    generator = tf.keras.Sequential([
        Dense(128, activation='relu', input_shape=(noise_dim,)),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(input_dim, activation='sigmoid')  # Generate traffic features
    ])
    return generator


# --- Class to handle generator training ---#
class GeneratorClient(fl.client.NumPyClient):
    def __init__(self, generator, discriminator, x_train, x_val, y_val, x_test, BATCH_SIZE, noise_dim, epochs, steps_per_epoch):
        self.generator = generator
        self.discriminator = discriminator  # Discriminator is fixed during generator training
        self.x_train = x_train
        self.x_val = x_val  # Validation data
        self.y_val = y_val  # Validation labels
        self.x_test = x_test
        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(self.BATCH_SIZE)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # loss based on generating fake data that get misclassified as real by discriminator
    def generator_loss(self, fake_output):
        # Generator aims to fool the discriminator by classifying fake samples as normal (0)
        return tf.keras.losses.sparse_categorical_crossentropy(
            tf.zeros_like(fake_output), fake_output
        )

    def get_parameters(self, config):
        return self.generator.get_weights()

    def fit(self, parameters, config):
        self.generator.set_weights(parameters)

        for epoch in range(self.epochs):
            for step in range(self.steps_per_epoch):
                # Generate noise to create fake samples
                noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

                # captures operations for the generator’s forward pass, computing gradients based on how well it generated fake samples that fooled the discriminator. These gradients then update the generator’s weights.
                # using tape to track trainable variables during generation and loss calculations
                with tf.GradientTape() as tape:
                    # Generate fake samples
                    generated_data = self.generator(noise, training=True)

                    # Discriminator output for fake data classifications
                    fake_output = self.discriminator(generated_data, training=False)

                    # Loss for generator to fool the discriminator based on its classifications
                    loss = self.generator_loss(fake_output)

                # calculate the gradient based on the loss respect to the weights of the model
                gradients = tape.gradient(loss, self.generator.trainable_variables)

                # Update the model based on the gradient of the loss respect to the weights of the model
                self.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

                if step % 100 == 0:
                    print(f"Epoch {epoch + 1}, Step {step}, G Loss: {loss.numpy()}")

            # After each epoch, evaluate on the validation set
            val_gen_loss = self.evaluate_validation()
            print(f'Epoch {epoch + 1}, Validation G Loss: {val_gen_loss}')

        return self.get_parameters(config={}), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.generator.set_weights(parameters)
        loss = 0
        for _ in self.x_test_ds:
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = self.generator(noise, training=False)
            fake_output = self.discriminator(generated_samples, training=False)
            loss += self.generator_loss(fake_output)
        return float(loss.numpy()), len(self.x_test), {}

    # Function to evaluate the generator on validation data
    def evaluate_validation(self):
        # Generate fake samples using the generator
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)

        # Pass the generated samples through the discriminator
        fake_output = self.discriminator(generated_samples, training=False)

        # Compute the generator loss (how well it fools the discriminator)
        gen_loss = self.generator_loss(fake_output)

        return float(gen_loss.numpy())