#########################################################
#    Imports / Env setup                                #
#########################################################

import os
import random
import time
from datetime import datetime
import argparse
import flwr as fl

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle


# Function for creating the discriminator model
def create_discriminator(input_dim):
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
        Dense(1, activation='sigmoid')
    ])

    return discriminator


def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# Define a class to handle discriminator training
class DiscriminatorClient(fl.client.NumPyClient):
    def __init__(self, discriminator, x_train, x_test, BATCH_SIZE, epochs, steps_per_epoch):
        self.discriminator = discriminator
        self.x_train = x_train
        self.x_test = x_test
        self.BATCH_SIZE = BATCH_SIZE
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(self.BATCH_SIZE)

    def get_parameters(self, config):
        return self.discriminator.get_weights()

    def fit(self, parameters, config):
        self.discriminator.set_weights(parameters)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        for epoch in range(self.epochs):
            for step, instances in enumerate(self.x_train_ds.take(self.steps_per_epoch)):
                noise = tf.random.normal([self.BATCH_SIZE, instances.shape[1]])

                with tf.GradientTape() as tape:
                    fake_data = noise  # Generator is not used here
                    real_output = self.discriminator(instances, training=True)
                    fake_output = self.discriminator(fake_data, training=True)

                    loss = discriminator_loss(real_output, fake_output)

                gradients = tape.gradient(loss, self.discriminator.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

                if step % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.numpy()}")

        return self.get_parameters(config={}), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.discriminator.set_weights(parameters)
        loss = 0
        for instances in self.x_test_ds:
            real_output = self.discriminator(instances, training=False)
            fake_output = self.discriminator(tf.random.normal([self.BATCH_SIZE, instances.shape[1]]), training=False)
            loss += discriminator_loss(real_output, fake_output)
        return float(loss.numpy()), len(self.x_test), {}


def main():
    # Load your dataset and preprocessing logic here (like in the original code)
    X_train_data, X_test_data = ...  # Load data

    BATCH_SIZE = 256
    input_dim = X_train_data.shape[1]
    epochs = 5
    steps_per_epoch = len(X_train_data) // BATCH_SIZE

    discriminator = create_discriminator(input_dim)

    client = DiscriminatorClient(discriminator, X_train_data, X_test_data, BATCH_SIZE, epochs, steps_per_epoch)

    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

    # Save the trained discriminator model
    discriminator.save("discriminator_model.h5")


if __name__ == "__main__":
    main()
