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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle


# Function for creating the generator model
def create_generator(input_dim, noise_dim):
    generator = tf.keras.Sequential([
        Dense(128, activation='relu', input_shape=(noise_dim,)),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(input_dim, activation='sigmoid')
    ])
    return generator


def generator_loss(fake_output):
    return tf.keras.losses.binary_crossentropy(tf.ones_like(fake_output), fake_output)


# Define a class to handle generator training
class GeneratorClient(fl.client.NumPyClient):
    def __init__(self, generator, discriminator, x_train, BATCH_SIZE, noise_dim, epochs, steps_per_epoch):
        self.generator = generator
        self.discriminator = discriminator
        self.x_train = x_train
        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.BATCH_SIZE)

    def get_parameters(self, config):
        return self.generator.get_weights()

    def fit(self, parameters, config):
        self.generator.set_weights(parameters)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        for epoch in range(self.epochs):
            for step in range(self.steps_per_epoch):
                noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

                with tf.GradientTape() as tape:
                    generated_samples = self.generator(noise, training=True)
                    fake_output = self.discriminator(generated_samples, training=False)
                    gen_loss = generator_loss(fake_output)

                gradients = tape.gradient(gen_loss, self.generator.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

                if step % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, G Loss: {gen_loss.numpy()}")

        return self.get_parameters(config={}), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.generator.set_weights(parameters)
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)
        print("Samples:", generated_samples)
        return 0.0, len(self.x_train), {}


def main():
    # Load your dataset and preprocessing logic here (like in the original code)
    X_train_data, _ = ...  # Load data

    # Load the pretrained discriminator
    discriminator = tf.keras.models.load_model("discriminator_model.h5")

    BATCH_SIZE = 256
    noise_dim = 100
    input_dim = X_train_data.shape[1]
    epochs = 5
    steps_per_epoch = len(X_train_data) // BATCH_SIZE

    generator = create_generator(input_dim, noise_dim)

    client = GeneratorClient(generator, discriminator, X_train_data, BATCH_SIZE, noise_dim, epochs, steps_per_epoch)

    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

    # Save the trained generator model
    generator.save("generator_model.h5")


if __name__ == "__main__":
    main()
