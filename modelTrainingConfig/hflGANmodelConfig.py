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
#                                               FL-GAN TRAINING Setup                                         #
################################################################################################################

def generate_and_save_network_traffic(model, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image.png')
    plt.show()


class GanClient(fl.client.NumPyClient):
    def __init__(self, generator, discriminator, nids, x_train, x_val, y_val, x_test, BATCH_SIZE, noise_dim, epochs,
                 steps_per_epoch, learning_rate):
        self.generator = generator
        self.discriminator = discriminator
        self.nids = nids

        self.x_train = x_train
        self.x_val = x_val  # Add validation data
        self.y_val = y_val
        self.x_test = x_test

        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.learning_rate = learning_rate

        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(self.BATCH_SIZE)

        self.gen_optimizer = Adam(self.learning_rate)
        self.disc_optimizer = Adam(self.learning_rate)

    def discriminator_loss(self, real_normal_output, real_intrusive_output, fake_output):
        # Assign labels: 0 for normal, 1 for intrusive, and 2 for fake
        real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.zeros_like(real_normal_output), real_normal_output
        )
        real_intrusive_loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.ones_like(real_intrusive_output), real_intrusive_output
        )
        fake_loss = tf.keras.losses.sparse_categorical_crossentropy(
            tf.ones_like(fake_output) * 2, fake_output
        )

        # Total loss as the mean of all three losses
        total_loss = tf.reduce_mean(real_normal_loss + real_intrusive_loss + fake_loss)
        return total_loss

    def generator_loss(self, fake_output):
        # Generator aims to fool the discriminator by classifying fake samples as normal (0)
        return tf.keras.losses.sparse_categorical_crossentropy(
            tf.zeros_like(fake_output), fake_output
        )

    def get_parameters(self, config):
        return {
            "generator": self.generator.get_weights(),
            "discriminator": self.discriminator.get_weights()
        }

    def evaluate_validation(self):
        # Generate fake samples using the generator
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)

        # Split validation data into normal and intrusive traffic
        normal_data = self.X_val_data[self.y_val_data == 1]  # Real normal traffic
        intrusive_data = self.X_val_data[self.y_val_data == 0]  # Real intrusive traffic

        # Pass real and fake data through the discriminator
        real_normal_output = self.discriminator(normal_data, training=False)
        real_intrusive_output = self.discriminator(intrusive_data, training=False)
        fake_output = self.discriminator(generated_samples, training=False)

        # Compute the discriminator loss using the real and fake outputs
        disc_loss = self.discriminator_loss(real_normal_output, real_intrusive_output, fake_output)

        # Compute the generator loss: How well does the generator fool the discriminator
        gen_loss = self.generator_loss(fake_output)

        return float(disc_loss.numpy()), float(gen_loss.numpy())

    def evaluate_validation_NIDS(self):
        # Generate fake samples using the generator
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)

        # Split validation data into normal and intrusive traffic
        normal_data = self.X_val_data[self.y_val_data == 1]  # Real normal traffic
        intrusive_data = self.X_val_data[self.y_val_data == 0]  # Real intrusive traffic

        # Step 3: Pass real and fake data through the NIDS model to get binary classification probabilities
        real_normal_output = self.nids(normal_data, training=False)  # Expected [P(normal), P(intrusive)]
        real_intrusive_output = self.nids(intrusive_data, training=False)  # Expected [P(normal), P(intrusive)]
        fake_output = self.nids(generated_samples, training=False)  # Expected [P(normal), P(intrusive)]

        # Step 5: Define target labels for binary cross-entropy loss
        # Real normal traffic should have high probability in the normal class
        real_normal_labels = tf.ones((real_normal_output.shape[0],),
                                     dtype=tf.int32)  # Shape matches the number of samples
        # Real intrusive traffic should have high probability in the intrusive class
        real_intrusive_labels = tf.zeros((real_intrusive_output.shape[0],),
                                         dtype=tf.int32)  # Shape matches the number of samples
        # Fake traffic (generated samples) should be labeled as normal (to fool the NIDS)
        fake_labels = tf.ones((fake_output.shape[0],), dtype=tf.int32)  # Shape matches the number of generated samples

        # Step 6: Calculate binary cross-entropy loss for each category
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_normal_loss = bce(real_normal_labels, real_normal_output[:, 0])  # Compare with "normal" class probability
        real_intrusive_loss = bce(real_intrusive_labels,
                                  real_intrusive_output[:, 1])  # Compare with "intrusive" class probability
        fake_loss = bce(fake_labels, fake_output[:, 0])  # Compare with "normal" class probability for generated samples

        # Step 7: Combine the losses
        nids_loss = real_normal_loss + real_intrusive_loss
        gen_loss = fake_loss  # Generator loss to fool the NIDS

        return float(nids_loss.numpy()), float(gen_loss.numpy())

    def fit(self, generator_parameters, discriminator_parameters, config):
        self.generator.set_weights(generator_parameters)
        self.discriminator.set_weights(discriminator_parameters)

        for epoch in range(self.epochs):
            for step, real_data in enumerate(self.x_train_ds.take(self.steps_per_epoch)):
                # Split real data into normal and intrusive traffic
                normal_data = real_data[real_data['label'] == 1]  # Real normal traffic
                intrusive_data = real_data[real_data['label'] == 0]  # Real intrusive traffic

                # Generate fake data
                noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
                generated_samples = self.generator(noise, training=True)

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    # Discriminator outputs
                    real_normal_output = self.discriminator(normal_data, training=True)
                    real_intrusive_output = self.discriminator(intrusive_data, training=True)
                    fake_output = self.discriminator(generated_samples, training=True)

                    # Calculate losses of both models
                    disc_loss = self.discriminator_loss(real_normal_output, real_intrusive_output, fake_output)
                    gen_loss = self.generator_loss(fake_output)

                # Apply gradients to both generator and discriminator (Training the models)
                # calculating gradiants of loss respect weights
                # (chain rule loss of model respect to outputs product of output respect to weights)
                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                # Applying new parameters given from gradients
                self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                if step % 100 == 0:
                    print(f'Epoch {epoch + 1}, Step {step}, D Loss: {disc_loss.numpy()}, G Loss: {gen_loss.numpy()}')

            # After each epoch, evaluate on the validation set
            val_disc_loss, val_gen_loss = self.evaluate_validation()
            print(f'Epoch {epoch + 1}, Validation D Loss: {val_disc_loss}, Validation G Loss: {val_gen_loss}')

            if self.nids is not None:
                val_nids_loss, val_gen_2_loss = self.evaluate_validation_NIDS()
                print(f'Epoch {epoch + 1}, Validation NIDS Loss: {val_nids_loss}, Validation G Loss: {val_gen_2_loss}')

            # Return parameters for both generator and discriminator
            return {
                "generator": self.generator.get_weights(),
                "discriminator": self.discriminator.get_weights()
            }, len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.generator.set_weights(parameters)
        self.discriminator.set_weights(parameters)

        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)

        # Split the real test data into normal and intrusive traffic
        normal_data = self.x_test[self.x_test['label'] == 1]  # Real normal traffic
        intrusive_data = self.x_test[self.x_test['label'] == 0]  # Real intrusive traffic

        print(self.x_test.shape)
        print(generated_samples.shape)

        real_normal_output = self.discriminator(normal_data, training=True)
        real_intrusive_output = self.discriminator(intrusive_data, training=True)
        fake_output = self.discriminator(generated_samples, training=True)

        disc_loss = self.discriminator_loss(real_normal_output, real_intrusive_output, fake_output)

        # Compute the generator loss: How well does the generator fool the discriminator
        gen_loss = self.generator_loss(fake_output)

        return {"discriminator_loss": float(disc_loss.numpy()), "generator_loss": float(gen_loss.numpy())}, len(self.x_test), {}
