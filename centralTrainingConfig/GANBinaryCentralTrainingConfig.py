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


class CentralBinaryGan:
    def __init__(self, model, nids, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 noise_dim, epochs, steps_per_epoch, learning_rate):
        self.model = model
        self.nids = nids
        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.x_train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(self.BATCH_SIZE)
        self.x_val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.BATCH_SIZE)

        lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0002, decay_steps=10000, decay_rate=0.98, staircase=True)

        lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.98, staircase=True)

        self.gen_optimizer = Adam(learning_rate=lr_schedule_gen, beta_1=0.5, beta_2=0.999)
        self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999)

        self.generator = self.model.layers[0]
        self.discriminator = self.model.layers[1]

    # -- Loss Calculation -- #
    def discriminator_loss(self, real_output, fake_output):
        # Create binary labels: 1 for real, 0 for fake
        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)

        # Compute binary cross-entropy loss
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_loss = bce(real_labels, real_output)
        fake_loss = bce(fake_labels, fake_output)

        return real_loss + fake_loss

    def generator_loss(self, fake_output):
        # Generator tries to make fake samples be classified as real (1)
        fake_labels = tf.ones_like(fake_output)  # Label 1 for fake samples
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        return bce(fake_labels, fake_output)

    # -- Train -- #
    def fit(self):
        for epoch in range(self.epochs):
            for step, (real_data, real_labels) in enumerate(self.x_train_ds.take(self.steps_per_epoch)):
                # Generate fake data
                noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    # Generate samples
                    generated_samples = self.generator(noise, training=True)

                    # Discriminator predictions
                    real_output = self.discriminator(real_data, training=True)
                    fake_output = self.discriminator(generated_samples, training=True)

                    # Compute losses
                    disc_loss = self.discriminator_loss(real_output, fake_output)
                    gen_loss = self.generator_loss(fake_output)

                # Compute gradients and apply updates
                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                # Apply gradient clipping
                gradients_of_generator, _ = tf.clip_by_global_norm(gradients_of_generator, 5.0)
                gradients_of_discriminator, _ = tf.clip_by_global_norm(gradients_of_discriminator, 5.0)

                self.gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                self.disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                if step % 100 == 0:
                    print(f'Epoch {epoch + 1}, Step {step}, D Loss: {disc_loss.numpy()}, G Loss: {gen_loss.numpy()}')

            # Validation after each epoch
            val_disc_loss = self.evaluate_validation_disc()
            print(f'Epoch {epoch + 1}, Validation D Loss: {val_disc_loss}')

            if self.nids is not None:
                val_nids_loss = self.evaluate_validation_NIDS()
                print(f'Epoch {epoch + 1}, Validation NIDS Loss: {val_nids_loss}')

    # -- Validate -- #
    def evaluate_validation_disc(self):
        total_disc_loss = 0.0
        num_batches = 0

        for step, (real_data, real_labels) in enumerate(self.x_val_ds):
            # Generate fake samples using the generator
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = self.generator(noise, training=False)

            # Pass real and fake data through the discriminator
            real_output = self.discriminator(real_data, training=False)
            fake_output = self.discriminator(generated_samples, training=False)

            # Compute the discriminator loss using the real and fake outputs
            disc_loss = self.discriminator_loss(real_output, fake_output)
            total_disc_loss += disc_loss.numpy()
            num_batches += 1

        # Average discriminator loss over all validation batches
        avg_disc_loss = total_disc_loss / num_batches
        return avg_disc_loss

    def evaluate_validation_gen(self):
        # Generate fake samples using the generator
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)

        # fake data through the discriminator
        fake_output = self.discriminator(generated_samples, training=False)

        # Compute the generator loss: How well does the generator fool the discriminator
        gen_loss = self.generator_loss(fake_output)

        return float(gen_loss.numpy())

    def evaluate_validation_NIDS(self):

        # Generate fake samples
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)

        # Real outputs
        real_output = self.nids(self.x_val_ds, training=False)  # Binary classification probabilities
        fake_output = self.nids(generated_samples, training=False)

        # Binary cross-entropy for real and fake samples
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        real_loss = bce(tf.ones_like(real_output), real_output)  # Label real samples as 1
        fake_loss = bce(tf.zeros_like(fake_output), fake_output)  # Label fake samples as 0

        nids_loss = real_loss
        gen_loss = fake_loss  # Generator loss to fool NIDS

        print(f'Validation GEN-NIDS Loss: {gen_loss}')
        return float(nids_loss.numpy())

    # -- Evaluate -- #
    def evaluate(self):
        total_disc_loss = 0.0
        total_gen_loss = 0.0
        num_batches = 0

        for step, (test_data_batch, test_labels_batch) in enumerate(self.x_test_ds):
            # Generate fake samples
            noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
            generated_samples = self.generator(noise, training=False)

            # Binary masks for separating normal and intrusive test data
            normal_mask = tf.equal(test_labels_batch, 1)  # Label 1 for normal
            intrusive_mask = tf.equal(test_labels_batch, 0)  # Label 0 for intrusive

            # Apply masks to create separate datasets
            normal_data = tf.boolean_mask(test_data_batch, normal_mask)
            intrusive_data = tf.boolean_mask(test_data_batch, intrusive_mask)

            print(f"Batch {step + 1}:")
            print(f"Normal data shape: {normal_data.shape}")
            print(f"Intrusive data shape: {intrusive_data.shape}")
            print(f"Generated samples shape: {generated_samples.shape}")

            # Discriminator predictions
            real_output = self.discriminator(test_data_batch, training=False)  # Real test data
            fake_output = self.discriminator(generated_samples, training=False)  # Generated samples

            # Binary cross-entropy loss for discriminator
            disc_loss = self.discriminator_loss(real_output, fake_output)
            total_disc_loss += disc_loss.numpy()

            # Binary cross-entropy loss for generator
            gen_loss = self.generator_loss(fake_output)
            total_gen_loss += gen_loss.numpy()

            num_batches += 1

        # Average losses over all test batches
        avg_disc_loss = total_disc_loss / num_batches
        avg_gen_loss = total_gen_loss / num_batches

        print(f"Final Evaluation Discriminator Loss: {avg_disc_loss}")
        print(f"Final Evaluation Generator Loss: {avg_gen_loss}")

        # Return average discriminator loss, number of test samples, and an empty dictionary (optional outputs)
        return avg_disc_loss, len(self.x_test_ds), {}
