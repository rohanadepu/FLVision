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
#                                      Discriminator Model Setup                                        #
################################################################################################################


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
    ])
    return discriminator


# loss based on correct classifications between normal, intrusive, and fake traffic
def discriminator_loss(self, real_normal_output, real_intrusive_output, fake_output):
    # Create labels matching the shape of the output logits
    real_normal_labels = tf.ones((tf.shape(real_normal_output)[0],), dtype=tf.int32)  # Label 1 for normal
    real_intrusive_labels = tf.zeros((tf.shape(real_intrusive_output)[0],), dtype=tf.int32)  # Label 0 for intrusive
    fake_labels = tf.fill([tf.shape(fake_output)[0]], 2)  # Label 2 for fake traffic

    # Calculate sparse categorical cross-entropy loss for each group separately
    real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(real_normal_labels, real_normal_output)
    real_intrusive_loss = tf.keras.losses.sparse_categorical_crossentropy(real_intrusive_labels,
                                                                          real_intrusive_output)
    fake_loss = tf.keras.losses.sparse_categorical_crossentropy(fake_labels, fake_output)

    # Compute the mean for each loss group independently
    mean_real_normal_loss = tf.reduce_mean(real_normal_loss)
    mean_real_intrusive_loss = tf.reduce_mean(real_intrusive_loss)
    mean_fake_loss = tf.reduce_mean(fake_loss)

    # Total loss as the average of mean losses for each group
    total_loss = (mean_real_normal_loss + mean_real_intrusive_loss + mean_fake_loss) / 3
    return total_loss


# Loss for intrusion training (normal and intrusive)
def discriminator_loss_intrusion(real_normal_output, real_intrusive_output):
    # Create labels matching the shape of the output logits
    real_normal_labels = tf.ones((tf.shape(real_normal_output)[0],), dtype=tf.int32)  # Label 0 for normal
    real_intrusive_labels = tf.zeros((tf.shape(real_intrusive_output)[0],), dtype=tf.int32)  # Label 1 for intrusive

    # Calculate sparse categorical cross-entropy loss for each group separately
    real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(real_normal_labels, real_normal_output)
    real_intrusive_loss = tf.keras.losses.sparse_categorical_crossentropy(real_intrusive_labels,
                                                                          real_intrusive_output)

    # Compute the mean for each loss group independently
    mean_real_normal_loss = tf.reduce_mean(real_normal_loss)
    mean_real_intrusive_loss = tf.reduce_mean(real_intrusive_loss)

    # Total loss as the average of mean losses for each group
    total_loss = (mean_real_normal_loss + mean_real_intrusive_loss) / 2

    # real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.zeros_like(real_normal_output), real_normal_output)  # Label 0 for normal
    # real_intrusive_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.ones_like(real_intrusive_output), real_intrusive_output)  # Label 1 for intrusive
    # total_loss = real_normal_loss + real_intrusive_loss

    return total_loss


# Loss for synthetic training (normal and fake)
def discriminator_loss_synthetic(real_normal_output, fake_output):
    # Create labels matching the shape of the output logits
    real_normal_labels = tf.ones((tf.shape(real_normal_output)[0],), dtype=tf.int32)  # Label 1 for normal
    fake_labels = tf.fill([tf.shape(fake_output)[0]], 2)  # Label 2 for fake traffic

    # Calculate sparse categorical cross-entropy loss for each group separately
    real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(real_normal_labels, real_normal_output)

    fake_loss = tf.keras.losses.sparse_categorical_crossentropy(fake_labels, fake_output)

    # Compute the mean for each loss group independently
    mean_real_normal_loss = tf.reduce_mean(real_normal_loss)
    mean_fake_loss = tf.reduce_mean(fake_loss)

    # Total loss as the average of mean losses for each group
    total_loss = (mean_real_normal_loss + mean_fake_loss) / 2

    # real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.zeros_like(real_normal_output), real_normal_output)  # Label 0 for normal
    # fake_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.fill(tf.shape(fake_output), 2), fake_output)  # Label 2 for fake
    # total_loss = real_normal_loss + fake_loss

    return total_loss


################################################################################################################
#                                       Discriminator INTRUSION Training                                       #
################################################################################################################

class DiscriminatorIntrusionClient(fl.client.NumPyClient):
    def __init__(self, discriminator, generator, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 noise_dim, epochs, steps_per_epoch):
        self.generator = generator
        self.discriminator = discriminator

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val  # Add validation data
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(self.BATCH_SIZE)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    def get_parameters(self, config):
        return self.discriminator.get_weights()

    def fit(self, parameters, config):
        self.discriminator.set_weights(parameters)

        for epoch in range(self.epochs):
            for step, real_data in enumerate(self.x_train_ds.take(self.steps_per_epoch)):
                # Assume real_data contains both normal and intrusive traffic
                # Split the real_data into normal and intrusive samples
                normal_data = real_data[real_data['Label' if self.dataset_used == "IOTBOTNET" else 'label'] == 1]  # Real normal traffic
                intrusive_data = real_data[real_data['Label' if self.dataset_used == "IOTBOTNET" else 'label'] == 0]  # Real malicious traffic

                # captures the discriminator’s operations to compute the gradients for adjusting its weights based on how well it classified real vs. fake data.
                # using tape to track trainable variables during discriminator classification and loss calculations
                with tf.GradientTape() as tape:
                    # Discriminator outputs for normal and intrusive data
                    real_normal_output = self.discriminator(normal_data, training=True)[:, :2]  # Only normal and intrusive
                    real_intrusive_output = self.discriminator(intrusive_data, training=True)[:, :2]

                    # Loss calculation for normal and intrusive data
                    loss = discriminator_loss_intrusion(real_normal_output, real_intrusive_output)

                gradients = tape.gradient(loss, self.discriminator.trainable_variables)

                self.optimize.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

                if step % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, D Loss: {loss.numpy()}")

            # After each epoch, evaluate on the validation set
            val_disc_loss = self.evaluate_validation()
            print(f'Epoch {epoch+1}, Validation D Loss: {val_disc_loss}')

        return self.get_parameters(config={}), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.discriminator.set_weights(parameters)
        loss = 0

        # Create a TensorFlow dataset that includes both test features and labels
        test_data = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(self.BATCH_SIZE)

        for instances, labels in test_data:
            # Filter normal and intrusive instances
            normal_data = tf.boolean_mask(instances, tf.equal(labels, 1))  # Assuming label 1 for normal
            intrusive_data = tf.boolean_mask(instances, tf.equal(labels, 0))  # Assuming label 0 for intrusive

            # Discriminator predictions
            real_normal_output = self.discriminator(normal_data, training=False)
            real_intrusive_output = self.discriminator(intrusive_data, training=False)

            # Compute the loss for this batch
            batch_loss = discriminator_loss_intrusion(real_normal_output, real_intrusive_output)
            loss += batch_loss

        return float(loss.numpy()), len(self.x_test), {}

    # Function to evaluate the discriminator on validation data
    def evaluate_validation(self):
        # Create a TensorFlow dataset for validation data
        val_data = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val)).batch(self.BATCH_SIZE)

        total_loss = 0
        for instances, labels in val_data:
            # Filter normal and intrusive instances
            normal_data = tf.boolean_mask(instances, tf.equal(labels, 1))  # Assuming label 1 for normal
            intrusive_data = tf.boolean_mask(instances, tf.equal(labels, 0))  # Assuming label 0 for intrusive

            # Discriminator predictions
            real_normal_output = self.discriminator(normal_data, training=False)
            real_intrusive_output = self.discriminator(intrusive_data, training=False)

            # Compute the loss for this batch
            batch_loss = discriminator_loss_intrusion(real_normal_output, real_intrusive_output)
            total_loss += batch_loss

        return float(total_loss.numpy())

################################################################################################################
#                                       Discriminator SYNTHETIC Training                                       #
################################################################################################################


class DiscriminatorSyntheticTrainingClient(fl.client.NumPyClient):
    def __init__(self, discriminator, generator, x_train, x_val, y_val, x_test, BATCH_SIZE, noise_dim, epochs, steps_per_epoch, dataset_used):
        self.discriminator = discriminator
        self.generator = generator  # Generator is fixed during discriminator training
        self.x_train = x_train
        self.x_val = x_val  # Validation data
        self.y_val = y_val  # Validation labels
        self.x_test = x_test
        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.dataset_used = dataset_used

        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(self.BATCH_SIZE)

        # # Compile the discriminator
        # self.discriminator.compile(
        #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        #     loss=discriminator_loss,  # Using the custom loss function
        #     metrics=['accuracy']
        # )

    def get_parameters(self, config):
        return self.discriminator.get_weights()

    def fit(self, parameters, config):
        self.discriminator.set_weights(parameters)

        train_data = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train)).batch(self.BATCH_SIZE)

        for epoch in range(self.epochs):
            for step, (real_data, real_labels) in enumerate(train_data.take(self.steps_per_epoch)):
                # Create masks for normal and intrusive traffic based on labels
                normal_mask = tf.equal(real_labels, 1)  # Assuming label 1 for normal

                # Filter data based on these masks
                normal_data = tf.boolean_mask(real_data, normal_mask)

                # Generate fake data using the generator
                noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
                generated_data = self.generator(noise, training=False)

                # captures the discriminator’s operations to compute the gradients for adjusting its weights based on how well it classified real vs. fake data.
                # using tape to track trainable variables during discriminator classification and loss calculations
                with tf.GradientTape() as tape:
                    # Discriminator outputs based on its classifications from inputted data in parameters
                    real_normal_output = self.discriminator(normal_data, training=True)
                    fake_output = self.discriminator(generated_data, training=True)

                    # Loss calculation for normal, intrusive, and fake data
                    loss = discriminator_loss_synthetic(real_normal_output, fake_output)

                # calculate the gradient based on the loss respect to the weights of the model
                gradients = tape.gradient(loss, self.discriminator.trainable_variables)

                # Update the model based on the gradient of the loss respect to the weights of the model
                self.optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

                if step % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, D Loss: {loss.numpy()}")

            # After each epoch, evaluate on the validation set
            val_disc_loss = self.evaluate_validation()
            print(f'Epoch {epoch+1}, Validation D Loss: {val_disc_loss}')

        return self.get_parameters(config={}), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.discriminator.set_weights(parameters)
        loss = 0

        # Create a TensorFlow dataset that includes both test features and labels
        test_data = tf.data.Dataset.from_tensor_slices((self.x_test, self.y_test)).batch(self.BATCH_SIZE)

        for instances, labels in test_data:
            # Filter normal and intrusive instances
            normal_data = tf.boolean_mask(instances, tf.equal(labels, 1))  # Assuming label 1 for normal
            # Generate fake data
            fake_data = self.generator(tf.random.normal([self.BATCH_SIZE, self.noise_dim]), training=False)

            # Discriminator predictions
            real_normal_output = self.discriminator(normal_data, training=False)
            fake_output = self.discriminator(fake_data, training=False)

            # Compute the loss for this batch
            batch_loss = discriminator_loss_synthetic(real_normal_output, fake_output)
            loss += batch_loss
        return float(loss.numpy()), len(self.x_test), {}

    # Function to evaluate the discriminator on validation data
    def evaluate_validation(self):
        # Generate fake samples
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)

        # Create a TensorFlow dataset for validation data
        val_data = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val)).batch(self.BATCH_SIZE)

        total_loss = 0
        for instances, labels in val_data:
            # Filter normal and intrusive instances
            normal_data = tf.boolean_mask(instances, tf.equal(labels, 1))  # Assuming label 1 for normal

            # Discriminator predictions
            real_normal_output = self.discriminator(normal_data, training=False)
            fake_output = self.discriminator(generated_samples, training=False)

            # Compute the loss for this batch
            batch_loss = discriminator_loss_synthetic(real_normal_output, fake_output)
            total_loss += batch_loss

        return float(total_loss.numpy())
