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
#                                       GAN Model Setup                                       #
################################################################################################################
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

    return generator,


def create_discriminator(input_dim):
    # Output will be a softmax over 3 classes: Normal (1), Intrusive (0), Fake (-1)
    discriminator = tf.keras.Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax')  # 3 classes: normal, intrusive, fake
    ])
    return discriminator


def create_model(input_dim, noise_dim):
    model = Sequential()

    model.add(create_generator(input_dim, noise_dim))
    model.add(create_discriminator(input_dim))

    return model


# def discriminator_loss(real_output, fake_output):
#     real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
#     fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
#     total_loss = real_loss + fake_loss
#     return total_loss

def discriminator_loss(real_normal_output, real_intrusive_output, fake_output):
    # Categorical cross-entropy for the three classes: normal, intrusive, and fake
    real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.ones_like(real_normal_output), real_normal_output)
    real_intrusive_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.zeros_like(real_intrusive_output), real_intrusive_output)
    fake_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.constant([-1], dtype=tf.float32), fake_output)
    total_loss = real_normal_loss + real_intrusive_loss + fake_loss
    return total_loss


# def generator_loss(fake_output):
#     return binary_crossentropy(tf.ones_like(fake_output), fake_output)

def generator_loss(fake_output):
    # Loss for generator is to fool the discriminator into classifying fake samples as real
    return tf.keras.losses.sparse_categorical_crossentropy(tf.ones_like(fake_output), fake_output)


def generate_and_save_network_traffic(model, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image.png')
    plt.show()


# hyperparameter
learning_rate = 0.0001  # 0.001 or .0001

# premade functions
# binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
seed = tf.random.normal([16, 100])


################################################################################################################
#                                               FL-GAN TRAINING Setup                                         #
################################################################################################################
# class GanClient(fl.client.NumPyClient):
#     def __init__(self, model, x_train, x_test, BATCH_SIZE, noise_dim, epochs, steps_per_epoch):
#         self.model = model
#         self.x_train = x_train
#         self.x_test = x_test
#
#         self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(BATCH_SIZE)
#         self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(BATCH_SIZE)
#
#         self.BATCH_SIZE = BATCH_SIZE
#         self.noise_dim = noise_dim
#         self.epochs = epochs
#         self.steps_per_epoch = steps_per_epoch
#
#     def get_parameters(self, config):
#         return self.model.get_weights()
#
#     # def fit(self, parameters, config):
#     #     self.model.set_weights(parameters)
#     #     generator = self.model.layers[0]
#     #     discriminator = self.model.layers[1]
#     #
#     #     for epoch in range(self.epochs):
#     #         for step, instances in enumerate(self.x_train_ds.take(self.steps_per_epoch)):
#     #
#     #             noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
#     #
#     #             with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
#     #                 generated_samples = generator(noise, training=True)
#     #
#     #                 real_output = discriminator(instances, training=True)
#     #                 fake_output = discriminator(generated_samples, training=True)
#     #
#     #                 gen_loss = generator_loss(fake_output)
#     #                 disc_loss = discriminator_loss(real_output, fake_output)
#     #
#     #             gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
#     #             gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
#     #
#     #             generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
#     #             discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
#     #
#     #             if step % 100 == 0:
#     #                 print(f'Epoch {epoch+1}, Step {step}, D Loss: {disc_loss.numpy()}, G Loss: {gen_loss.numpy()}')
#     #
#     #     return self.get_parameters(config={}), len(self.x_train), {}
#
#     def evaluate(self, parameters, config):
#         self.model.set_weights(parameters)
#         generator = self.model.layers[0]
#         discriminator = self.model.layers[1]
#
#         noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
#
#         generated_samples = generator(noise, training=False)
#
#         real_output = discriminator(self.x_test, training=False)
#         fake_output = discriminator(generated_samples, training=False)
#
#         loss = discriminator_loss(real_output, fake_output)
#
#         return float(loss.numpy()), len(self.x_test), {}

class GanClient(fl.client.NumPyClient):
    def __init__(self, generator, discriminator, x_train, x_val, y_val, x_test, BATCH_SIZE, noise_dim, epochs, steps_per_epoch):
        self.generator = generator
        self.discriminator = discriminator
        self.x_train = x_train
        self.x_val = x_val  # Add validation data
        self.y_val = y_val
        self.x_test = x_test
        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(self.BATCH_SIZE)

    def get_parameters(self, config):
        return self.generator.get_weights()

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
        disc_loss = discriminator_loss(real_normal_output, real_intrusive_output, fake_output)

        # Compute the generator loss: How well does the generator fool the discriminator
        gen_loss = generator_loss(fake_output)

        return float(disc_loss.numpy()), float(gen_loss.numpy())

    def evaluate_validation_NIDS(self, pretrained_nids_path):
        # Generate fake samples using the generator
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)

        # Split validation data into normal and intrusive traffic
        normal_data = self.X_val_data[self.y_val_data == 1]  # Real normal traffic
        intrusive_data = self.X_val_data[self.y_val_data == 0]  # Real intrusive traffic

        # load Pretrained NIDS model
        nids = tf.keras.models.load_model(pretrained_nids_path)

        # Step 4: Pass real and fake data through the NIDS model to get binary classification probabilities
        real_normal_output = nids(normal_data, training=False)  # Expected [P(normal), P(intrusive)]
        real_intrusive_output = nids(intrusive_data, training=False)  # Expected [P(normal), P(intrusive)]
        fake_output = nids(generated_samples, training=False)  # Expected [P(normal), P(intrusive)]

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

    def fit(self, parameters, config):
        self.generator.set_weights(parameters)

        gen_optimizer = Adam(learning_rate=0.0001)
        disc_optimizer = Adam(learning_rate=0.0001)

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
                    disc_loss = discriminator_loss(real_normal_output, real_intrusive_output, fake_output)
                    gen_loss = generator_loss(fake_output)

                # Apply gradients to both generator and discriminator (Training the models)
                # calculating gradiants of loss respect weights
                # (chain rule loss of model respect to outputs product of output respect to weights)
                gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                # Applying new parameters given from gradients
                gen_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                disc_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                if step % 100 == 0:
                    print(f'Epoch {epoch + 1}, Step {step}, D Loss: {disc_loss.numpy()}, G Loss: {gen_loss.numpy()}')

            # After each epoch, evaluate on the validation set
            val_disc_loss, val_gen_loss = self.evaluate_validation()
            print(f'Epoch {epoch + 1}, Validation D Loss: {val_disc_loss}, Validation G Loss: {val_gen_loss}')

        return self.get_parameters(config={}), len(self.x_train), {}

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

        disc_loss = discriminator_loss(real_normal_output, real_intrusive_output, fake_output)

        # Compute the generator loss: How well does the generator fool the discriminator
        gen_loss = generator_loss(fake_output)

        return {"discriminator_loss": float(disc_loss.numpy()), "generator_loss": float(gen_loss.numpy())}, len(self.x_test), {}
