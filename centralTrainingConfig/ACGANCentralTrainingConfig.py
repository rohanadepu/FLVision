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
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import f1_score, classification_report
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

class CentralACGan:
    def __init__(self, discriminator, generator, ACGAN, nids, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch, learning_rate):
        #-- models
        self.generator = generator
        self.discriminator = discriminator
        self.ACGAN = ACGAN
        self.nids = nids

        #-- I/O Specs for models
        self.batch_size = BATCH_SIZE
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.input_dim = input_dim

        #-- training duration
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        #-- Data
        # Features
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        # Categorical Labels
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        #-- Optimizers
        # LR decay
        lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0002, decay_steps=10000, decay_rate=0.98, staircase=True)

        lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.98, staircase=True)

        # Compile optimizer
        self.gen_optimizer = Adam(learning_rate=lr_schedule_gen, beta_1=0.5, beta_2=0.999)
        self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999)

        #-- Model Compilations
        # Compile Discriminator separately (before freezing)
        self.discriminator.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            optimizer=self.disc_optimizer,
            metrics=['accuracy', 'binary_accuracy', 'categorical_accuracy', 'AUC']
        )

        # Freeze Discriminator only for AC-GAN training
        self.discriminator.trainable = False

        # Define AC-GAN (Generator + Frozen Discriminator)
        # I/O
        noise_input = tf.keras.Input(shape=(latent_dim,))
        label_input = tf.keras.Input(shape=(1,), dtype='int32')
        generated_data = self.generator([noise_input, label_input])
        validity, class_pred = self.discriminator(generated_data)
        # Compile Combined Model
        self.ac_gan = Model([noise_input, label_input], [validity, class_pred])
        self.ac_gan.compile(
            loss=['binary_crossentropy', 'categorical_crossentropy'],
            optimizer=self.gen_optimizer,
            metrics={'Discriminator': 'accuracy', 'Discriminator_1': 'categorical_accuracy'}
        )

    # -- Train -- #
    def train(self, X_train=None, y_train=None):
        if X_train is None or y_train is None:
            X_train = self.x_train
            y_train = self.y_train

        valid = tf.ones((self.batch_size, 1))
        fake = tf.zeros((self.batch_size, 1))

        for epoch in range(self.epochs):
            print(f'\n=== Epoch {epoch}/{self.epochs} ===\n')
            # --------------------------
            # Train Discriminator
            # --------------------------

            # Sample a batch of real data
            idx = tf.random.shuffle(tf.range(len(X_train)))[:self.batch_size]
            real_data = tf.gather(X_train, idx)
            real_labels = tf.gather(y_train, idx)

            # Ensure labels are one-hot encoded
            if len(real_labels.shape) == 1:
                real_labels_onehot = tf.one_hot(tf.cast(real_labels, tf.int32), depth=self.num_classes)
            else:
                real_labels_onehot = real_labels

            # Sample the noise data
            noise = tf.random.normal((self.batch_size, self.latent_dim))
            fake_labels = tf.random.uniform((self.batch_size,), minval=0, maxval=self.num_classes, dtype=tf.int32)
            fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)

            # Generate fake data
            generated_data = self.generator.predict([noise, fake_labels])

            # Train discriminator on real and fake data
            d_loss_real = self.discriminator.train_on_batch(real_data, [valid, real_labels_onehot])
            d_loss_fake = self.discriminator.train_on_batch(generated_data, [fake, fake_labels_onehot])
            d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)

            # Extract discriminator metrics
            d_validity_loss = d_loss[1]
            d_class_loss = d_loss[2]
            d_acc = d_loss[3] * 100
            d_auc = d_loss[4] * 100

            print(f"\nTraining Discriminator")
            print(
                f"Discriminator Loss: {d_loss[0]:.4f} | Validity Loss: {d_validity_loss:.4f} | Class Loss: {d_class_loss:.4f}")
            print(f"Discriminator Accuracy: {d_acc:.2f}% | AUC: {d_auc:.2f}%")

            # --------------------------
            # Train Generator (AC-GAN)
            # --------------------------

            # Generate noise and label inputs for ACGAN
            noise = tf.random.normal((self.batch_size, self.latent_dim))
            sampled_labels = tf.random.uniform((self.batch_size,), minval=0, maxval=self.num_classes,
                                               dtype=tf.int32)
            sampled_labels_onehot = tf.one_hot(sampled_labels, depth=self.num_classes)

            # Train ACGAN with sampled noise data
            g_loss = self.ac_gan.train_on_batch([noise, sampled_labels], [valid, sampled_labels_onehot])

            # Extract generator metrics
            g_validity_loss = g_loss[1]
            g_class_loss = g_loss[2]
            g_acc = g_loss[3] * 100

            print("\nTraining Generator with ACGAN FLOW")
            print(
                f"AC-GAN Generator Loss: {g_loss[0]:.4f} | Validity Loss: {g_validity_loss:.4f} | Class Loss: {g_class_loss:.4f}")
            print(f"Generator Accuracy: {g_acc:.2f}%")

            # --------------------------
            # Validation every 1 epochs
            # --------------------------
            if epoch % 1 == 0:
                print(f"\n=== Epoch {epoch} Validation ===")

                # Validate Discriminator
                d_val_loss, d_val_metrics = self.validation_disc()
                print(f"Validation Discriminator Avg Loss: {d_val_loss:.4f}")

                # Validate Generator (via AC-GAN)
                g_val_loss, g_val_metrics = self.validation_gen()
                print(f"Validation Generator Total Loss: {g_val_loss:.4f}")

                # Validate NIDS if defined
                if self.nids is not None:
                    nids_custom_loss, nids_val_metrics = self.validation_NIDS()
                    print(f"Validation NIDS Custom Loss: {nids_custom_loss:.4f}")

                # Optionally, print a summary of the training losses
                print(f"\nEpoch {epoch}: D Loss: {d_loss[0]:.4f}, G Loss: {g_loss[0]:.4f}, D Acc: {d_acc:.2f}%")

        # -- Loss Calculation -- #

    def nids_loss(self, real_output, fake_output):
        """
        Compute the NIDS loss on real and fake samples.
        For real samples, the target is 1 (benign), and for fake samples, 0 (attack).
        Returns a scalar loss value.
        """
        # define labels
        real_labels = tf.ones_like(real_output)
        fake_labels = tf.zeros_like(fake_output)

        # define loss function
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)

        # calculate outputs
        real_loss = bce(real_labels, real_output)
        fake_loss = bce(fake_labels, fake_output)

        # sum up total loss
        total_loss = real_loss + fake_loss
        return total_loss.numpy()

    # -- Validate -- #
    def validation_disc(self):
        """
        Evaluate the discriminator on the validation set.
        First, evaluate on real data (with labels = 1) and then on fake data (labels = 0).
        Prints and returns the average total loss along with a metrics dictionary.
        """
        # --- Evaluate on real validation data ---
        val_valid_labels = np.ones((len(self.x_val), 1))

        # Ensure y_val is one-hot encoded if needed
        if self.y_val.ndim == 1 or self.y_val.shape[1] != self.num_classes:
            y_val_onehot = tf.one_hot(self.y_val, depth=self.num_classes)
        else:
            y_val_onehot = self.y_val

        d_loss_real = self.discriminator.evaluate(
            self.x_val, [val_valid_labels, y_val_onehot], verbose=0
        )

        # --- Evaluate on generated (fake) data ---
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        fake_labels = tf.random.uniform(
            (len(self.x_val),), minval=0, maxval=self.num_classes, dtype=tf.int32
        )
        fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)
        fake_valid_labels = np.zeros((len(self.x_val), 1))
        generated_data = self.generator.predict([noise, fake_labels])
        d_loss_fake = self.discriminator.evaluate(
            generated_data, [fake_valid_labels, fake_labels_onehot], verbose=0
        )

        # --- Compute average loss ---
        avg_total_loss = 0.5 * (d_loss_real[0] + d_loss_fake[0])

        print("\nValidation Discriminator Evaluation:")
        print(f"Real Data -> Total Loss: {d_loss_real[0]:.4f}, "
              f"Validity Loss: {d_loss_real[1]:.4f}, Class Loss: {d_loss_real[2]:.4f}, "
              f"Accuracy: {d_loss_real[3] * 100:.2f}%, AUC: {d_loss_real[4] * 100:.2f}%")
        print(f"Fake Data -> Total Loss: {d_loss_fake[0]:.4f}, "
              f"Validity Loss: {d_loss_fake[1]:.4f}, Class Loss: {d_loss_fake[2]:.4f}, "
              f"Accuracy: {d_loss_fake[3] * 100:.2f}%, AUC: {d_loss_fake[4] * 100:.2f}%")
        print(f"Average Discriminator Loss: {avg_total_loss:.4f}")

        metrics = {
            "real": {"total_loss": d_loss_real[0],
                     "validity_loss": d_loss_real[1],
                     "class_loss": d_loss_real[2],
                     "accuracy": d_loss_real[3],
                     "auc": d_loss_real[4]},
            "fake": {"total_loss": d_loss_fake[0],
                     "validity_loss": d_loss_fake[1],
                     "class_loss": d_loss_fake[2],
                     "accuracy": d_loss_fake[3],
                     "auc": d_loss_fake[4]},
            "average_total_loss": avg_total_loss
        }
        return avg_total_loss, metrics

    def validation_gen(self):
        """
        Evaluate the generator (via the AC-GAN) using a validation batch.
        The generator is evaluated by its ability to “fool” the discriminator.
        Prints and returns the total generator loss along with key metrics.
        """
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        sampled_labels = tf.random.uniform(
            (len(self.x_val),), minval=0, maxval=self.num_classes, dtype=tf.int32
        )
        sampled_labels_onehot = tf.one_hot(sampled_labels, depth=self.num_classes)
        valid_labels = np.ones((len(self.x_val), 1))

        g_loss = self.ac_gan.evaluate(
            [noise, sampled_labels],
            [valid_labels, sampled_labels_onehot],
            verbose=0
        )

        print("\nValidation Generator (AC-GAN) Evaluation:")
        print(f"Total Loss: {g_loss[0]:.4f}, Validity Loss: {g_loss[1]:.4f}, "
              f"Class Loss: {g_loss[2]:.4f}, Accuracy: {g_loss[3] * 100:.2f}%")

        metrics = {"total_loss": g_loss[0],
                   "validity_loss": g_loss[1],
                   "class_loss": g_loss[2],
                   "accuracy": g_loss[3]}
        return g_loss[0], metrics

    def validation_NIDS(self):
        """
        Evaluate the NIDS model on validation data augmented with generated fake samples.
        Real data is labeled as 1 (benign) and fake/generated data as 0 (attack).
        Prints detailed metrics including a classification report and returns the custom
        NIDS loss along with a metrics dictionary.
        """
        if not hasattr(self, 'nids'):
            print("NIDS model is not defined.")
            return None

        # --- Prepare Real Data ---
        X_real = self.x_val
        y_real = np.ones((len(self.x_val),), dtype="int32")  # Real samples labeled 1

        # --- Generate Fake Data ---
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        fake_labels = tf.random.uniform(
            (len(self.x_val),), minval=0, maxval=self.num_classes, dtype=tf.int32
        )
        generated_samples = self.generator.predict([noise, fake_labels])
        X_fake = generated_samples
        y_fake = np.zeros((len(self.x_val),), dtype="int32")  # Fake samples labeled 0

        # --- Compute custom NIDS loss ---
        real_output = self.nids.predict(X_real)
        fake_output = self.nids.predict(X_fake)
        custom_nids_loss = self.nids_loss(real_output, fake_output)

        # --- Evaluate on the Combined Dataset ---
        X_combined = np.vstack([X_real, X_fake])
        y_combined = np.hstack([y_real, y_fake])
        nids_eval = self.nids.evaluate(X_combined, y_combined, verbose=0)
        # Expected order: [loss, accuracy, precision, recall, auc, logcosh]

        # --- Compute Additional Metrics ---
        y_pred_probs = self.nids.predict(X_combined)
        y_pred = (y_pred_probs > 0.5).astype("int32")
        f1 = f1_score(y_combined, y_pred)
        class_report = classification_report(
            y_combined, y_pred, target_names=["Attack (Fake)", "Benign (Real)"]
        )

        print("\nValidation NIDS Evaluation with Augmented Data:")
        print(f"Custom NIDS Loss (Real vs Fake): {custom_nids_loss:.4f}")
        print(f"Overall NIDS Loss: {nids_eval[0]:.4f}, Accuracy: {nids_eval[1]:.4f}, "
              f"Precision: {nids_eval[2]:.4f}, Recall: {nids_eval[3]:.4f}, "
              f"AUC: {nids_eval[4]:.4f}, LogCosh: {nids_eval[5]:.4f}")
        print("\nClassification Report:")
        print(class_report)
        print(f"F1 Score: {f1:.4f}")

        metrics = {
            "custom_nids_loss": custom_nids_loss,
            "loss": nids_eval[0],
            "accuracy": nids_eval[1],
            "precision": nids_eval[2],
            "recall": nids_eval[3],
            "auc": nids_eval[4],
            "logcosh": nids_eval[5],
            "f1_score": f1
        }
        return custom_nids_loss, metrics

    # -- Evaluate -- #
    def evaluate(self, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            X_test = self.x_test
            y_test = self.y_test

        print("Evaluating Discriminator...")
        d_loss, d_validity_loss, d_class_loss, d_acc, d_auc = self.discriminator.evaluate(X_test, [
            tf.ones((len(y_test), 1)), y_test], verbose=0)
        print(
            f"Discriminator Loss: {d_loss:.4f} | Validity Loss: {d_validity_loss:.4f} | Class Loss: {d_class_loss:.4f}")
        print(f"Discriminator Accuracy: {d_acc * 100:.2f}% | AUC: {d_auc * 100:.2f}%")

        print("Evaluating Generator...")
        noise = tf.random.normal((len(y_test), self.latent_dim))
        sampled_labels = tf.random.uniform((len(y_test),), minval=0, maxval=self.num_classes, dtype=tf.int32)
        g_loss = self.ac_gan.evaluate([noise, sampled_labels], [tf.ones((len(y_test), 1)),
                                                                tf.one_hot(sampled_labels, depth=self.num_classes)],
                                      verbose=0)

        print(f"Generator Loss: {g_loss[0]:.4f} | Validity Loss: {g_loss[1]:.4f} | Class Loss: {g_loss[2]:.4f}")
        print(f"Generator Accuracy: {g_loss[3] * 100:.2f}%")
