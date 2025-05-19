#########################################################
#    Imports / Env setup                                #
#########################################################

import os
import random
import time
import logging
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
from collections import Counter
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
    def __init__(self, discriminator, generator, nids, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch, learning_rate,
                 log_file="training.log"):
        #-- models
        self.generator = generator
        self.discriminator = discriminator
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

        # -- Setup Logging
        self.setup_logger(log_file)

        #-- Optimizers
        # LR decay
        lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.00001, decay_steps=10000, decay_rate=0.97, staircase=False)

        lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.00003, decay_steps=10000, decay_rate=0.97, staircase=False)

        # Compile optimizer
        self.gen_optimizer = Adam(learning_rate=lr_schedule_gen, beta_1=0.5, beta_2=0.999)
        self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999)

        print("Discriminator Output:", self.discriminator.output_names)

        #-- Model Compilations
        # Compile Discriminator separately (before freezing)
        # self.discriminator.compile(
        #     loss={'validity': 'binary_crossentropy', 'class': 'categorical_crossentropy'},
        #     optimizer=self.disc_optimizer,
        #     metrics={
        #         'validity': ['binary_accuracy'],
        #         'class': ['categorical_accuracy']
        #     }
        # )

        # Freeze Discriminator only for AC-GAN training
        self.discriminator.trainable = False

        # Define AC-GAN (Generator + Frozen Discriminator)
        # I/O
        noise_input = tf.keras.Input(shape=(latent_dim,))
        label_input = tf.keras.Input(shape=(1,), dtype='int32')
        generated_data = self.generator([noise_input, label_input])
        validity, class_pred = self.discriminator(generated_data)
        # Compile Combined Model
        self.ACGAN = Model([noise_input, label_input], [validity, class_pred])

        print("ACGAN Output:", self.ACGAN.output_names)

        self.ACGAN.compile(
            loss={'Discriminator': 'binary_crossentropy', 'Discriminator_1': 'categorical_crossentropy'},
            optimizer=self.gen_optimizer,
            metrics={
                'Discriminator': ['binary_accuracy'],
                'Discriminator_1': ['categorical_accuracy']
            }
        )

    # Saving function for ACGAN
    def setACGAN(self):
        return self.ACGAN

    # -- logging Functions -- #
    def setup_logger(self, log_file):
        """Set up a logger that records both to a file and to the console."""
        self.logger = logging.getLogger("CentralACGan")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # Avoid adding duplicate handlers if logger already has them.
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log_model_settings(self):
        """Logs model names, structures, and hyperparameters."""
        self.logger.info("=== Model Settings ===")

        self.logger.info("Generator Model Summary:")
        generator_summary = []
        self.generator.summary(print_fn=lambda x: generator_summary.append(x))
        for line in generator_summary:
            self.logger.info(line)

        self.logger.info("Discriminator Model Summary:")
        discriminator_summary = []
        self.discriminator.summary(print_fn=lambda x: discriminator_summary.append(x))
        for line in discriminator_summary:
            self.logger.info(line)

        if self.nids is not None:
            self.logger.info("NIDS Model Summary:")
            nids_summary = []
            self.nids.summary(print_fn=lambda x: nids_summary.append(x))
            for line in nids_summary:
                self.logger.info(line)
        else:
            self.logger.info("NIDS Model is not defined.")

        self.logger.info("--- Hyperparameters ---")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Noise Dimension: {self.noise_dim}")
        self.logger.info(f"Latent Dimension: {self.latent_dim}")
        self.logger.info(f"Number of Classes: {self.num_classes}")
        self.logger.info(f"Input Dimension: {self.input_dim}")
        self.logger.info(f"Epochs: {self.epochs}")
        self.logger.info(f"Steps per Epoch: {self.steps_per_epoch}")
        self.logger.info(f"Learning Rate (Generator): {self.gen_optimizer.learning_rate}")
        self.logger.info(f"Learning Rate (Discriminator): {self.disc_optimizer.learning_rate}")
        self.logger.info("=" * 50)

    def log_epoch_metrics(self, epoch, d_metrics, g_metrics, nids_metrics=None, fusion_metrics=None):
        """Logs a formatted summary of the metrics for this epoch."""
        self.logger.info(f"=== Epoch {epoch} Metrics Summary ===")
        self.logger.info("Discriminator Metrics:")
        for key, value in d_metrics.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("Generator Metrics:")
        for key, value in g_metrics.items():
            self.logger.info(f"  {key}: {value}")
        if nids_metrics is not None:
            self.logger.info("NIDS Metrics:")
            for key, value in nids_metrics.items():
                self.logger.info(f"  {key}: {value}")
        if fusion_metrics is not None:
            self.logger.info("Probabilistic Fusion Metrics:")
            for key, value in fusion_metrics.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    def log_evaluation_metrics(self, d_eval, g_eval, nids_eval=None, fusion_eval=None):
        """Logs a formatted summary of evaluation metrics."""
        self.logger.info("=== Evaluation Metrics Summary ===")
        self.logger.info("Discriminator Evaluation:")
        for key, value in d_eval.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("Generator Evaluation:")
        for key, value in g_eval.items():
            self.logger.info(f"  {key}: {value}")
        if nids_eval is not None:
            self.logger.info("NIDS Evaluation:")
            for key, value in nids_eval.items():
                self.logger.info(f"  {key}: {value}")
        if fusion_eval is not None:
            self.logger.info("Probabilistic Fusion Evaluation:")
            for key, value in fusion_eval.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    # -- Train -- #
    def fit(self, X_train=None, y_train=None, d_to_g_ratio=3):
        """
        Train the AC-GAN with a configurable ratio between discriminator and generator training steps.

        Parameters:
        -----------
        X_train : array-like, optional
            Training features. If None, uses self.x_train.
        y_train : array-like, optional
            Training labels. If None, uses self.y_train.
        d_to_g_ratio : int, optional
            Ratio of discriminator training steps to generator training steps.
            Default is 3 (train discriminator 3 times for each generator training step).
        """
        if X_train is None or y_train is None:
            X_train = self.x_train
            y_train = self.y_train

        # -- Make sure discriminator is trainable for individual training -- #
        self.discriminator.trainable = True
        # Ensure all layers within discriminator are trainable
        for layer in self.discriminator.layers:
            layer.trainable = True

        # -- Re-compile discriminator with trainable weights -- #
        self.discriminator.compile(
            loss={'validity': 'binary_crossentropy', 'class': 'categorical_crossentropy'},
            optimizer=self.disc_optimizer,
            metrics={
                'validity': ['binary_accuracy'],
                'class': ['categorical_accuracy']
            }
        )

        # Log model settings at the start
        self.log_model_settings()
        self.logger.info(f"Training with discriminator-to-generator ratio: {d_to_g_ratio}:1")

        # -- Apply Class split for Class Specific Training
        # Separate data by class
        benign_indices = tf.where(tf.equal(tf.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train, 0))
        attack_indices = tf.where(tf.equal(tf.argmax(y_train, axis=1) if y_train.ndim > 1 else y_train, 1))

        # -- Apply label smoothing -- #
        # Create smoothed labels for discriminator training
        valid_smoothing_factor = 0.08
        valid_smooth = tf.ones((self.batch_size, 1)) * (1 - valid_smoothing_factor)

        fake_smoothing_factor = 0.05
        fake_smooth = tf.zeros((self.batch_size, 1)) + fake_smoothing_factor

        # For generator training, we use a slightly different smoothing
        # to keep the generator from becoming too confident
        gen_smoothing_factor = 0.08
        valid_smooth_gen = tf.ones((self.batch_size, 1)) * (1 - gen_smoothing_factor)  # Slightly less than 1.0

        self.logger.info(f"Using valid label smoothing with factor: {valid_smoothing_factor}")
        self.logger.info(f"Using fake label smoothing with factor: {fake_smoothing_factor}")
        self.logger.info(f"Using gen label smoothing with factor: {gen_smoothing_factor}")

        # -- Initialize metrics tracking -- #
        d_metrics_history = []
        g_metrics_history = []

        # -- Training Loop -- #
        for epoch in range(self.epochs):
            print(f'\n=== Epoch {epoch + 1}/{self.epochs} ===\n')
            self.logger.info(f'=== Epoch {epoch}/{self.epochs} ===')

            epoch_d_losses = []
            epoch_g_losses = []

            # Determine how many steps per epoch based on batch size
            actual_steps = min(self.steps_per_epoch, len(X_train) // self.batch_size)

            for step in range(actual_steps):
                # --------------------------
                # Train Discriminator (multiple times per generator step)
                # --------------------------
                d_step_losses = []

                for d_step in range(d_to_g_ratio):
                    # -- Train on real data -- #
                    # - BATCH 1: Train on benign data - #
                    if len(benign_indices) > self.batch_size:
                        # Select a new batch of benign samples
                        benign_idx = tf.random.shuffle(benign_indices)[:self.batch_size]
                        benign_data = tf.gather(X_train, benign_idx)
                        benign_labels = tf.gather(y_train, benign_idx)

                        # Fix the shape issue - ensure benign_data is 2D
                        if len(benign_data.shape) > 2:
                            benign_data = tf.reshape(benign_data, (benign_data.shape[0], -1))

                        # Ensure one-hot encoding with correct shape
                        if len(benign_labels.shape) == 1:
                            benign_labels_onehot = tf.one_hot(tf.cast(benign_labels, tf.int32), depth=self.num_classes)
                        else:
                            benign_labels_onehot = benign_labels

                        # Ensure benign_labels_onehot has shape (batch_size, num_classes)
                        if len(benign_labels_onehot.shape) > 2:
                            benign_labels_onehot = tf.reshape(benign_labels_onehot,
                                                              (benign_labels_onehot.shape[0], self.num_classes))

                        # Create valid labels with correct shape
                        valid_smooth_benign = tf.ones((benign_data.shape[0], 1)) * (1 - valid_smoothing_factor)

                        # Train discriminator on real benign data
                        d_loss_benign = self.discriminator.train_on_batch(benign_data,
                                                                          [valid_smooth_benign, benign_labels_onehot])

                    # - BATCH 2: Train on attack data - #
                    if len(attack_indices) > self.batch_size:
                        # Select a new batch of attack samples
                        attack_idx = tf.random.shuffle(attack_indices)[:self.batch_size]
                        attack_data = tf.gather(X_train, attack_idx)
                        attack_labels = tf.gather(y_train, attack_idx)

                        # Fix the shape issue - ensure attack_data is 2D
                        if len(attack_data.shape) > 2:
                            attack_data = tf.reshape(attack_data, (attack_data.shape[0], -1))

                        # Ensure one-hot encoding with correct shape
                        if len(attack_labels.shape) == 1:
                            attack_labels_onehot = tf.one_hot(tf.cast(attack_labels, tf.int32), depth=self.num_classes)
                        else:
                            attack_labels_onehot = attack_labels

                        # Ensure attack_labels_onehot has shape (batch_size, num_classes)
                        if len(attack_labels_onehot.shape) > 2:
                            attack_labels_onehot = tf.reshape(attack_labels_onehot,
                                                              (attack_labels_onehot.shape[0], self.num_classes))

                        # Create valid labels with correct shape
                        valid_smooth_attack = tf.ones((attack_data.shape[0], 1)) * (1 - valid_smoothing_factor)

                        # Train discriminator on real attack data
                        d_loss_attack = self.discriminator.train_on_batch(attack_data,
                                                                          [valid_smooth_attack, attack_labels_onehot])

                    # -- Train on fake data -- #
                    # -- BATCH 3: Generate fake data -- #
                    # Generate a new batch of fake data
                    # Sample the noise data
                    noise = tf.random.normal((self.batch_size, self.latent_dim))
                    fake_labels = tf.random.uniform((self.batch_size,), minval=0, maxval=self.num_classes,
                                                    dtype=tf.int32)
                    fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)

                    # generate data from noise and desired labels
                    generated_data = self.generator.predict([noise, fake_labels], verbose=0)

                    # -- Train discriminator on fake data -- #
                    d_loss_fake = self.discriminator.train_on_batch(generated_data, [fake_smooth, fake_labels_onehot])

                    # Inside the fit method after training on benign, attack, and fake data
                    d_loss, d_metrics = self.calculate_weighted_loss(
                        d_loss_benign,
                        d_loss_attack,
                        d_loss_fake,
                        attack_weight=0.5,  # Adjust as needed
                        benign_weight=0.5,  # Adjust as needed
                        validity_weight=0.5,  # Adjust as needed
                        class_weight=0.5  # Adjust as needed
                    )

                    d_step_losses.append(float(d_metrics['Total Loss']))

                # Store average discriminator loss for this step
                avg_d_loss = sum(d_step_losses) / len(d_step_losses)
                epoch_d_losses.append(avg_d_loss)

                # --------------------------
                # Train Generator (AC-GAN) - only once per multiple discriminator steps
                # --------------------------
                # -- Freeze discriminator for generator training -- #
                self.discriminator.trainable = False

                # -- BATCH 4: NEW GENERATOR INPUT -- #
                # -- Generate new noise and label inputs for ACGAN (generator) training -- #
                noise = tf.random.normal((self.batch_size, self.latent_dim))
                sampled_labels = tf.random.uniform((self.batch_size,), minval=0, maxval=self.num_classes,
                                                   dtype=tf.int32)
                sampled_labels_onehot = tf.one_hot(sampled_labels, depth=self.num_classes)

                # -- Train ACGAN with sampled noise data -- #
                g_loss = self.ACGAN.train_on_batch([noise, sampled_labels], [valid_smooth_gen, sampled_labels_onehot])
                epoch_g_losses.append(g_loss[0])

                # -- Unfreeze discriminator for next steps -- #
                self.discriminator.trainable = True

                # Print progress every few steps
                if step % max(1, actual_steps // 10) == 0:
                    print(f"Step {step}/{actual_steps} - D loss: {avg_d_loss:.4f}, G loss: {g_loss[0]:.4f}")

            # Collect metrics for this epoch
            avg_epoch_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
            avg_epoch_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)

            # Collect generator metrics from the last step
            g_metrics = {
                "Total Loss": f"{g_loss[0]:.4f}",
                "Validity Loss": f"{g_loss[1]:.4f}",  # This is Discriminator_loss
                "Class Loss": f"{g_loss[2]:.4f}",  # This is Discriminator_1_loss
                "Validity Binary Accuracy": f"{g_loss[3] * 100:.2f}%",  # Discriminator_binary_accuracy
                "Class Categorical Accuracy": f"{g_loss[4] * 100:.2f}%"  # Discriminator_1_categorical_accuracy
            }

            # Log metrics
            self.logger.info(f"Epoch {epoch + 1} Summary:")
            self.logger.info(f"Discriminator Average Loss: {avg_epoch_d_loss:.4f}")
            self.logger.info(f"Generator Loss: {avg_epoch_g_loss:.4f}")
            self.logger.info(f"Generator Validity Accuracy: {g_loss[3] * 100:.2f}%")
            self.logger.info(f"Generator Class Accuracy: {g_loss[4] * 100:.2f}%")

            # Store metrics history
            d_metrics_history.append(avg_epoch_d_loss)
            g_metrics_history.append(avg_epoch_g_loss)

            # Calculate current d_g_ratio based on performance
            # If discriminator loss is much lower than generator loss, consider reducing d_to_g_ratio
            # If discriminator loss is much higher, consider increasing it
            d_g_loss_ratio = avg_epoch_d_loss / avg_epoch_g_loss

            # Adaptive ratio adjustment (optional)
            if epoch > 0 and epoch % 5 == 0:  # Adjust every 5 epochs
                if d_g_loss_ratio < 0.5:  # Discriminator getting too good
                    d_to_g_ratio = max(1, d_to_g_ratio - 1)
                    self.logger.info(f"Adjusting d_to_g_ratio down to {d_to_g_ratio}:1")
                elif d_g_loss_ratio > 2.0:  # Discriminator struggling
                    d_to_g_ratio += 1
                    self.logger.info(f"Adjusting d_to_g_ratio up to {d_to_g_ratio}:1")

            # --------------------------
            # Validation every epoch
            # --------------------------
            self.logger.info(f"=== Epoch {epoch} Validation ===")
            # -- GAN Validation --#
            d_val_loss, d_val_metrics = self.validation_disc()
            g_val_loss, g_val_metrics = self.validation_gen()

            # -- Probabilistic Fusion Validation -- #
            self.logger.info("=== Probabilistic Fusion Validation on Real Data ===")
            fusion_results, fusion_metrics = self.validate_with_probabilistic_fusion(self.x_val, self.y_val)
            self.logger.info(f"Probabilistic Fusion Accuracy: {fusion_metrics['accuracy'] * 100:.2f}%")
            self.logger.info(f"Predicted Class Distribution: {fusion_metrics['predicted_class_distribution']}")

            # -- NIDS Validation -- #
            nids_val_metrics = None
            if self.nids is not None:
                nids_custom_loss, nids_val_metrics = self.validation_NIDS()
                self.logger.info(f"Validation NIDS Custom Loss: {nids_custom_loss:.4f}")

            # -- Log the metrics for epoch -- #
            self.log_epoch_metrics(epoch, d_val_metrics, g_val_metrics, nids_val_metrics, fusion_metrics)

        # Return the training history for analysis
        return {
            "discriminator_loss": d_metrics_history,
            "generator_loss": g_metrics_history
        }

        # -- Loss Calculation -- #

    def calculate_weighted_loss(self, d_loss_benign, d_loss_attack, d_loss_fake,
                                attack_weight=0.7, benign_weight=0.3,
                                validity_weight=0.4, class_weight=0.6):
        """
        Calculate weighted discriminator loss combining benign, attack, and fake samples.

        Parameters:
        -----------
        d_loss_benign : list
            Loss components for benign samples [total, validity, class, validity_acc, class_acc]
        d_loss_attack : list
            Loss components for attack samples [total, validity, class, validity_acc, class_acc]
        d_loss_fake : list
            Loss components for fake samples [total, validity, class, validity_acc, class_acc]
        attack_weight : float, optional
            Weight to apply to attack samples, default 0.7
        benign_weight : float, optional
            Weight to apply to benign samples, default 0.3
        validity_weight : float, optional
            Weight to apply to validity task, default 0.4
        class_weight : float, optional
            Weight to apply to classification task, default 0.6

        Returns:
        --------
        tuple
            (d_loss, d_metrics) where d_loss is the final weighted loss and
            d_metrics is a dictionary of loss components for logging
        """
        # Unpack loss components
        # Benign samples
        d_loss_benign_validity = d_loss_benign[1]  # Validity loss for benign samples
        d_loss_benign_class = d_loss_benign[2]  # Class loss for benign samples
        d_benign_valid_acc = d_loss_benign[3]  # Validity accuracy for benign
        d_benign_class_acc = d_loss_benign[4]  # Class accuracy for benign

        # Attack samples
        d_loss_attack_validity = d_loss_attack[1]  # Validity loss for attack samples
        d_loss_attack_class = d_loss_attack[2]  # Class loss for attack samples
        d_attack_valid_acc = d_loss_attack[3]  # Validity accuracy for attack
        d_attack_class_acc = d_loss_attack[4]  # Class accuracy for attack

        # Fake samples
        d_loss_fake_validity = d_loss_fake[1]  # Validity loss for fake samples
        d_loss_fake_class = d_loss_fake[2]  # Class loss for fake samples
        d_fake_valid_acc = d_loss_fake[3]  # Validity accuracy for fake
        d_fake_class_acc = d_loss_fake[4]  # Class accuracy for fake

        # Calculate weighted validity loss
        d_loss_validity_real = (benign_weight * d_loss_benign_validity) + (attack_weight * d_loss_attack_validity)
        d_loss_validity = 0.5 * (d_loss_validity_real + d_loss_fake_validity)

        # Calculate weighted class loss
        d_loss_class_real = (benign_weight * d_loss_benign_class) + (attack_weight * d_loss_attack_class)
        d_loss_class = 0.5 * (d_loss_class_real + d_loss_fake_class)

        # Calculate combined loss with task weights
        d_loss = (validity_weight * d_loss_validity) + (class_weight * d_loss_class)

        # For logging/display, calculate total losses for each sample type
        d_loss_benign_total = benign_weight * (d_loss_benign[0])
        d_loss_attack_total = attack_weight * (d_loss_attack[0])
        d_loss_fake_total = 0.5 * (d_loss_fake[0])
        d_loss_total = d_loss_benign_total + d_loss_attack_total + d_loss_fake_total

        # Calculate weighted accuracies
        d_valid_acc_real = (benign_weight * d_benign_valid_acc + attack_weight * d_attack_valid_acc)
        d_class_acc_real = (benign_weight * d_benign_class_acc + attack_weight * d_attack_class_acc)

        # Create metrics dictionary for logging
        d_metrics = {
            "Total Loss": f"{d_loss_total:.4f}",
            "Benign Loss": f"{d_loss_benign[0]:.4f}",
            "Attack Loss": f"{d_loss_attack[0]:.4f}",
            "Fake Loss": f"{d_loss_fake[0]:.4f}",
            "Validity Loss": f"{d_loss_validity:.4f}",
            "Class Loss": f"{d_loss_class:.4f}",
            "Benign Validity Acc": f"{d_benign_valid_acc * 100:.2f}%",
            "Attack Validity Acc": f"{d_attack_valid_acc * 100:.2f}%",
            "Fake Validity Acc": f"{d_fake_valid_acc * 100:.2f}%",
            "Benign Class Acc": f"{d_benign_class_acc * 100:.2f}%",
            "Attack Class Acc": f"{d_attack_class_acc * 100:.2f}%",
            "Fake Class Acc": f"{d_fake_class_acc * 100:.2f}%"
        }

        return d_loss, d_metrics

    def calculate_jsd_loss(self, real_outputs, fake_outputs):
        """
        Calculate Jensen-Shannon Divergence Loss between real and fake sample distributions.
        This can be used as an alternative or supplement to binary cross entropy for
        measuring discriminator performance.

        Parameters:
        -----------
        real_outputs : tensor
            Discriminator outputs for real samples
        fake_outputs : tensor
            Discriminator outputs for fake samples

        Returns:
        --------
        float : JSD loss value
        """
        # Average probabilities
        p_real = tf.reduce_mean(real_outputs, axis=0)
        p_fake = tf.reduce_mean(fake_outputs, axis=0)

        # Calculate mixtures
        p_mixture = 0.5 * (p_real + p_fake)

        # Calculate JS divergence
        kl_real_mix = tf.reduce_sum(p_real * tf.math.log(p_real / p_mixture + 1e-10))
        kl_fake_mix = tf.reduce_sum(p_fake * tf.math.log(p_fake / p_mixture + 1e-10))

        # JS divergence
        jsd = 0.5 * (kl_real_mix + kl_fake_mix)

        return jsd

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

    # -- Probabilistic Fusion Methods -- #
    def probabilistic_fusion(self, input_data):
        """
        Apply probabilistic fusion to combine validity and class predictions.
        Returns combined probabilities for all four possible outcomes.
        """
        # Get discriminator predictions
        validity_scores, class_predictions = self.discriminator.predict(input_data)

        total_samples = len(input_data)
        results = []

        for i in range(total_samples):
            # Validity probabilities: P(valid) and P(invalid)
            p_valid = validity_scores[i][0]  # Probability of being valid/real
            p_invalid = 1 - p_valid  # Probability of being invalid/fake

            # Class probabilities: 2 classes (benign=0, attack=1)
            p_benign = class_predictions[i][0]  # Probability of being benign
            p_attack = class_predictions[i][1]  # Probability of being attack

            # Calculate joint probabilities for all combinations
            p_valid_benign = p_valid * p_benign
            p_valid_attack = p_valid * p_attack
            p_invalid_benign = p_invalid * p_benign
            p_invalid_attack = p_invalid * p_attack

            # Store all probabilities in a dictionary
            probabilities = {
                "valid_benign": p_valid_benign,
                "valid_attack": p_valid_attack,
                "invalid_benign": p_invalid_benign,
                "invalid_attack": p_invalid_attack
            }

            # Find the most likely outcome
            most_likely = max(probabilities, key=probabilities.get)

            # For analysis, add the actual probabilities alongside the classification
            result = {
                "classification": most_likely,
                "probabilities": probabilities
            }

            results.append(result)

        return results

    def validate_with_probabilistic_fusion(self, validation_data, validation_labels=None):
        """
        Evaluate model using probabilistic fusion and calculate metrics if labels are available.
        """
        fusion_results = self.probabilistic_fusion(validation_data)

        # Extract classifications
        classifications = [result["classification"] for result in fusion_results]

        # Count occurrences of each class
        predicted_class_distribution = Counter(classifications)
        self.logger.info(f"Predicted Class Distribution: {dict(predicted_class_distribution)}")

        # If we have ground truth labels, calculate accuracy
        if validation_labels is not None:
            correct_predictions = 0
            correct_classifications = []
            true_classifications = []

            for i, result in enumerate(fusion_results):
                # Get the true label (assuming 0=benign, 1=attack)
                if isinstance(validation_labels, np.ndarray) and validation_labels.ndim > 1:
                    true_class_idx = np.argmax(validation_labels[i])
                else:
                    true_class_idx = validation_labels[i]

                true_class = "benign" if true_class_idx == 0 else "attack"

                # For validation data (which is real), expected validity is "valid"
                true_validity = "valid"  # Since validation data is real data

                # Construct the true combined label
                true_combined = f"{true_validity}_{true_class}"

                # Add to true classifications list
                true_classifications.append(true_combined)

                # Check if prediction matches
                if result["classification"] == true_combined:
                    correct_predictions += 1
                    correct_classifications.append(result["classification"])

            # Count distribution of correctly classified samples
            correct_class_distribution = Counter(correct_classifications)

            # Count distribution of true classes
            true_class_distribution = Counter(true_classifications)
            self.logger.info(f"True Class Distribution: {dict(true_class_distribution)}")

            accuracy = correct_predictions / len(validation_data)
            self.logger.info(f"Accuracy: {accuracy:.4f}")

            metrics = {
                "accuracy": accuracy,
                "total_samples": len(validation_data),
                "correct_predictions": correct_predictions,
                "predicted_class_distribution": dict(predicted_class_distribution),
                "correct_class_distribution": dict(correct_class_distribution),
                "true_class_distribution": dict(true_class_distribution)
            }

            return classifications, metrics

        return classifications, {"predicted_class_distribution": dict(predicted_class_distribution)}

    def analyze_fusion_results(self, fusion_results):
        """Analyze the distribution of probabilities from fusion results"""
        # Extract probabilities for each category
        valid_benign_probs = [r["probabilities"]["valid_benign"] for r in fusion_results]
        valid_attack_probs = [r["probabilities"]["valid_attack"] for r in fusion_results]
        invalid_benign_probs = [r["probabilities"]["invalid_benign"] for r in fusion_results]
        invalid_attack_probs = [r["probabilities"]["invalid_attack"] for r in fusion_results]

        # Calculate summary statistics
        categories = ["Valid Benign", "Valid Attack", "Invalid Benign", "Invalid Attack"]
        all_probs = [valid_benign_probs, valid_attack_probs, invalid_benign_probs, invalid_attack_probs]

        for cat, probs in zip(categories, all_probs):
            self.logger.info(
                f"{cat}: Mean={np.mean(probs):.4f}, Median={np.median(probs):.4f}, Max={np.max(probs):.4f}")

        # You could add additional visualizations or analysis here

    # -- Validate -- #
    def validation_disc(self):
        """
        Evaluate the discriminator on the validation set.
        Separates benign and attack samples for more detailed metrics.
        """
        # --- Prepare validation data ---
        val_valid_labels = np.ones((len(self.x_val), 1))

        # Ensure y_val is one-hot encoded if needed
        if self.y_val.ndim == 1 or self.y_val.shape[1] != self.num_classes:
            y_val_onehot = tf.one_hot(self.y_val, depth=self.num_classes)
        else:
            y_val_onehot = self.y_val

        # --- Separate benign and attack samples ---
        if self.y_val.ndim > 1:
            val_labels_idx = tf.argmax(self.y_val, axis=1)
        else:
            val_labels_idx = self.y_val

        benign_indices = tf.where(tf.equal(val_labels_idx, 0))
        attack_indices = tf.where(tf.equal(val_labels_idx, 1))

        # Create benign and attack validation sets
        x_val_benign = tf.gather(self.x_val, benign_indices[:, 0])
        y_val_benign_onehot = tf.gather(y_val_onehot, benign_indices[:, 0])
        x_val_attack = tf.gather(self.x_val, attack_indices[:, 0])
        y_val_attack_onehot = tf.gather(y_val_onehot, attack_indices[:, 0])

        # Fix shape issues
        if len(x_val_benign.shape) > 2:
            x_val_benign = tf.reshape(x_val_benign, (x_val_benign.shape[0], -1))
        if len(x_val_attack.shape) > 2:
            x_val_attack = tf.reshape(x_val_attack, (x_val_attack.shape[0], -1))

        # --- Generate fake data ---
        noise = tf.random.normal((len(self.x_val), self.latent_dim))
        fake_labels = tf.random.uniform(
            (len(self.x_val),), minval=0, maxval=self.num_classes, dtype=tf.int32
        )
        fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)
        fake_valid_labels = np.zeros((len(self.x_val), 1))
        generated_data = self.generator.predict([noise, fake_labels])

        # --- Evaluate on each data type ---
        # Benign evaluation
        benign_valid_labels = np.ones((len(x_val_benign), 1))
        d_loss_benign = self.discriminator.evaluate(
            x_val_benign, [benign_valid_labels, y_val_benign_onehot], verbose=0
        )

        # Attack evaluation
        attack_valid_labels = np.ones((len(x_val_attack), 1))
        d_loss_attack = self.discriminator.evaluate(
            x_val_attack, [attack_valid_labels, y_val_attack_onehot], verbose=0
        )

        # Fake data evaluation
        d_loss_fake = self.discriminator.evaluate(
            generated_data, [fake_valid_labels, fake_labels_onehot], verbose=0
        )

        # --- Apply weighted loss calculation ---
        # Log distribution
        benign_ratio = len(x_val_benign) / (len(x_val_benign) + len(x_val_attack))
        attack_ratio = len(x_val_attack) / (len(x_val_benign) + len(x_val_attack))
        self.logger.info(f"Validation data distribution: Benign {benign_ratio:.2f}, Attack {attack_ratio:.2f}")

        # Use the same weighted loss function as training
        total_loss, metrics = self.calculate_weighted_loss(
            d_loss_benign,
            d_loss_attack,
            d_loss_fake,
            attack_weight=0.7,
            benign_weight=0.3,
            validity_weight=0.4,
            class_weight=0.6
        )

        # Additional logging that might be specific to validation
        self.logger.info("Validation Discriminator Evaluation:")
        self.logger.info(f"Weighted Total Loss: {total_loss:.4f}")

        return total_loss, metrics

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

        g_loss = self.ACGAN.evaluate(
            [noise, sampled_labels],
            [valid_labels, sampled_labels_onehot],
            verbose=0
        )

        # Log detailed metrics
        self.logger.info("Validation Generator (AC-GAN) Evaluation:")
        self.logger.info(
            f"Total Loss: {g_loss[0]:.4f}, Validity Loss: {g_loss[1]:.4f}, Class Loss: {g_loss[2]:.4f}")
        self.logger.info(
            f"Validity Binary Accuracy: {g_loss[3] * 100:.2f}%")
        self.logger.info(
            f"Class Categorical Accuracy: {g_loss[4] * 100:.2f}%")

        # Create a metrics dictionary with the full set of metrics
        g_metrics = {
            "Total Loss": f"{g_loss[0]:.4f}",
            "Validity Loss": f"{g_loss[1]:.4f}",
            "Class Loss": f"{g_loss[2]:.4f}",
            "Validity Binary Accuracy": f"{g_loss[3] * 100:.2f}%",
            "Class Categorical Accuracy": f"{g_loss[4] * 100:.2f}%"
        }
        return g_loss[0], g_metrics

    def validation_NIDS(self):
        """
        Evaluate the NIDS model on validation data augmented with generated fake samples.
        Real data is labeled as 1 (benign) and fake/generated data as 0 (attack).
        Prints detailed metrics including a classification report and returns the custom
        NIDS loss along with a metrics dictionary.
        """
        if self.nids is None:
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

        self.logger.info("Validation NIDS Evaluation with Augmented Data:")
        self.logger.info(f"Custom NIDS Loss (Real vs Fake): {custom_nids_loss:.4f}")
        self.logger.info(f"Overall NIDS Loss: {nids_eval[0]:.4f}, Accuracy: {nids_eval[1]:.4f}, "
                         f"Precision: {nids_eval[2]:.4f}, Recall: {nids_eval[3]:.4f}, "
                         f"AUC: {nids_eval[4]:.4f}, LogCosh: {nids_eval[5]:.4f}")
        self.logger.info("Classification Report:")
        self.logger.info(class_report)
        self.logger.info(f"F1 Score: {f1:.4f}")

        metrics = {
            "Custom NIDS Loss": f"{custom_nids_loss:.4f}",
            "Loss": f"{nids_eval[0]:.4f}",
            "Accuracy": f"{nids_eval[1]:.4f}",
            "Precision": f"{nids_eval[2]:.4f}",
            "Recall": f"{nids_eval[3]:.4f}",
            "AUC": f"{nids_eval[4]:.4f}",
            "LogCosh": f"{nids_eval[5]:.4f}",
            "F1 Score": f"{f1:.4f}"
        }
        return custom_nids_loss, metrics

    # -- Evaluate -- #
    def evaluate(self, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            X_test = self.x_test
            y_test = self.y_test

        # --------------------------
        # Test Discriminator with Class Separation
        # --------------------------
        self.logger.info("-- Evaluating Discriminator --")

        # Separate benign and attack test samples
        if y_test.ndim > 1:
            test_labels_idx = tf.argmax(y_test, axis=1)
        else:
            test_labels_idx = y_test

        benign_indices = tf.where(tf.equal(test_labels_idx, 0))
        attack_indices = tf.where(tf.equal(test_labels_idx, 1))

        # Create benign and attack test sets
        x_test_benign = tf.gather(X_test, benign_indices[:, 0]) if len(benign_indices) > 0 else tf.zeros(
            (0, X_test.shape[1]))
        y_test_benign = tf.gather(y_test, benign_indices[:, 0]) if len(benign_indices) > 0 else tf.zeros(
            (0, y_test.shape[1] if y_test.ndim > 1 else 0))
        x_test_attack = tf.gather(X_test, attack_indices[:, 0]) if len(attack_indices) > 0 else tf.zeros(
            (0, X_test.shape[1]))
        y_test_attack = tf.gather(y_test, attack_indices[:, 0]) if len(attack_indices) > 0 else tf.zeros(
            (0, y_test.shape[1] if y_test.ndim > 1 else 0))

        # Ensure correct one-hot encoding
        if y_test.ndim == 1 or y_test.shape[1] != self.num_classes:
            if len(benign_indices) > 0:
                y_test_benign_onehot = tf.one_hot(tf.cast(y_test_benign, tf.int32), depth=self.num_classes)
            else:
                y_test_benign_onehot = tf.zeros((0, self.num_classes))

            if len(attack_indices) > 0:
                y_test_attack_onehot = tf.one_hot(tf.cast(y_test_attack, tf.int32), depth=self.num_classes)
            else:
                y_test_attack_onehot = tf.zeros((0, self.num_classes))
        else:
            y_test_benign_onehot = y_test_benign
            y_test_attack_onehot = y_test_attack

        # Log distribution of test data
        benign_count = len(x_test_benign)
        attack_count = len(x_test_attack)
        total_count = benign_count + attack_count
        benign_ratio = benign_count / total_count if total_count > 0 else 0
        attack_ratio = attack_count / total_count if total_count > 0 else 0

        self.logger.info(f"Test Data Distribution: {benign_count} Benign ({benign_ratio:.2f}), "
                         f"{attack_count} Attack ({attack_ratio:.2f})")

        # Generate fake test data for evaluation
        fake_count = min(benign_count, attack_count) * 2  # Generate a balanced number of fake samples
        if fake_count > 0:
            noise = tf.random.normal((fake_count, self.latent_dim))
            fake_labels = tf.random.uniform((fake_count,), minval=0, maxval=self.num_classes, dtype=tf.int32)
            fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)
            fake_valid_labels = np.zeros((fake_count, 1))
            generated_data = self.generator.predict([noise, fake_labels])
        else:
            self.logger.warning("No test samples available to generate fake data")
            generated_data = tf.zeros((0, X_test.shape[1]))
            fake_labels_onehot = tf.zeros((0, self.num_classes))
            fake_valid_labels = np.zeros((0, 1))

        # Evaluate on each data type separately
        d_loss_benign = None
        d_loss_attack = None
        d_loss_fake = None

        if benign_count > 0:
            benign_valid_labels = np.ones((benign_count, 1))
            d_loss_benign = self.discriminator.evaluate(
                x_test_benign, [benign_valid_labels, y_test_benign_onehot], verbose=0
            )
            self.logger.info(
                f"Benign Test -> Total Loss: {d_loss_benign[0]:.4f}, "
                f"Validity Loss: {d_loss_benign[1]:.4f}, "
                f"Class Loss: {d_loss_benign[2]:.4f}, "
                f"Validity Accuracy: {d_loss_benign[3] * 100:.2f}%, "
                f"Class Accuracy: {d_loss_benign[4] * 100:.2f}%"
            )

        if attack_count > 0:
            attack_valid_labels = np.ones((attack_count, 1))
            d_loss_attack = self.discriminator.evaluate(
                x_test_attack, [attack_valid_labels, y_test_attack_onehot], verbose=0
            )
            self.logger.info(
                f"Attack Test -> Total Loss: {d_loss_attack[0]:.4f}, "
                f"Validity Loss: {d_loss_attack[1]:.4f}, "
                f"Class Loss: {d_loss_attack[2]:.4f}, "
                f"Validity Accuracy: {d_loss_attack[3] * 100:.2f}%, "
                f"Class Accuracy: {d_loss_attack[4] * 100:.2f}%"
            )

        if fake_count > 0:
            d_loss_fake = self.discriminator.evaluate(
                generated_data, [fake_valid_labels, fake_labels_onehot], verbose=0
            )
            self.logger.info(
                f"Fake Test -> Total Loss: {d_loss_fake[0]:.4f}, "
                f"Validity Loss: {d_loss_fake[1]:.4f}, "
                f"Class Loss: {d_loss_fake[2]:.4f}, "
                f"Validity Accuracy: {d_loss_fake[3] * 100:.2f}%, "
                f"Class Accuracy: {d_loss_fake[4] * 100:.2f}%"
            )

        # Apply weighted loss calculation if all data types are available
        if d_loss_benign is not None and d_loss_attack is not None and d_loss_fake is not None:
            weighted_loss, d_eval_metrics = self.calculate_weighted_loss(
                d_loss_benign,
                d_loss_attack,
                d_loss_fake,
                attack_weight=0.7,
                benign_weight=0.3,
                validity_weight=0.4,
                class_weight=0.6
            )
            self.logger.info(f"Weighted Total Discriminator Loss: {weighted_loss:.4f}")
        else:
            # Fall back to simple average if not all data types are available
            available_losses = []
            if d_loss_benign is not None:
                available_losses.append(d_loss_benign[0])
            if d_loss_attack is not None:
                available_losses.append(d_loss_attack[0])
            if d_loss_fake is not None:
                available_losses.append(d_loss_fake[0])

            if available_losses:
                avg_loss = sum(available_losses) / len(available_losses)
                self.logger.info(f"Average Discriminator Loss: {avg_loss:.4f}")
            else:
                self.logger.warning("No loss data available for discriminator")
                avg_loss = 0.0

            # Create basic metrics dictionary
            d_eval_metrics = {
                "Loss": f"{avg_loss:.4f}",
                "Note": "Limited metrics available due to insufficient test data",
            }
            if d_loss_benign is not None:
                d_eval_metrics.update({
                    "Benign Total Loss": f"{d_loss_benign[0]:.4f}",
                    "Benign Validity Acc": f"{d_loss_benign[3] * 100:.2f}%",
                    "Benign Class Acc": f"{d_loss_benign[4] * 100:.2f}%"
                })
            if d_loss_attack is not None:
                d_eval_metrics.update({
                    "Attack Total Loss": f"{d_loss_attack[0]:.4f}",
                    "Attack Validity Acc": f"{d_loss_attack[3] * 100:.2f}%",
                    "Attack Class Acc": f"{d_loss_attack[4] * 100:.2f}%"
                })
            if d_loss_fake is not None:
                d_eval_metrics.update({
                    "Fake Total Loss": f"{d_loss_fake[0]:.4f}",
                    "Fake Validity Acc": f"{d_loss_fake[3] * 100:.2f}%",
                    "Fake Class Acc": f"{d_loss_fake[4] * 100:.2f}%"
                })

        # --------------------------
        # Test Generator (ACGAN)
        # --------------------------
        self.logger.info("-- Evaluating Generator --")

        # get the noise samples
        noise = tf.random.normal((len(y_test), self.latent_dim))
        sampled_labels = tf.random.uniform((len(y_test),), minval=0, maxval=self.num_classes, dtype=tf.int32)

        # run the model
        g_loss = self.ACGAN.evaluate([noise, sampled_labels],
                                     [tf.ones((len(y_test), 1)),
                                      tf.one_hot(sampled_labels, depth=self.num_classes)],
                                     verbose=0)

        # Using the updated ordering for ACGAN:
        g_loss_total = g_loss[0]
        g_loss_validity = g_loss[1]
        g_loss_class = g_loss[2]
        g_validity_bin_acc = g_loss[3]
        g_class_cat_acc = g_loss[4]

        g_eval_metrics = {
            "Loss": f"{g_loss_total:.4f}",
            "Validity Loss": f"{g_loss_validity:.4f}",
            "Class Loss": f"{g_loss_class:.4f}",
            "Validity Binary Accuracy": f"{g_validity_bin_acc * 100:.2f}%",
            "Class Categorical Accuracy": f"{g_class_cat_acc * 100:.2f}%"
        }
        self.logger.info(
            f"Generator Total Loss: {g_loss_total:.4f} | Validity Loss: {g_loss_validity:.4f} | Class Loss: {g_loss_class:.4f}"
        )
        self.logger.info(
            f"Validity Binary Accuracy: {g_validity_bin_acc * 100:.2f}%"
        )
        self.logger.info(
            f"Class Categorical Accuracy: {g_class_cat_acc * 100:.2f}%"
        )

        # --------------------------
        # Test NIDS
        # --------------------------
        nids_eval_metrics = None
        if self.nids is not None:
            self.logger.info("-- Evaluating NIDS --")
            # Prepare real test data (labeled as benign, 1)
            X_real = X_test
            y_real = np.ones((len(X_test),), dtype="int32")

            # Generate fake test data (labeled as attack, 0)
            noise = tf.random.normal((len(X_test), self.latent_dim))
            fake_labels = tf.random.uniform((len(X_test),), minval=0, maxval=self.num_classes, dtype=tf.int32)
            generated_samples = self.generator.predict([noise, fake_labels])
            X_fake = generated_samples
            y_fake = np.zeros((len(X_test),), dtype="int32")

            # Compute custom NIDS loss on real and fake outputs
            real_output = self.nids.predict(X_real)
            fake_output = self.nids.predict(X_fake)
            custom_nids_loss = self.nids_loss(real_output, fake_output)

            # Combine real and fake data for evaluation
            X_combined = np.vstack([X_real, X_fake])
            y_combined = np.hstack([y_real, y_fake])
            nids_eval_results = self.nids.evaluate(X_combined, y_combined, verbose=0)
            # Expected order: [loss, accuracy, precision, recall, auc, logcosh]

            # Compute additional metrics
            y_pred_probs = self.nids.predict(X_combined)
            y_pred = (y_pred_probs > 0.5).astype("int32")
            f1 = f1_score(y_combined, y_pred)
            class_report = classification_report(
                y_combined, y_pred, target_names=["Attack (Fake)", "Benign (Real)"]
            )

            nids_eval_metrics = {
                "Custom NIDS Loss": f"{custom_nids_loss:.4f}",
                "Loss": f"{nids_eval_results[0]:.4f}",
                "Accuracy": f"{nids_eval_results[1]:.4f}",
                "Precision": f"{nids_eval_results[2]:.4f}",
                "Recall": f"{nids_eval_results[3]:.4f}",
                "AUC": f"{nids_eval_results[4]:.4f}",
                "LogCosh": f"{nids_eval_results[5]:.4f}",
                "F1 Score": f"{f1:.4f}"
            }
            self.logger.info(f"NIDS Custom Loss: {custom_nids_loss:.4f}")
            self.logger.info(
                f"NIDS Evaluation -> Loss: {nids_eval_results[0]:.4f}, Accuracy: {nids_eval_results[1]:.4f}, "
                f"Precision: {nids_eval_results[2]:.4f}, Recall: {nids_eval_results[3]:.4f}, "
                f"AUC: {nids_eval_results[4]:.4f}, LogCosh: {nids_eval_results[5]:.4f}")
            self.logger.info("NIDS Classification Report:")
            self.logger.info(class_report)

        # --------------------------
        # Probabilistic Fusion Evaluation
        # --------------------------
        self.logger.info("-- Evaluating Probabilistic Fusion --")
        fusion_results, fusion_metrics = self.validate_with_probabilistic_fusion(X_test, y_test)
        self.logger.info(f"Probabilistic Fusion Accuracy: {fusion_metrics['accuracy'] * 100:.2f}%")
        self.logger.info(f"Predicted Class Distribution: {fusion_metrics['predicted_class_distribution']}")

        # Log the overall evaluation metrics using our logging function
        self.log_evaluation_metrics(d_eval_metrics, g_eval_metrics, nids_eval_metrics, fusion_metrics)

    # -- Saving Models -- #
    def save(self, save_name):
        # Save each submodel separately
        self.generator.save(f"../pretrainedModels/generator_local_ACGAN_{save_name}.h5")
        self.discriminator.save(f"../pretrainedModels/discriminator_local_ACGAN_{save_name}.h5")
