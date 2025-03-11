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
#                                               FEDERATED DISCRIMINATOR CLIENT                              #
################################################################################################################

class ACDiscriminatorClient(fl.client.NumPyClient):
    def __init__(self, discriminator, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE,
                 num_classes, input_dim, epochs, steps_per_epoch, learning_rate,
                 log_file="training.log"):
        # -- models
        self.discriminator = discriminator

        # -- I/O Specs for models
        self.batch_size = BATCH_SIZE
        self.num_classes = num_classes
        self.input_dim = input_dim

        # -- training duration
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

        # -- Data
        self.x_train = x_train
        self.x_test = x_test
        self.x_val = x_val
        self.y_train = y_train
        self.y_test = y_test
        self.y_val = y_val

        # -- Setup Logging
        self.setup_logger(log_file)

        # -- Optimizers and Learning Rate Scheduling
        lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate, decay_steps=10000, decay_rate=0.98, staircase=True)
        self.disc_optimizer = Adam(learning_rate=lr_schedule_disc, beta_1=0.5, beta_2=0.999)

        # -- Model Compilation: only the discriminator is used.
        # Note: In federated learning, the loss weight for the validity branch is set to 0
        # so that only the class prediction is learned.
        self.discriminator.compile(
            optimizer=self.disc_optimizer,
            loss={'validity': 'binary_crossentropy', 'class': 'categorical_crossentropy'},
            loss_weights={'validity': 0.0, 'class': 1.0},
            metrics={
                'validity': ['accuracy', 'AUC'],
                'class': ['accuracy']
            }
        )

    def get_parameters(self, config):
        # Return the discriminator's weights
        return self.discriminator.get_weights()

    # -- Logging Functions -- #

    def setup_logger(self, log_file):
        """Set up a logger that records both to a file and to the console."""
        self.logger = logging.getLogger("ACDiscriminatorClient")
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

        # Avoid duplicate handlers
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log_model_settings(self):
        """Logs model summary and hyperparameters."""
        self.logger.info("=== Model Settings ===")
        self.logger.info("Discriminator Model Summary:")
        disc_summary = []
        self.discriminator.summary(print_fn=lambda x: disc_summary.append(x))
        for line in disc_summary:
            self.logger.info(line)

        self.logger.info("--- Hyperparameters ---")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Number of Classes: {self.num_classes}")
        self.logger.info(f"Input Dimension: {self.input_dim}")
        self.logger.info(f"Epochs: {self.epochs}")
        self.logger.info(f"Steps per Epoch: {self.steps_per_epoch}")
        self.logger.info(f"Learning Rate (Discriminator): {self.disc_optimizer.learning_rate}")
        self.logger.info("=" * 50)

    def log_epoch_metrics(self, epoch, d_metrics):
        """Logs a formatted summary of the metrics for the current epoch."""
        self.logger.info(f"=== Epoch {epoch} Metrics Summary ===")
        self.logger.info("Discriminator Metrics:")
        for key, value in d_metrics.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    def log_evaluation_metrics(self, d_eval):
        """Logs a formatted summary of evaluation metrics."""
        self.logger.info("=== Evaluation Metrics Summary ===")
        self.logger.info("Discriminator Evaluation:")
        for key, value in d_eval.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    # -- Train -- #
    def fit(self, parameters, config):
        self.discriminator.set_weights(parameters)
        X_train = self.x_train
        y_train = self.y_train

        # Log model settings at the start
        self.log_model_settings()

        valid = tf.ones((self.batch_size, 1))

        for epoch in range(self.epochs):
            print(f'\n=== Epoch {epoch}/{self.epochs} ===\n')
            self.logger.info(f'=== Epoch {epoch}/{self.epochs} ===')

            # Sample a batch of real data
            idx = tf.random.shuffle(tf.range(len(X_train)))[:self.batch_size]
            real_data = tf.gather(X_train, idx)
            real_labels = tf.gather(y_train, idx)

            # Ensure one-hot encoding
            if len(real_labels.shape) == 1:
                real_labels_onehot = tf.one_hot(tf.cast(real_labels, tf.int32), depth=self.num_classes)
            else:
                real_labels_onehot = real_labels

            # Train the discriminator on real data only
            d_loss = self.discriminator.train_on_batch(real_data, [valid, real_labels_onehot])

            # Assuming the metrics ordering from:
            # ['loss', 'validity_loss', 'class_loss', 'validity_accuracy', 'validity_AUC', 'class_accuracy']
            # Note: Since loss_weights for validity is 0, only class loss is optimized.
            d_metrics = {
                "Total Loss": f"{d_loss[0]:.4f}",
                "Validity Loss": f"{d_loss[1]:.4f}",
                "Class Loss": f"{d_loss[2]:.4f}",
                "Validity Accuracy": f"{d_loss[3] * 100:.2f}%",
                "Validity AUC": f"{d_loss[4] * 100:.2f}%",
                "Class Accuracy": f"{d_loss[5] * 100:.2f}%"
            }

            self.logger.info("Training Discriminator")
            self.logger.info(
                f"Discriminator Total Loss: {d_loss[0]:.4f} | Validity Loss: {d_loss[1]:.4f} | Class Loss: {d_loss[2]:.4f}")
            self.logger.info(
                f"Validity Accuracy: {d_loss[3] * 100:.2f}% | Validity AUC: {d_loss[4] * 100:.2f}% | Class Accuracy: {d_loss[5] * 100:.2f}%")

            # Optionally, perform validation every epoch
            if epoch % 1 == 0:
                self.logger.info(f"=== Epoch {epoch} Validation ===")
                d_val_loss, d_val_metrics = self.validation_disc()
                self.log_epoch_metrics(epoch, d_val_metrics)
                self.logger.info(f"Epoch {epoch}: Discriminator Loss: {d_loss[0]:.4f}, Accuracy: {d_loss[5] * 100:.2f}%")

        return self.discriminator.get_weights(), len(self.x_train), {}

    # -- Validate -- #
    def validation_disc(self):
        """
        Evaluate the discriminator on the validation set using real data and generated fake data.
        Returns the average total loss and a metrics dictionary.
        """
        # Evaluate on real validation data
        val_valid_labels = np.ones((len(self.x_val), 1))
        if self.y_val.ndim == 1 or self.y_val.shape[1] != self.num_classes:
            y_val_onehot = tf.one_hot(self.y_val, depth=self.num_classes)
        else:
            y_val_onehot = self.y_val

        d_loss_real = self.discriminator.evaluate(
            self.x_val, [val_valid_labels, y_val_onehot], verbose=0
        )

        # Evaluate on generated (fake) data
        noise = tf.random.normal((len(self.x_val), 1))  # Not used in training; using only real data
        # For federated learning with only real data, we don't generate fake samples.
        # But if desired, you could simulate fake data here.
        # For now, we'll only evaluate on real data.
        avg_total_loss = d_loss_real[0]

        self.logger.info("Validation Discriminator Evaluation (Real Data Only):")
        self.logger.info(
            f"Real Data -> Total Loss: {d_loss_real[0]:.4f}, Validity Loss: {d_loss_real[1]:.4f}, "
            f"Class Loss: {d_loss_real[2]:.4f}, Validity Accuracy: {d_loss_real[3] * 100:.2f}%, "
            f"Validity AUC: {d_loss_real[4] * 100:.2f}%, Class Accuracy: {d_loss_real[5] * 100:.2f}%"
        )

        metrics = {
            "Real Total Loss": f"{d_loss_real[0]:.4f}",
            "Real Validity Loss": f"{d_loss_real[1]:.4f}",
            "Real Class Loss": f"{d_loss_real[2]:.4f}",
            "Real Validity Accuracy": f"{d_loss_real[3] * 100:.2f}%",
            "Real Validity AUC": f"{d_loss_real[4] * 100:.2f}%",
            "Real Class Accuracy": f"{d_loss_real[5] * 100:.2f}%"
        }
        return avg_total_loss, metrics

    # -- Evaluate -- #
    def evaluate(self, X_test=None, y_test=None):
        if X_test is None or y_test is None:
            X_test = self.x_test
            y_test = self.y_test

        self.logger.info("-- Evaluating Discriminator --")
        results = self.discriminator.evaluate(X_test, [tf.ones((len(y_test), 1)), y_test], verbose=0)
        d_loss_total = results[0]
        d_loss_validity = results[1]
        d_loss_class = results[2]
        d_validity_acc = results[3]
        d_validity_auc = results[4]
        d_class_acc = results[5]

        d_eval_metrics = {
            "Loss": f"{d_loss_total:.4f}",
            "Validity Loss": f"{d_loss_validity:.4f}",
            "Class Loss": f"{d_loss_class:.4f}",
            "Validity Accuracy": f"{d_validity_acc * 100:.2f}%",
            "Validity AUC": f"{d_validity_auc * 100:.2f}%",
            "Class Accuracy": f"{d_class_acc * 100:.2f}%"
        }
        self.logger.info(
            f"Discriminator Total Loss: {d_loss_total:.4f} | Validity Loss: {d_loss_validity:.4f} | Class Loss: {d_loss_class:.4f}"
        )
        self.logger.info(
            f"Validity Accuracy: {d_validity_acc * 100:.2f}% | Validity AUC: {d_validity_auc * 100:.2f}% | Class Accuracy: {d_class_acc * 100:.2f}%"
        )

        # Log overall evaluation metrics
        self.log_evaluation_metrics(d_eval_metrics)  # Only discriminator metrics for federated config

        return float(results.numpy()), len(self.x_test), {}

    def save(self, save_name):
        # Save each submodel separately
        self.discriminator.save(f"../pretrainedModels/discriminator_fed_ACGAN_{save_name}.h5")
