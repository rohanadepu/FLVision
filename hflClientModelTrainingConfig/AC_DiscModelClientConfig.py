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
from collections import Counter
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
                 log_file="training.log", use_class_labels=True):
        # -- models
        self.discriminator = discriminator

        # -- I/O Specs for models
        self.batch_size = BATCH_SIZE
        self.num_classes = num_classes
        self.input_dim = input_dim
        self.use_class_labels = use_class_labels  # Whether to use class labels in training

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

        # -- Model Compilations based on whether we use class labels
        if self.use_class_labels:
            self.discriminator.compile(
                loss={'validity': 'binary_crossentropy', 'class': 'categorical_crossentropy'},
                optimizer=self.disc_optimizer,
                metrics={
                    'validity': ['binary_accuracy'],
                    'class': ['categorical_accuracy']
                }
            )
        else:
            # If not using class labels, we only care about validity output
            self.discriminator.compile(
                loss={'validity': 'binary_crossentropy'},
                optimizer=self.disc_optimizer,
                metrics={'validity': ['binary_accuracy']}
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

        # -- make sure discriminator is trainable for individual training -- #
        self.discriminator.trainable = True
        # Ensure all layers within discriminator are trainable
        for layer in self.discriminator.layers:
            layer.trainable = True

        # -- Re-compile discriminator with trainable weights -- #
        self.discriminator.compile(
            loss={'validity': 'binary_crossentropy', 'class': 'categorical_crossentropy'},
            optimizer=self.disc_optimizer,
            metrics={
                'validity': [ 'binary_accuracy'],
                'class': ['categorical_accuracy']
            }
        )

        # load data as a new variable
        X_train = self.x_train
        y_train = self.y_train

        # Log model settings at the start
        self.log_model_settings()

        # -- Apply label smoothing -- #
        # Create smoothed labels for discriminator training
        valid_smoothing_factor = 0.15
        valid_smooth = tf.ones((self.batch_size, 1)) * (1 - valid_smoothing_factor)

        self.logger.info(f"Using valid label smoothing with factor: {valid_smoothing_factor}")

        for epoch in range(self.epochs):
            print(f'\n=== Epoch {epoch}/{self.epochs} ===\n')
            self.logger.info(f'=== Epoch {epoch}/{self.epochs} ===')

            # --------------------------
            # Train Discriminator (REAL DATA ONLY)
            # --------------------------

            # Calculate number of batches per epoch
            total_batches = min(self.steps_per_epoch, len(X_train) // self.batch_size)

            # Track metrics across all batches
            epoch_loss = 0
            epoch_validity_loss = 0
            epoch_class_loss = 0
            epoch_validity_acc = 0
            epoch_class_acc = 0

            for batch in range(total_batches):
                # Sample a batch of real data
                idx = tf.random.shuffle(tf.range(len(X_train)))[:self.batch_size]
                real_data = tf.gather(X_train, idx)
                real_labels = tf.gather(y_train, idx)

                # Prepare the training data based on whether we use class labels
                if self.use_class_labels:
                    # Ensure labels are one-hot encoded
                    if len(real_labels.shape) == 1:
                        real_labels_onehot = tf.one_hot(tf.cast(real_labels, tf.int32), depth=self.num_classes)
                    else:
                        real_labels_onehot = real_labels

                    # Train discriminator on real data with both validity and class labels
                    d_loss_real = self.discriminator.train_on_batch(real_data, [valid_smooth, real_labels_onehot])
                else:
                    # Train discriminator on real data with only validity labels
                    d_loss_real = self.discriminator.train_on_batch(real_data, valid_smooth)

                # Accumulate metrics based on whether we use class labels
                if self.use_class_labels:
                    # With class labels, we have more metrics to track
                    epoch_loss += d_loss_real[0]
                    epoch_validity_loss += d_loss_real[1]
                    epoch_class_loss += d_loss_real[2]
                    epoch_validity_acc += d_loss_real[3]
                    epoch_class_acc += d_loss_real[4]  # Index 4 for categorical accuracy
                else:
                    # Without class labels, we have fewer metrics
                    epoch_loss += d_loss_real[0]
                    epoch_validity_loss += d_loss_real[0]  # Total loss is validity loss
                    epoch_validity_acc += d_loss_real[1]  # Index 1 for binary accuracy

            # Calculate average metrics for the epoch
            batch_count = total_batches
            avg_loss = epoch_loss / batch_count
            avg_validity_loss = epoch_validity_loss / batch_count
            avg_validity_acc = epoch_validity_acc / batch_count

            # Log training metrics
            self.logger.info("Training Discriminator (REAL DATA ONLY)")

            if self.use_class_labels:
                avg_class_loss = epoch_class_loss / batch_count
                avg_class_acc = epoch_class_acc / batch_count

                self.logger.info(
                    f"Discriminator Loss: {avg_loss:.4f} | Validity Loss: {avg_validity_loss:.4f} | Class Loss: {avg_class_loss:.4f}")
                self.logger.info(
                    f"Validity Binary Accuracy: {avg_validity_acc * 100:.2f}%")
                self.logger.info(
                    f"Class Categorical Accuracy: {avg_class_acc * 100:.2f}%")

                # Collect discriminator metrics with class information
                d_metrics = {
                    "Total Loss": f"{avg_loss:.4f}",
                    "Validity Loss": f"{avg_validity_loss:.4f}",
                    "Class Loss": f"{avg_class_loss:.4f}",
                    "Validity Binary Accuracy": f"{avg_validity_acc * 100:.2f}%",
                    "Class Categorical Accuracy": f"{avg_class_acc * 100:.2f}%"
                }

                # Store last epoch metrics to return to server
                if epoch == self.epochs - 1:
                    epoch_metrics = {
                        "loss": float(avg_loss),
                        "validity_loss": float(avg_validity_loss),
                        "class_loss": float(avg_class_loss),
                        "validity_accuracy": float(avg_validity_acc),
                        "class_accuracy": float(avg_class_acc)
                    }
            else:
                self.logger.info(
                    f"Discriminator Loss: {avg_loss:.4f} (Validity Loss)")
                self.logger.info(
                    f"Validity Binary Accuracy: {avg_validity_acc * 100:.2f}%")

                # Collect discriminator metrics without class information
                d_metrics = {
                    "Total Loss": f"{avg_loss:.4f}",
                    "Validity Loss": f"{avg_validity_loss:.4f}",
                    "Validity Binary Accuracy": f"{avg_validity_acc * 100:.2f}%"
                }

                # Store last epoch metrics to return to server
                if epoch == self.epochs - 1:
                    epoch_metrics = {
                        "loss": float(avg_loss),
                        "validity_accuracy": float(avg_validity_acc)
                    }

            # --------------------------
            # Validation every 1 epochs
            # --------------------------
            if epoch % 1 == 0:
                self.logger.info(f"=== Epoch {epoch} Validation ===")

                d_val_loss, d_val_metrics = self.validation_disc()

                # -- Probabilistic Fusion Validation -- #
                self.logger.info("=== Probabilistic Fusion Validation on Real Data ===")
                fusion_results, fusion_metrics = self.validate_with_probabilistic_fusion(self.x_val, self.y_val)
                self.logger.info(f"Probabilistic Fusion Accuracy: {fusion_metrics['accuracy'] * 100:.2f}%")

                # Log distribution of classifications
                self.logger.info(f"Predicted Class Distribution: {fusion_metrics['predicted_class_distribution']}")
                self.logger.info(f"Correct Class Distribution: {fusion_metrics['correct_class_distribution']}")
                self.logger.info(f"True Class Distribution: {fusion_metrics['true_class_distribution']}")

                # Log the metrics for this epoch
                self.log_epoch_metrics(epoch, d_val_metrics)
                self.logger.info(
                    f"Epoch {epoch}: D Loss: {avg_loss:.4f}, D Validity Acc: {avg_validity_acc * 100:.2f}%")

        return self.discriminator.get_weights(), len(self.x_train), {}

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
        Evaluate the discriminator on the validation set using real data and generated fake data.
        Returns the average total loss and a metrics dictionary.
        """
        # --- Evaluate on real validation data ---
        val_valid_labels = np.ones((len(self.x_val), 1))

        # Different evaluation approach based on whether we use class labels
        if self.use_class_labels:
            # Ensure y_val is one-hot encoded if needed
            if self.y_val.ndim == 1 or self.y_val.shape[1] != self.num_classes:
                y_val_onehot = tf.one_hot(self.y_val, depth=self.num_classes)
            else:
                y_val_onehot = self.y_val

            d_loss_real = self.discriminator.evaluate(
                self.x_val, [val_valid_labels, y_val_onehot], verbose=0
            )
        else:
            # Only validate with validity labels
            d_loss_real = self.discriminator.evaluate(
                self.x_val, val_valid_labels, verbose=0
            )

        # Evaluate on generated (fake) data
        # For federated learning with only real data, we don't generate fake samples.
        # But if desired, you could simulate fake data here.
        # For now, we'll only evaluate on real data.
        avg_total_loss = d_loss_real[0]

        self.logger.info("Validation Discriminator Evaluation (Real Data Only):")
        self.logger.info(
            f"Real Data -> Total Loss: {d_loss_real[0]:.4f}, Validity Loss: {d_loss_real[1]:.4f}, "
            f"Class Loss: {d_loss_real[2]:.4f}, Validity Accuracy: {d_loss_real[3] * 100:.2f}%, "
            f"Class Accuracy: {d_loss_real[4] * 100:.2f}%"
        )

        metrics = {
            "Real Total Loss": f"{d_loss_real[0]:.4f}",
            "Real Validity Loss": f"{d_loss_real[1]:.4f}",
            "Real Class Loss": f"{d_loss_real[2]:.4f}",
            "Real Validity Accuracy": f"{d_loss_real[3] * 100:.2f}%",
            "Real Class Accuracy": f"{d_loss_real[4] * 100:.2f}%"
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
        d_validity_bin_acc = results[3]
        d_class_cat_acc = results[4]

        d_eval_metrics = {
            "Loss": f"{d_loss_total:.4f}",
            "Validity Loss": f"{d_loss_validity:.4f}",
            "Class Loss": f"{d_loss_class:.4f}",
            "Validity Binary Accuracy": f"{d_validity_bin_acc * 100:.2f}%",
            "Class Categorical Accuracy": f"{d_class_cat_acc * 100:.2f}%"
        }
        self.logger.info(
            f"Discriminator Total Loss: {d_loss_total:.4f} | Validity Loss: {d_loss_validity:.4f} | Class Loss: {d_loss_class:.4f}"
        )
        self.logger.info(
            f"Validity Binary Accuracy: {d_validity_bin_acc * 100:.2f}%"
        )
        self.logger.info(
            f"Class Categorical Accuracy: {d_class_cat_acc * 100:.2f}%"
        )

        # Log overall evaluation metrics
        self.log_evaluation_metrics(d_eval_metrics)  # Only discriminator metrics for federated config

        return float(results.numpy()), len(self.x_test), {}

    def save(self, save_name):
        # Save each submodel separately
        self.discriminator.save(f"../pretrainedModels/discriminator_fed_ACGAN_{save_name}.h5")
