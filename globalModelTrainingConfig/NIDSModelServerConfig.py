# Custom FedAvg strategy with server-side model training and saving
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
import tensorflow_privacy as tfp
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
from hflNIDSModelConfig import create_CICIOT_Model, create_IOTBOTNET_Model


class NIDSAdvGANStrategy(fl.server.strategy.FedAvg):
    def __init__(self, discriminator, generator, dataset_used, node, adversarialTrainingEnabled, earlyStopEnabled, DP_enabled,
                 X_train_data, y_train_data,X_test_data, y_test_data, X_val_data, y_val_data, l2_norm_clip,
                 noise_multiplier, num_microbatches,batch_size, epochs, steps_per_epoch, learning_rate, synth_portion, adv_portion,
                 metric_to_monitor_es, es_patience,restor_best_w, metric_to_monitor_l2lr, l2lr_patience,
                 save_best_only, metric_to_monitor_mc, checkpoint_mode, **kwargs):

        super().__init__(**kwargs)
        self.discriminator = discriminator
        self.generator = generator
        self.nids = None

        self.data_used = dataset_used
        self.node = node

        # flags
        self.adversarialTrainingEnabled = adversarialTrainingEnabled
        self.DP_enabled = DP_enabled
        self.earlyStopEnabled = earlyStopEnabled

        # data
        self.X_train_data = X_train_data
        self.y_train_data = y_train_data
        self.X_test_data = X_test_data
        self.y_test_data = y_test_data
        self.X_val_data = X_val_data
        self.y_val_data = y_val_data

        # hyperparams
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        # dp
        self.num_microbatches = num_microbatches
        self.l2_norm_clip = l2_norm_clip
        self.noise_multiplier = noise_multiplier
        # adversarial
        self.adv_portion = adv_portion

        # callback params
        # early stop
        self.metric_to_monitor_es = metric_to_monitor_es
        self.es_patience = es_patience
        self.restor_best_w = restor_best_w
        # lr schedule
        self.metric_to_monitor_l2lr = metric_to_monitor_l2lr
        self.l2lr_factor = l2lr_patience
        self.l2lr_patience = es_patience
        # model checkpoint
        self.save_best_only = save_best_only
        self.metric_to_monitor_mc = metric_to_monitor_mc
        self.checkpoint_mode = checkpoint_mode

        # counters
        self.roundCount = 0
        self.evaluateCount = 0

        self.callbackFunctions = []

        # init callback functions based on inputs

        if self.earlyStopEnabled:
            early_stopping = EarlyStopping(monitor=self.metric_to_monitor_es, patience=self.es_patience,
                                           restore_best_weights=self.restor_best_w)

            self.callbackFunctions.append(early_stopping)

        if self.lrSchedRedEnabled:
            lr_scheduler = ReduceLROnPlateau(monitor=self.metric_to_monitor_l2lr, factor=self.l2lr_factor,
                                             patience=self.l2lr_patience)

            self.callbackFunctions.append(lr_scheduler)

        if self.modelCheckpointEnabled:
            model_checkpoint = ModelCheckpoint(f'best_model_{self.model_name}.h5', save_best_only=self.save_best_only,
                                               monitor=self.metric_to_monitor_mc, mode=self.checkpoint_mode)

            # add to callback functions list being added during fitting
            self.callbackFunctions.append(model_checkpoint)

        # Create model
        if self.dataset_used == "IOTBOTNET":
            print("No pretrained discriminator provided. Creating a new model.")

            self.nids = create_IOTBOTNET_Model(self.input_dim, self.regularizationEnabled, self.l2_alpha)

        else:
            print("No pretrained discriminator provided. Creating a new mdoel.")

            self.nids = create_CICIOT_Model(self.input_dim, self.regularizationEnabled, self.DP_enabled, self.l2_alpha)

    def on_fit_end(self, server_round, aggregated_weights, failures):
        # increment round count
        self.roundCount += 1

        # debug print
        print("Round:", self.roundCount, "\n")

        # Record start time
        start_time = time.time()

        # set model weights
        self.nids.set_weights(aggregated_weights)

        # ---         Differential Privacy Engine Model Compile              --- #

        if self.DP_enabled:
            print("\nIncluding DP into optimizer...\n")

            # Making Custom Optimizer Component with Differential Privacy
            dp_optimizer = tfp.DPKerasAdamOptimizer(
                l2_norm_clip=self.l2_norm_clip,
                noise_multiplier=self.noise_multiplier,
                num_microbatches=self.num_microbatches,
                learning_rate=self.learning_rate
            )

            # compile model with custom dp optimizer
            self.nids.compile(optimizer=dp_optimizer,
                          loss=tf.keras.losses.binary_crossentropy,
                          metrics=['accuracy', Precision(), Recall(), AUC(), LogCosh()])

        # ---              Normal Model Compile                        --- #

        else:
            print("\nDefault optimizer...\n")

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

            self.nids.compile(optimizer=optimizer,
                              loss=tf.keras.losses.binary_crossentropy,
                              metrics=['accuracy', Precision(), Recall(), AUC(), LogCosh()]
                              )

        # generate new balanced data from generator to add to training dataset
        if self.generator is not None and self.discriminator is not None:
            print("\nGenerating balanced data using generator...\n")

            # Generate balanced examples using the generator
            generated_data_size = int(self.synth_portion * len(self.X_train_data))
            X_generated = self.generator.predict(
                np.random.normal(size=(generated_data_size, self.X_train_data.shape[1])))

            # Use the pre-trained discriminator to generate realistic labels for synthetic data
            y_generated_probs = self.discriminator.predict(X_generated)
            y_generated = (y_generated_probs > 0.5).astype(int)  # Assign label 1 if probability > 0.5, else label 0

            # Concatenate original training data with generated balanced data
            self.X_train_data = np.vstack((self.X_train_data, X_generated))
            self.y_train_data = np.vstack((self.y_train_data, y_generated))

        # Further train model on server-side data
        if self.X_train_data is not None and self.y_train_data is not None:
            print(f"Training aggregated model on server-side data for {self.epochs} epochs...")
            history = self.nids.fit(self.X_train_data, self.y_train_data,
                                    validation_data=(self.X_val_data, self.y_val_data),
                                    epochs=self.epochs, batch_size=self.batch_size,
                                    steps_per_epoch=self.steps_per_epoch,
                                    callbacks=self.callbackFunctions,
                                    verbose=1)

        # Cross-validate the NIDS model using the discriminator
        if self.discriminator is not None:
            print("\nCross-validating the NIDS model using the discriminator...\n")
            # Use the discriminator to evaluate the NIDS model on validation data
            y_val_pred_probs = self.discriminator.predict(self.X_val_data)
            y_val_pred_labels = (y_val_pred_probs > 0.5).astype(int)
            validation_accuracy = np.mean(y_val_pred_labels == self.y_val_data)
            print(f"Validation accuracy based on discriminator: {validation_accuracy:.2f}")

        # Save the fine-tuned model
        self.nids.save("federated_model_fine_tuned.h5")
        print(f"Model fine-tuned and saved after round {server_round}.")

        # Record end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Debugging: Print the shape of the loss
        loss_tensor = history.history['loss']
        val_loss_tensor = history.history['val_loss']
        # print(f"Loss tensor shape: {tf.shape(loss_tensor)}")
        # print(f"Validation Loss tensor shape: {tf.shape(val_loss_tensor)}")

        # Save metrics to file
        logName = self.trainingLog
        # logName = f'training_metrics_{dataset_used}_optimized_{l2_norm_clip}_{noise_multiplier}.txt'
        self.recordTraining(logName, history, elapsed_time, self.roundCount, val_loss_tensor)

        # Send updated weights back to clients
        return self.nids.get_weights(), {}

    #########################################################
    #    Metric Saving Functions                           #
    #########################################################

    def recordTraining(self, name, history, elapsed_time, roundCount, val_loss):
        with open(name, 'a') as f:
            f.write(f"Node|{self.node}| Round: {roundCount}\n")
            f.write(f"Training Time Elapsed: {elapsed_time} seconds\n")
            for epoch in range(self.epochs):
                f.write(f"Epoch {epoch + 1}/{self.epochs}\n")
                for metric, values in history.history.items():
                    # Debug: print the length of values list and the current epoch
                    print(f"Metric: {metric}, Values Length: {len(values)}, Epoch: {epoch}")
                    if epoch < len(values):
                        f.write(f"{metric}: {values[epoch]}\n")
                    else:
                        print(f"Skipping metric {metric} for epoch {epoch} due to out-of-range error.")
                if epoch < len(val_loss):
                    f.write(f"Validation Loss: {val_loss[epoch]}\n")
                else:
                    print(f"Skipping Validation Loss for epoch {epoch} due to out-of-range error.")
                f.write("\n")

    def recordEvaluation(self, name, elapsed_time, evaluateCount, loss, accuracy, precision, recall, auc, logcosh):
        with open(name, 'a') as f:
            f.write(f"Node|{self.node}| Round: {evaluateCount}\n")
            f.write(f"Evaluation Time Elapsed: {elapsed_time} seconds\n")
            f.write(f"Loss: {loss}\n")
            f.write(f"Precision: {precision}\n")
            f.write(f"Recall: {recall}\n")
            f.write(f"AUC: {auc}\n")
            f.write(f"LogCosh: {logcosh}\n")
            f.write("\n")
