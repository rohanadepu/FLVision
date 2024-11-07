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

class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, server_data, server_labels, epochs=5, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.server_data = server_data
        self.server_labels = server_labels
        self.epochs = epochs
        self.batch_size = batch_size

    def on_fit_end(self, server_round, aggregated_weights, failures, input_dim, dataset_used, DP_enabled, regularizationEnabled, l2_alpha):
        # Create model and set aggregated weights
        if dataset_used == "IOTBOTNET":
            print("No pretrained discriminator provided. Creating a new model.")

            model = create_IOTBOTNET_Model(input_dim, regularizationEnabled, l2_alpha)

        else:
            print("No pretrained discriminator provided. Creating a new mdoel.")

            model = create_CICIOT_Model(input_dim, regularizationEnabled, DP_enabled, l2_alpha)

        model.set_weights(aggregated_weights)

        # Compile model for server-side training
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

        # Further train model on server-side data
        if self.server_data is not None and self.server_labels is not None:
            print(f"Training aggregated model on server-side data for {self.epochs} epochs...")
            model.fit(self.server_data, self.server_labels, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

        # Save the fine-tuned model
        model.save("federated_model_fine_tuned.h5")
        print(f"Model fine-tuned and saved after round {server_round}.")

        # Send updated weights back to clients
        return model.get_weights(), {}