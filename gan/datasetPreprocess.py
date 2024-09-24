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
#                                       Preprocessing & Assigning the dataset                                  #
################################################################################################################

def preprocess_dataset(dataset_used, ciciot_train_data=None, ciciot_test_data=None, all_attacks_train=None,
                       all_attacks_test=None, irrelevant_features_ciciot=None, relevant_features_iotbotnet=None):
    print("\nSelecting Features...")
    if dataset_used == "CICIOT":
        # Drop the irrelevant features (Feature selection)
        ciciot_train_data = ciciot_train_data.drop(columns=irrelevant_features_ciciot)
        ciciot_test_data = ciciot_test_data.drop(columns=irrelevant_features_ciciot)

        # Shuffle data
        ciciot_train_data = shuffle(ciciot_train_data, random_state=47)
        ciciot_test_data = shuffle(ciciot_test_data, random_state=47)

        train_data = ciciot_train_data
        test_data = ciciot_test_data

    elif dataset_used == "IOTBOTNET":
        # Select the relevant features in the dataset and labels
        all_attacks_train = all_attacks_train[relevant_features_iotbotnet + ['Label']]
        all_attacks_test = all_attacks_test[relevant_features_iotbotnet + ['Label']]

        # Shuffle data
        all_attacks_train = shuffle(all_attacks_train, random_state=47)
        all_attacks_test = shuffle(all_attacks_test, random_state=47)

        train_data = all_attacks_train
        test_data = all_attacks_test

    print("Features Selected...")

    # --- Encoding ---
    print("\nEncoding...")

    # Print instances before encoding and scaling
    unique_labels = train_data['Label' if dataset_used == "IOTBOTNET" else 'label'].unique()
    for label in unique_labels:
        print(f"First instance of {label}:")
        print(train_data[train_data['Label' if dataset_used == "IOTBOTNET" else 'label'] == label].iloc[0])

    # Print the amount of instances for each label
    class_counts = train_data['Label' if dataset_used == "IOTBOTNET" else 'label'].value_counts()
    print(class_counts)

    # Encoding
    label_encoder = LabelEncoder()
    train_data['Label' if dataset_used == "IOTBOTNET" else 'label'] = label_encoder.fit_transform(
        train_data['Label' if dataset_used == "IOTBOTNET" else 'label'])
    test_data['Label' if dataset_used == "IOTBOTNET" else 'label'] = label_encoder.transform(
        test_data['Label' if dataset_used == "IOTBOTNET" else 'label'])

    # showing the label mappings
    label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print("Label mappings:", label_mapping)

    print("Labels Encoded...")

    # --- Normalizing ---
    print("\nNormalizing...")

    # Normalizing
    scaler = MinMaxScaler(feature_range=(0, 1))
    relevant_num_cols = train_data.columns.difference(['Label' if dataset_used == "IOTBOTNET" else 'label'])

    scaler.fit(train_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet])
    train_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet] = scaler.transform(train_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet])
    test_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet] = scaler.transform(test_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet])

    print("Data Normalized...")

    # DEBUG DISPLAY
    print("\nTraining Data After Normalization:")
    print(train_data.head())
    print(train_data.shape)
    print("\nTest Data After Normalization:")
    print(test_data.head())
    print(test_data.shape)

    # --- Assigning and Splitting ---
    print("\nAssigning Data to Models...")

    # Train & Validation data
    X_data = train_data.drop(columns=['Label' if dataset_used == "IOTBOTNET" else 'label'])
    y_data = train_data['Label' if dataset_used == "IOTBOTNET" else 'label']

    # Split into Train & Validation data
    X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(X_data, y_data, test_size=0.2,
                                                                          random_state=47, stratify=y_data)
    # Test data
    X_test_data = test_data.drop(columns=['Label' if dataset_used == "IOTBOTNET" else 'label'])
    y_test_data = test_data['Label' if dataset_used == "IOTBOTNET" else 'label']

    print("Data Assigned...")
    return X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data
