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

# def underfit_preprocessing():

def preprocess_dataset(dataset_used, ciciot_train_data=None, ciciot_test_data=None, all_attacks_train=None,
                       all_attacks_test=None, irrelevant_features_ciciot=None, relevant_features_iotbotnet=None):
    # --- 1 Feature Selection ---#
    print("\nSelecting Features...")
    if dataset_used == "CICIOT":
        # Drop the irrelevant features (Feature selection)
        ciciot_train_data = ciciot_train_data.drop(columns=irrelevant_features_ciciot)
        ciciot_test_data = ciciot_test_data.drop(columns=irrelevant_features_ciciot)

        # Shuffle data
        ciciot_train_data = shuffle(ciciot_train_data, random_state=47)
        ciciot_test_data = shuffle(ciciot_test_data, random_state=47)

        # initiate model training and test data
        train_data = ciciot_train_data
        test_data = ciciot_test_data

    elif dataset_used == "IOTBOTNET":
        # Select the relevant features in the dataset and labels
        all_attacks_train = all_attacks_train[relevant_features_iotbotnet + ['Label']]
        all_attacks_test = all_attacks_test[relevant_features_iotbotnet + ['Label']]

        # Shuffle data
        all_attacks_train = shuffle(all_attacks_train, random_state=47)
        all_attacks_test = shuffle(all_attacks_test, random_state=47)

        # initiate model training and test data
        train_data = all_attacks_train
        test_data = all_attacks_test

    else:
        raise ValueError("Unsupported dataset type.")

    print("Features Selected...")

    # --- 2 Encoding ---#
    print("\nEncoding...")

    # Print instances before encoding and scaling
    unique_labels = train_data['Label' if dataset_used == "IOTBOTNET" else 'label'].unique()
    for label in unique_labels:
        print(f"First instance of {label}:")
        print(train_data[train_data['Label' if dataset_used == "IOTBOTNET" else 'label'] == label].iloc[0])

    # Print the amount of instances for each label
    class_counts = train_data['Label' if dataset_used == "IOTBOTNET" else 'label'].value_counts()
    print(class_counts)

    # initiate encoder
    label_encoder = LabelEncoder()

    # fit and encode training data
    train_data['Label' if dataset_used == "IOTBOTNET" else 'label'] = label_encoder.fit_transform(
        train_data['Label' if dataset_used == "IOTBOTNET" else 'label'])

    # encode test data
    test_data['Label' if dataset_used == "IOTBOTNET" else 'label'] = label_encoder.transform(
        test_data['Label' if dataset_used == "IOTBOTNET" else 'label'])

    # showing the label mappings
    label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print("Label mappings:", label_mapping)

    print("Labels Encoded...")

    # --- 3 Normalizing ---#
    print("\nNormalizing...")

    # DEBUG DISPLAY Before
    print("\nTraining Data Before Normalization:")
    print(train_data.head())
    print(train_data.shape)
    print("\nTest Data Before Normalization:")
    print(test_data.head())
    print(test_data.shape)

    # initiate scaler and colums to scale
    # review this part - doesnt matter since encoded data is 0-1 already
    scaler = MinMaxScaler(feature_range=(0, 1))
    relevant_num_cols = train_data.columns.difference(['Label' if dataset_used == "IOTBOTNET" else 'label'])

    # fit scaler
    scaler.fit(train_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet])

    # Normalize data
    train_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet] = scaler.transform(train_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet])
    test_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet] = scaler.transform(test_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet])

    print("\nData Normalized...")

    # DEBUG DISPLAY After
    print("\nTraining Data After Normalization:")
    print(train_data.head())
    print(train_data.shape)
    print("\nTest Data After Normalization:")
    print(test_data.head())
    print(test_data.shape)

    # --- 4 Assigning and Splitting ---#
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

    # Print dataset distributions
    print(f"\nTraining label distribution:")
    print(pd.Series(y_train_data).value_counts())
    print(f"\nValidation label distribution:")
    print(pd.Series(y_val_data).value_counts())
    print(f"\nTest label distribution:")
    print(pd.Series(y_test_data).value_counts())

    print("Data Assigned...")
    return X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data


# Time-series transformation using sliding window
def create_sliding_window(X, y, time_steps):
    X_windowed, y_windowed = [], []
    for i in range(len(X) - time_steps):
        X_windowed.append(X.iloc[i:i + time_steps].values)
        y_windowed.append(y.iloc[i + time_steps])
    return np.array(X_windowed), np.array(y_windowed)


def preprocess_timeseries_dataset(X, y, time_steps=10, test_size=0.2, val_size=0.1, random_state=42):
    """
    Preprocesses a dataset for time-series analysis, including train, validation, and test splits.

    Parameters:
    - X (pd.DataFrame): Features dataframe.
    - y (pd.Series or pd.DataFrame): Target dataframe or series.
    - time_steps (int): Number of time steps for sliding window transformation.
    - test_size (float): Proportion of the dataset to include in the test split.
    - val_size (float): Proportion of the training data to include in the validation split.
    - random_state (int): Random seed for reproducibility.

    Returns:
    - X_train, X_val, X_test (np.array): Preprocessed train, validation, and test features.
    - y_train, y_val, y_test (np.array): Preprocessed train, validation, and test labels.
    """
    # Check for missing values and impute (mean for numerical columns)
    if X.isnull().sum().sum() > 0:
        X.fillna(X.mean(), inplace=True)
    if y.isnull().sum() > 0:
        y.fillna(y.mode()[0], inplace=True)  # Replace with mode for categorical

    # Encode categorical target labels (if needed)
    if y.dtypes == 'object' or isinstance(y.iloc[0], str):
        le = LabelEncoder()
        y = le.fit_transform(y)

    # Normalize features
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X)
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)

    # Time-series transformation using sliding window
    X_ts, y_ts = create_sliding_window(X_normalized, pd.Series(y), time_steps)

    # Split into training + validation and test sets
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_ts, y_ts, test_size=test_size, random_state=random_state
    )

    # Split training + validation into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size, random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
