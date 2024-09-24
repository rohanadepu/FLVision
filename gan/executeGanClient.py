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

from ciciotDatasetLoad import loadCICIOT
from iotbotnetDatasetLoad import loadIOTBOTNET
from datasetPreprocess import preprocess_dataset
from modelConfig import create_model, GanClient

################################################################################################################
#                                                   Execute                                                   #
################################################################################################################
def main():
    print("\n ////////////////////////////// \n")
    print("Federated Learning Training Demo:", "\n")

    # Generate a static timestamp at the start of the script
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Argument Parsing --- #
    parser = argparse.ArgumentParser(description='Select dataset, model selection, and to enable DP respectively')
    parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET"], default="CICIOT",
                        help='Datasets to use: CICIOT, IOTBOTNET')

    parser.add_argument("--node", type=int, choices=[1, 2, 3, 4, 5, 6], default=1, help="Client node number 1-6")
    parser.add_argument("--fixedServer", type=int, choices=[1, 2, 3, 4], default=1, help="Fixed Server node number 1-4")

    parser.add_argument("--pData", type=str, choices=["LF33", "LF66", "FN33", "FN66", None], default=None,
                        help="Label Flip: LF33, LF66")

    parser.add_argument('--reg', action='store_true', help='Enable Regularization')  # tested

    parser.add_argument("--evalLog", type=str, default=f"evaluation_metrics_{timestamp}.txt",
                        help="Name of the evaluation log file")
    parser.add_argument("--trainLog", type=str, default=f"training_metrics_{timestamp}.txt",
                        help="Name of the training log file")

    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model")

    # init variables to handle arguments
    args = parser.parse_args()

    dataset_used = args.dataset
    fixedServer = args.fixedServer
    node = args.node
    poisonedDataType = args.pData
    regularizationEnabled = args.reg
    epochs = args.epochs


    # display selected arguments
    print("|MAIN CONFIG|", "\n")
    # main experiment config
    print("Selected Fixed Server:", fixedServer, "\n")
    print("Selected Node:", node, "\n")
    print("Selected DATASET:", dataset_used, "\n")
    print("Poisoned Data:", poisonedDataType, "\n")

    # --- Load Data ---
    if dataset_used == "CICIOT":
        # Load CICIOT data
        ciciot_train_data = loadCICIOT  # Replace with actual loading
        ciciot_test_data = pd.DataFrame()  # Replace with actual loading
        all_attacks_train = None
        all_attacks_test = None

    elif dataset_used == "IOTBOTNET":
        # Load IoTBotNet data
        ciciot_train_data = None
        ciciot_test_data = None
        all_attacks_train = pd.DataFrame()  # Replace with actual loading
        all_attacks_test = pd.DataFrame()  # Replace with actual loading

    # --- Preprocess Dataset ---
    X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = preprocess_dataset(
        dataset_used, ciciot_train_data, ciciot_test_data, all_attacks_train, all_attacks_test)

    # --- Set up model ---
    # Hyperparameters
    BATCH_SIZE = 256
    noise_dim = 100

    if regularizationEnabled:
        l2_alpha = 0.01  # Increase if overfitting, decrease if underfitting
    betas = [0.9, 0.999]  # Stable

    steps_per_epoch = len(X_train_data) // BATCH_SIZE

    input_dim = X_train_data.shape[1]

    model = create_model(input_dim, noise_dim)

    client = GanClient(model,
                       X_train_data, X_test_data,
                       BATCH_SIZE, noise_dim,
                       epochs, steps_per_epoch)

    # --- Initiate Training ---
    if fixedServer == 4:
        server_address = "192.168.129.8:8080"
    elif fixedServer == 2:
        server_address = "192.168.129.6:8080"
    elif fixedServer == 3:
        server_address = "192.168.129.7:8080"
    else:
        server_address = "192.168.129.2:8080"

    # Train GAN model
    fl.client.start_numpy_client(server_address=server_address, client=client)


if __name__ == "__main__":
    main()
