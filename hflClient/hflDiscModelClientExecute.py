#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
import random
from datetime import datetime
import argparse
sys.path.append(os.path.abspath('..'))

if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']

import flwr as fl

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd
import matplotlib.pyplot as plt

# import math
# import glob

# from tqdm import tqdm

# import seaborn as sns

# import pickle
# import joblib

from sklearn.utils import shuffle

from datasetLoadProcess.loadCiciotOptimized import loadCICIOT
from datasetLoadProcess.iotbotnetDatasetLoad import loadIOTBOTNET
from datasetLoadProcess.datasetPreprocess import preprocess_dataset
from modelTrainingConfig.hflDiscModelConfig import DiscriminatorClient, create_discriminator
from modelTrainingConfig.hflGenModelConfig import create_generator
################################################################################################################
#                                       Abstract                                       #
################################################################################################################


def main():
    print("\n ////////////////////////////// \n")
    print("Federated Learning Discriminator Client Training:", "\n")

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

    parser.add_argument('--pretrained_generator', type=str, help="Path to pretrained generator model (optional)",
                        default=None)
    parser.add_argument('--pretrained_discriminator', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)

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

    # --- Load Data ---#

    # Initiate CICIOT to none
    ciciot_train_data = None
    ciciot_test_data = None
    irrelevant_features_ciciot = None

    # Initiate iotbonet to none
    all_attacks_train = None
    all_attacks_test = None
    relevant_features_iotbotnet = None

    # load ciciot data if selected
    if dataset_used == "CICIOT":
        # Load CICIOT data
        ciciot_train_data, ciciot_test_data, irrelevant_features_ciciot = loadCICIOT()

    # load iotbotnet data if selected
    elif dataset_used == "IOTBOTNET":
        # Load IOTbotnet data
        all_attacks_train, all_attacks_test, relevant_features_iotbotnet = loadIOTBOTNET()

    # --- Preprocess Dataset ---#
    X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = preprocess_dataset(
        dataset_used, ciciot_train_data, ciciot_test_data, all_attacks_train, all_attacks_test,
        irrelevant_features_ciciot, relevant_features_iotbotnet)

    # --- Model setup --- #

    # --- Hyperparameters ---#
    BATCH_SIZE = 256
    input_dim = X_train_data.shape[1]
    noise_dim = 100
    epochs = epochs
    steps_per_epoch = len(X_train_data) // BATCH_SIZE

    print("\nBase Hyperparameters:")
    print("Input Dim (Feature Size):", input_dim)
    print("Epochs:", epochs)
    print("Batch Size:", BATCH_SIZE)
    print(f"Steps per epoch (({len(X_train_data)} // {BATCH_SIZE})):", steps_per_epoch)

    # --- Load or Create model ----#

    # Load or create the discriminator model
    if args.pretrained_discriminator:
        print(f"Loading pretrained discriminator from {args.pretrained_discriminator}")
        discriminator = tf.keras.models.load_model(args.pretrained_discriminator)
    else:
        print("No pretrained discriminator provided. Creating a new discriminator.")
        discriminator = create_discriminator(input_dim)

    # Load or create the generator model
    if args.pretrained_generator:
        print(f"Loading pretrained generator from {args.pretrained_generator}")
        generator = tf.keras.models.load_model(args.pretrained_generator)
    else:
        print("No pretrained generator provided. Creating a new generator.")
        generator = create_generator(input_dim, noise_dim)

    #--- Initiate client with models, data, and parameters ---#
    client = DiscriminatorClient(discriminator, generator, X_train_data, X_val_data, y_train_data, y_val_data,
                                 X_test_data, y_test_data, BATCH_SIZE, noise_dim, epochs, steps_per_epoch)

    # --- Initiate Training ---#
    if fixedServer == 4:
        server_address = "192.168.129.8:8080"
    elif fixedServer == 2:
        server_address = "192.168.129.6:8080"
    elif fixedServer == 3:
        server_address = "192.168.129.7:8080"
    else:
        server_address = "192.168.129.2:8080"

    # Train discriminator model
    fl.client.start_client(server_address=server_address, client=client.to_client())

    # --- Save the trained discriminator model ---#
    discriminator.save("discriminator_model.h5")


if __name__ == "__main__":
    main()
