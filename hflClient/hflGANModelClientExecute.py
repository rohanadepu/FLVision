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
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
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
from sklearn.utils import shuffle

from datasetLoadProcess.loadCiciotOptimized import loadCICIOT
from datasetLoadProcess.iotbotnetDatasetLoad import loadIOTBOTNET
from datasetLoadProcess.datasetPreprocess import preprocess_dataset
from modelTrainingConfig.hflGANmodelConfig import GanClient, create_model, load_GAN_model
from modelTrainingConfig.hflDiscModelConfig import create_discriminator
from modelTrainingConfig.hflGenModelConfig import create_generator

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

    parser.add_argument('--pretrained_generator', type=str, help="Path to pretrained generator model (optional)",
                        default=None)
    parser.add_argument('--pretrained_discriminator', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)

    parser.add_argument('--pretrained_GAN', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)

    parser.add_argument('--pretrained_nids', type=str,
                        help="Path to pretrained nids model (optional)", default=None)

    # init variables to handle arguments
    args = parser.parse_args()
    # argument variables
    dataset_used = args.dataset
    fixedServer = args.fixedServer
    node = args.node
    poisonedDataType = args.pData
    regularizationEnabled = args.reg
    epochs = args.epochs
    pretrainedGan = args.pretrained_GAN
    pretrainedGenerator = args.pretrained_generator
    pretrainedDiscriminator = args.pretrained_discriminator
    pretrainedNids = args.pretrained_nids

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
    noise_dim = 100

    if regularizationEnabled:
        l2_alpha = 0.01  # Increase if overfitting, decrease if underfitting

    betas = [0.9, 0.999]  # Stable

    learning_rate = 0.0001

    steps_per_epoch = len(X_train_data) // BATCH_SIZE

    input_dim = X_train_data.shape[1]

    # --- Load or Create model ----#

    # Load or create the discriminator, generator, or whole gan model
    if pretrainedGan:
        print(f"Loading pretrained GAN Model from {pretrainedGan}")
        model = tf.keras.models.load_model(args.pretrained_discriminator)

    elif pretrainedGenerator and not pretrainedDiscriminator:

        print(f"Pretrained Generator provided from {pretrainedGenerator}. Creating a new Discriminator model.")
        generator = tf.keras.models.load_model(args.pretrained_generator)

        discriminator = create_discriminator(input_dim)

        model = load_GAN_model(generator, discriminator)

    elif pretrainedDiscriminator and not pretrainedGenerator:
        print(f"Pretrained Discriminator provided from {pretrainedDiscriminator}. Creating a new Generator model.")
        discriminator = tf.keras.models.load_model(args.pretrained_discriminator)

        generator = create_generator(input_dim, noise_dim)

        model = load_GAN_model(generator, discriminator)

    elif pretrainedDiscriminator and pretrainedGenerator:
        print(f"Pretrained Generator and Discriminator provided from {pretrainedGenerator} , {pretrainedDiscriminator}")
        discriminator = tf.keras.models.load_model(args.pretrained_discriminator)
        generator = tf.keras.models.load_model(args.pretrained_generator)

        model = load_GAN_model(generator, discriminator)

    else:
        print("No pretrained GAN provided. Creating a new GAN model.")
        model = create_model(input_dim, noise_dim)

    # Optionally load the pretrained nids model
    nids = None
    if pretrainedNids:
        print(f"Loading pretrained NIDS from {args.pretrained_nids}")
        nids = tf.keras.models.load_model(args.pretrained_nids)

    client = GanClient(model, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data, BATCH_SIZE,
                       noise_dim, epochs, steps_per_epoch, learning_rate)

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
    fl.client.start_client(server_address=server_address, client=client.to_client())

    # --- Save the trained generator model ---#
    model.save("GAN_V1.h5")

    # Assuming `self.model` is the GAN model created with Sequential([generator, discriminator])
    generator = model.layers[0]
    discriminator = model.layers[1]

    # Save each submodel separately
    generator.save("generator_GAN_V1.h5")
    discriminator.save("discriminator_GAN_V1.h5")


if __name__ == "__main__":
    main()
