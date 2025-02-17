#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
import random
import logging
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

from datasetHandling.loadCiciotOptimized import loadCICIOT
from datasetHandling.iotbotnetDatasetLoad import loadIOTBOTNET
from datasetHandling.datasetPreprocess import preprocess_dataset, preprocess_AC_dataset
from centralTrainingConfig.ACGANCentralTrainingConfig import CentralACGan
from modelStructures.discriminatorStruct import create_discriminator, build_AC_discriminator
from modelStructures.generatorStruct import create_generator, build_AC_generator
from modelStructures.ganStruct import create_model, load_GAN_model

################################################################################################################
#                                                   Execute                                                   #
################################################################################################################
def main():
    print("\n ////////////////////////////// \n")
    print("ACGAN Training:", "\n")

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
    parser.add_argument('--save_name', type=str,
                        help="name of model files you save as", default=f"{timestamp}")

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
    save_name = args.save_name

    # display selected arguments
    print("|Training CONFIG|", "\n")
    # main experiment config
    print("Selected Fixed Server:", fixedServer, "\n")
    print("Selected Node:", node, "\n")
    print("Selected DATASET:", dataset_used, "\n")
    print("Poisoned Data:", poisonedDataType, "\n")

    # --- Load Data ---#
    print("\n === Load Data Samples === \n")

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
    print("\n === Process Data Samples === \n")

    X_train_data, X_val_data, y_train_categorical, y_val_categorical, X_test_data, y_test_categorical = preprocess_AC_dataset(
        dataset_used, ciciot_train_data, ciciot_test_data, all_attacks_train, all_attacks_test,
        irrelevant_features_ciciot, relevant_features_iotbotnet)

    # --- Model setup --- #
    print("\n === MODEL SETUP === \n")
    # --- Hyperparameters ---#
    BATCH_SIZE = 256
    noise_dim = 100
    latent_dim = 100
    steps_per_epoch = len(X_train_data) // BATCH_SIZE
    input_dim = X_train_data.shape[1]
    # num_classes = len(np.unique(y_train_categorical))
    num_classes = 3

    if regularizationEnabled:
        l2_alpha = 0.01  # Increase if overfitting, decrease if underfitting

    betas = [0.9, 0.999]  # Stable
    learning_rate = 0.0001

    # --- Load or Create model ----#
    print("\n -- Load or create Discriminator and/or Generator MODELs -- \n")
    ACGAN = None
    generator = None
    discriminator = None

    if pretrainedGenerator and not pretrainedDiscriminator:
        print(f"Pretrained Generator provided from {pretrainedGenerator}. Creating a new Discriminator model.")
        generator = tf.keras.models.load_model(args.pretrained_generator)

        discriminator = build_AC_discriminator(input_dim, num_classes)

    elif pretrainedDiscriminator and not pretrainedGenerator:
        print(f"Pretrained Discriminator provided from {pretrainedDiscriminator}. Creating a new Generator model.")
        discriminator = tf.keras.models.load_model(args.pretrained_discriminator)

        generator = build_AC_generator(latent_dim, num_classes, input_dim)

    elif pretrainedDiscriminator and pretrainedGenerator:
        print(f"Pretrained Generator and Discriminator provided from {pretrainedGenerator} , {pretrainedDiscriminator}")
        discriminator = tf.keras.models.load_model(args.pretrained_discriminator)

        generator = tf.keras.models.load_model(args.pretrained_generator)

    else:
        print("No pretrained ACGAN provided. Creating a new ACGAN model.")
        generator = build_AC_generator(latent_dim, num_classes, input_dim)

        discriminator = build_AC_discriminator(input_dim, num_classes)

    # Optionally load the pretrained nids model
    nids = None
    if pretrainedNids:
        print(f"Loading pretrained NIDS from {pretrainedNids}")
        with tf.keras.utils.custom_object_scope({'LogCosh': LogCosh}):
            nids = tf.keras.models.load_model(pretrainedNids)

    # --- Train the Model ----#
    print("\n === TRAINING MODEL === \n")
    client = CentralACGan(discriminator, generator, nids, X_train_data, X_val_data, y_train_categorical, y_val_categorical, X_test_data, y_test_categorical, BATCH_SIZE,
                 noise_dim, latent_dim, num_classes, input_dim, epochs, steps_per_epoch, learning_rate)

    # train model
    client.train()

    # evaluate model
    print("\n === EVALUATING MODEL === \n")
    client.evaluate()

    # --- Load or Create model ----#
    print("\n === Saving MODELS === \n")

    # Save each submodel separately
    generator.save(f"../pretrainedModels/ACGAN_generator_{save_name}.h5")
    discriminator.save(f"../pretrainedModels/ACGAN_discriminator_{save_name}.h5")

if __name__ == "__main__":
    main()
