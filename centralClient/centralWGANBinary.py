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

from datasetHandling.loadCiciotOptimized import loadCICIOT
from datasetHandling.iotbotnetDatasetLoad import loadIOTBOTNET
from datasetHandling.datasetPreprocess import preprocess_dataset
from centralTrainingConfig.WGANBinaryCentralTrainingConfig import CentralBinaryWGan
from modelStructures.discriminatorStruct import create_discriminator_binary, create_discriminator_binary_optimized, create_discriminator_binary
from modelStructures.generatorStruct import create_generator, create_generator_optimized
from modelStructures.ganStruct import create_model, load_GAN_model, create_model_binary, create_model_binary_optimized, create_model_W_binary
from tensorflow_addons.layers import SpectralNormalization

################################################################################################################
#                                                   Execute                                                   #
################################################################################################################
def main():
    print("\n ////////////////////////////// \n")
    print("WGAN Training:", "\n")

    # Generate a static timestamp at the start of the script
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 1 Argument Parsing --- #
    parser = argparse.ArgumentParser(description='Select dataset, model selection, and to enable DP respectively')
    parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET"], default="CICIOT",
                        help='Datasets to use: CICIOT, IOTBOTNET')

    parser.add_argument("--evalLog", type=str, default=f"evaluation_metrics_{timestamp}.txt",
                        help="Name of the evaluation log file")
    parser.add_argument("--trainLog", type=str, default=f"training_metrics_{timestamp}.txt",
                        help="Name of the training log file")

    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs to train the model")

    parser.add_argument('--pretrained_GAN', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)
    parser.add_argument('--pretrained_generator', type=str, help="Path to pretrained generator model (optional)",
                        default=None)
    parser.add_argument('--pretrained_discriminator', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)

    parser.add_argument('--pretrained_nids', type=str,
                        help="Path to pretrained nids model (optional)", default=None)

    parser.add_argument('--save_name', type=str,
                        help="name of model files you save as", default=f"{timestamp}")

    # init variables to handle arguments
    args = parser.parse_args()

    # argument variables
    dataset_used = args.dataset

    epochs = args.epochs

    pretrainedGan = args.pretrained_GAN
    pretrainedGenerator = args.pretrained_generator
    pretrainedDiscriminator = args.pretrained_discriminator

    pretrainedNids = args.pretrained_nids

    save_name = args.save_name

    # display selected arguments
    print("|MAIN CONFIG|", "\n")
    print("Selected DATASET:", dataset_used, "\n")

    # --- 2 Load Data ---#

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

    # --- 3 Preprocess Dataset ---#
    X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = preprocess_dataset(
        dataset_used, ciciot_train_data, ciciot_test_data, all_attacks_train, all_attacks_test,
        irrelevant_features_ciciot, relevant_features_iotbotnet)

    # --- 4 Model setup --- #

    # Hyperparameters
    BATCH_SIZE = 256
    noise_dim = 100
    steps_per_epoch = len(X_train_data) // BATCH_SIZE
    input_dim = X_train_data.shape[1]

    learning_rate = 0.0001

    # Load or Create model

    # Load or create the discriminator, generator, or whole gan model
    if pretrainedGan:
        print(f"Loading pretrained GAN Model from {pretrainedGan}")
        with tf.keras.utils.custom_object_scope({"SpectralNormalization": SpectralNormalization}):
            model = tf.keras.models.load_model(pretrainedGan)

    elif pretrainedGenerator and not pretrainedDiscriminator:
        print(f"Pretrained Generator provided from {pretrainedGenerator}. Creating a new Discriminator model.")
        generator = tf.keras.models.load_model(args.pretrained_generator)

        discriminator = create_discriminator_binary(input_dim)

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
        model = create_model_W_binary(input_dim, noise_dim)

    # Optionally load the pretrained nids model
    nids = None
    if pretrainedNids:
        print(f"Loading pretrained NIDS from {args.pretrained_nids}")
        with tf.keras.utils.custom_object_scope({'LogCosh': LogCosh}):
            model = tf.keras.models.load_model(pretrainedNids)

    # --- 5 Start Training ---#
    client = CentralBinaryWGan(model, nids, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data, BATCH_SIZE,
                              noise_dim, epochs, steps_per_epoch, learning_rate)

    # train and evaluate
    client.fit()
    client.evaluate()

    # --- 6 Save the trained generator model ---#
    model.save(f"../pretrainedModels/WGAN_{save_name}.h5")

    # Assuming `model` is the GAN model created with Sequential([generator, discriminator])
    generator = model.layers[0]
    discriminator = model.layers[1]

    # Save each submodel separately
    generator.save(f"../pretrainedModels/generator_WGAN_{save_name}.h5")
    discriminator.save(f"../pretrainedModels/discriminator_WGAN_{save_name}.h5")


if __name__ == "__main__":
    main()
