#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
import random
from datetime import datetime
import argparse
sys.path.append(os.path.abspath('..'))
# TensorFlow & Flower
if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']
import flwr as fl
import tensorflow as tf
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# other plugins
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import expand_dims
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# import math
# import glob
# from tqdm import tqdm
# import seaborn as sns
# import pickle
# import joblib

from datasetHandling.loadCiciotOptimized import loadCICIOT
from datasetHandling.iotbotnetDatasetLoad import loadIOTBOTNET

from datasetHandling.datasetPreprocess import preprocess_dataset

from centralTrainingConfig.GANBinaryCentralTrainingConfig import CentralBinaryGan

from modelStructures.discriminatorStruct import create_discriminator_binary, create_discriminator_binary_optimized, create_discriminator_binary
from modelStructures.generatorStruct import create_generator, create_generator_optimized
from modelStructures.ganStruct import create_model, load_GAN_model, create_model_binary, create_model_binary_optimized

################################################################################################################
#                                                   Execute                                                   #
################################################################################################################

def main():
    print("\n ////////////////////////////// \n")
    print("Centralized Training:", "\n")

    # Generate a static timestamp at the start of the script
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- 1. Argument Parsing --- #
    parser = argparse.ArgumentParser(description='Select dataset, model selection, and to enable DP respectively')

    # -- Dataset Settings -- #
    parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET"], default="CICIOT",
                        help='Datasets to use: CICIOT, IOTBOTNET')
    parser.add_argument('--dataset_processing', type=str, choices=["Default", "MM[-1,-1]", "AC-GAN"], default="CICIOT",
                        help='Datasets to use: Default, MM[-1,-1], AC-GAN')

    # -- Training / Model Parameters -- #
    parser.add_argument('--model_type', type=str, choices=["NIDS", "GAN", "WGAN-GP", "AC-GAN"],
                        help='Please select NIDS ,GAN, WGAN-GP, or AC-GAN as the model type to train')

    parser.add_argument('--model_training', type=str, choices=["NIDS","Generator", "Discriminator", "Both"],
                        help='Please select NIDS ,GAN, WGAN-GP, or AC-GAN as the model type to train')

    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model")

    # -- Loading Models -- +
    parser.add_argument('--model_init', type=str, choices=["Client", "Server"],
                        help='Please select Client or Server to select where the model is being initialized from to train')

    parser.add_argument('--pretrained_GAN', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)
    parser.add_argument('--pretrained_generator', type=str, help="Path to pretrained generator model (optional)",
                        default=None)
    parser.add_argument('--pretrained_discriminator', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)
    parser.add_argument('--pretrained_nids', type=str,
                        help="Path to pretrained nids model (optional)", default=None)

    # -- Saving Models -- +
    parser.add_argument('--save_name', type=str,
                        help="name of model files you save as", default=f"{timestamp}")

    # init variables to handle arguments
    args = parser.parse_args()
    # argument variables
    dataset_used = args.dataset
    dataset_processing = args.dataset_processing
    model_type = args.model_type
    train_type = args.model_training
    epochs = args.epochs
    pretrainedGan = args.pretrained_GAN
    pretrainedGenerator = args.pretrained_generator
    pretrainedDiscriminator = args.pretrained_discriminator
    pretrainedNids = args.pretrained_nids
    save_name = args.save_name

    if model_type == "AC-GAN":
        dataset_processing = "AC-GAN"

    # -- Display selected arguments --#
    print("|MAIN CONFIG|", "\n")
    # main experiment config
    print("Selected DATASET:", dataset_used, "\n")
    print("Selected Preprocessing:", dataset_processing, "\n")
    print("Selected Model Type:", model_type, "\n")
    print("Selected Model Training:", train_type, "\n")
    print("Selected Epochs:", epochs, "\n")
    print("Loaded GAN/GAN-Variant:", dataset_used, "\n")
    print("Loaded Generator Model:", pretrainedGenerator, "\n")
    print("Loaded Discriminator Model:", pretrainedDiscriminator, "\n")
    print("Loaded NIDS Model:", pretrainedNids, "\n")
    print("Save Name for the models in Trained in this session:", save_name, "\n")

    # --- 2 Load & Preprocess Data ---#

    X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = datasetLoadProcess(dataset_used, dataset_processing)

    # --- 3 Model Hyperparameter & Training Parameters ---#
    BATCH_SIZE = 256
    noise_dim = 100
    steps_per_epoch = len(X_train_data) // BATCH_SIZE
    input_dim = X_train_data.shape[1]

    l2_alpha = 0.01  # Increase if overfitting, decrease if underfitting
    betas = [0.9, 0.999]  # Stable
    learning_rate = 0.0001

    # --- 4 Model Loading & Creation ---#

    # --- 5 Load Training Config ---#

    # --- 6 Train Model ---#

    # --- 7 Evaluate Model ---#

    # --- 8 Save Model ---#