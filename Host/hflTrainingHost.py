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

from datasetHandling.datasetLoadProcess import datasetLoadProcess

from Client.overheadConfig.hyperparameterLoading import hyperparameterLoading
from Client.overheadConfig.modelCreateLoad import modelCreateLoad
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
    parser.add_argument('--dataset_processing', type=str, choices=["Default", "MM[-1,-1]", "AC-GAN"], default="Default",
                        help='Datasets to use: Default, MM[-1,1], AC-GAN')

    # -- Training / Model Parameters -- #
    parser.add_argument('--clientBased', action='store_true',
                        help='Only load the model structure and get the weights from the server')
    parser.add_argument('--serverSave', action='store_true',
                        help='Only load the model structure and get the weights from the server')

    parser.add_argument('--model_type', type=str, choices=["NIDS", "GAN", "WGAN-GP", "AC-GAN"],
                        help='Please select NIDS ,GAN, WGAN-GP, or AC-GAN as the model type to train')

    parser.add_argument('--model_training', type=str, choices=["NIDS", "Generator", "Discriminator", "Both"],
                        help='Please select NIDS, Generator, Discriminator, Both as the sub-model type to train')

    # Optional
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs to train the model")
    parser.add_argument("--rounds", type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], default=1,
                        help="Rounds of training 1-10")
    parser.add_argument("--min_clients", type=int, choices=[1, 2, 3, 4, 5, 6], default=2,
                        help="Minimum number of clients required for training")

    # -- Loading Models Optional -- +
    parser.add_argument('--pretrained_GAN', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)
    parser.add_argument('--pretrained_generator', type=str, help="Path to pretrained generator model (optional)",
                        default=None)
    parser.add_argument('--pretrained_discriminator', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)
    parser.add_argument('--pretrained_nids', type=str,
                        help="Path to pretrained nids model (optional)", default=None)

    # -- Saving Models Optional -- +
    parser.add_argument('--save_name', type=str,
                        help="name of model files you save as", default=f"{timestamp}")

    # init variables to handle arguments
    args = parser.parse_args()

    # argument variables
    dataset_used = args.dataset
    dataset_processing = args.dataset_processing

    # Model Spec
    clientBased = args.clientBased
    serverSave = args.serverSave
    model_type = args.model_type
    train_type = args.model_training
    if model_type == "AC-GAN":
        dataset_processing = "AC-GAN"

    # Training / Hyper Param
    epochs = args.epochs
    regularizationEnabled = True
    DP_enabled = None
    earlyStopEnabled = None
    lrSchedRedEnabled = None
    modelCheckpointEnabled = None

    roundInput = args.rounds
    minClients = args.min_clients

    # Whole GAN model
    pretrainedGan = args.pretrained_GAN

    # Individual GAN Submodels
    pretrainedGenerator = args.pretrained_generator
    pretrainedDiscriminator = args.pretrained_discriminator

    # Optional NIDS
    pretrainedNids = args.pretrained_nids

    # Save/Record Param
    save_name = args.save_name
    evaluationLog = timestamp
    trainingLog = timestamp
    node = 1

    # -- Display selected arguments --#
    print("|MAIN CONFIG|", "\n")
    # main experiment config
    if clientBased is False:
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

    if clientBased is True:
        if serverSave is False:
            # --- Default ---#
            fl.server.start_server(
                config=fl.server.ServerConfig(num_rounds=roundInput),
                strategy=fl.server.strategy.FedAvg(
                    min_fit_clients=minClients,
                    min_evaluate_clients=minClients,
                    min_available_clients=minClients
                )
            )
        else:
            # --- Save Model ---#
            x = None
    else:
        # --- Server Based ---#
        x = None

        # --- 2 Load & Preprocess Data ---#
        X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = datasetLoadProcess(dataset_used,
                                                                                                          dataset_processing)

        # --- 3 Model Hyperparameter & Training Parameters ---#
        (BATCH_SIZE, noise_dim, steps_per_epoch, input_dim, num_classes, latent_dim, betas, learning_rate, l2_alpha,
         l2_norm_clip, noise_multiplier, num_microbatches, metric_to_monitor_es, es_patience, restor_best_w,
         metric_to_monitor_l2lr, l2lr_patience, save_best_only,
         metric_to_monitor_mc, checkpoint_mode) = hyperparameterLoading(model_type, X_train_data,
                                                                        regularizationEnabled, DP_enabled,
                                                                        earlyStopEnabled,
                                                                        lrSchedRedEnabled, modelCheckpointEnabled)

        # --- 4 Model Loading & Creation ---#
        nids, discriminator, generator, GAN = modelCreateLoad(model_type, train_type, pretrainedNids, pretrainedGan,
                                                              pretrainedGenerator, pretrainedDiscriminator,
                                                              dataset_used,
                                                              input_dim, noise_dim, regularizationEnabled, DP_enabled,
                                                              l2_alpha, latent_dim, num_classes)
        # --- 5 Load Server Strat ---#

        # --- 6 Start Hosting ---#
