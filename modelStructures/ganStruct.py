#########################################################
#    Imports / Env setup                                #
#########################################################

import sys
import os
sys.path.append(os.path.abspath('..'))
import random
import time
from datetime import datetime
import argparse

from modelStructures.discriminatorStruct import create_discriminator, create_discriminator_binary_optimized, create_discriminator_binary, create_discriminator_binary_optimized_spectral, create_W_discriminator_binary_optimized
from modelStructures.generatorStruct import create_generator, create_generator_optimized, create_W_generator

if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']

import flwr as fl

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, LSTM, Conv1D, MaxPooling1D, GRU
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GAN
def create_model(input_dim, noise_dim):
    model = Sequential()

    model.add(create_generator(input_dim, noise_dim))
    model.add(create_discriminator(input_dim))

    return model

# GAN Binary
def create_model_binary_optimized(input_dim, noise_dim):
    model = Sequential()

    model.add(create_generator_optimized(input_dim, noise_dim))
    model.add(create_discriminator_binary_optimized(input_dim))

    return model

def create_model_binary(input_dim, noise_dim):
    model = Sequential()

    model.add(create_generator(input_dim, noise_dim))
    model.add(create_discriminator_binary(input_dim))

    return model

# ACGAN
# def create_model_AC(latent_dim, num_classes, input_dim):
#     model =

# WGAN Binary
def create_model_W_binary(input_dim, noise_dim):
    model = Sequential()

    model.add(create_W_generator(input_dim, noise_dim))
    model.add(create_W_discriminator_binary_optimized(input_dim))

    return model


# model Loading
def load_GAN_model(generator, discriminator):
    model = Sequential([generator, discriminator])

    return model


# Submodel
def split_GAN_model(model):
    # Assuming `self.model` is the GAN model created with Sequential([generator, discriminator])
    generator = model.layers[0]
    discriminator = model.layers[1]

    return generator, discriminator
