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
#                                       GAN Model Setup                                       #
################################################################################################################
def create_generator(input_dim, noise_dim):
    generator = tf.keras.Sequential([
        Dense(128, activation='relu', input_shape=(noise_dim,)),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(input_dim, activation='sigmoid')
    ])

    return generator


def create_discriminator(input_dim):
    discriminator = tf.keras.Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dense(1, activation='sigmoid')
    ])

    return discriminator


def create_model(input_dim, noise_dim):
    model = Sequential()

    model.add(create_generator(input_dim, noise_dim))
    model.add(create_discriminator(input_dim))

    return model


def discriminator_loss(real_output, fake_output):
    real_loss = binary_crossentropy(tf.ones_like(real_output), real_output)
    fake_loss = binary_crossentropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return binary_crossentropy(tf.ones_like(fake_output), fake_output)


def generate_and_save_network_traffic(model, test_input):
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')

    plt.savefig('image.png')
    plt.show()


# hyperparameter
learning_rate = 0.0001  # 0.001 or .0001

# premade functions
binary_crossentropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
generator_optimizer = tf.keras.optimizers.Adam(learning_rate)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate)
seed = tf.random.normal([16, 100])


################################################################################################################
#                                               FL-GAN TRAINING Setup                                         #
################################################################################################################
class GanClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, x_test, BATCH_SIZE, noise_dim, epochs, steps_per_epoch):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test

        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(BATCH_SIZE)

        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        generator = self.model.layers[0]
        discriminator = self.model.layers[1]

        for epoch in range(self.epochs):
            for step, instances in enumerate(self.x_train_ds.take(self.steps_per_epoch)):

                noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

                with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                    generated_samples = generator(noise, training=True)

                    real_output = discriminator(instances, training=True)
                    fake_output = discriminator(generated_samples, training=True)

                    gen_loss = generator_loss(fake_output)
                    disc_loss = discriminator_loss(real_output, fake_output)

                gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
                gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

                generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

                if step % 100 == 0:
                    print(f'Epoch {epoch+1}, Step {step}, D Loss: {disc_loss.numpy()}, G Loss: {gen_loss.numpy()}')

        return self.get_parameters(config={}), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        generator = self.model.layers[0]
        discriminator = self.model.layers[1]

        # Convert x_test to tensor
        self.x_test = tf.convert_to_tensor(self.x_test.values, dtype=tf.float32)

        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])

        generated_samples = generator(noise, training=False)

        print(self.x_test.shape)
        print(generated_samples.shape)

        real_output = discriminator(self.x_test, training=False)
        fake_output = discriminator(generated_samples, training=False)

        loss = discriminator_loss(real_output, fake_output)

        return float(loss.numpy()), len(self.x_test), {}


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
    fl.client.start_client(server_address=server_address, client=client.to_client())


if __name__ == "__main__":
    main()