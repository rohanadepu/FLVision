import flwr as fl
import sys
import os
from datetime import datetime
import argparse
sys.path.append(os.path.abspath('..'))
import tensorflow as tf

from datasetLoadProcess.loadCiciotOptimized import loadCICIOT
from datasetLoadProcess.iotbotnetDatasetLoad import loadIOTBOTNET
from datasetLoadProcess.datasetPreprocess import preprocess_dataset
from globalModelTrainingConfig.DiscModelServerConfig import DiscriminatorFullStrategy
from clientModelTrainingConfig.GenModelClientConfig import create_generator

def main():


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Federated Learning Server with Model Saving")
    parser.add_argument("--rounds", type=int, choices=range(1, 11), default=8, help="Rounds of training 1-10")
    parser.add_argument("--min_clients", type=int, choices=range(1, 7), default=2, help="Minimum number of clients required for training")

    parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET"], default="CICIOT",
                        help='Datasets to use: CICIOT, IOTBOTNET')

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
    roundInput = args.rounds
    minClients = args.min_clients
    dataset_used = args.dataset


    poisonedDataType = args.pData
    regularizationEnabled = args.reg
    epochs = args.epochs

    # display selected arguments
    print("|MAIN SERVER CONFIG|", "\n")

    # main experiment config
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
    # Hyperparameters
    BATCH_SIZE = 256
    input_dim = X_train_data.shape[1]
    noise_dim = 100
    epochs = 5
    steps_per_epoch = len(X_train_data) // BATCH_SIZE

    # Load or create the generator model
    if args.pretrained_generator:
        print(f"Loading pretrained generator from {args.pretrained_generator}")
        generator = tf.keras.models.load_model(args.pretrained_generator)
    else:
        print("No pretrained generator provided. Creating a new generator.")
        generator = create_generator(input_dim, noise_dim)

    # Start the federated server with custom strategy
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=roundInput),
        strategy=DiscriminatorFullStrategy(
            generator, X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data, BATCH_SIZE,
            noise_dim, epochs, steps_per_epoch, dataset_used, input_dim,
            min_fit_clients=minClients,
            min_evaluate_clients=minClients,
            min_available_clients=minClients
        )
    )


if __name__ == "__main__":
    main()