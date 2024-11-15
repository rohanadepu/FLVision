import flwr as fl
import sys
import os
import random
from datetime import datetime
import argparse
sys.path.append(os.path.abspath('..'))
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from datasetLoadProcess.loadCiciotOptimized import loadCICIOT
from datasetLoadProcess.iotbotnetDatasetLoad import loadIOTBOTNET
from datasetLoadProcess.datasetPreprocess import preprocess_dataset
from modelTrainingConfig.hflNIDSModelServerConfig import NIDSAdvGANStrategy
from modelTrainingConfig.hflGenModelConfig import create_generator

def main():


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Parsing command line arguments
    parser = argparse.ArgumentParser(description="Federated Learning Server with Model Saving")
    parser.add_argument("--rounds", type=int, choices=range(1, 11), default=8, help="Rounds of training 1-10")
    parser.add_argument("--min_clients", type=int, choices=range(1, 7), default=2, help="Minimum number of clients required for training")

    parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET"], default="CICIOT",
                        help='Datasets to use: CICIOT, IOTBOTNET, CIFAR')

    parser.add_argument('--pretrained_model', type=str, help="Path to pretrained discriminator model (optional)",
                        default=None)

    parser.add_argument("--node", type=int, choices=[1, 2, 3, 4, 5, 6], default=1, help="Client node number 1-6")
    parser.add_argument("--fixedServer", type=int, choices=[1, 2, 3, 4], default=1, help="Fixed Server node number 1-4")

    parser.add_argument("--pData", type=str, choices=["LF33", "LF66", "FN33", "FN66", None], default=None,
                        help="Label Flip: LF33, LF66")

    parser.add_argument('--reg', action='store_true', help='Enable Regularization')  # tested
    parser.add_argument('--dp', action='store_true',
                        help='Enable Differential Privacy with TFP')  # untested but working plz tune
    parser.add_argument('--adversarial', action='store_true',
                        help='Enable model adversarial training with gradients')  # bugged

    parser.add_argument('--eS', action='store_true', help='Enable model early stop training')  # callback unessary
    parser.add_argument('--lrSched', action='store_true',
                        help='Enable model lr scheduling training')  # callback unessary
    parser.add_argument('--mChkpnt', action='store_true',
                        help='Enable model model checkpoint training')  # store false irelevent

    parser.add_argument("--evalLog", type=str, default=f"evaluation_metrics_{timestamp}.txt",
                        help="Name of the evaluation log file")
    parser.add_argument("--trainLog", type=str, default=f"training_metrics_{timestamp}.txt",
                        help="Name of the training log file")

    args = parser.parse_args()

    dataset_used = args.dataset
    fixedServer = args.fixedServer
    node = args.node
    poisonedDataType = args.pData
    regularizationEnabled = args.reg
    epochs = args.epochs

    roundInput = args.rounds
    minClients = args.min_clients

    dataset_used = args.dataset
    pretrained_model = args.pretrained_model

    fixedServer = args.fixedServer
    node = args.node

    poisonedDataType = args.pData

    regularizationEnabled = args.reg
    DP_enabled = args.dp
    adversarialTrainingEnabled = args.adversarial

    earlyStopEnabled = args.eS
    lrSchedRedEnabled = args.lrSched
    modelCheckpointEnabled = args.mChkpnt

    evaluationLog = args.evalLog  # input into evaluation method if you want to input name
    trainingLog = args.trainLog  # input into train method if you want to input name

    # display selected arguments

    print("|MAIN CONFIG|", "\n")
    # main experiment config
    print("Selected Fixed Server:", fixedServer, "\n")
    print("Selected Node:", node, "\n")

    print("Selected DATASET:", dataset_used, "\n")
    print("Poisoned Data:", poisonedDataType, "\n")

    print("|DEFENSES|", "\n")
    # defense settings display
    if regularizationEnabled:
        print("Regularization Enabled", "\n")
    else:
        print("Regularization Disabled", "\n")

    if DP_enabled:

        print("Differential Privacy Engine Enabled", "\n")
    else:
        print("Differential Privacy Disabled", "\n")

    if adversarialTrainingEnabled:
        print("Adversarial Training Enabled", "\n")
    else:
        print("Adversarial Training Disabled", "\n")

    print("|CALL-BACK FUNCTIONS|", "\n")
    # callback functions display
    if earlyStopEnabled:
        print("early stop training Enabled", "\n")
    else:
        print("early stop training Disabled", "\n")

    if lrSchedRedEnabled:
        print("lr scheduler  Enabled", "\n")
    else:
        print("lr scheduler Disabled", "\n")

    if modelCheckpointEnabled:
        print("Model Check Point Enabled", "\n")
    else:
        print("Model Check Point Disabled", "\n")

    # --- Load Data ---#

    # load ciciot data if selected
    if dataset_used == "CICIOT":
        # set iotbonet to none
        all_attacks_train = None
        all_attacks_test = None
        relevant_features_iotbotnet = None

        # Load CICIOT data
        ciciot_train_data, ciciot_test_data, irrelevant_features_ciciot = loadCICIOT()

    # load iotbotnet data if selected
    elif dataset_used == "IOTBOTNET":
        # Set CICIOT to none
        ciciot_train_data = None
        ciciot_test_data = None
        irrelevant_features_ciciot = None

        # Load IOTbotnet data
        all_attacks_train, all_attacks_test, relevant_features_iotbotnet = loadIOTBOTNET()

    # --- Preprocess Dataset ---#
    X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = preprocess_dataset(
        dataset_used, ciciot_train_data, ciciot_test_data, all_attacks_train, all_attacks_test,
        irrelevant_features_ciciot, relevant_features_iotbotnet)

    # --- Model setup --- #

    # --- Hyperparameters ---#
    print("\n /////////////////////////////////////////////// \n")

    # base hyperparameters for most models
    model_name = dataset_used  # name for file

    input_dim = X_train_data.shape[1]  # dependant for feature size

    batch_size = 64  # 32 - 128; try 64, 96, 128; maybe intervals of 16, maybe even 256
    noise_dim = 100

    epochs = 5  # 1, 2 , 3 or 5 epochs

    # steps_per_epoch = (len(X_train_data) // batch_size) // epochs  # dependant  # debug
    steps_per_epoch = len(
        X_train_data) // batch_size  # dependant between sample size of the dataset and the batch size chosen

    learning_rate = 0.0001  # 0.001 or .0001
    betas = [0.9, 0.999]  # Stable

    # initiate optional variables
    l2_alpha = None
    l2_norm_clip = None
    noise_multiplier = None
    num_microbatches = None
    adv_portion = None
    metric_to_monitor_es = None
    es_patience = None
    restor_best_w = None
    metric_to_monitor_l2lr = None
    l2lr_patience = None
    save_best_only = None
    metric_to_monitor_mc = None
    checkpoint_mode = None

    # regularization param
    if regularizationEnabled:
        l2_alpha = 0.01  # Increase if overfitting, decrease if underfitting

        if DP_enabled:
            l2_alpha = 0.001  # Increase if overfitting, decrease if underfitting

        print("\nRegularization Parameter:")
        print("L2_alpha:", l2_alpha)

    if DP_enabled:
        num_microbatches = 1  # this is bugged keep at 1

        noise_multiplier = 0.3  # need to optimize noise budget and determine if noise is properly added
        l2_norm_clip = 1.5  # determine if l2 needs to be tuned as well 1.0 - 2.0

        epochs = 10
        learning_rate = 0.0007  # will be optimized

        print("\nDifferential Privacy Parameters:")
        print("L2_norm clip:", l2_norm_clip)
        print("Noise Multiplier:", noise_multiplier)
        print("MicroBatches", num_microbatches)

    if adversarialTrainingEnabled:
        adv_portion = 0.05  # in intervals of 0.05 until to 0.20
        # adv_portion = 0.1
        learning_rate = 0.0001  # will be optimized

        print("\nAdversarial Training Parameter:")
        print("Adversarial Sample %:", adv_portion * 100, "%")

    # set hyperparameters for callback

    # early stop
    if earlyStopEnabled:
        es_patience = 5  # 3 -10 epochs
        restor_best_w = True
        metric_to_monitor_es = 'val_loss'

        print("\nEarly Stop Callback Parameters:")
        print("Early Stop Patience:", es_patience)
        print("Early Stop Restore best weights?", restor_best_w)
        print("Early Stop Metric Monitored:", metric_to_monitor_es)

    # lr sched
    if lrSchedRedEnabled:
        l2lr_patience = 3  # eppoch when metric stops imporving
        l2lr_factor = 0.1  # Reduce lr to 10%
        metric_to_monitor_l2lr = 'val_auc'
        if DP_enabled:
            metric_to_monitor_l2lr = 'val_loss'

        print("\nLR sched Callback Parameters:")
        print("LR sched Patience:", l2lr_patience)
        print("LR sched Factor:", l2lr_factor)
        print("LR sched Metric Monitored:", metric_to_monitor_l2lr)

    # save best model
    if modelCheckpointEnabled:
        save_best_only = True
        checkpoint_mode = "min"
        metric_to_monitor_mc = 'val_loss'

        print("\nModel Checkpoint Callback Parameters:")
        print("Model Checkpoint Save Best only?", save_best_only)
        print("Model Checkpoint mode:", checkpoint_mode)
        print("Model Checkpoint Metric Monitored:", metric_to_monitor_mc)

    # 'val_loss' for general error, 'val_auc' for eval trade off for TP and TF rate for BC problems, "precision", "recall", ""F1-Score for imbalanced data

    print("\nBase Hyperparameters:")
    print("Input Dim (Feature Size):", input_dim)
    print("Epochs:", epochs)
    print("Batch Size:", batch_size)
    print(f"Steps per epoch (({len(X_train_data)} // {batch_size})):", steps_per_epoch)
    # print(f"Steps per epoch (({len(X_train_data)} // {batch_size}) // {epochs}):", steps_per_epoch)  ## Debug
    print("Betas:", betas)
    print("Learning Rate:", learning_rate)

    # --- Load or Create model ----#
    if args.pretrained_generator:
        print(f"Loading pretrained generator from {args.pretrained_generator}")
        generator = tf.keras.models.load_model(args.pretrained_generator)
    else:
        print("No pretrained generator provided. Creating a new generator.")
        generator = create_generator(input_dim, noise_dim)

    # Start the federated server with custom strategy
    fl.server.start_server(
        config=fl.server.ServerConfig(num_rounds=roundInput),
        strategy=NIDSAdvGANStrategy(
            generator, dataset_used, node, adversarialTrainingEnabled, earlyStopEnabled, DP_enabled,
            X_train_data, y_train_data, X_test_data, y_test_data, X_val_data, y_val_data, l2_norm_clip,
            noise_multiplier, num_microbatches, batch_size, epochs, steps_per_epoch, learning_rate, adv_portion,
            metric_to_monitor_es, es_patience, restor_best_w, metric_to_monitor_l2lr, l2lr_patience,
            save_best_only, metric_to_monitor_mc, checkpoint_mode,
            min_fit_clients=minClients,
            min_evaluate_clients=minClients,
            min_available_clients=minClients
        )
    )


if __name__ == "__main__":
    main()
