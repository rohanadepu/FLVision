from datasetHandling.loadCiciotOptimized import loadCICIOT
from datasetHandling.iotbotnetDatasetLoad import loadIOTBOTNET

from datasetHandling.datasetPreprocess import preprocess_dataset, preprocess_AC_dataset

# --- Load Data ---#

def datasetLoadProcess(dataset_used, dataset_preprocessing):
    # --- 1 Sample Dataset ---#
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
        print("Loading CICIOT")
        # Load CICIOT data
        ciciot_train_data, ciciot_test_data, irrelevant_features_ciciot = loadCICIOT()

    # load iotbotnet data if selected
    elif dataset_used == "IOTBOTNET":
        # Load IOTbotnet data
        all_attacks_train, all_attacks_test, relevant_features_iotbotnet = loadIOTBOTNET()

    # --- 2 Preprocess Dataset ---#
    if dataset_preprocessing == "Default":
        X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = preprocess_dataset(
            dataset_used, ciciot_train_data, ciciot_test_data, all_attacks_train, all_attacks_test,
            irrelevant_features_ciciot, relevant_features_iotbotnet)
        return X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data

    elif dataset_preprocessing == "MM[-1,1]":
        X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data = preprocess_dataset(
            dataset_used, dataset_preprocessing, ciciot_train_data, ciciot_test_data, all_attacks_train, all_attacks_test,
            irrelevant_features_ciciot, relevant_features_iotbotnet)
        return X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data

    elif dataset_preprocessing == "AC-GAN":
        X_train_data, X_val_data, y_train_categorical, y_val_categorical, X_test_data, y_test_categorical = preprocess_AC_dataset(
            dataset_used, ciciot_train_data, ciciot_test_data, all_attacks_train, all_attacks_test,
            irrelevant_features_ciciot, relevant_features_iotbotnet)
        return X_train_data, X_val_data, y_train_categorical, y_val_categorical, X_test_data, y_test_categorical