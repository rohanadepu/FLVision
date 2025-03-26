from datasetHandling.loadCiciotOptimized import loadCICIOT
from datasetHandling.iotbotnetDatasetLoad import loadIOTBOTNET
from datasetHandling.IotLoadProcess import load_and_preprocess_data, feature_selection, prepare_data, prepare_data_min_max

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

    # Initiate IoT to None
    X = None
    y = None

    # load ciciot data if selected
    if dataset_used == "CICIOT":
        print("Loading CICIOT")
        # Load CICIOT data
        ciciot_train_data, ciciot_test_data, irrelevant_features_ciciot = loadCICIOT()

    # load iotbotnet data if selected
    elif dataset_used == "IOTBOTNET":
        # Load IOTbotnet data
        all_attacks_train, all_attacks_test, relevant_features_iotbotnet = loadIOTBOTNET()

    elif dataset_used == "IOT":
        '''Load IOT -- Kiran's Dataset loading function'''
        # -- file path to preprocessed dataset
        file_path = 'datasets/50000_5000_IOT112andAllfields_Preprocessed.csv'
        # file_path = 'datasets/combined_edgeIIot_500k_custom_DDos.csv'
        X, y = load_and_preprocess_data(file_path)


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

    elif dataset_preprocessing == "IOT":
        selected_features = feature_selection(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(X, y, selected_features)

        return X_train, X_val, X_test, y_train, y_val, y_test

    elif dataset_preprocessing == "IOT-MinMax":
        selected_features = feature_selection(X, y)
        X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data_min_max(X, y, selected_features)

        return X_train, X_val, X_test, y_train, y_val, y_test
