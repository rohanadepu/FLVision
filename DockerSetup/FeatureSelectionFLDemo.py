#########################################################
#    Imports / Env setup                                #
#########################################################

import os
import flwr as fl
import numpy as np
import tensorflow as tf

# import numpy as np
import pandas as pd
# import math
# import glob
import random
# from tqdm import tqdm
#from IPython.display import clear_output
import os
# import time
# import matplotlib.pyplot as plt
# import seaborn as sns
# import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, mutual_info_classif
from sklearn.model_selection import train_test_split

# import sklearn.cluster as cluster
# from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import cross_val_predict
# from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle
# from sklearn.impute import SimpleImputer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import matthews_corrcoef
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset_used = "CICIOT"

print("DATASET BEING USED:", dataset_used, "\n")

#########################################################
#    Loading Dataset For CICIOT 2023                    #
#########################################################
if dataset_used == "CICIOT":
    ### Inputs ###
    ciciot_sample_size = 2  # input: 2 at minimum
    # label classes 33+1 7+1 1+1
    ciciot_label_class = "1+1"


    DATASET_DIRECTORY = '../../trainingDataset/'

    # List the files in the dataset
    csv_filepaths = [filename for filename in os.listdir(DATASET_DIRECTORY) if filename.endswith('.csv')]
    print(csv_filepaths)

    # If there are more than X CSV files, randomly select X files from the list
    if len(csv_filepaths) > ciciot_sample_size:
        csv_filepaths = random.sample(csv_filepaths, ciciot_sample_size)
        print(csv_filepaths)
    csv_filepaths.sort()
    split_index = int(len(csv_filepaths) * 0.5)

    training_data_sets = csv_filepaths[:split_index]
    test_data_sets = csv_filepaths[split_index:]

    print("Training Sets:\n", training_data_sets, "\n")
    print("Test Sets:\n", test_data_sets)

    # Mapping Features
    # num_cols = [
    #     'flow_duration', 'Header_Length', 'Duration',
    #     'Rate', 'Srate', 'ack_count', 'syn_count',
    #     'fin_count', 'urg_count', 'rst_count', 'Tot sum',
    #     'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    #     'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight',
    # ]
    num_cols = ['Duration','Rate', 'Srate', 'ack_count', 'syn_count','fin_count','Tot size', 'IAT', 'Number','Weight']

    cat_cols = [
        'Protocol Type', 'Drate', 'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
        'psh_flag_number', 'ack_flag_number', 'ece_flag_number',
        'cwr_flag_number', 'HTTP', 'HTTPS', 'DNS', 'Telnet',
        'SMTP', 'SSH', 'IRC', 'TCP', 'UDP', 'DHCP', 'ARP',
        'ICMP', 'IPv', 'LLC', 'label'
    ]

    irrelevant_features = ['ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'Magnitue', 'Radius', 'Covariance',
                           'Variance', 'flow_duration', 'Header_Length', 'urg_count', 'rst_count', 'Tot sum', 'Min',
                           'Max', 'AVG', 'Std']

    # Mapping Labels
    dict_7classes = {'DDoS-RSTFINFlood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS',
                     'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS',
                     'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS', 'DDoS-UDP_Fragmentation': 'DDoS',
                     'DDoS-ICMP_Fragmentation': 'DDoS', 'DDoS-SlowLoris': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS',
                     'DoS-UDP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS',
                     'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai', 'Mirai-udpplain': 'Mirai',
                     'Recon-PingSweep': 'Recon', 'Recon-OSScan': 'Recon', 'Recon-PortScan': 'Recon',
                     'VulnerabilityScan': 'Recon', 'Recon-HostDiscovery': 'Recon', 'DNS_Spoofing': 'Spoofing',
                     'MITM-ArpSpoofing': 'Spoofing', 'BenignTraffic': 'Benign', 'BrowserHijacking': 'Web',
                     'Backdoor_Malware': 'Web', 'XSS': 'Web', 'Uploading_Attack': 'Web', 'SqlInjection': 'Web',
                     'CommandInjection': 'Web', 'DictionaryBruteForce': 'BruteForce'}

    dict_2classes = {'DDoS-RSTFINFlood': 'Attack', 'DDoS-PSHACK_Flood': 'Attack', 'DDoS-SYN_Flood': 'Attack',
                     'DDoS-UDP_Flood': 'Attack', 'DDoS-TCP_Flood': 'Attack', 'DDoS-ICMP_Flood': 'Attack',
                     'DDoS-SynonymousIP_Flood': 'Attack', 'DDoS-ACK_Fragmentation': 'Attack',
                     'DDoS-UDP_Fragmentation': 'Attack', 'DDoS-ICMP_Fragmentation': 'Attack', 'DDoS-SlowLoris': 'Attack',
                     'DDoS-HTTP_Flood': 'Attack', 'DoS-UDP_Flood': 'Attack', 'DoS-SYN_Flood': 'Attack',
                     'DoS-TCP_Flood': 'Attack', 'DoS-HTTP_Flood': 'Attack', 'Mirai-greeth_flood': 'Attack',
                     'Mirai-greip_flood': 'Attack', 'Mirai-udpplain': 'Attack', 'Recon-PingSweep': 'Attack',
                     'Recon-OSScan': 'Attack', 'Recon-PortScan': 'Attack', 'VulnerabilityScan': 'Attack',
                     'Recon-HostDiscovery': 'Attack', 'DNS_Spoofing': 'Attack', 'MITM-ArpSpoofing': 'Attack',
                     'BenignTraffic': 'Benign', 'BrowserHijacking': 'Attack', 'Backdoor_Malware': 'Attack', 'XSS': 'Attack',
                     'Uploading_Attack': 'Attack', 'SqlInjection': 'Attack', 'CommandInjection': 'Attack',
                     'DictionaryBruteForce': 'Attack'}

    # Extracting data from csv to input into data frame
    ciciot_train_data = pd.DataFrame()
    for data_set in training_data_sets:
        print(f"data set {data_set} out of {len(training_data_sets)} \n")
        data_path = os.path.join(DATASET_DIRECTORY, data_set)
        df = pd.read_csv(data_path)
        ciciot_train_data = pd.concat([ciciot_train_data, df])  # dataframe to manipulate

    ciciot_test_data = pd.DataFrame()
    for test_set in test_data_sets:
        print(f"Testing set {test_set} out of {len(test_data_sets)} \n")
        data_path = os.path.join(DATASET_DIRECTORY, test_set)
        df = pd.read_csv(data_path)
        ciciot_test_data = pd.concat([df])

#########################################################
#    Process Dataset For CICIOT 2023                    #
#########################################################

    ## Remapping for other Classifications ##

    if ciciot_label_class == "7+1":
        # Relabel the 'label' column using dict_7classes
        ciciot_train_data['label'] = ciciot_train_data['label'].map(dict_7classes)
        ciciot_test_data['label'] = ciciot_test_data['label'].map(dict_7classes)

    if ciciot_label_class == "1+1":
        # Relabel the 'label' column using dict_2classes
        ciciot_train_data['label'] = ciciot_train_data['label'].map(dict_2classes)
        ciciot_test_data['label'] = ciciot_test_data['label'].map(dict_2classes)

    # # Drop the irrelevant features
    # ciciot_train_data = ciciot_train_data.drop(columns=irrelevant_features)
    # ciciot_test_data = ciciot_test_data.drop(columns=irrelevant_features)

    # Shuffle data
    ciciot_train_data = shuffle(ciciot_train_data, random_state=47)

    # prints an instance of each class
    print("Before Encoding and Scaling:")
    unique_labels = ciciot_train_data['label'].unique()
    for label in unique_labels:
        print(f"First instance of {label}:")
        print(ciciot_train_data[ciciot_train_data['label'] == label].iloc[0])


# ---                   Labeling                     --- #

    # Assuming 'label' is the column name for the labels in the DataFrame `synth_data`
    unique_labels = ciciot_train_data['label'].nunique()

    # Print the number of unique labels
    print(f"There are {unique_labels} unique labels in the dataset.")

    # print the amount of instances for each label
    class_counts = ciciot_train_data['label'].value_counts()
    print(class_counts)

    # Display the first few entries to verify the changes
    print(ciciot_train_data.head())

    # Encodes the training label
    label_encoder = LabelEncoder()
    ciciot_train_data['label'] = label_encoder.fit_transform(ciciot_train_data['label'])
    ciciot_test_data['label'] = label_encoder.fit_transform(ciciot_test_data['label'])

    # Store label mappings
    label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print("Label mappings:", label_mapping)

    # Retrieve the numeric codes for classes
    class_codes = {label: label_encoder.transform([label])[0] for label in label_encoder.classes_}

    # Print specific instances after label encoding
    print("Real data After Encoding:")
    for label, code in class_codes.items():
        # Print the first instance of each class
        print(f"First instance of {label} (code {code}):")
        print(ciciot_train_data[ciciot_train_data['label'] == code].iloc[0])
    print(ciciot_train_data.head(), "\n")

    # Feature / Label Split (X y split)
    X_train_data = ciciot_train_data.drop(columns=['label'])
    y_train_data = ciciot_train_data['label']

    X_test_data = ciciot_test_data.drop(columns=['label'])
    y_test_data = ciciot_test_data['label']

    # Print the shapes of the resulting splits
    print("X_train shape:", X_train_data.shape)
    print("y_train shape:", y_train_data.shape)

    #########################################################
    # Step 2: Remove Correlated Features                    #
    #########################################################

    correlation_matrix = X_train_data.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > 0.70)]

    X_train_reduced = X_train_data.drop(to_drop, axis=1)
    X_test_reduced = X_test_data.drop(to_drop, axis=1) # make sure to process the test data as well
    print(f"Removed correlated features: {to_drop}")

    #########################################################
    # Step 3A: Apply Mutual Information                      #
    #########################################################

    mi = mutual_info_classif(X_train_reduced, y_train_data, random_state=42)
    mi_series = pd.Series(mi, index=X_train_reduced.columns)

    top_features_mi = mi_series.sort_values(ascending=False).head(30).index
    print(f"Top features by mutual information: {top_features_mi}")

    #########################################################
    # Step 1C: Scale the Features                            #
    #########################################################

    # Setting up Scaler for Features
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Scale the numeric features present in X_reduced
    scaled_num_cols = [col for col in num_cols if col in X_train_reduced.columns]
    X_train_reduced[scaled_num_cols] = scaler.fit_transform(X_train_reduced[scaled_num_cols])
    X_test_reduced[scaled_num_cols] = scaler.fit_transform(X_test_reduced[scaled_num_cols])

    # prove if the data is loaded properly
    print("Real data After Scaling (TRAIN):")
    print(X_train_reduced.head())
    print(X_train_reduced.shape)

    print("Real data After Scaling (TEST):")
    print(X_test_reduced.head())
    print(X_test_reduced.shape)

    #########################################################
    # Step 3B: Tree-Based Feature Importance                #
    #########################################################

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_reduced, y_train_data)

    importances = model.feature_importances_
    feature_importances = pd.Series(importances, index=X_train_reduced.columns)

    top_features_rf = feature_importances.sort_values(ascending=False).head(30).index
    print(f"Top features by Random Forest importance: {top_features_rf}")

    #########################################################
    # Step 3C: Apply RFE with Random Forest                  #
    #########################################################

    rfe = RFE(estimator=model, n_features_to_select=16, step=1)
    rfe.fit(X_train_reduced, y_train_data)

    top_features_rfe = X_train_reduced.columns[rfe.support_]
    print(f"Top features by RFE: {top_features_rfe}")

    #########################################################
    # Step 3D: Combine features from different methods               #
    #########################################################

    combined_features = list(set(top_features_mi) | set(top_features_rf) | set(top_features_rfe))
    # combined_features = list(set(top_features_mi) | set(top_features_rf))
    print(f"Combined top features: {combined_features}")

    X_selected = X_train_reduced[combined_features]

    X_train_data = X_selected
    X_test_data = X_test_data[combined_features]  # Ensure test data has the same features

    print("Final X_train shape:", X_train_data.shape)
    print("Final X_test shape:", X_test_data.shape)

#########################################################
#    Model Initialization & Setup                       #
#########################################################

# Define a dense neural network for anomaly detection based on the dataset
if dataset_used == "CICIOT":

    input_dim = X_train_data.shape[1]
    print("///////////////////////////////////////////////")
    print("Unique Labels:", unique_labels)
    print("Input Dim:", input_dim)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(unique_labels, activation='sigmoid')  # unique_labels is the number of classes
    ])


model.compile(optimizer='adam',
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=['accuracy'])

model.summary()

#########################################################
#    Federated Learning Setup                           #
#########################################################

class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(X_train_data, y_train_data, epochs=1, batch_size=32, steps_per_epoch=3)  # Change dataset here
        return model.get_weights(), len(X_train_data), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(X_test_data, y_test_data)  # change dataset here
        return loss, len(X_test_data), {"accuracy": float(accuracy)}

#########################################################
#    Start the client                                   #
#########################################################

fl.client.start_client(server_address="192.168.117.3:8080", client=FLClient().to_client())
