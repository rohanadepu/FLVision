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


##########################################################################################
#                          Load Config for CICIOT 2023 Dataset                           #
##########################################################################################
def loadCICIOT(poisonedDataType=None):
    # ---                   Data Loading Settings                     --- #

    # Sample size for train and test datasets
    ciciot_train_sample_size = 25  # input: 3 samples for training
    ciciot_test_sample_size = 10  # input: 1 sample for testing

    # label classes 33+1 7+1 1+1
    ciciot_label_class = "1+1"

    if poisonedDataType == "LF33":
        DATASET_DIRECTORY = '/root/datasets/CICIOT2023_POISONEDLF33'

    elif poisonedDataType == "LF66":
        DATASET_DIRECTORY = '/root/datasets/CICIOT2023_POISONEDLF66'

    elif poisonedDataType == "FN33":
        DATASET_DIRECTORY = '/root/datasets/CICIOT2023_POISONEDFN33'

    elif poisonedDataType == "FN66":
        DATASET_DIRECTORY = '/root/datasets/CICIOT2023_POISONEDFN66'

    else:
        # directory of the stored data samples
        DATASET_DIRECTORY = '../../datasets/CICIOT2023'

    # ---    CICIOT Feature Mapping for numerical and categorical features       --- #

    num_cols = ['flow_duration', 'Header_Length', 'Rate', 'Srate', 'Drate', 'ack_count', 'syn_count', 'fin_count',
                'urg_count', 'rst_count', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
                'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
                ]

    cat_cols = [
        'Protocol Type', 'Duration', 'fin flag number', 'syn flag number', 'rst flag number', 'psh flag number',
        'ack flag number', 'ece flag number', 'cwr flag number', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
        'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC'
    ]

    # ---                   CICIOT irrelevant & relevant features mappings                    --- #

    irrelevant_features = ['Srate', 'ece_flag_number', 'rst_flag_number', 'ack_flag_number', 'cwr_flag_number',
                           'ack_count', 'syn_count', 'fin_count', 'rst_count', 'LLC', 'Min', 'Max', 'AVG', 'Std',
                           'Tot size', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight',
                           'flow_duration', 'Header_Length', 'urg_count', 'Tot sum'
                           ]

    # select the num cols that are relevant
    relevant_num_cols = [col for col in num_cols if col not in irrelevant_features]

    ## Debug ##
    # relevant_num_cols = [col for col in num_cols if col in relevant_features]
    ## EOF DEBUG ##

    # ---                   Label Mapping for 1+1 and 7+1                      --- #

    dict_7classes = {'DDoS-RSTFINFlood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS',
                     'DDoS-UDP_Flood': 'DDoS', 'DDoS-TCP_Flood': 'DDoS', 'DDoS-ICMP_Flood': 'DDoS',
                     'DDoS-SynonymousIP_Flood': 'DDoS', 'DDoS-ACK_Fragmentation': 'DDoS',
                     'DDoS-UDP_Fragmentation': 'DDoS',
                     'DDoS-ICMP_Fragmentation': 'DDoS', 'DDoS-SlowLoris': 'DDoS', 'DDoS-HTTP_Flood': 'DDoS',
                     'DoS-UDP_Flood': 'DoS', 'DoS-SYN_Flood': 'DoS', 'DoS-TCP_Flood': 'DoS', 'DoS-HTTP_Flood': 'DoS',
                     'Mirai-greeth_flood': 'Mirai', 'Mirai-greip_flood': 'Mirai', 'Mirai-udpplain': 'Mirai',
                     'Recon-PingSweep': 'Recon', 'Recon-OSScan': 'Recon', 'Recon-PortScan': 'Recon',
                     'VulnerabilityScan': 'Recon', 'Recon-HostDiscovery': 'Recon', 'DNS_Spoofing': 'Spoofing',
                     'MITM-ArpSpoofing': 'Spoofing', 'BenignTraffic': 'Benign', 'BrowserHijacking': 'Web',
                     'Backdoor_Malware': 'Web', 'XSS': 'Web', 'Uploading_Attack': 'Web', 'SqlInjection': 'Web',
                     'CommandInjection': 'Web', 'DictionaryBruteForce': 'BruteForce'
                     }

    dict_2classes = {'DDoS-RSTFINFlood': 'Attack', 'DDoS-PSHACK_Flood': 'Attack', 'DDoS-SYN_Flood': 'Attack',
                     'DDoS-UDP_Flood': 'Attack', 'DDoS-TCP_Flood': 'Attack', 'DDoS-ICMP_Flood': 'Attack',
                     'DDoS-SynonymousIP_Flood': 'Attack', 'DDoS-ACK_Fragmentation': 'Attack',
                     'DDoS-UDP_Fragmentation': 'Attack', 'DDoS-ICMP_Fragmentation': 'Attack',
                     'DDoS-SlowLoris': 'Attack',
                     'DDoS-HTTP_Flood': 'Attack', 'DoS-UDP_Flood': 'Attack', 'DoS-SYN_Flood': 'Attack',
                     'DoS-TCP_Flood': 'Attack', 'DoS-HTTP_Flood': 'Attack', 'Mirai-greeth_flood': 'Attack',
                     'Mirai-greip_flood': 'Attack', 'Mirai-udpplain': 'Attack', 'Recon-PingSweep': 'Attack',
                     'Recon-OSScan': 'Attack', 'Recon-PortScan': 'Attack', 'VulnerabilityScan': 'Attack',
                     'Recon-HostDiscovery': 'Attack', 'DNS_Spoofing': 'Attack', 'MITM-ArpSpoofing': 'Attack',
                     'BenignTraffic': 'Benign', 'BrowserHijacking': 'Attack', 'Backdoor_Malware': 'Attack',
                     'XSS': 'Attack',
                     'Uploading_Attack': 'Attack', 'SqlInjection': 'Attack', 'CommandInjection': 'Attack',
                     'DictionaryBruteForce': 'Attack'
                     }

    # ---      Functions    --- #

    def load_and_balance_data(file_path, label_class_dict, current_benign_size, benign_size_limit):
        # load data to dataframe
        data = pd.read_csv(file_path)

        # remap labels to new classification
        data['label'] = data['label'].map(label_class_dict)

        # Separate the classes
        attack_samples = data[data['label'] == 'Attack']
        benign_samples = data[data['label'] == 'Benign']

        # Limit the benign samples to avoid overloading
        remaining_benign_quota = benign_size_limit - current_benign_size
        if len(benign_samples) > remaining_benign_quota:
            benign_samples = benign_samples.sample(remaining_benign_quota, random_state=47)

        # Balance the samples
        min_samples = min(len(attack_samples), len(benign_samples))  # choosing the smallest number from both samples
        if min_samples > 0:  # if min sample exist sample out the samples
            attack_samples = attack_samples.sample(min_samples, random_state=47)
            benign_samples = benign_samples.sample(min_samples, random_state=47)

        # Bring both samples together
        balanced_data = pd.concat([attack_samples, benign_samples])

        return balanced_data, len(benign_samples)

    def reduce_attack_samples(data, attack_ratio):
        # sort samples
        attack_samples = data[data['label'] == 'Attack']
        benign_samples = data[data['label'] == 'Benign']

        # samples from pool of attack samples
        reduced_attack_samples = attack_samples.sample(frac=attack_ratio, random_state=47)

        # make new dataset from benign pool and reduced attack samples
        combined_data = pd.concat([benign_samples, reduced_attack_samples])

        return combined_data

    # ---     Load in two separate sets of file samples for the train and test datasets --- #

    print("\nLoading Network Traffic Data Files...")

    # List the files in the dataset
    csv_filepaths = [filename for filename in os.listdir(DATASET_DIRECTORY) if filename.endswith('.csv')]
    print(csv_filepaths)

    # Randomly select the specified number of files for training and testing
    if len(csv_filepaths) > (ciciot_train_sample_size + ciciot_test_sample_size):
        train_sample_files = random.sample(csv_filepaths,
                                           ciciot_train_sample_size)  # samples the file names from filepaths list
        remaining_files = [file for file in csv_filepaths if
                           file not in train_sample_files]  # takes the remaining files not from the training set
        test_sample_files = random.sample(remaining_files,
                                          ciciot_test_sample_size)  # samples from the remaining set of files
    else:
        # If there are not enough files, use all available files for training and testing
        train_sample_files = csv_filepaths[:ciciot_train_sample_size]
        test_sample_files = csv_filepaths[ciciot_train_sample_size:ciciot_train_sample_size + ciciot_test_sample_size]

    # Sort the file paths (optional)
    train_sample_files.sort()
    test_sample_files.sort()

    print("\nTraining Sets:\n", train_sample_files, "\n")
    print("\nTest Sets:\n", test_sample_files, "\n")

    # ---      Load the data from the sampled sets of files into train and test dataframes respectively    --- #

    # Train Dataframe
    ciciot_train_data = pd.DataFrame()
    train_normal_traffic_total_size = 0
    train_normal_traffic_size_limit = 110000

    print("\nLoading Training Data...")
    for data_set in train_sample_files:

        # Load Data from sampled files until enough benign traffic is loaded
        if train_normal_traffic_total_size >= train_normal_traffic_size_limit:
            break

        print(f"Training dataset sample {data_set} \n")

        # find the path for sample
        data_path = os.path.join(DATASET_DIRECTORY, data_set)

        # load the dataset, remap, then balance
        balanced_data, benign_count = load_and_balance_data(data_path, dict_2classes, train_normal_traffic_total_size,
                                                            train_normal_traffic_size_limit)
        train_normal_traffic_total_size += benign_count  # adding to quota count

        print(
            f"Benign Traffic Train Samples: {benign_count} | {train_normal_traffic_total_size} |LIMIT| {train_normal_traffic_size_limit}")

        # add to train dataset
        ciciot_train_data = pd.concat([ciciot_train_data, balanced_data])  # dataframe to manipulate

    # Test Dataframe
    ciciot_test_data = pd.DataFrame()
    test_normal_traffic_total_size = 0
    test_normal_traffic_size_limit = 44000
    attack_ratio = 0.1  # Adjust this ratio as needed to achieve desired imbalance

    print("\nLoading Testing Data...")
    for data_set in test_sample_files:

        # Load Data from sampled files until enough benign traffic is loaded
        if test_normal_traffic_total_size >= test_normal_traffic_size_limit:
            break

        print(f"Testing dataset sample {data_set} \n")

        # find the path for sample
        data_path = os.path.join(DATASET_DIRECTORY, data_set)

        # load the dataset, remap, then balance
        balanced_data, benign_count = load_and_balance_data(data_path, dict_2classes, test_normal_traffic_total_size,
                                                            test_normal_traffic_size_limit)

        test_normal_traffic_total_size += benign_count  # adding to quota count

        print(
            f"Benign Traffic Test Samples: {benign_count} | {test_normal_traffic_total_size} |LIMIT| {test_normal_traffic_size_limit}")

        # add to train dataset
        ciciot_test_data = pd.concat([ciciot_test_data, balanced_data])  # dataframe to manipulate

    # Reduce the number of attack samples in the test set
    ciciot_test_data = reduce_attack_samples(ciciot_test_data, attack_ratio)

    print("\nTrain & Test Attack Data Loaded (Attack Data already Combined)...")

    print("CICIOT Combined Data (Train):")
    print(ciciot_train_data.head())

    print("CICIOT Combined Data (Test):")
    print(ciciot_test_data.head())

    return ciciot_train_data, ciciot_test_data, irrelevant_features


###################################################################################
#               Load Config for IOTBOTNET 2023 Dataset                            #
###################################################################################
def loadIOTBOTNET(poisonedDataType=None):

    # ---                  Data loading Settings                  --- #
    # sample size to select for some attacks with multiple files; MAX is 3, MIN is 2
    sample_size = 1

    if poisonedDataType == "LF33":
        DATASET_DIRECTORY = '/root/datasets/IOTBOTNET2020_POISONEDLF33'

    elif poisonedDataType == "LF66":
        DATASET_DIRECTORY = '/root/datasets/IOTBOTNET2020_POISONEDLF66'

    elif poisonedDataType == "FN33":
        DATASET_DIRECTORY = '/root/datasets/IOTBOTNET2020_POISONEDFN33'

    elif poisonedDataType == "FN66":
        DATASET_DIRECTORY = '/root/datasets/IOTBOTNET2020_POISONEDFN66'

    else:
        DATASET_DIRECTORY = '/root/datasets/IOTBOTNET2020'

    # ---                   IOTBOTNET relevant features/attribute mappings                    --- #
    relevant_features_iotbotnet = [
        'Src_Port', 'Pkt_Size_Avg', 'Bwd_Pkts/s', 'Pkt_Len_Mean', 'Dst_Port', 'Bwd_IAT_Max', 'Flow_IAT_Mean',
        'ACK_Flag_Cnt', 'Flow_Duration', 'Flow_IAT_Max', 'Flow_Pkts/s', 'Fwd_Pkts/s', 'Bwd_IAT_Tot', 'Bwd_Header_Len',
        'Bwd_IAT_Mean', 'Bwd_Seg_Size_Avg'
    ]

    relevant_attributes_iotbotnet = [
        'Src_Port', 'Pkt_Size_Avg', 'Bwd_Pkts/s', 'Pkt_Len_Mean', 'Dst_Port', 'Bwd_IAT_Max', 'Flow_IAT_Mean',
        'ACK_Flag_Cnt', 'Flow_Duration', 'Flow_IAT_Max', 'Flow_Pkts/s', 'Fwd_Pkts/s', 'Bwd_IAT_Tot', 'Bwd_Header_Len',
        'Bwd_IAT_Mean', 'Bwd_Seg_Size_Avg', 'Label'
    ]

    # ---                   Functions                   --- #
    def load_files_from_directory(directory, file_extension=".csv", sample_size=None):
        # Check if the directory exists
        if not os.path.exists(directory):
            print(f"Directory '{directory}' does not exist.")
            return []

        dataframes = []
        all_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.endswith(file_extension):
                    file_path = os.path.join(root, file)
                    all_files.append(file_path)

        # Shuffle and sample files if sample_size is specified
        if sample_size is not None:
            random.shuffle(all_files)
            all_files = all_files[:sample_size]
            print("Sample(s) Selected:", all_files)

        for file_path in all_files:
            df = pd.read_csv(file_path)  # Modify this line if files are in a different format
            dataframes.append(df)
            print("Sample(s) Selected:", file_path)

        print("Data Loaded...")
        return dataframes

    # Function to split a DataFrame into train and test sets
    def split_train_test(dataframe, test_size=0.2):
        train_df, test_df = train_test_split(dataframe, test_size=test_size)
        print("Dataset Test Train Split...")
        return train_df, test_df

    # Function to combine subcategories into general classes
    def combine_general_attacks(ddos_dataframes, dos_dataframes, scan_dataframes, theft_dataframes):
        ddos_combined = pd.concat(ddos_dataframes, ignore_index=True)
        dos_combined = pd.concat(dos_dataframes, ignore_index=True)
        scan_combined = pd.concat(scan_dataframes, ignore_index=True)
        theft_combined = pd.concat(theft_dataframes, ignore_index=True)
        print("attacks combined...")
        return ddos_combined, dos_combined, scan_combined, theft_combined

    # Function to combine all dataframes into one
    def combine_all_attacks(dataframes):
        combined_df = pd.concat(dataframes, ignore_index=True)
        print("attacks combined...")
        return combined_df

    # Function to balance the dataset via undersampling
    def balance_data(dataframe, label_column='Label'):
        # Get the counts of each label
        label_counts = dataframe[label_column].value_counts()
        print("Label counts before balancing:", label_counts)

        # Determine the minimum count of samples across all labels
        min_count = label_counts.min()

        # Sample min_count number of samples from each label
        balanced_dataframes = []
        for label in label_counts.index:
            label_df = dataframe[dataframe[label_column] == label]
            balanced_label_df = label_df.sample(min_count, random_state=47)
            balanced_dataframes.append(balanced_label_df)

        # Concatenate the balanced dataframes
        balanced_dataframe = pd.concat(balanced_dataframes)
        balanced_dataframe = shuffle(balanced_dataframe, random_state=47)

        # Print the new counts of each label
        print("Label counts after balancing:", balanced_dataframe[label_column].value_counts())

        return balanced_dataframe

    def reduce_attack_samples(data, attack_ratio):
        # sort samples
        attack_samples = data[data['Label'] == 'Anomaly']
        benign_samples = data[data['Label'] == 'Normal']

        # sample the attack samples
        reduced_attack_samples = attack_samples.sample(frac=attack_ratio, random_state=47)

        # make new pool with benign and reduced attack pool
        combined_data = pd.concat([benign_samples, reduced_attack_samples])

        return combined_data

    # ---                   Load Each Attack Dataset                 --- #

    print("\nLoading Each IOT Network Attack Class...")
    print("Loading DDOS Data...")
    # Load DDoS UDP files
    ddos_udp_directory = DATASET_DIRECTORY + '/ddos/ddos_udp'
    ddos_udp_dataframes = load_files_from_directory(ddos_udp_directory, sample_size=sample_size)

    # Load DDoS TCP files
    ddos_tcp_directory = DATASET_DIRECTORY + '/ddos/ddos_tcp'
    ddos_tcp_dataframes = load_files_from_directory(ddos_tcp_directory, sample_size=sample_size)

    # Load DDoS HTTP files
    ddos_http_directory = DATASET_DIRECTORY + '/ddos/ddos_http'
    ddos_http_dataframes = load_files_from_directory(ddos_http_directory)

    print("Loading DOS Data...")
    # Load DoS UDP files
    dos_udp_directory = DATASET_DIRECTORY + '/dos/dos_udp'
    dos_udp_dataframes = load_files_from_directory(dos_udp_directory, sample_size=sample_size)

    # Load DDoS TCP files
    dos_tcp_directory = DATASET_DIRECTORY + '/dos/dos_tcp'
    dos_tcp_dataframes = load_files_from_directory(dos_tcp_directory, sample_size=sample_size)

    # Load DDoS HTTP files
    dos_http_directory = DATASET_DIRECTORY + '/dos/dos_http'
    dos_http_dataframes = load_files_from_directory(dos_http_directory)

    print("Loading SCAN Data...")
    # Load scan_os files
    scan_os_directory = DATASET_DIRECTORY + '/scan/os'
    scan_os_dataframes = load_files_from_directory(scan_os_directory)

    # Load scan_service files
    scan_service_directory = DATASET_DIRECTORY + '/scan/service'
    scan_service_dataframes = load_files_from_directory(scan_service_directory)

    print("Loading THEFT Data...")
    # Load theft_data_exfiltration files
    theft_data_exfiltration_directory = DATASET_DIRECTORY + '/theft/data_exfiltration'
    theft_data_exfiltration_dataframes = load_files_from_directory(theft_data_exfiltration_directory)

    # Load theft_keylogging files
    theft_keylogging_directory = DATASET_DIRECTORY + '/theft/keylogging'
    theft_keylogging_dataframes = load_files_from_directory(theft_keylogging_directory)

    print("Loading Finished...")

    # ---                   Combine all classes                    --- #
    print("\nCombining Attack Data...")

    # concatenate all dataframes
    ddos_udp_data = pd.concat(ddos_udp_dataframes, ignore_index=True)
    ddos_tcp_data = pd.concat(ddos_tcp_dataframes, ignore_index=True)
    ddos_http_data = pd.concat(ddos_http_dataframes, ignore_index=True)

    dos_udp_data = pd.concat(dos_udp_dataframes, ignore_index=True)
    dos_tcp_data = pd.concat(dos_tcp_dataframes, ignore_index=True)
    dos_http_data = pd.concat(dos_http_dataframes, ignore_index=True)

    scan_os_data = pd.concat(scan_os_dataframes, ignore_index=True)
    scan_service_data = pd.concat(scan_service_dataframes, ignore_index=True)

    theft_data_exfiltration_data = pd.concat(theft_data_exfiltration_dataframes, ignore_index=True)
    theft_keylogging_data = pd.concat(theft_keylogging_dataframes, ignore_index=True)

    # Combine subcategories into general classes
    ddos_combined, dos_combined, scan_combined, theft_combined = combine_general_attacks(
        [ddos_udp_data, ddos_tcp_data, ddos_http_data],
        [dos_udp_data, dos_tcp_data, dos_http_data],
        [scan_os_data, scan_service_data],
        [theft_data_exfiltration_data, theft_keylogging_data]
    )

    # Combine all attacks into one DataFrame
    all_attacks_combined = combine_all_attacks([
        ddos_combined, dos_combined, scan_combined, theft_combined
    ])

    print("Attack Data Loaded & Combined...")

    ################################################################################################################
    #                        First Steps of Preprocessing IOTBOTNET 2020 Dataset                             #
    ################################################################################################################

    # ---                   Cleaning                     --- #

    print("\nCleaning Dataset...")

    # Replace inf values with NaN and then drop them
    all_attacks_combined.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

    # Clean the dataset by dropping rows with missing values
    all_attacks_combined = all_attacks_combined.dropna()

    print("Nan and inf values Removed...")

    # ---                   Train Test Split                  --- #

    print("\nTrain Test Split...")

    # Split each combined DataFrame into train and test sets
    all_attacks_train, all_attacks_test = split_train_test(all_attacks_combined)

    print("IOTBOTNET Combined Data (Train):")
    print(all_attacks_train.head())

    print("IOTBOTNET Combined Data (Test):")
    print(all_attacks_test.head())

    # --- Provide proper repesentation of classes in the train and test datasets --- #

    print("\nBalance Training Dataset...")

    all_attacks_train = balance_data(all_attacks_train, label_column='Label')

    print("\nBalance Testing Dataset...")

    all_attacks_test = balance_data(all_attacks_test, label_column='Label')

    # ---                   Correcting test set representation                    --- #

    print("\nTurning attacks into anomalies...")
    print("Testing sample size:", all_attacks_test.shape[0])

    # Count the number of benign and attack samples in the test set
    benign_count = (all_attacks_test['Label'] == 'Normal').sum()
    attack_count = (all_attacks_test['Label'] != 'Normal').sum()
    print("Testing benign sample size:", benign_count)
    print("Testing attack sample size:", attack_count)

    print("Adjust Test Set Representation...")

    attack_ratio = 0.25  # Adjust this ratio as needed to achieve desired imbalance
    all_attacks_test = reduce_attack_samples(all_attacks_test, attack_ratio)

    print("Testing sample size:", all_attacks_test.shape[0])

    # Count the number of benign and attack samples in the test set
    benign_count = (all_attacks_test['Label'] == 'Normal').sum()
    attack_count = (all_attacks_test['Label'] != 'Normal').sum()
    print("Testing benign sample size:", benign_count)
    print("Testing attack sample size:", attack_count)

    print("Test Set Representation Adjusted...")

    return all_attacks_train, all_attacks_test, relevant_features_iotbotnet


################################################################################################################
#                          Preprocessing & Assigning the dataset                                  #
################################################################################################################

def preprocess_dataset(dataset_used, ciciot_train_data=None, ciciot_test_data=None, all_attacks_train=None,
                       all_attacks_test=None, irrelevant_features_ciciot=None, relevant_features_iotbotnet=None):
    print("\nSelecting Features...")
    if dataset_used == "CICIOT":
        # Drop the irrelevant features (Feature selection)
        ciciot_train_data = ciciot_train_data.drop(columns=irrelevant_features_ciciot)
        ciciot_test_data = ciciot_test_data.drop(columns=irrelevant_features_ciciot)

        # Shuffle data
        ciciot_train_data = shuffle(ciciot_train_data, random_state=47)
        ciciot_test_data = shuffle(ciciot_test_data, random_state=47)

        # initiate model training and test data
        train_data = ciciot_train_data
        test_data = ciciot_test_data

    elif dataset_used == "IOTBOTNET":
        # Select the relevant features in the dataset and labels
        all_attacks_train = all_attacks_train[relevant_features_iotbotnet + ['Label']]
        all_attacks_test = all_attacks_test[relevant_features_iotbotnet + ['Label']]

        # Shuffle data
        all_attacks_train = shuffle(all_attacks_train, random_state=47)
        all_attacks_test = shuffle(all_attacks_test, random_state=47)

        # initiate model training and test data
        train_data = all_attacks_train
        test_data = all_attacks_test

    print("Features Selected...")

    # --- Encoding ---
    print("\nEncoding...")

    # Print instances before encoding and scaling
    unique_labels = train_data['Label' if dataset_used == "IOTBOTNET" else 'label'].unique()
    for label in unique_labels:
        print(f"First instance of {label}:")
        print(train_data[train_data['Label' if dataset_used == "IOTBOTNET" else 'label'] == label].iloc[0])

    # Print the amount of instances for each label
    class_counts = train_data['Label' if dataset_used == "IOTBOTNET" else 'label'].value_counts()
    print(class_counts)

    # initiate encoder
    label_encoder = LabelEncoder()

    # fit and encode training data
    train_data['Label' if dataset_used == "IOTBOTNET" else 'label'] = label_encoder.fit_transform(
        train_data['Label' if dataset_used == "IOTBOTNET" else 'label'])

    # encode test data
    test_data['Label' if dataset_used == "IOTBOTNET" else 'label'] = label_encoder.transform(
        test_data['Label' if dataset_used == "IOTBOTNET" else 'label'])

    # showing the label mappings
    label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print("Label mappings:", label_mapping)

    print("Labels Encoded...")

    # --- Normalizing ---
    print("\nNormalizing...")

    # initiate scaler and colums to scale
    scaler = MinMaxScaler(feature_range=(0, 1))
    relevant_num_cols = train_data.columns.difference(['Label' if dataset_used == "IOTBOTNET" else 'label'])

    # fit scaler
    scaler.fit(train_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet])

    # Normalize data
    train_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet] = scaler.transform(train_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet])
    test_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet] = scaler.transform(test_data[relevant_num_cols if dataset_used=="CICIOT" else relevant_features_iotbotnet])

    print("Data Normalized...")

    # DEBUG DISPLAY
    print("\nTraining Data After Normalization:")
    print(train_data.head())
    print(train_data.shape)
    print("\nTest Data After Normalization:")
    print(test_data.head())
    print(test_data.shape)

    # --- Assigning and Splitting ---
    print("\nAssigning Data to Models...")

    # Train & Validation data
    X_data = train_data.drop(columns=['Label' if dataset_used == "IOTBOTNET" else 'label'])
    y_data = train_data['Label' if dataset_used == "IOTBOTNET" else 'label']

    # Split into Train & Validation data
    X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(X_data, y_data, test_size=0.2,
                                                                          random_state=47, stratify=y_data)
    # Test data
    X_test_data = test_data.drop(columns=['Label' if dataset_used == "IOTBOTNET" else 'label'])
    y_test_data = test_data['Label' if dataset_used == "IOTBOTNET" else 'label']

    print("Data Assigned...")
    return X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data

################################################################################################################
#                                       GAN Model Setup (Discriminator Training)                                       #
################################################################################################################

# Function for creating the discriminator model
def create_discriminator(input_dim):
    # Discriminator is designed to classify three classes:
    # - Normal (Benign) traffic
    # - Intrusive (Malicious) traffic
    # - Generated (Fake) traffic from the generator
    discriminator = tf.keras.Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(3, activation='softmax')  # 3 classes: Normal, Intrusive, Fake
    ])
    return discriminator


# Function for creating the generator model
def create_generator(input_dim, noise_dim):
    generator = tf.keras.Sequential([
        Dense(128, activation='relu', input_shape=(noise_dim,)),
        BatchNormalization(),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dense(input_dim, activation='sigmoid')  # Generate traffic features
    ])
    return generator

# loss based on correct classifications between normal, intrusive, and fake traffic
def discriminator_loss(real_normal_output, real_intrusive_output, fake_output):
    # Categorical cross-entropy loss for 3 classes: Normal, Intrusive, and Fake
    real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.ones_like(real_normal_output), real_normal_output)
    real_intrusive_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.zeros_like(real_intrusive_output), real_intrusive_output)
    fake_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.constant([-1], dtype=tf.float32), fake_output)
    total_loss = real_normal_loss + real_intrusive_loss + fake_loss
    return total_loss


# --- Class to handle discriminator training ---#
class DiscriminatorClient(fl.client.NumPyClient):
    def __init__(self, discriminator, generator, x_train, x_val, y_val, x_test, BATCH_SIZE, noise_dim, epochs, steps_per_epoch, dataset_used):
        self.discriminator = discriminator
        self.generator = generator # Generator is fixed during discriminator training
        self.x_train = x_train
        self.x_val = x_val  # Validation data
        self.y_val = y_val  # Validation labels
        self.x_test = x_test
        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.dataset_used = dataset_used

        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(self.BATCH_SIZE)

    def get_parameters(self, config):
        return self.discriminator.get_weights()

    def fit(self, parameters, config):
        self.discriminator.set_weights(parameters)

        # initiate optimizers
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        for epoch in range(self.epochs):
            for step, real_data in enumerate(self.x_train_ds.take(self.steps_per_epoch)):
                # Assume real_data contains both normal and intrusive traffic
                # Split the real_data into normal and intrusive samples
                normal_data = real_data[real_data['Label' if self.dataset_used == "IOTBOTNET" else 'label'] == 1]  # Real normal traffic
                intrusive_data = real_data[real_data['Label' if self.dataset_used == "IOTBOTNET" else 'label'] == 0]  # Real malicious traffic

                # Generate fake data using the generator
                noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
                generated_data = self.generator(noise, training=False)

                # captures the discriminator’s operations to compute the gradients for adjusting its weights based on how well it classified real vs. fake data.
                # using tape to track trainable variables during discriminator classification and loss calculations
                with tf.GradientTape() as tape:
                    # Discriminator outputs based on its classifications from inputted data in parameters
                    real_normal_output = self.discriminator(normal_data, training=True)
                    real_intrusive_output = self.discriminator(intrusive_data, training=True)
                    fake_output = self.discriminator(generated_data, training=True)

                    # Loss calculation for normal, intrusive, and fake data
                    loss = discriminator_loss(real_normal_output, real_intrusive_output, fake_output)

                # calculate the gradient based on the loss respect to the weights of the model
                gradients = tape.gradient(loss, self.discriminator.trainable_variables)

                # Update the model based on the gradient of the loss respect to the weights of the model
                optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

                if step % 100 == 0:
                    print(f"Epoch {epoch+1}, Step {step}, D Loss: {loss.numpy()}")

            # After each epoch, evaluate on the validation set
            val_disc_loss = self.evaluate_validation()
            print(f'Epoch {epoch+1}, Validation D Loss: {val_disc_loss}')

        return self.get_parameters(config={}), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.discriminator.set_weights(parameters)
        loss = 0
        for instances in self.x_test_ds:
            real_normal_output = self.discriminator(instances[instances['label'] == 1], training=False)
            real_intrusive_output = self.discriminator(instances[instances['label'] == 0], training=False)
            fake_output = self.discriminator(self.generator(tf.random.normal([self.BATCH_SIZE, self.noise_dim]), training=False), training=False)
            loss += discriminator_loss(real_normal_output, real_intrusive_output, fake_output)
        return float(loss.numpy()), len(self.x_test), {}

    # Function to evaluate the discriminator on validation data
    def evaluate_validation(self):
        # Generate fake samples using the generator
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)

        # Split validation data into normal and intrusive traffic
        normal_data = self.x_val[self.y_val == 1]  # Real normal traffic
        intrusive_data = self.x_val[self.y_val == 0]  # Real intrusive traffic

        # Pass real and fake data through the discriminator
        real_normal_output = self.discriminator(normal_data, training=False)
        real_intrusive_output = self.discriminator(intrusive_data, training=False)
        fake_output = self.discriminator(generated_samples, training=False)

        # Compute the discriminator loss using the real and fake outputs
        disc_loss = discriminator_loss(real_normal_output, real_intrusive_output, fake_output)

        return float(disc_loss.numpy())

################################################################################################################
#                                       Abstract                                       #
################################################################################################################


def main():
    print("\n ////////////////////////////// \n")
    print("Federated Learning Discriminator Client Training:", "\n")

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
    parser.add_argument('--pretrained_generator', type=str, help="Path to pretrained generator model (optional)",
                        default=None)
    parser.add_argument('--pretrained_discriminator', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)

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
    # Hyperparameters
    BATCH_SIZE = 256
    input_dim = X_train_data.shape[1] - 1  # Exclude label column
    noise_dim = 100
    epochs = 5
    steps_per_epoch = len(X_train_data) // BATCH_SIZE

    # Load or create the discriminator model
    if args.pretrained_discriminator:
        print(f"Loading pretrained discriminator from {args.pretrained_discriminator}")
        discriminator = tf.keras.models.load_model(args.pretrained_discriminator)
    else:
        print("No pretrained discriminator provided. Creating a new discriminator.")
        discriminator = create_discriminator(input_dim)

    # Load or create the generator model
    if args.pretrained_generator:
        print(f"Loading pretrained generator from {args.pretrained_generator}")
        generator = tf.keras.models.load_model(args.pretrained_generator)
    else:
        print("No pretrained generator provided. Creating a new generator.")
        generator = create_generator(input_dim, noise_dim)

    # initiate client with models, data, and parameters
    client = DiscriminatorClient(discriminator, generator, X_train_data, X_val_data, y_val_data, X_test_data, BATCH_SIZE
                                 , noise_dim, epochs, steps_per_epoch, dataset_used)

    # --- initiate federated training ---#
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

    # --- Save the trained discriminator model ---#
    discriminator.save("discriminator_model.h5")


if __name__ == "__main__":
    main()
