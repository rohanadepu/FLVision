#########################################################
#    Imports / Env setup                                #
#########################################################

import os
import random
# import time

import flwr as fl

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.losses import LogCosh
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import KFold
import tensorflow_privacy as tfp

import numpy as np
import pandas as pd

# import math
# import glob

# from IPython.display import clear_output
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns

# import pickle
import joblib

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
# from sklearn.metrics import accuracy_score, f1_score

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

dataset_used = "CICIOT"

print("DATASET BEING USED:", dataset_used, "\n")

#########################################################
#    Loading Dataset For CICIOT 2023                    #
#########################################################

if dataset_used == "CICIOT":

    # ---                   Settings                     --- #

    ciciot_train_sample_size = 2  # input: 3 samples for training
    ciciot_test_sample_size = 1  # input: 1 sample for testing

    # label classes 33+1 7+1 1+1
    ciciot_label_class = "1+1"

    DATASET_DIRECTORY = '../../trainingDataset/'

    # ---     Load in two separate sets of file samples for the train and test datasets --- #

    print("Loading Network Traffic Data...")

    # List the files in the dataset
    csv_filepaths = [filename for filename in os.listdir(DATASET_DIRECTORY) if filename.endswith('.csv')]
    print(csv_filepaths)

    # Randomly select the specified number of files for training and testing
    if len(csv_filepaths) > (ciciot_train_sample_size + ciciot_test_sample_size):
        train_sample_files = random.sample(csv_filepaths, ciciot_train_sample_size)  # samples the file names from filepaths list
        remaining_files = [file for file in csv_filepaths if file not in train_sample_files]  # takes the remaining files not from the training set
        test_sample_files = random.sample(remaining_files, ciciot_test_sample_size)  # samples from the remaining set of files
    else:
        # If there are not enough files, use all available files for training and testing
        train_sample_files = csv_filepaths[:ciciot_train_sample_size]
        test_sample_files = csv_filepaths[ciciot_train_sample_size:ciciot_train_sample_size + ciciot_test_sample_size]

    # Sort the file paths
    train_sample_files.sort()
    test_sample_files.sort()

    print("Training Sets:\n", train_sample_files, "\n")
    print("Test Sets:\n", test_sample_files)

    # ---                   Feature Mapping for numerical and categorical features       --- #

    # DEBUG
    # num_cols = [
    #     'flow_duration', 'Header_Length', 'Duration',
    #     'Rate', 'Srate', 'ack_count', 'syn_count',
    #     'fin_count', 'urg_count', 'rst_count', 'Tot sum',
    #     'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
    #     'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight',
    # ]

    # num_cols = ['Duration', 'Rate', 'Srate', 'ack_count', 'syn_count', 'fin_count', 'Tot size', 'IAT', 'Number',
    #             'Weight']

    # END OF DEBUG

    num_cols = ['flow_duration', 'Header_Length', 'Rate', 'Srate', 'Drate', 'ack_count', 'syn_count', 'fin_count',
                'urg_count', 'rst_count', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
                'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
                ]

    cat_cols = [
        'Protocol Type', 'Duration', 'fin flag number', 'syn flag number', 'rst flag number', 'psh flag number',
        'ack flag number', 'ece flag number', 'cwr flag number', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
        'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC'
    ]

    # ---                   irrelevant features and relevant features mappings                    --- #

    # DEBUG
    # irrelevant_features = ['ack_flag_number', 'ece_flag_number', 'cwr_flag_number', 'Magnitue', 'Radius',
    #                        'Covariance', 'Variance', 'flow_duration', 'Header_Length', 'urg_count', 'rst_count',
    #                        'Tot sum', 'Min', 'Max', 'AVG', 'Std']
    #
    # irrelevant_features = ['Srate', 'rst_flag_number', 'ack_flag_number', 'ack_count', 'syn_count', 'fin_count',
    #                        'rst_count', 'LLC', 'Max', 'AVG', 'Std', 'Tot size', 'Number', 'Magnitue', 'Radius',
    #                        'Covariance', 'Variance', 'Weight']
    # ---
    # relevant_features = ['Duration', 'IAT', 'urg_count', 'flow_duration', 'Min', 'Tot sum', 'Protocol Type',
    #                      'Header_Length', 'IPv', 'TCP', 'HTTPS', 'Rate', 'syn_flag_number', 'UDP', 'ICMP',
    #                      'fin_flag_number', 'psh_flag_number', 'HTTP', 'ece_flag_number', 'SMTP', 'IRC', 'DNS', 'SSH',
    #                      'Telnet', 'DHCP', 'ARP', 'Drate', 'cwr_flag_number', 'label']
    # END OF DEBUG

    irrelevant_features = ['Srate', 'ece_flag_number', 'rst_flag_number', 'ack_flag_number', 'cwr_flag_number',
                           'ack_count', 'syn_count', 'fin_count', 'rst_count', 'LLC', 'Min', 'Max', 'AVG', 'Std',
                           'Tot size', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight',
                           'flow_duration', 'Header_Length', 'urg_count', 'Tot sum'
                           ]  # being used

    # ---                   Label Mapping for 1+1 and 7+1                      --- #

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
                     'CommandInjection': 'Web', 'DictionaryBruteForce': 'BruteForce'
                     }

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
                     'DictionaryBruteForce': 'Attack'
                     }

    # ---      Functions    --- #

    def load_and_balance_data(file_path, label_class_dict, load_balanced=True):

        # Load the data
        print("Load Sample...")
        data = pd.read_csv(file_path)

        # Remap the labels to the right classification
        print(f"Remap Sample Labels into {ciciot_label_class} Categories...")
        data['label'] = data['label'].map(label_class_dict)

        if load_balanced == True:

            print("Balancing Sample Data...")

            # Separate the classes
            attack_samples = data[data['label'] == 'Attack']
            benign_samples = data[data['label'] == 'Benign']

            # Balance the classes
            min_samples = min(len(attack_samples), len(benign_samples))
            if min_samples > 0:
                attack_samples = attack_samples.sample(min_samples, random_state=47)
                benign_samples = benign_samples.sample(min_samples, random_state=47)

            balanced_data = pd.concat([attack_samples, benign_samples])

            print("Done with Sample")
            return balanced_data

        print("Done with Sample")
        return data

        # ---      Load the data from the sampled sets of files into train and test dataframes respectively    --- #

    # Train Dataframe
    ciciot_train_data = pd.DataFrame()
    for data_set in train_sample_files:
        print(f"Training dataset sample {data_set} out of {len(train_sample_files)} \n")
        data_path = os.path.join(DATASET_DIRECTORY, data_set)
        sampled_data = load_and_balance_data(data_path, dict_2classes if ciciot_label_class == "1+1" else dict_7classes
                                             , load_balanced=False)
        ciciot_train_data = pd.concat([ciciot_train_data, sampled_data])  # dataframe to manipulate

    # Test Dataframe
    ciciot_test_data = pd.DataFrame()
    for test_set in test_sample_files:
        print(f"Testing dataset sample {test_set} out of {len(test_sample_files)} \n")
        data_path = os.path.join(DATASET_DIRECTORY, test_set)
        sampled_data = load_and_balance_data(data_path, dict_2classes if ciciot_label_class == "1+1" else dict_7classes
                                             , load_balanced=False)
        ciciot_test_data = pd.concat([ciciot_test_data, sampled_data])  # dataframe to manipulate


#########################################################
#    Process Dataset For CICIOT 2023                    #
#########################################################

        # ---                   Feature Selection                --- #

        print("Selecting Features...")

        # Drop the irrelevant features (Feature selection)
        ciciot_train_data = ciciot_train_data.drop(columns=irrelevant_features)
        ciciot_test_data = ciciot_test_data.drop(columns=irrelevant_features)

        # Shuffle data
        ciciot_train_data = shuffle(ciciot_train_data, random_state=47)
        ciciot_test_data = shuffle(ciciot_test_data, random_state=47)

        # prints an instance of each class in training data
        print("Before Encoding and Scaling:")
        unique_labels = ciciot_train_data['label'].unique()
        for label in unique_labels:
            print(f"First instance of {label}:")
            print(ciciot_train_data[ciciot_train_data['label'] == label].iloc[0])

        # ---                   Encoding                     --- #

        print("Encoding...")

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
        print("Training Data After Encoding:")
        for label, code in class_codes.items():
            # Print the first instance of each class
            print(f"First instance of {label} (code {code}):")
            print(ciciot_train_data[ciciot_train_data['label'] == code].iloc[0])
        print(ciciot_train_data.head(), "\n")

        # ---                    Normalizing                      --- #

        print("Normalizing...")

        # Setting up Scaler for Features
        scaler = MinMaxScaler(feature_range=(0, 1))

        # select the num cols that are relevant
        relevant_num_cols = [col for col in num_cols if col not in irrelevant_features]
        # Debug
        #relevant_num_cols = [col for col in num_cols if col in relevant_features]

        # train the scalar on train data features
        scaler.fit(ciciot_train_data[relevant_num_cols])

        # Save the Scaler for use in other files
        # joblib.dump(scaler, f'./MinMaxScaler.pkl')

        # Scale the features in the train test dataframe
        ciciot_train_data[relevant_num_cols] = scaler.transform(ciciot_train_data[relevant_num_cols])
        ciciot_test_data[relevant_num_cols] = scaler.transform(ciciot_test_data[relevant_num_cols])

        # prove if the data is loaded properly
        print("Training Data After Normalization:")
        print(ciciot_train_data.head())
        print(ciciot_train_data.shape)

        # # DEBUG prove if the data is loaded properly
        # print("Test Data After Normalization:")
        # print(ciciot_test_data.head())
        # print(ciciot_test_data.shape)

        # ---                   Assigning / X y Split                   --- #

        # Feature / Label Split (X y split)
        X_train_data = ciciot_train_data.drop(columns=['label'])
        y_train_data = ciciot_train_data['label']

        X_test_data = ciciot_test_data.drop(columns=['label'])
        y_test_data = ciciot_test_data['label']

        # Print the shapes of the resulting splits
        print("X_train shape:", X_train_data.shape)
        print("y_train shape:", y_train_data.shape)

        # Get the sample size
        ciciot_df_size = X_train_data.shape[0]
        print("Sample size:", ciciot_df_size)

        print("Datasets Ready...")

#########################################################
#    Load Dataset For IOTBOTNET 2023                    #
#########################################################

if dataset_used == "IOTBOTNET":

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

        print("Files Loaded...")
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


    # ---                   Loading Specific Classes                    --- #

    # sample size to select for some attacks with multiple files; MAX is 3, MIN is 2
    sample_size = 1

    print("Loading DDOS Data..")
    # Load DDoS UDP files
    ddos_udp_directory = '/root/trainingDataset/iotbotnet2020/ddos/ddos_udp'
    ddos_udp_dataframes = load_files_from_directory(ddos_udp_directory, sample_size=sample_size)

    # Load DDoS TCP files
    ddos_tcp_directory = '/root/trainingDataset/iotbotnet2020/ddos/ddos_tcp'
    ddos_tcp_dataframes = load_files_from_directory(ddos_tcp_directory, sample_size=sample_size)

    # Load DDoS HTTP files
    ddos_http_directory = '/root/trainingDataset/iotbotnet2020/ddos/ddos_http'
    ddos_http_dataframes = load_files_from_directory(ddos_http_directory)

    print("Loading DOS Data..")
    # Load DoS UDP files
    # dos_udp_directory = './iotbotnet2020_archive/dos/dos_udp'
    # dos_udp_dataframes = load_files_from_directory(dos_udp_directory, sample_size=sample_size)

    # # Load DDoS TCP files
    # dos_tcp_directory = './iotbotnet2020_archive/dos/dos_tcp'
    # dos_tcp_dataframes = load_files_from_directory(dos_tcp_directory, sample_size=sample_size)
    #
    # # Load DDoS HTTP files
    # dos_http_directory = './iotbotnet2020_archive/dos/dos_http'
    # dos_http_dataframes = load_files_from_directory(dos_http_directory)

    print("Loading SCAN Data..")
    # Load scan_os files
    # scan_os_directory = '/root/trainingDataset/iotbotnet2020/scan/os'
    # scan_os_dataframes = load_files_from_directory(scan_os_directory, sample_size=sample_size)
    #
    # # Load scan_service files
    # scan_service_directory = './iotbotnet2020_archive/scan/service'
    # scan_service_dataframes = load_files_from_directory(scan_service_directory)

    print("Loading THEFT Data..")
    # # Load theft_data_exfiltration files
    # theft_data_exfiltration_directory = './iotbotnet2020_archive/theft/data_exfiltration'
    # theft_data_exfiltration_dataframes = load_files_from_directory(theft_data_exfiltration_directory)
    #
    # # Load theft_keylogging files
    # theft_keylogging_directory = './iotbotnet2020_archive/theft/keylogging'
    # theft_keylogging_dataframes = load_files_from_directory(theft_keylogging_directory)

    # ---                   Concatenations to combine all classes                    --- #
    print("Further Concatonation...")

    # Optionally, concatenate all dataframes if needed
    ddos_udp_data = pd.concat(ddos_udp_dataframes, ignore_index=True)
    ddos_tcp_data = pd.concat(ddos_tcp_dataframes, ignore_index=True)
    ddos_http_data = pd.concat(ddos_http_dataframes, ignore_index=True)
    # dos_udp_data = pd.concat(dos_udp_dataframes, ignore_index=True)
    # dos_tcp_data = pd.concat(dos_tcp_dataframes, ignore_index=True)
    # dos_http_data = pd.concat(dos_http_dataframes, ignore_index=True)
    # scan_os_data = pd.concat(scan_os_dataframes, ignore_index=True)
    # scan_service_data = pd.concat(scan_service_dataframes, ignore_index=True)
    # theft_data_exfiltration_data = pd.concat(theft_data_exfiltration_dataframes, ignore_index=True)
    # theft_keylogging_data = pd.concat(theft_keylogging_dataframes, ignore_index=True)

    # # Combine subcategories into general classes
    # ddos_combined, dos_combined, scan_combined, theft_combined = combine_general_attacks(
    #     [ddos_udp_data, ddos_tcp_data, ddos_http_data],
    #     [dos_udp_data, dos_tcp_data, dos_http_data],
    #     [scan_os_data, scan_service_data],
    #     [theft_data_exfiltration_data, theft_keylogging_data]
    # )
    #
    # # Combine all attacks into one DataFrame
    # all_attacks_combined = combine_all_attacks([
    #     ddos_combined, dos_combined, scan_combined, theft_combined
    # ])

    # Combine all attacks into one DataFrame
    all_attacks_combined = combine_all_attacks([
        ddos_udp_data, ddos_tcp_data, ddos_http_data
    ])

    # all_attacks_combined = scan_os_data

    # ---                   Train Test Split                  --- #

    print("Train Test Split...")

    # Split each combined DataFrame into train and test sets
    # ddos_train, ddos_test = split_train_test(ddos_combined)
    # dos_train, dos_test = split_train_test(dos_combined)
    # scan_train, scan_test = split_train_test(scan_combined)
    # theft_train, theft_test = split_train_test(theft_combined)
    all_attacks_train, all_attacks_test = split_train_test(all_attacks_combined)

    ## Debug ##

    # # Display the first few rows of the combined DataFrames
    # print("DDoS UDP Data:")
    # print(ddos_udp_data.head())
    #
    # print("DDoS TCP Data:")
    # print(ddos_tcp_data.head())
    #
    # print("DDoS HTTP Data:")
    # print(ddos_http_data.head())
    #
    # print("DoS UDP Data:")
    # print(dos_udp_data.head())
    #
    # print("DoS TCP Data:")
    # print(dos_tcp_data.head())
    #
    # print("DoS HTTP Data:")
    # print(dos_http_data.head())
    #
    # print("Scan OS Data:")
    # print(scan_os_data.head())
    #
    # print("Scan Service Data:")
    # print(scan_service_data.head())
    #
    # print("Theft Data Exfiltration Data:")
    # print(theft_data_exfiltration_data.head())
    #
    # print("Theft Keylogging Data:")
    # print(theft_keylogging_data.head())
    #
    # # Display the first few rows of each combined DataFrame
    # print("DDoS Combined Data (Train):")
    # print(ddos_train.head())
    #
    # print("DDoS Combined Data (Test):")
    # print(ddos_test.head())
    #
    # print("DoS Combined Data (Train):")
    # print(dos_train.head())
    #
    # print("DoS Combined Data (Test):")
    # print(dos_test.head())
    #
    # print("Scan Combined Data (Train):")
    # print(scan_train.head())
    #
    # print("Scan Combined Data (Test):")
    # print(scan_test.head())
    #
    # print("Theft Combined Data (Train):")
    # print(theft_train.head())
    #
    # print("Theft Combined Data (Test):")
    # print(theft_test.head())
    #
    # print("All Attacks Combined Data (Train):")
    # print(all_attacks_train.head())
    #
    # print("All Attacks Combined Data (Test):")
    # print(all_attacks_test.head())

    ## end of debug ##

#########################################################
#    Process Dataset For IOTBOTNET 2020                 #
#########################################################

    # ---                   Feature Selection                     --- #

    print("Selecting Features and X y Split...")

    relevant_features_iotbotnet = [
        'Src_Port', 'Pkt_Size_Avg', 'Bwd_Pkts/s', 'Pkt_Len_Mean', 'Dst_Port', 'Bwd_IAT_Max', 'Flow_IAT_Mean',
        'ACK_Flag_Cnt', 'Flow_Duration', 'Flow_IAT_Max', 'Flow_Pkts/s', 'Fwd_Pkts/s', 'Bwd_IAT_Tot', 'Bwd_Header_Len',
        'Bwd_IAT_Mean', 'Bwd_Seg_Size_Avg'
    ]

    # Split the dataset into features and labels
    X_train = all_attacks_train[relevant_features_iotbotnet]
    y_train = all_attacks_train['Label']

    X_test = all_attacks_test[relevant_features_iotbotnet]
    y_test = all_attacks_test['Label']
    print("Features Selected and Dataframes Split...")

    # ---                   Cleaning                     --- #

    print("Cleaning...")

    # Replace inf values with NaN and then drop them
    X_train.replace([float('inf'), -float('inf')], float('nan'), inplace=True)
    X_test.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

    # Clean the dataset by dropping rows with missing values
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]  # Ensure labels match the cleaned data

    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]  # Ensure labels match the cleaned data
    print("Nan and inf values Removed...")

    # ---                   Encoding                      --- #

    print("Encoding...")

    # Initialize the encoder
    label_encoder = LabelEncoder()

    # Fit and transform the training labels
    y_train_encoded = label_encoder.fit_transform(y_train)

    # Transform the test labels
    y_test_encoded = label_encoder.transform(y_test)

    # Optionally, shuffle the training data
    X_train, y_train_encoded = shuffle(X_train, y_train_encoded, random_state=47)

    # Create a DataFrame for y_train_encoded to align indices
    y_train_encoded_df = pd.Series(y_train_encoded, index=X_train.index)

    print("Labels Encoded...")

    # ---                   Normalizing                     --- #

    print("Normalizing...")
    # Initialize the scaler
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit and transform the training data
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the test data
    X_test_scaled = scaler.transform(X_test)

    # Convert scaled arrays back to DataFrames to maintain alignment
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=relevant_features_iotbotnet, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=relevant_features_iotbotnet, index=X_test.index)

    # ---                   Assigning                    --- #

    # Print an instance of each class in the new train data
    print("After Encoding and Normalization:")
    unique_labels = y_train_encoded_df.unique()
    for label in unique_labels:
        print(f"First instance of {label}:")
        print(X_train_scaled[y_train_encoded_df == label].iloc[0])

    # train
    X_train_data = X_train_scaled
    X_test_data = X_test_scaled

    # test
    y_train_data = y_train_encoded
    y_test_data = y_test_encoded

    # Print the shapes of the resulting splits
    print("X_train shape:", X_train_data.shape)
    print("y_train shape:", y_train_data.shape)

    # Get the sample size
    iotbotnet_df_size = X_train_data.shape[0]
    print("Sample size:", iotbotnet_df_size)

    print("Datasets Ready...")

#########################################################
#    Model Initialization & Setup Default DEMO Cifar10  #
#########################################################

if dataset_used == "CIFAR":

    # Creates the train and test dataset from calling cifar10 in TF
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    input_dim = X_train.shape[1]  # feature size

    #### DEMO MODEL ######
    model = tf.keras.applications.MobileNetV2((input_dim), classes=2, weights=None)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=[tf.keras.metrics.BinaryAccuracy(), Precision(), Recall(), AUC()]
                  )

#########################################################
#    Model Initialization & Setup                    #
#########################################################

# ---                  Hyper Parameters                  --- #

model_name = dataset_used  # name for file

noise_multiplier = 1.1  # Privacy param - noise budget:

l2_norm_clip = 1.0  # privacy param:

batch_size = 32  # 32 - 128; try 64, 96, 128; maybe intervals of 16
num_microbatches = 1  # this is bugged keep at 1

learning_rate = 0.001  # will be optimized
betas = [0.9, 0.999]  # Best to keep as is
l2_alpha = 0.01  # Increase if overfitting, decrease if underfitting

epochs = 5  # will be optimized
steps_per_epoch = (len(X_train_data) // batch_size) // epochs  # dependant

input_dim = X_train_data.shape[1]  # dependant

print("///////////////////////////////////////////////")
print("HyperParameters:")
print("Input Dim:", input_dim)
print("Epochs:", epochs)
print("Batch Size:", input_dim)
print("MicroBatches", num_microbatches)
print(f"Steps per epoch (({len(X_train_data)} // {batch_size}) // {epochs}):", steps_per_epoch)
print("Betas:", betas)
print("Learning Rate:", learning_rate)
print("L2_alpha:", l2_alpha)
print("L2_norm clip:", l2_norm_clip)
print("Noise Multiplier:", noise_multiplier)

# ---                   CICIOT Model                   --- #

if dataset_used == "CICIOT":
    print("model for ciciot dataset")

    # --- Model Definition --- #
    model_selection = 3

    if model_selection == 1:
        print("Picked 1st model")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.5),  # Dropout layer with 50% dropout rate
            Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

    if model_selection == 2:
        print("Picked 2nd model")
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.5),  # Dropout layer with 50% dropout rate
            Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])


        if model_selection == 3:
            print("Picked 3rd model")
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),  # Dropout layer with 50% dropout rate
                Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(2, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])

        if model_selection == 4:
            print("Picked 4th model")

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                Dense(28, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),  # Dropout layer with 50% dropout rate
                Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(2, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])

        if model_selection == 5:
            print("Picked 5th model")

            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                Dense(21, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),  # Dropout layer with 50% dropout rate
                Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(2, activation='relu', kernel_regularizer=l2(l2_alpha)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(1, activation='sigmoid')
            ])

# ---                   IOTBOTNET Model                  --- #

if dataset_used == "IOTBOTNET":

    # --- Model Definition --- #
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.5),  # Dropout layer with 50% dropout rate
        Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(2, activation='relu', kernel_regularizer=l2(l2_alpha)),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

# ---                   Differential Privacy                   --- #

# Making Custom Optimizer Component with Differential Privacy
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
dp_optimizer = tfp.DPKerasAdamOptimizer(
    l2_norm_clip=l2_norm_clip,
    noise_multiplier=noise_multiplier,
    num_microbatches=num_microbatches,
    learning_rate=learning_rate
)

# ---                   Model Compile                    --- #

model.compile(optimizer=dp_optimizer,
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy', Precision(), Recall(), AUC(), LogCosh()]
              )

# ---                   Callback components                   --- #

# set hyperparameters for callback
es_patience = 5
restor_best_w = True

l2lr_patience = 3
l2lr_factor = 0.1

metric_to_monitor = 'auc'

save_best_only = True
checkpoint_mode = "min"

# early_stopping = EarlyStopping(monitor=metric_to_monitor, patience=es_patience, restore_best_weights=restor_best_w)
# lr_scheduler = ReduceLROnPlateau(monitor=metric_to_monitor, factor=l2lr_factor, patience=l2lr_patience)
model_checkpoint = ModelCheckpoint(f'best_model_{model_name}.h5', save_best_only=save_best_only,
                                   monitor=metric_to_monitor, mode=checkpoint_mode)

# ---                   Model Analysis                   --- #

model.summary()

#########################################################
#    Federated Learning Setup                           #
#########################################################


class FLClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)

        # Train Model
        history = model.fit(X_train_data, y_train_data, epochs=epochs, batch_size=batch_size, steps_per_epoch=steps_per_epoch,
                            callbacks=[model_checkpoint])

        # Debugging: Print the shape of the loss
        loss_tensor = history.history['loss']
        print(f"Loss tensor shape: {tf.shape(loss_tensor)}")

        return model.get_weights(), len(X_train_data), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)

        # Test the model
        loss, accuracy, precision, recall, auc, LogCosh = model.evaluate(X_test_data, y_test_data)
        return loss, len(X_test_data), {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc,
                                        "LogCosh": LogCosh
                                        }

#########################################################
#    Start the client                                   #
#########################################################


fl.client.start_client(server_address="192.168.117.3:8080", client=FLClient().to_client())
