#########################################################
#    Imports / Env setup                                #
#########################################################

import os
import random
import time
import argparse

if 'TF_USE_LEGACY_KERAS' in os.environ:
    del os.environ['TF_USE_LEGACY_KERAS']

import flwr as fl
from flwr.client.mod import fixedclipping_mod
from flwr.client.mod.localdp_mod import LocalDpMod

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

print("\n ////////////////////////////// \n")
print("Federated Learning Training Demo:", "\n")

#########################################################
#    Script Parameters                               #
#########################################################

# --- Argument Parsing --- #
parser = argparse.ArgumentParser(description='Select dataset, model selection, and to enable DP respectively')
parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET", "CIFAR"], default="CICIOT", help='Datasets to use: CICIOT, IOTBOTNET, CIFAR')

parser.add_argument('--reg', action='store_true', help='Enable Regularization')  # tested
parser.add_argument('--dp', type=int, default=0, choices=[0, 1, 2], help='Differential Privacy 0: None, 1: TFP Engine, 2: Flwr Mod') # untested but working plz tune
parser.add_argument('--prune', action='store_true', help='Enable model pruning')  # broken
parser.add_argument('--adversarial', action='store_true', help='Enable model adversarial training') # bugged

parser.add_argument('--eS', action='store_true', help='Enable model early stop training') # callback unessary
parser.add_argument('--lrSched', action='store_true', help='Enable model lr scheduling training') # callback unessary
parser.add_argument('--mChkpnt', action='store_true', help='Enable model model checkpoint training') # store false irelevent

# i got bored and started planning to add the poisoned pipeline into this file cause its less work
parser.add_argument("--node", type=int, choices=[1,2,3,4,5,6], default=1, help="Client node number 1-6")
parser.add_argument("--pData", type=str, choices=["LF33", "LF66", "FN33", "FN66", None], default=None, help="Label Flip: LF33, LF66")

parser.add_argument("--evalLog", type=str, default=f"training_metrics", help="Name of the training log file")
parser.add_argument("--trainLog", type=str, default=f"evaluation_metrics", help="Name of the evaluation log file")

# init variables to handle arguments
args = parser.parse_args()

dataset_used = args.dataset
regularizationEnabled = args.reg

DP_enabled = args.dp

pruningEnabled = args.prune
adversarialTrainingEnabled = args.adversarial

earlyStopEnabled = args.eS
lrSchedRedEnabled = args.lrSched
modelCheckpointEnabled = args.mChkpnt

node = args.node
poisonedDataType = args.pData
evaluationLog = args.evalLog  # input into evaluation method if you want to input name
trainingLog = args.trainLog  # input into train method if you want to input name

# display selected arguments
print("Selected DATASET:", dataset_used, "\n")

if regularizationEnabled:
    print("Regularization Enabled", "\n")
else:
    print("Regularization Disabled", "\n")

if DP_enabled == 1:
    print("Differential Privacy Engine Enabled", "\n")
elif DP_enabled == 2:
    print("Differential Privacy Mod Enabled", "\n")
else:
    print("Differential Privacy Disabled", "\n")

if pruningEnabled:
    print("Pruning Enabled", "\n")
else:
    print("Pruning Disabled", "\n")

if adversarialTrainingEnabled:
    print("Adversarial Training Enabled", "\n")
else:
    print("Adversarial Training Disabled", "\n")

#########################################################
#    Loading Dataset For CICIOT 2023                    #
#########################################################

if dataset_used == "CICIOT":

    # ---                   Data Loading Settings                     --- #

    # Sample size for train and test datasets
    ciciot_train_sample_size = 20  # input: 3 samples for training
    ciciot_test_sample_size = 1  # input: 1 sample for testing

    # label classes 33+1 7+1 1+1
    ciciot_label_class = "1+1"

    if poisonedDataType == "LF33":
        DATASET_DIRECTORY = '../../datasets/CICIOT2023_POISONED33'

    elif poisonedDataType == "LF66":
        DATASET_DIRECTORY = '../../datasets/CICIOT2023_POISONED66'

    # elif poisonedDataType == "FN33":
    #     DATASET_DIRECTORY = '../../datasets/CICIOT2023_POISONED33'
    #
    # elif poisonedDataType == "FN66":
    #     DATASET_DIRECTORY = '../../datasets/CICIOT2023_POISONED33'

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

    # ---     Load in two separate sets of file samples for the train and test datasets --- #

    print("Loading Network Traffic Data Files...")

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

    print("Training Sets:\n", train_sample_files, "\n")
    print("Test Sets:\n", test_sample_files, "\n")

    # ---      Load the data from the sampled sets of files into train and test dataframes respectively    --- #

    # Train Dataframe
    ciciot_train_data = pd.DataFrame()
    normal_traffic_total_size = 0
    normal_traffic_size_limit = 100000

    print("Loading Training Data...")
    for data_set in train_sample_files:

        # Load Data from sampled files until enough benign traffic is loaded
        if normal_traffic_total_size >= normal_traffic_size_limit:
            break

        print(f"Training dataset sample {data_set} \n")

        # find the path for sample
        data_path = os.path.join(DATASET_DIRECTORY, data_set)

        # load the dataset, remap, then balance
        balanced_data, benign_count = load_and_balance_data(data_path, dict_2classes, normal_traffic_total_size,
                                                            normal_traffic_size_limit)
        normal_traffic_total_size += benign_count  # adding to quota count

        print(f"Benign Traffic Train Samples: {benign_count} | {normal_traffic_total_size} |LIMIT| {normal_traffic_size_limit}")

        # add to train dataset
        ciciot_train_data = pd.concat([ciciot_train_data, balanced_data])  # dataframe to manipulate

    # Test Dataframe
    ciciot_test_data = pd.DataFrame()

    print("Loading Testing Data...")
    for test_set in test_sample_files:

        # load the test dataset without balancing
        print(f"Testing dataset sample {test_set} out of {len(test_sample_files)} \n")

        # find the path for the sample
        data_path = os.path.join(DATASET_DIRECTORY, test_set)

        # read data
        test_data = pd.read_csv(data_path)

        # remap labels to new classification
        test_data['label'] = test_data['label'].map(dict_2classes)

        # add to test dataset
        ciciot_test_data = pd.concat([ciciot_test_data, test_data])

    print("Train & Test Attack Data Loaded (Attack Data already Combined)...")

    print("CICIOT Combined Data (Train):")
    print(ciciot_train_data.head())

    print("CICIOT Combined Data (Test):")
    print(ciciot_test_data.head())

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

    print("Features Selected...")

    # prints an instance of each class in training data
    print("Before Encoding and Scaling:")
    unique_labels = ciciot_train_data['label'].unique()
    for label in unique_labels:
        print(f"First instance of {label}:")
        print(ciciot_train_data[ciciot_train_data['label'] == label].iloc[0])

    # ---                   Encoding                     --- #

    print("Encoding...")

    # get each label in dataset
    unique_labels = ciciot_train_data['label'].nunique()

    # Print the number of unique labels
    print(f"There are {unique_labels} unique labels in the dataset.")

    # print the amount of instances for each label
    class_counts = ciciot_train_data['label'].value_counts()
    print(class_counts)

    # Encodes the labels
    label_encoder = LabelEncoder()  # Initialize the encoder
    ciciot_train_data['label'] = label_encoder.fit_transform(ciciot_train_data['label'])  # Fit and encode the training labels
    ciciot_test_data['label'] = label_encoder.transform(ciciot_test_data['label'])  # encode the test labels

    # Store label mappings
    label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print("Label mappings:", label_mapping)

    # Retrieve the numeric codes for classes
    class_codes = {label: label_encoder.transform([label])[0] for label in label_encoder.classes_}

    # Print specific instances
    print("Training Data After Encoding:")
    for label, code in class_codes.items():
        # Check if there are any instances of the current label
        if not ciciot_train_data[ciciot_train_data['label'] == code].empty:
            # Print the first instance of each class
            print(f"First instance of {label} (code {code}):")
            print(ciciot_train_data[ciciot_train_data['label'] == code].iloc[0])
        else:
            print(f"No instances found for {label} (code {code})")
    print(ciciot_train_data.head(), "\n")

    print("Labels Encoded...")

    # ---                    Normalizing                      --- #

    print("Normalizing...")

    # Setting up Scaler for Features normalization
    scaler = MinMaxScaler(feature_range=(0, 1))

    # train the scalar on train data features
    scaler.fit(ciciot_train_data[relevant_num_cols])

    # Save the Scaler for use in other files
    # joblib.dump(scaler, f'./MinMaxScaler.pkl')

    # Normalize the features in the train test dataframes
    ciciot_train_data[relevant_num_cols] = scaler.transform(ciciot_train_data[relevant_num_cols])
    ciciot_test_data[relevant_num_cols] = scaler.transform(ciciot_test_data[relevant_num_cols])

    # prove if the data is loaded properly
    print("Training Data After Normalization:")
    print(ciciot_train_data.head())
    print(ciciot_train_data.shape)

    # DEBUG prove if the data is loaded properly
    print("Test Data After Normalization:")
    print(ciciot_test_data.head())
    print(ciciot_test_data.shape)

    # ---                   Assigning / X y Split                   --- #

    # Feature / Label Split (X y split)
    X_train_data = ciciot_train_data.drop(columns=['label'])
    y_train_data = ciciot_train_data['label']

    X_test_data = ciciot_test_data.drop(columns=['label'])
    y_test_data = ciciot_test_data['label']

    # Print the shapes of the resulting splits
    print("X_train shape:", X_train_data.shape)
    print("y_train shape:", y_train_data.shape)

    print("X_test shape:", X_test_data.shape)
    print("y_test shape:", y_test_data.shape)

    # Get the sample size
    ciciot_train_df_size = X_train_data.shape[0]
    ciciot_test_df_size = X_test_data.shape[0]

    print("Training sample size:", ciciot_train_df_size)
    print("Testing sample size:", ciciot_test_df_size)

    print("Datasets Ready...")

#########################################################
#    Load Dataset For IOTBOTNET 2023                    #
#########################################################

if dataset_used == "IOTBOTNET":

    # ---                  Data loading Settings                  --- #

    # sample size to select for some attacks with multiple files; MAX is 3, MIN is 2
    sample_size = 1

    if poisonedDataType == "LF33":
        DATASET_DIRECTORY = '/root/dataset/IOTBOTNET2020_POISONED33'

    elif poisonedDataType == "LF66":
        DATASET_DIRECTORY = '/root/datasets/IOTBOTNET2020_POISONED66'

    # elif poisonedDataType == "FN33":
    #     DATASET_DIRECTORY = '/root/datasets/IOTBOTNET2020_POISONED66'
    #
    # elif poisonedDataType == "FN66":
    #     DATASET_DIRECTORY = '/root/datasets/IOTBOTNET2020_POISONED66'

    else:
        DATASET_DIRECTORY = '/root/attack/IOTBOTNET2020/IOTBOTNET2020/iotbotnet2020_archive'

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


    # ---                   Load Each Attack Dataset                 --- #

    print("Loading DDOS Data...")
    # Load DDoS UDP files
    ddos_udp_directory = DATASET_DIRECTORY + '/ddos/DDOS UDP'
    ddos_udp_dataframes = load_files_from_directory(ddos_udp_directory, sample_size=sample_size)

    # Load DDoS TCP files
    ddos_tcp_directory = DATASET_DIRECTORY + '/ddos/DDOS TCP'
    ddos_tcp_dataframes = load_files_from_directory(ddos_tcp_directory, sample_size=sample_size)

    # Load DDoS HTTP files
    ddos_http_directory = DATASET_DIRECTORY + '/ddos/DDOS HTTP'
    ddos_http_dataframes = load_files_from_directory(ddos_http_directory)

    print("Loading DOS Data...")
    # Load DoS UDP files
    dos_udp_directory = DATASET_DIRECTORY + '/dos/dos udp'
    dos_udp_dataframes = load_files_from_directory(dos_udp_directory, sample_size=sample_size)

    # Load DDoS TCP files
    dos_tcp_directory = DATASET_DIRECTORY + '/dos/dos tcp'
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
    print("Combining Attack Data...")

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

    #########################################################
    #    Process Dataset For IOTBOTNET 2020                 #
    #########################################################

    # ---                   Cleaning                     --- #

    print("Cleaning Dataset...")

    # Replace inf values with NaN and then drop them
    all_attacks_combined.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

    # Clean the dataset by dropping rows with missing values
    all_attacks_combined = all_attacks_combined.dropna()

    print("Nan and inf values Removed...")

    # ---                   Train Test Split                  --- #

    print("Train Test Split...")

    # Split each combined DataFrame into train and test sets
    all_attacks_train, all_attacks_test = split_train_test(all_attacks_combined)

    print("IOTBOTNET Combined Data (Train):")
    print(all_attacks_train.head())

    print("IOTBOTNET Combined Data (Test):")
    print(all_attacks_test.head())

    # --- Balance Training dataset --- #

    print("Balance Training Dataset...")

    all_attacks_train = balance_data(all_attacks_train, label_column='Label')

    print("Dataset Training Balanced...")

    # ---                   Feature Selection                --- #

    print("Selecting Features...")

    # Select the relevant features in the dataset and labels
    all_attacks_train = all_attacks_train[relevant_attributes_iotbotnet]
    all_attacks_test = all_attacks_test[relevant_attributes_iotbotnet]

    # Shuffle data
    all_attacks_train = shuffle(all_attacks_train, random_state=47)
    all_attacks_test = shuffle(all_attacks_test, random_state=47)

    print("Features Selected...")

    # prints an instance of each class in training data
    print("Before Encoding and Scaling:")
    unique_labels = all_attacks_train['Label'].unique()
    for label in unique_labels:
        print(f"First instance of {label}:")
        print(all_attacks_train[all_attacks_train['Label'] == label].iloc[0])

    # ---                   Encoding                      --- #

    print("Encoding...")

    # get each label in dataset
    unique_labels = all_attacks_train['Label'].nunique()

    # Print the number of unique labels
    print(f"There are {unique_labels} unique labels in the dataset.")

    # print the amount of instances for each label
    class_counts = all_attacks_train['Label'].value_counts()
    print(class_counts)

    # Encodes the labels
    label_encoder = LabelEncoder()  # Initialize the encoder
    all_attacks_train['Label'] = label_encoder.fit_transform(all_attacks_train['Label'])  # Fit and encode the training labels
    all_attacks_test['Label'] = label_encoder.transform(all_attacks_test['Label'])  # encode the test labels

    # Store label mappings
    label_mapping = {index: label for index, label in enumerate(label_encoder.classes_)}
    print("Label mappings:", label_mapping)

    # Retrieve the numeric codes for classes
    class_codes = {label: label_encoder.transform([label])[0] for label in label_encoder.classes_}

    # Print specific instances
    print("Training Data After Encoding:")
    for label, code in class_codes.items():
        # Check if there are any instances of the current label
        if not all_attacks_train[all_attacks_train['Label'] == code].empty:
            # Print the first instance of each class
            print(f"First instance of {label} (code {code}):")
            print(all_attacks_train[all_attacks_train['Label'] == code].iloc[0])
        else:
            print(f"No instances found for {label} (code {code})")
    print(all_attacks_train.head(), "\n")

    print("Labels Encoded...")

    # ---                   Normalizing                     --- #

    print("Normalizing...")

    # Setting up Scaler for Features normalization
    scaler = MinMaxScaler(feature_range=(0, 1))

    # Fit and normalize the training data
    scaler.fit(all_attacks_train[relevant_features_iotbotnet])

    # Save the Scaler for use in other files
    # joblib.dump(scaler, f'./MinMaxScaler.pkl')

    # Normalize the features in the train test dataframes
    all_attacks_train[relevant_features_iotbotnet] = scaler.transform(all_attacks_train[relevant_features_iotbotnet])
    all_attacks_test[relevant_features_iotbotnet] = scaler.transform(all_attacks_test[relevant_features_iotbotnet])

    # prove if the data is loaded properly
    print("Training Data After Normalization:")
    print(all_attacks_train.head())
    print(all_attacks_train.shape)

    # DEBUG prove if the data is loaded properly
    print("Test Data After Normalization:")
    print(all_attacks_test.head())
    print(all_attacks_test.shape)

    # ---                   Assigning                    --- #

    # Feature / Label Split (X y split)
    X_train_data = all_attacks_train.drop(columns=['Label'])
    y_train_data = all_attacks_train['Label']

    X_test_data = all_attacks_test.drop(columns=['Label'])
    y_test_data = all_attacks_test['Label']

    # Print the shapes of the resulting splits
    print("X_train shape:", X_train_data.shape)
    print("y_train shape:", y_train_data.shape)

    print("X_test shape:", X_test_data.shape)
    print("y_test shape:", y_test_data.shape)

    # Get the sample size
    iotbotnet_train_df_size = X_train_data.shape[0]
    iotbotnet_test_df_size = X_train_data.shape[0]

    print("Training sample size:", iotbotnet_train_df_size)
    print("Testing sample size:", iotbotnet_test_df_size)

    print("Datasets Ready...")

#########################################################
#    Model Initialization & Setup Default DEMO Cifar10  #
#########################################################

if dataset_used == "CIFAR":

    # Creates the train and test dataset from calling cifar10 in TF
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    input_dim = x_train.shape[1]  # feature size

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

noise_multiplier = 0.1  # Privacy param - noise budget: 0, none; 1, some noise; >1, more noise TFP ENGINE ONLY

l2_norm_clip = 1.0  # privacy param: 0.1 - 10: larger value, larger gradients, smaller value, more clipping TFP ENGINE ONly

batch_size = 64  # 32 - 128; try 64, 96, 128; maybe intervals of 16
num_microbatches = 1  # this is bugged keep at 1

learning_rate = 0.0001  # will be optimized
betas = [0.9, 0.999]  # Best to keep as is
l2_alpha = 0.01  # Increase if overfitting, decrease if underfitting

epochs = 10  # will be optimized
# steps_per_epoch = (len(X_train_data) // batch_size) // epochs  # dependant  # debug
steps_per_epoch = len(X_train_data) // batch_size   # dependant

input_dim = X_train_data.shape[1]  # dependant

if pruningEnabled:
    import tensorflow_model_optimization as tfmot

    # Define the pruning parameters
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,
            final_sparsity=0.5,
            begin_step=0,
            end_step=np.ceil(1.0 * len(X_train_data) / batch_size).astype(np.int32) * epochs
        )
    }

# set hyperparameters for callback
es_patience = 5
restor_best_w = True

l2lr_patience = 3
l2lr_factor = 0.1

metric_to_monitor_es = 'loss'
metric_to_monitor_l2lr = 'loss'
metric_to_monitor_mc = 'auc'

save_best_only = True
checkpoint_mode = "min"

print("\n /////////////////////////////////////////////// \n")
print("Hyperparameters:")
print("Input Dim (Feature Size):", input_dim)
print("Epochs:", epochs)
print("Batch Size:", batch_size)
print("MicroBatches", num_microbatches)
print(f"Steps per epoch (({len(X_train_data)} // {batch_size})):", steps_per_epoch)
# print(f"Steps per epoch (({len(X_train_data)} // {batch_size}) // {epochs}):", steps_per_epoch)  ## Debug
print("Betas:", betas)
print("Learning Rate:", learning_rate)
print("L2_alpha:", l2_alpha)
print("L2_norm clip:", l2_norm_clip)
print("Noise Multiplier:", noise_multiplier)

# ---             !!! INITIALIZE MODEL !!!                --- #

# ---                   CICIOT Models                   --- #

if dataset_used == "CICIOT":

    # --- Model Definition --- #
    if not pruningEnabled and regularizationEnabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(64, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.5),  # Dropout layer with 50% dropout rate
            Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.5),
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

    elif not pruningEnabled and not regularizationEnabled:
        # without regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),  # Dropout layer with 50% dropout rate
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(4, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

    elif pruningEnabled and regularizationEnabled:
        model = tf.keras.Sequential([
            tfmot.sparsity.keras.prune_low_magnitude(Dense(64, activation='relu', kernel_regularizer=l2(l2_alpha)), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(2, activation='relu', kernel_regularizer=l2(l2_alpha)), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_alpha)), **pruning_params),
        ])

    elif pruningEnabled and not regularizationEnabled:
        model = tf.keras.Sequential([
            tfmot.sparsity.keras.prune_low_magnitude(Dense(64, activation='relu'), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(32, activation='relu'), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(16, activation='relu'), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(8, activation='relu'), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(4, activation='relu'), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(2, activation='relu'), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(1, activation='sigmoid'), **pruning_params),
        ])

# ---                   IOTBOTNET Models                  --- #

if dataset_used == "IOTBOTNET":

    # --- Model Definitions --- #

    if not pruningEnabled and regularizationEnabled:
        # with regularization
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

    elif not pruningEnabled and not regularizationEnabled:
        # without regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),  # Dropout layer with 50% dropout rate
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(4, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(2, activation='relu'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='sigmoid')
        ])

    elif pruningEnabled and regularizationEnabled:
        model = tf.keras.Sequential([
            tfmot.sparsity.keras.prune_low_magnitude(Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(2, activation='relu', kernel_regularizer=l2(l2_alpha)), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_alpha)), **pruning_params),
        ])

    elif pruningEnabled and not regularizationEnabled:
        model = tf.keras.Sequential([
            tfmot.sparsity.keras.prune_low_magnitude(Dense(16, activation='relu'), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(8, activation='relu'), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(4, activation='relu'), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(2, activation='relu'), **pruning_params),
            BatchNormalization(),
            Dropout(0.5),
            tfmot.sparsity.keras.prune_low_magnitude(Dense(1, activation='sigmoid'), **pruning_params),
        ])

# ---         Differential Privacy Engine Model Compile              --- #

if DP_enabled == 1:
    print("\nIncluding DP into optimizer...\n")
    # Making Custom Optimizer Component with Differential Privacy
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    dp_optimizer = tfp.DPKerasAdamOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=num_microbatches,
        learning_rate=learning_rate
    )

    # compile model with custom dp optimizer
    model.compile(optimizer=dp_optimizer,
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy', Precision(), Recall(), AUC(), LogCosh()]
                  )

# ---              Normal Model Compile                        --- #

else:
    print("\nDefault optimizer...\n")
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer= optimizer,
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy', Precision(), Recall(), AUC(), LogCosh()])


# ---                   Callback components                   --- #

# init main call back functions list
callbackFunctions = []

# init callback functions based on inputs

if pruningEnabled:
    # Add pruning callbacks if enabled
    pruning_callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),
        tfmot.sparsity.keras.PruningSummaries(log_dir='/tmp/pruning_logs')
    ]

    # add it to main callback function list
    callbackFunctions += pruning_callbacks


if earlyStopEnabled:
    early_stopping = EarlyStopping(monitor=metric_to_monitor_es, patience=es_patience, restore_best_weights=restor_best_w)

    callbackFunctions += early_stopping


if lrSchedRedEnabled:
    lr_scheduler = ReduceLROnPlateau(monitor=metric_to_monitor_l2lr, factor=l2lr_factor, patience=l2lr_patience)

    callbackFunctions += lr_scheduler


if modelCheckpointEnabled:
    model_checkpoint = ModelCheckpoint(f'best_model_{model_name}.h5', save_best_only=save_best_only,
                                       monitor=metric_to_monitor_mc, mode=checkpoint_mode)

    # add to callback functions list being added during fitting
    callbackFunctions += model_checkpoint

# ---                   Model Analysis                   --- #

model.summary()

#########################################################
#    Adversarial Training Functions                     #
#########################################################

# Function to generate adversarial examples using FGSM
def create_adversarial_example(model, x, y, epsilon=0.01):
    # Ensure x is a tensor
    x = tf.convert_to_tensor(x, dtype=tf.float32)
    y = tf.convert_to_tensor(y, dtype=tf.float32)

    # Create a gradient tape context to record operations for automatic differentiation
    with tf.GradientTape() as tape:
        tape.watch(x)  # Adds the tensor x to the list of watched tensors, allowing its gradients to be computed
        prediction = model(x)  # Passes x through the model to get predictions
        loss = tf.keras.losses.binary_crossentropy(y, prediction)  # Computes the binary crossentropy loss between true labels y and predictions

    # Computes the gradient of the loss with respect to the input x
    gradient = tape.gradient(loss, x)

    # Creates the perturbation using the sign of the gradient and scales it by epsilon
    perturbation = epsilon * tf.sign(gradient)

    # Adds the perturbation to the original input to create the adversarial example
    adversarial_example = x + perturbation

    return adversarial_example


#########################################################
#    Metric Saving Functions                           #
#########################################################

def recordTraining(name, history, elapsed_time, roundCount):
    # f'training_metrics_{dataset_used}_optimized_{l2_norm_clip}_{noise_multiplier}.txt
    with open(name, 'a') as f:
        f.write(f"Node|{node}| Round: {roundCount}\n")
        f.write(f"Training Time Elapsed: {elapsed_time} seconds\n")
        for epoch in range(epochs):
            f.write(f"Epoch {epoch + 1}/{epochs}\n")
            for metric, values in history.history.items():
                f.write(f"{metric}: {values[epoch]}\n")
            f.write("\n")


def recordEvaluation(name, elapsed_time, evaluateCount, loss, accuracy, precision, recall, auc, logcosh):
    # f'evaluation_metrics_{dataset_used}_optimized_{l2_norm_clip}_{noise_multiplier}.txt
    with open(name, 'a') as f:
        f.write(f"Node|{node}| Round: {evaluateCount}\n")
        f.write(f"Evaluation Time Elapsed: {elapsed_time} seconds\n")
        f.write(f"Loss: {loss}\n")
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"AUC: {auc}\n")
        f.write(f"LogCosh: {logcosh}\n")
        f.write("\n")

#########################################################
#    Federated Learning Setup                           #
#########################################################


class FLClient(fl.client.NumPyClient):

    def __init__(self, model_used):
        self.model = model_used
        self.roundCount = 0
        self.evaluateCount = 0

    def get_parameters(self, config):
        return model.get_weights()

    def fit(self, parameters, config):
        # increment round count
        self.roundCount += 1

        # debug print
        print("Round:", self.roundCount, "\n")

        # Record start time
        start_time = time.time()

        self.model.set_weights(parameters)

        if adversarialTrainingEnabled:
            # Adversarial Training
            adv_X_train_data = np.array(
                [create_adversarial_example(self.model, x, y) for x, y in
                 zip(X_train_data.to_numpy(), y_train_data.to_numpy())]
            )

            # Convert adversarial examples to DataFrame
            adv_X_train_data = pd.DataFrame(adv_X_train_data, columns=X_train_data.columns)

            # Combine original and adversarial data
            combined_X_train_data = pd.concat([X_train_data, adv_X_train_data])
            combined_y_train_data = pd.concat([y_train_data, y_train_data])

            # train with the combined data
            history = model.fit(combined_X_train_data, combined_y_train_data, epochs=epochs, batch_size=batch_size,
                                steps_per_epoch=steps_per_epoch, callbacks=callbackFunctions)

        else:
            # Train Model
            history = self.model.fit(X_train_data, y_train_data, epochs=epochs, batch_size=batch_size,
                                steps_per_epoch=steps_per_epoch, callbacks=callbackFunctions)

        if pruningEnabled:
            # Strip the pruning wrappers and save the pruned model
            self.model = tfmot.sparsity.keras.strip_pruning(self.model)

        # Record end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Debugging: Print the shape of the loss
        loss_tensor = history.history['loss']
        print(f"Loss tensor shape: {tf.shape(loss_tensor)}")

        # Save metrics to file
        logName = trainingLog
        #logName = f'training_metrics_{dataset_used}_optimized_{l2_norm_clip}_{noise_multiplier}.txt'
        recordTraining(logName, history, elapsed_time, self.roundCount)

        return model.get_weights(), len(X_train_data), {}

    def evaluate(self, parameters, config):
        # increment evaluate count
        self.evaluateCount += 1

        # debug print
        print("Evaluate Round:", self.evaluateCount, "\n")

        # Record start time
        start_time = time.time()

        # set the weights given from server
        self.model.set_weights(parameters)

        # Test the model
        loss, accuracy, precision, recall, auc, logcosh = self.model.evaluate(X_test_data, y_test_data)

        # Record end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Save metrics to file
        logName = evaluationLog
        #logName = f'evaluation_metrics_{dataset_used}_optimized_{l2_norm_clip}_{noise_multiplier}.txt'
        recordEvaluation(logName, elapsed_time, self.evaluateCount, loss, accuracy, precision, recall, auc, logcosh)

        return loss, len(X_test_data), {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc,
                                        "LogCosh": logcosh}


#########################################################
#    Start the client                                   #
#########################################################

if DP_enabled == 2:

    # Create an instance of the LocalDpMod with the required params
    local_dp_obj = LocalDpMod(
        l2_norm_clip, 1.0, 1.0, 1e-6  # TUNE values for sensitivity, epsilon, and delta for DP 2
    )

    # Add local_dp_obj to the client-side mods
    app = fl.client.ClientApp(
        client_fn= FLClient,
        mods=[fixedclipping_mod, local_dp_obj],
    )
    app.start("192.168.129.2:8080")

else:
    fl.client.start_client(server_address="192.168.129.2:8080", client=FLClient(model).to_client())
