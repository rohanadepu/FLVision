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

import numpy as np
import pandas as pd

# import math
# import glob

# from tqdm import tqdm
# import matplotlib.pyplot as plt
# import seaborn as sns

# import pickle
# import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer, LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("\n ////////////////////////////// \n")
print("Federated Learning Training Demo:", "\n")

#########################################################
#    Script Parameters                               #
#########################################################

# Generate a static timestamp at the start of the script
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- Argument Parsing --- #
parser = argparse.ArgumentParser(description='Select dataset, model selection, and to enable DP respectively')
parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET"], default="CICIOT", help='Datasets to use: CICIOT, IOTBOTNET, CIFAR')

parser.add_argument("--node", type=int, choices=[1,2,3,4,5,6], default=1, help="Client node number 1-6")
parser.add_argument("--fixedServer", type=int, choices=[1,2,3,4], default=1, help="Fixed Server node number 1-4")

parser.add_argument("--pData", type=str, choices=["LF33", "LF66", "FN33", "FN66", None], default=None, help="Label Flip: LF33, LF66")

parser.add_argument('--reg', action='store_true', help='Enable Regularization')  # tested
parser.add_argument('--dp', action='store_true', help='Enable Differential Privacy with TFP') # untested but working plz tune
parser.add_argument('--adversarial', action='store_true', help='Enable model adversarial training with gradients') # bugged

parser.add_argument('--eS', action='store_true', help='Enable model early stop training') # callback unessary
parser.add_argument('--lrSched', action='store_true', help='Enable model lr scheduling training') # callback unessary
parser.add_argument('--mChkpnt', action='store_true', help='Enable model model checkpoint training') # store false irelevent

parser.add_argument("--evalLog", type=str, default=f"evaluation_metrics_{timestamp}.txt", help="Name of the evaluation log file")
parser.add_argument("--trainLog", type=str, default=f"training_metrics_{timestamp}.txt", help="Name of the training log file")

# init variables to handle arguments
args = parser.parse_args()

dataset_used = args.dataset

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
    import tensorflow_privacy as tfp
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

#########################################################
#    Loading Dataset For CICIOT 2023                    #
#########################################################

if dataset_used == "CICIOT":

    # ---                   Data Loading Settings                     --- #

    # Sample size for train and test datasets
    ciciot_train_sample_size = 50  # input: 3 samples for training
    ciciot_test_sample_size = 10  # input: 1 sample for testing

    # label classes 33+1 7+1 1+1
    ciciot_label_class = "1+1"

    # keep in mind evaluate dataset is being grab from same pool of training when unpoisoned
    DATASET_DIRECTORY_EVALUATE = '/root/datasets/CICIOT2023'

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
        DATASET_DIRECTORY = '/root/datasets/CICIOT2023'

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
        attack_samples = data[data['label'] == 'Attack']
        benign_samples = data[data['label'] == 'Benign']

        reduced_attack_samples = attack_samples.sample(frac=attack_ratio, random_state=47)
        combined_data = pd.concat([benign_samples, reduced_attack_samples])

        return combined_data

    # ---     Load in two separate sets of file samples for the train and test datasets --- #

    print("\nLoading Network Traffic Data Files...")

    if not poisonedDataType:
        # List the files in the dataset
        print("\nDATA IS UNCOMPROMISED...")

    # this is grabbing the file names of the samples (this should be the same across all ciciot dataset dir
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
        data_path = os.path.join(DATASET_DIRECTORY, data_set)  # this will grab the data from the given pool

        # load the dataset, remap, then balance
        balanced_data, benign_count = load_and_balance_data(data_path, dict_2classes, train_normal_traffic_total_size,
                                                            train_normal_traffic_size_limit)
        train_normal_traffic_total_size += benign_count  # adding to quota count

        print(f"Benign Traffic Train Samples: {benign_count} | {train_normal_traffic_total_size} |LIMIT| {train_normal_traffic_size_limit}")

        # add to train dataset
        ciciot_train_data = pd.concat([ciciot_train_data, balanced_data])  # dataframe to manipulate

    # Test Dataframe
    ciciot_test_data = pd.DataFrame()
    test_normal_traffic_total_size = 0
    test_normal_traffic_size_limit = 44000
    attack_ratio = 0.1  # Adjust this ratio as needed to achieve desired imbalance

    # load balance samples and reduce attack samples
    print("\nLoading Testing Data...")
    for data_set in test_sample_files:

        # Load Data from sampled files until enough benign traffic is loaded
        if test_normal_traffic_total_size >= test_normal_traffic_size_limit:
            break

        print(f"Testing dataset sample {data_set} \n")

        # find the path for sample
        data_path = os.path.join(DATASET_DIRECTORY_EVALUATE, data_set)  # ensures that the pool of test data isnt from the compromised pool (its the same dir as unpoisoned)

        # load the dataset, remap, then balance
        balanced_data, benign_count = load_and_balance_data(data_path, dict_2classes, test_normal_traffic_total_size,
                                                            test_normal_traffic_size_limit)

        test_normal_traffic_total_size += benign_count  # adding to quota count

        print(f"Benign Traffic Test Samples: {benign_count} | {test_normal_traffic_total_size} |LIMIT| {test_normal_traffic_size_limit}")

        # add to train dataset
        ciciot_test_data = pd.concat([ciciot_test_data, balanced_data])  # dataframe to manipulate

    # Reduce the number of attack samples in the test set
    ciciot_test_data = reduce_attack_samples(ciciot_test_data, attack_ratio)

    print("\nTrain & Test Attack Data Loaded (Attack Data already Combined)...")

    print("CICIOT Combined Data (Train):")
    print(ciciot_train_data.head())

    print("CICIOT Combined Data (Test):")
    print(ciciot_test_data.head())

#########################################################
#    Process Dataset For CICIOT 2023                    #
#########################################################

# ---                   Feature Selection                --- #

    print("\nSelecting Features...")

    # Drop the irrelevant features (Feature selection)
    ciciot_train_data = ciciot_train_data.drop(columns=irrelevant_features)
    ciciot_test_data = ciciot_test_data.drop(columns=irrelevant_features)

    # Shuffle data
    ciciot_train_data = shuffle(ciciot_train_data, random_state=47)
    ciciot_test_data = shuffle(ciciot_test_data, random_state=47)

    print("Features Selected...")

    # prints an instance of each class in training data
    print("\nBefore Encoding and Scaling:")
    unique_labels = ciciot_train_data['label'].unique()
    for label in unique_labels:
        print(f"First instance of {label}:")
        print(ciciot_train_data[ciciot_train_data['label'] == label].iloc[0])

    # ---                   Encoding                     --- #

    print("\nEncoding...")

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

    print("\nNormalizing...")

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

    print("\nAssign dataset to model...")
    # Feature / Label Split (X y split)
    X_data = ciciot_train_data.drop(columns=['label'])
    y_data = ciciot_train_data['label']

    # Split the data into training and validation sets
    X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(X_data, y_data, test_size=0.2,
                                                                          random_state=47, stratify=y_data)

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

    DATASET_DIRECTORY_EVALUATE = '/root/datasets/IOTBOTNET2020'

    if poisonedDataType == "LF33":
        print("Training Dataset Compromised!!!")
        DATASET_DIRECTORY = '/root/datasets/IOTBOTNET2020_POISONEDLF33'
    
    elif poisonedDataType == "LF66":
        print("Training Dataset Compromised!!!")
        DATASET_DIRECTORY = '/root/datasets/IOTBOTNET2020_POISONEDLF66'
    
    elif poisonedDataType == "FN33":
        print("Training Dataset Compromised!!!")
        DATASET_DIRECTORY = '/root/datasets/IOTBOTNET2020_POISONEDFN33'
    
    elif poisonedDataType == "FN66":
        print("Training Dataset Compromised!!!")
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
    def balance_data(dataframe, label_column='Label', benign_limit=None):
        # Get the counts of each label
        label_counts = dataframe[label_column].value_counts()
        print("Label counts before balancing:", label_counts)

        # Determine the minimum count of samples across all anomaly labels
        anomally_count = label_counts.drop('Normal').min()

        if benign_limit is not None:
            anomally_count = min(anomally_count, benign_limit)

        # Sample min_count number of samples from each label
        balanced_dataframes = []
        for label in label_counts.index:
            label_df = dataframe[dataframe[label_column] == label]
            if label == 'Normal' and benign_limit is not None:
                balanced_label_df = label_df.sample(min(benign_limit, len(label_df)), random_state=47)
            else:
                balanced_label_df = label_df.sample(min(anomally_count, len(label_df)), random_state=47)
            balanced_dataframes.append(balanced_label_df)

        # Concatenate the balanced dataframes
        balanced_dataframe = pd.concat(balanced_dataframes)
        balanced_dataframe = shuffle(balanced_dataframe, random_state=47)

        # Print the new counts of each label
        print("Label counts after balancing:", balanced_dataframe[label_column].value_counts())

        return balanced_dataframe


    def reduce_attack_samples(data, attack_ratio):
        attack_samples = data[data['Label'] == 'Anomaly']
        benign_samples = data[data['Label'] == 'Normal']

        reduced_attack_samples = attack_samples.sample(frac=attack_ratio, random_state=47)
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

    # ---                   ENSURE EVALUATION SET IS UNCOMPROMISED                   --- #
    if poisonedDataType:
        print("\n Loading uncompromised Attack Data for evaluation...")

        print("Loading DDOS Data...")
        # Load DDoS UDP files
        ddos_udp_directory = DATASET_DIRECTORY_EVALUATE + '/ddos/ddos_udp'
        ddos_udp_dataframes = load_files_from_directory(ddos_udp_directory, sample_size=sample_size)

        # Load DDoS TCP files
        ddos_tcp_directory = DATASET_DIRECTORY_EVALUATE + '/ddos/ddos_tcp'
        ddos_tcp_dataframes = load_files_from_directory(ddos_tcp_directory, sample_size=sample_size)

        # Load DDoS HTTP files
        ddos_http_directory = DATASET_DIRECTORY_EVALUATE + '/ddos/ddos_http'
        ddos_http_dataframes = load_files_from_directory(ddos_http_directory)

        print("Loading DOS Data...")
        # Load DoS UDP files
        dos_udp_directory = DATASET_DIRECTORY_EVALUATE + '/dos/dos_udp'
        dos_udp_dataframes = load_files_from_directory(dos_udp_directory, sample_size=sample_size)

        # Load DDoS TCP files
        dos_tcp_directory = DATASET_DIRECTORY_EVALUATE + '/dos/dos_tcp'
        dos_tcp_dataframes = load_files_from_directory(dos_tcp_directory, sample_size=sample_size)

        # Load DDoS HTTP files
        dos_http_directory = DATASET_DIRECTORY_EVALUATE + '/dos/dos_http'
        dos_http_dataframes = load_files_from_directory(dos_http_directory)

        print("Loading SCAN Data...")
        # Load scan_os files
        scan_os_directory = DATASET_DIRECTORY_EVALUATE + '/scan/os'
        scan_os_dataframes = load_files_from_directory(scan_os_directory)

        # Load scan_service files
        scan_service_directory = DATASET_DIRECTORY_EVALUATE + '/scan/service'
        scan_service_dataframes = load_files_from_directory(scan_service_directory)

        print("Loading THEFT Data...")
        # Load theft_data_exfiltration files
        theft_data_exfiltration_directory = DATASET_DIRECTORY_EVALUATE + '/theft/data_exfiltration'
        theft_data_exfiltration_dataframes = load_files_from_directory(theft_data_exfiltration_directory)

        # Load theft_keylogging files
        theft_keylogging_directory = DATASET_DIRECTORY_EVALUATE + '/theft/keylogging'
        theft_keylogging_dataframes = load_files_from_directory(theft_keylogging_directory)

        # --- Combine all classes
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

        # Combine all attacks into one DataFrame for !!! evaluation set !!!
        all_attacks_combined_evaluate = combine_all_attacks([
            ddos_combined, dos_combined, scan_combined, theft_combined
        ])

        print("Attack Data Loaded & Combined...")

    #########################################################
    #    Process Dataset For IOTBOTNET 2020                 #
    #########################################################

    # ---                   Cleaning                     --- #

    print("\nCleaning Dataset...")

    # Replace inf values with NaN and then drop them
    all_attacks_combined.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

    # Clean the dataset by dropping rows with missing values
    all_attacks_combined = all_attacks_combined.dropna()

    if poisonedDataType:
        print("\nCleaning uncompromised evaluation Dataset...")

        all_attacks_combined_evaluate.replace([float('inf'), -float('inf')], float('nan'), inplace=True)

        # Clean the dataset by dropping rows with missing values
        all_attacks_combined_evaluate = all_attacks_combined.dropna()

    print("Nan and inf values Removed...")

    # ---                   Train Test Split                  --- #

    print("\nTrain Test Split...")

    # Split each combined DataFrame into train and test sets
    all_attacks_train, all_attacks_test = split_train_test(all_attacks_combined)

    print("IOTBOTNET Combined Data (Train):")
    print(all_attacks_train.head())

    if poisonedDataType:  # overwrite compromised test data with uncompromised version and continue down pipeline as usual
        all_attacks_test = all_attacks_combined_evaluate

    print("IOTBOTNET Combined Data (Test):")
    print(all_attacks_test.head())

    # --- Provide proper repesentation of classes in the train and test datasets --- #

    print("\nBalance Training Dataset...")

    all_attacks_train = balance_data(all_attacks_train, label_column='Label', benign_limit=110000)

    print("\nBalance Testing Dataset...")

    all_attacks_test = balance_data(all_attacks_test, label_column='Label', benign_limit=44000)

    # ---                   Correcting test set representation                    --- #

    print("\nTurning attacks into anomalies...")
    print("Testing sample size:", all_attacks_test.shape[0])

    # Count the number of benign and attack samples in the test set
    benign_count = (all_attacks_test['Label'] == 'Normal').sum()
    attack_count = (all_attacks_test['Label'] != 'Normal').sum()
    print("Testing benign sample size:", benign_count)
    print("Testing attack sample size:", attack_count)

    print("Adjust Test Set Representation...")

    attack_ratio = 0.1  # Adjust this ratio as needed to achieve desired imbalance
    all_attacks_test = reduce_attack_samples(all_attacks_test, attack_ratio)

    print("Testing sample size:", all_attacks_test.shape[0])

    # Count the number of benign and attack samples in the test set
    benign_count = (all_attacks_test['Label'] == 'Normal').sum()
    attack_count = (all_attacks_test['Label'] != 'Normal').sum()
    print("Testing benign sample size:", benign_count)
    print("Testing attack sample size:", attack_count)

    print("Test Set Representation Adjusted...")

    # ---                   Feature Selection                --- #

    print("\nSelecting Features...")

    # Select the relevant features in the dataset and labels
    all_attacks_train = all_attacks_train[relevant_attributes_iotbotnet]
    all_attacks_test = all_attacks_test[relevant_attributes_iotbotnet]

    # Shuffle data
    all_attacks_train = shuffle(all_attacks_train, random_state=47)
    all_attacks_test = shuffle(all_attacks_test, random_state=47)

    print("Features Selected...")

    # prints an instance of each class in training data
    print("\nBefore Encoding and Scaling:")
    unique_labels = all_attacks_train['Label'].unique()
    for label in unique_labels:
        print(f"First instance of {label}:")
        print(all_attacks_train[all_attacks_train['Label'] == label].iloc[0])

    # ---                   Encoding                      --- #

    print("\nEncoding...")

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

    print("\nNormalizing...")

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

    print("\nAssign dataset to model...")

    # Feature / Label Split (X y split)
    X_data = all_attacks_train.drop(columns=['Label'])
    y_data = all_attacks_train['Label']

    X_train_data, X_val_data, y_train_data, y_val_data = train_test_split(X_data, y_data, test_size=0.2,
                                                                          random_state=47, stratify=y_data)

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


################################################################################################################
#                                       Model Initialization & Setup                                           #
################################################################################################################

# ---                  Hyper Parameters                  --- #

print("\n /////////////////////////////////////////////// \n")

# base hyperparameters for most models
model_name = dataset_used  # name for file

input_dim = X_train_data.shape[1]  # dependant for feature size

batch_size = 64  # 32 - 128; try 64, 96, 128; maybe intervals of 16

epochs = 5  # 1, 2 , 3 or 5 epochs

# steps_per_epoch = (len(X_train_data) // batch_size) // epochs  # dependant  # debug
steps_per_epoch = len(X_train_data) // batch_size   # dependant between sample size of the dataset and the batch size chosen

learning_rate = 0.0001  # 0.001 or .0001
betas = [0.9, 0.999]  # Stable

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

# ---             !!! INITIALIZE MODEL !!!                --- #

# ---                   CICIOT Models                   --- #
if dataset_used == "CICIOT":

    # --- Model Definition --- #
    if regularizationEnabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(64, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),  # Dropout layer with 50% dropout rate
            Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])

    elif regularizationEnabled and DP_enabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(32, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.4),
            Dense(1, activation='sigmoid')
        ])

    elif DP_enabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(4, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

    else:
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

# ---                   IOTBOTNET Models                  --- #

if dataset_used == "IOTBOTNET":

    # --- Model Definition --- #
    if regularizationEnabled:
        # with regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(16, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.3),  # Dropout layer with 30% dropout rate
            Dense(8, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(4, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(2, activation='relu', kernel_regularizer=l2(l2_alpha)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

    else:
        # without regularization
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(input_dim,)),
            Dense(16, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),  # Dropout layer with 50% dropout rate
            Dense(8, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(4, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(2, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

# ---         Differential Privacy Engine Model Compile              --- #

if DP_enabled:
    print("\nIncluding DP into optimizer...\n")

    # Making Custom Optimizer Component with Differential Privacy
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

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.binary_crossentropy,
                  metrics=['accuracy', Precision(), Recall(), AUC(), LogCosh()]
                  )

# ---                   Callback components                   --- #

# init main call back functions list
callbackFunctions = []

# init callback functions based on inputs

if earlyStopEnabled:
    early_stopping = EarlyStopping(monitor=metric_to_monitor_es, patience=es_patience, restore_best_weights=restor_best_w)

    callbackFunctions.append(early_stopping)


if lrSchedRedEnabled:
    lr_scheduler = ReduceLROnPlateau(monitor=metric_to_monitor_l2lr, factor=l2lr_factor, patience=l2lr_patience)

    callbackFunctions.append(lr_scheduler)


if modelCheckpointEnabled:
    model_checkpoint = ModelCheckpoint(f'best_model_{model_name}.h5', save_best_only=save_best_only,
                                       monitor=metric_to_monitor_mc, mode=checkpoint_mode)

    # add to callback functions list being added during fitting
    callbackFunctions.append(model_checkpoint)

# ---                   Model Analysis                   --- #

model.summary()

#########################################################
#    Adversarial Training Functions                     #
#########################################################

if adversarialTrainingEnabled:

    # Function to generate adversarial examples using FGSM
    def create_adversarial_example(model, x, y, epsilon=0.01):
        # Ensure x is a tensor and has the correct shape (batch_size, input_dim)
        # print("Original x shape:", x.shape)
        # print("Original y shape:", y.shape)

        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x = tf.expand_dims(x, axis=0)  # Adding batch dimension
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        y = tf.expand_dims(y, axis=0)  # Adding batch dimension to match prediction shape

        # print("Expanded x shape:", x.shape)
        # print("Expanded y shape:", y.shape)

        # Create a gradient tape context to record operations for automatic differentiation
        with tf.GradientTape() as tape:
            tape.watch(x)  # Adds the tensor x to the list of watched tensors, allowing its gradients to be computed
            prediction = model(x)  # Passes x through the model to get predictions
            y = tf.reshape(y, prediction.shape)  # Reshape y to match the shape of prediction
            # print("Reshaped y shape:", y.shape)
            loss = tf.keras.losses.binary_crossentropy(y, prediction)  # Computes the binary crossentropy loss between true labels y and predictions

        # Computes the gradient of the loss with respect to the input x
        gradient = tape.gradient(loss, x)

        # Creates the perturbation using the sign of the gradient and scales it by epsilon
        perturbation = epsilon * tf.sign(gradient)

        # Adds the perturbation to the original input to create the adversarial example
        adversarial_example = x + perturbation
        adversarial_example = tf.clip_by_value(adversarial_example, 0, 1)  # Ensure values are within valid range
        adversarial_example = tf.squeeze(adversarial_example, axis=0)  # Removing the batch dimension

        return adversarial_example

#########################################################
#    Metric Saving Functions                           #
#########################################################


def recordConfig(name):
    with open(name, 'a') as f:
        f.write(f"Dataset Used: {dataset_used}\n")
        f.write(f"Defenses Enabled: DP - {DP_enabled}, Adversarial Training - {adversarialTrainingEnabled}, Regularization - {regularizationEnabled}\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"Input Dim (Feature Size): {input_dim}\n")
        f.write(f"Epochs: {epochs}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Steps per epoch: {steps_per_epoch}\n")
        f.write(f"Betas: {betas}\n")
        f.write(f"Learning Rate: {learning_rate}\n")
        if DP_enabled:
            f.write(f"L2 Norm Clip: {l2_norm_clip}\n")
            f.write(f"Noise Multiplier: {noise_multiplier}\n")
            f.write(f"MicroBatches: {num_microbatches}\n")
        if adversarialTrainingEnabled:
            f.write(f"Adversarial Sample %: {adv_portion * 100}%\n")
        if regularizationEnabled:
            f.write(f"L2 Alpha: {l2_alpha}\n")
        f.write(f"Model Layer Structure:\n")
        for layer in model.layers:
            f.write(f"Layer: {layer.name}, Type: {layer.__class__.__name__}, Output Shape: {layer.output_shape}, Params: {layer.count_params()}\n")
        f.write("\n")


def recordTraining(name, history, elapsed_time, roundCount, val_loss):
    with open(name, 'a') as f:
        f.write(f"Node|{node}| Round: {roundCount}\n")
        f.write(f"Training Time Elapsed: {elapsed_time} seconds\n")
        for epoch in range(epochs):
            f.write(f"Epoch {epoch + 1}/{epochs}\n")
            for metric, values in history.history.items():
                # Debug: print the length of values list and the current epoch
                print(f"Metric: {metric}, Values Length: {len(values)}, Epoch: {epoch}")
                if epoch < len(values):
                    f.write(f"{metric}: {values[epoch]}\n")
                else:
                    print(f"Skipping metric {metric} for epoch {epoch} due to out-of-range error.")
            if epoch < len(val_loss):
                f.write(f"Validation Loss: {val_loss[epoch]}\n")
            else:
                print(f"Skipping Validation Loss for epoch {epoch} due to out-of-range error.")
            f.write("\n")


def recordEvaluation(name, elapsed_time, evaluateCount, loss, accuracy, precision, recall, auc, logcosh):
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
        return self.model.get_weights()

    def fit(self, parameters, config):
        # increment round count
        self.roundCount += 1

        # debug print
        print("Round:", self.roundCount, "\n")

        # Record start time
        start_time = time.time()

        self.model.set_weights(parameters)

        if adversarialTrainingEnabled:

            total_examples = len(X_train_data)
            print_every = max(total_examples // 10000, 1)  # Print progress every 0.1%

            # Define proportion of data to use for adversarial training (e.g., 10%)
            adv_proportion = adv_portion

            num_adv_examples = int(total_examples * adv_proportion)
            print("# of adversarial examples", num_adv_examples)
            adv_indices = random.sample(range(total_examples), num_adv_examples)

            adv_examples = []
            for idx, (x, y) in enumerate(zip(X_train_data.to_numpy(), y_train_data.to_numpy())):
                if idx in adv_indices:
                    adv_example = create_adversarial_example(model, x, y)
                    adv_examples.append(adv_example)
                else:
                    adv_examples.append(x)

                if (idx + 1) % print_every == 0 or (idx + 1) == total_examples:
                    print(f"Progress: {(idx + 1) / total_examples * 100:.2f}%")

            adv_X_train_data = np.array(adv_examples)

            adv_X_train_data = pd.DataFrame(adv_X_train_data, columns=X_train_data.columns)
            combined_X_train_data = pd.concat([X_train_data, adv_X_train_data])
            combined_y_train_data = pd.concat([y_train_data, y_train_data])

            history = self.model.fit(combined_X_train_data, combined_y_train_data,
                                     validation_data=(X_val_data, y_val_data),
                                     epochs=epochs, batch_size=batch_size,
                                     steps_per_epoch=steps_per_epoch,
                                     callbacks=callbackFunctions)
        else:
            # Train Model
            history = self.model.fit(X_train_data, y_train_data,
                                     validation_data=(X_val_data, y_val_data),
                                     epochs=epochs, batch_size=batch_size,
                                     steps_per_epoch=steps_per_epoch,
                                     callbacks=callbackFunctions)

        # Record end time and calculate elapsed time
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Debugging: Print the shape of the loss
        loss_tensor = history.history['loss']
        val_loss_tensor = history.history['val_loss']
        # print(f"Loss tensor shape: {tf.shape(loss_tensor)}")
        # print(f"Validation Loss tensor shape: {tf.shape(val_loss_tensor)}")

        # Save metrics to file
        logName = trainingLog
        #logName = f'training_metrics_{dataset_used}_optimized_{l2_norm_clip}_{noise_multiplier}.txt'
        recordTraining(logName, history, elapsed_time, self.roundCount, val_loss_tensor)

        return self.model.get_weights(), len(X_train_data), {}

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
        logName1 = evaluationLog
        #logName = f'evaluation_metrics_{dataset_used}_optimized_{l2_norm_clip}_{noise_multiplier}.txt'
        recordEvaluation(logName1, elapsed_time, self.evaluateCount, loss, accuracy, precision, recall, auc, logcosh)

        return loss, len(X_test_data), {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc,
                                        "LogCosh": logcosh}


#########################################################
#    Start the client                                   #
#########################################################

# Record initial configuration before training starts
logName = trainingLog
recordConfig(logName)
logName1 = evaluationLog
recordConfig(logName1)

if fixedServer == 1:
    server_address = "192.168.129.2:8080"
elif fixedServer == 2:
    server_address = "192.168.129.6:8080"
elif fixedServer == 3:
    server_address = "192.168.129.7:8080"
else:
    server_address = "192.168.129.8:8080"

fl.client.start_client(server_address=server_address, client=FLClient(model).to_client())
