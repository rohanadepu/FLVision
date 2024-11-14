import os
import random
import pandas as pd
from sklearn.utils import shuffle

#---                 Constants               ---#

DICT_2CLASSES = {'DDoS-RSTFINFlood': 'Attack', 'DDoS-PSHACK_Flood': 'Attack', 'DDoS-SYN_Flood': 'Attack',
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

DICT_7CLASSES = {'DDoS-RSTFINFlood': 'DDoS', 'DDoS-PSHACK_Flood': 'DDoS', 'DDoS-SYN_Flood': 'DDoS',
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

NUM_COLS = ['flow_duration', 'Header_Length', 'Rate', 'Srate', 'Drate', 'ack_count', 'syn_count', 'fin_count',
                'urg_count', 'rst_count', 'Tot sum', 'Min', 'Max', 'AVG', 'Std', 'Tot size', 'IAT', 'Number',
                'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight'
                ]

CAT_COLS = [
        'Protocol Type', 'Duration', 'fin flag number', 'syn flag number', 'rst flag number', 'psh flag number',
        'ack flag number', 'ece flag number', 'cwr flag number', 'HTTP', 'HTTPS', 'DNS', 'Telnet', 'SMTP', 'SSH', 'IRC',
        'TCP', 'UDP', 'DHCP', 'ARP', 'ICMP', 'IPv', 'LLC'
    ]

IRRELEVANT_FEATURES = ['Srate', 'ece_flag_number', 'rst_flag_number', 'ack_flag_number', 'cwr_flag_number',
                       'ack_count', 'syn_count', 'fin_count', 'rst_count', 'LLC', 'Min', 'Max', 'AVG', 'Std',
                       'Tot size', 'Number', 'Magnitue', 'Radius', 'Covariance', 'Variance', 'Weight',
                       'flow_duration', 'Header_Length', 'urg_count', 'Tot sum']

# select the num cols that are relevant
RElEVANT_NUM_COLS = [col for col in NUM_COLS if col not in IRRELEVANT_FEATURES]


# ---                 Helper  Functions                   --- #
def map_labels(data, label_class_dict):

    data['label'] = data['label'].map(label_class_dict)

    return data


def load_and_balance_data(file_path, label_class_dict, current_benign_size, benign_size_limit):

    data = pd.read_csv(file_path)

    data = map_labels(data, label_class_dict)

    attack_samples = data[data['label'] == 'Attack']
    benign_samples = data[data['label'] == 'Benign']

    remaining_benign_quota = benign_size_limit - current_benign_size
    if len(benign_samples) > remaining_benign_quota:
        benign_samples = benign_samples.sample(remaining_benign_quota, random_state=47)

    min_samples = min(len(attack_samples), len(benign_samples))
    balanced_data = pd.concat([attack_samples.sample(min_samples, random_state=47),
                               benign_samples.sample(min_samples, random_state=47)])

    return balanced_data, len(benign_samples)


def reduce_attack_samples(data, attack_ratio):

    attack_samples = data[data['label'] == 'Attack'].sample(frac=attack_ratio, random_state=47)

    benign_samples = data[data['label'] == 'Benign']

    return pd.concat([benign_samples, attack_samples])


###################################################################################
#               Load Config for CICIOT 2023 Dataset                            #
###################################################################################
def loadCICIOT(poisonedDataType=None, verbose=True, train_sample_size=25, test_sample_size=10,
               training_dataset_size=220000, testing_dataset_size=80000, attack_eval_samples_ratio=0.1):

    DATASET_DIRECTORY = f'/root/datasets/CICIOT2023_POISONED{poisonedDataType}' if poisonedDataType else '../../datasets/CICIOT2023'

    training_benign_size = training_dataset_size // 2
    testing_benign_size = testing_dataset_size // 2

    benign_size_limits = {'train': training_benign_size, 'test': testing_benign_size}

    attack_ratio = attack_eval_samples_ratio  # For reducing attacks in test data
    attack_percentage = attack_ratio * 10

    #--- File Paths ---#
    if verbose:
        print("\nLoading Network Traffic Data Files...")

    # List and sorts the files in the dataset directory
    csv_filepaths = sorted([filename for filename in os.listdir(DATASET_DIRECTORY) if filename.endswith('.csv')])
    train_files = random.sample(csv_filepaths, train_sample_size)
    test_files = random.sample([f for f in csv_filepaths if f not in train_files], test_sample_size)

    if verbose:
        print("\nTraining Sets:\n", train_files, "\n")
        print("\nTest Sets:\n", test_files, "\n")

    #--- Load Train Data ---#

    if verbose:
        print("\nLoading Training Data...")

    ciciot_train_data = pd.DataFrame()
    train_benign_count = 0

    for file in train_files:

        # break loop once limit is reached
        if train_benign_count >= benign_size_limits['train']:
            break

        if verbose:
            print(f"Training dataset sample: {file} \n")

        data, benign_count = load_and_balance_data(os.path.join(DATASET_DIRECTORY, file), DICT_2CLASSES,
                                                   train_benign_count, benign_size_limits['train'])

        ciciot_train_data = pd.concat([ciciot_train_data, data])

        train_benign_count += benign_count

        if verbose:
            print(
                f"Benign Traffic Train Samples | Samples in File: {benign_count} | Total: {train_benign_count} | LIMIT: {benign_size_limits['train']}")

    #--- Load Test Data ---#

    if verbose:
        print("\nLoading Testing Data...")

    ciciot_test_data = pd.DataFrame()
    test_benign_count = 0

    for file in test_files:

        if test_benign_count >= benign_size_limits['test']:
            break

        if verbose:
            print(f"Testing dataset sample: {file} \n")

        data, benign_count = load_and_balance_data(os.path.join(DATASET_DIRECTORY, file), DICT_2CLASSES,
                                                   test_benign_count, benign_size_limits['test'])

        ciciot_test_data = pd.concat([ciciot_test_data, data])

        test_benign_count += benign_count

        if verbose:
            print(
                f"Benign Traffic Test Samples | Samples in File: {benign_count} | Total: {test_benign_count} | LIMIT: {benign_size_limits['test']}")

    # --- Reduce attack samples in test data  ---#
    if verbose:
        print("\nReducing Attack Samples in Testing Data to", attack_percentage, "%")

    ciciot_test_data = reduce_attack_samples(ciciot_test_data, attack_ratio)

    if verbose:
        print("\n(CICIOT) Train & Test Attack Data Loaded (Attack Data already Combined)...")

        print("Training Data Sample:")
        print(ciciot_train_data.head())

        print("Testing Data Sample:")
        print(ciciot_test_data.head())

    return ciciot_train_data, ciciot_test_data, IRRELEVANT_FEATURES
