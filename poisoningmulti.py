import pandas as pd
import numpy as np
import os
import shutil
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

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


def poison_dataset(file_path, output_folder, label):
    df = pd.read_csv(file_path)
    
    # Assuming we want to poison 100% of the selected files using the least frequent label
    least_frequent_label = df[label].value_counts().idxmin()
    df[label] = least_frequent_label
    
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    output_file_name = os.path.join(output_folder, os.path.basename(file_path))
    df.to_csv(output_file_name, index=False)
    print(f"Poisoned dataset saved as {output_file_name}")

def handle_file(file_path, output_folder, label, poison_probability):
    relative_path = os.path.relpath(file_path, input_folder)
    new_output_folder = os.path.join(output_folder, os.path.dirname(relative_path))

    # Randomly decide whether to poison the file
    if np.random.rand() < poison_probability:
        poison_dataset(file_path, new_output_folder, label)
    else:
        # Copy the file if not poisoning
        os.makedirs(new_output_folder, exist_ok=True)
        shutil.copy(file_path, os.path.join(new_output_folder, os.path.basename(file_path)))

def poison_all_files(input_folder, poison_percentage):


    label = 'label' if 'CICIOT2023' in input_folder else 'Label'
    output_folder = input_folder + '_POISONED'

    #if CICIOT remap labels to dict_2classes
    
    # Ensure the output directory does not exist to start fresh
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    csv_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(input_folder) for f in filenames if f.endswith('.csv')]
    poison_probability = poison_percentage / 100

    # Use a number of threads based on the number of CPUs
    num_threads = multiprocessing.cpu_count()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(handle_file, csv_files, [output_folder]*len(csv_files), [label]*len(csv_files), [poison_probability]*len(csv_files))

# Example usage
input_folder = 'CICIOT2023'  # or 'CICIOT2023'
poison_percentage = 33  # Percentage of files to poison
poison_all_files(input_folder, poison_percentage)
