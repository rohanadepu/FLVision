import argparse
import os
import subprocess
import time

# Define the datasets and their paths
datasets = {
    "IOTBOTNET_33": "/root/attacks/IOTBOTNET2020_POISONED33/IOTBOTNET2020/iotbotnet2020_archive",
    "IOTBOTNET_66": "/root/attacks/IOTBOTNET2020_POISONED66/IOTBOTNET2020/iotbotnet2020_archive",
    "CICIOT_33": "/root/attacks/CICIOT2023_POISONED33/CICIOT2023/ciciot2023_archive",
    "CICIOT_66": "/root/attacks/CICIOT2023_POISONED66/CICIOT2023/ciciot2023_archive",
}

# Clean datasets paths for node 2
clean_datasets = {
    "IOTBOTNET_33": "/root/attacks/IOTBOTNET2020/IOTBOTNET2020/iotbotnet2020_archive",
    "IOTBOTNET_66": "/root/attacks/IOTBOTNET2020/IOTBOTNET2020/iotbotnet2020_archive",
    "CICIOT_33": "/root/attacks/CICIOT2023/CICIOT2023/ciciot2023_archive",
    "CICIOT_66": "/root/attacks/CICIOT2023/CICIOT2023/ciciot2023_archive",
}

def run_command(command):
    print(f"Executing: {command}")
    process = subprocess.Popen(command, shell=True)
    process.communicate()

def run_training_node1(dataset_name):
    dataset_path = datasets[dataset_name]
    evaluation_log = f"evaluation_metrics_{dataset_name}.log"
    training_log = f"training_metrics_{dataset_name}.log"
    flag_file = "/tmp/node_1_completed.flag"
    dataset_type = "IOTBOTNET" if "IOTBOTNET" in dataset_name else "CICIOT"

    command = f"python3 clientPoisoned.py --dataset {dataset_type} --node 1 --dataset_path {dataset_path} --evaluation_log {evaluation_log} --training_log {training_log}"

    print(f"Running training on node 1 with dataset {dataset_name}")
    run_command(command)

    # Wait for the flag file to indicate completion
    while not os.path.isfile(flag_file):
        time.sleep(10)

    print(f"Node 1 completed training with dataset {dataset_name}")

def run_training_node2(dataset_name):
    dataset_path = clean_datasets[dataset_name]
    evaluation_log = f"evaluation_metrics_{dataset_name}_CLEAN.log"
    training_log = f"training_metrics_{dataset_name}_CLEAN.log"
    flag_file = "/tmp/node_2_completed.flag"
    dataset_type = "IOTBOTNET" if "IOTBOTNET" in dataset_name else "CICIOT"

    command = f"python3 clientPoisoned.py --dataset {dataset_type} --node 2 --dataset_path {dataset_path} --evaluation_log {evaluation_log} --training_log {training_log}"

    print(f"Running training on node 2 with dataset {dataset_name} (clean)")
    run_command(command)

    # Wait for the flag file to indicate completion
    while not os.path.isfile(flag_file):
        time.sleep(10)

    print(f"Node 2 completed training with dataset {dataset_name}")

def run_server():
    while True:
        print("Starting the server node")
        command = "python3 server.py"
        run_command(command)
        print("Server node completed. Restarting...")

def parse_datasets(dataset_input):
    dataset_map = {
        'i33': 'IOTBOTNET_33',
        'i66': 'IOTBOTNET_66',
        'c33': 'CICIOT_33',
        'c66': 'CICIOT_66',
    }
    return [dataset_map[ds] for ds in dataset_input]

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Training Script")
    parser.add_argument("--node", type=int, required=True, help="Node number (1 or 2) or '3' for the server node")
    parser.add_argument('--datasets', type=str, required=True, help='Datasets to use, e.g., "i33c33c66"')
    args = parser.parse_args()

    node_number = args.node
    dataset_input = args.datasets
    datasets_to_use = parse_datasets(dataset_input.split())

    if node_number == 1:
        for dataset_name in datasets_to_use:
            run_training_node1(dataset_name)
    elif node_number >= 2:
        for dataset_name in datasets_to_use:
            run_training_node2(dataset_name)
    elif node_number == 0:
        run_server()
    else:
        print(f"Unknown node number: {node_number}")
        exit(1)

if __name__ == "__main__":
    main()
