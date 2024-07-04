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
    "IOTBOTNET": "/root/attacks/IOTBOTNET2020/IOTBOTNET2020/iotbotnet2020_archive",
    "CICIOT": "/root/attacks/CICIOT2023/CICIOT2023/ciciot2023_archive",
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

    command = f"python3 clientPoisoned.py --node 1 --dataset_path {dataset_path} --evaluation_log {evaluation_log} --training_log {training_log}"

    print(f"Running training on node 1 with dataset {dataset_name}")
    run_command(command)

    # Wait for the flag file to indicate completion
    while not os.path.isfile(flag_file):
        time.sleep(10)

    print(f"Node 1 completed training with dataset {dataset_name}")

def run_training_node2(dataset_name):
    if "IOTBOTNET" in dataset_name:
        clean_dataset_name = "IOTBOTNET"
    else:
        clean_dataset_name = "CICIOT"

    dataset_path = clean_datasets[clean_dataset_name]
    evaluation_log = f"evaluation_metrics_{clean_dataset_name}_CLEAN.log"
    training_log = f"training_metrics_{clean_dataset_name}_CLEAN.log"
    flag_file = "/tmp/node_2_completed.flag"

    command = f"python3 clientPoisoned.py --node 2 --dataset_path {dataset_path} --evaluation_log {evaluation_log} --training_log {training_log}"

    print(f"Running training on node 2 with dataset {clean_dataset_name} (clean)")
    run_command(command)

    # Wait for the flag file to indicate completion
    while not os.path.isfile(flag_file):
        time.sleep(10)

    print(f"Node 2 completed training with dataset {clean_dataset_name}")

def run_server():
    print("Starting the server node")
    command = "python3 server.py"
    run_command(command)
    print("Server node completed")

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Training Script")
    parser.add_argument("--node", type=int, required=True, help="Node number (1 or 2) or 'server' for the server node")
    args = parser.parse_args()

    node_number = args.node

    if node_number == 1:
        for dataset_name in ["IOTBOTNET_33", "IOTBOTNET_66", "CICIOT_33", "CICIOT_66"]:
            run_training_node1(dataset_name)
    elif node_number == 2:
        for dataset_name in ["IOTBOTNET", "IOTBOTNET", "CICIOT", "CICIOT"]:
            run_training_node2(dataset_name)
    elif node_number == "server":
        run_server()
    else:
        print(f"Unknown node number: {node_number}")
        exit(1)

if __name__ == "__main__":
    main()
