import argparse
import time
from fabric import Connection
import threading

# Define the defense strategies including the option for no defenses
defense_strategies = [
    "none",
    "differential_privacy",
    "regularization",
    "adversarial_training",
    "model_pruning",
    "all"
]

# Define the datasets and poisoned variants
datasets = ["IOTBOTNET", "CICIOT"]
poisoned_variants = ["LF33", "LF66", "FN33", "FN66"]

# IP addresses for each node
node_ips = {
    1: "192.168.129.1",
    2: "192.168.129.3",
    3: "192.168.129.4",
    4: "192.168.129.5",
    5: "192.168.129.9",
    6: "192.168.129.10",
    "server": "192.168.129.2"
}

# Path to SSH key
ssh_key = "aerpaw_id_rsa"

def run_command_on_host(ip, command):
    print(f"Executing on {ip}: {command}")
    conn = Connection(host=ip, user='root', connect_kwargs={"key_filename": ssh_key})
    result = conn.run(command, hide=True, warn=True)
    if result.stdout:
        print(f"[{ip}] STDOUT:\n{result.stdout.strip()}")
    if result.stderr:
        print(f"[{ip}] STDERR:\n{result.stderr.strip()}")
    conn.close()

def run_server():
    print("Starting the server node")
    server_ip = node_ips["server"]
    command = 'cd FLVision/X_REUSummer2024Programs && nohup python3 server.py &'
    run_command_on_host(server_ip, command)

def run_client_experiments_on_node(node, datasets, poisoned_variants, defense_strategies, num_clean_nodes_list):
    for dataset in datasets:
        for poisoned_variant in poisoned_variants:
            for strategy in defense_strategies:
                for num_nodes in num_clean_nodes_list:
                    nodes_to_use = [1] + [i for i in range(2, 7)][:num_nodes]  # Select clean nodes from 2 to 6
                    if node in nodes_to_use:
                        poisoned_data = poisoned_variant if node == 1 else None
                        log_file = f"log_node{node}_dataset{dataset}_poisoned{poisoned_variant}_strategy{strategy}_clean{num_nodes}.txt"
                        run_client(node, dataset, poisoned_data, strategy, log_file)
                        # Wait for a bit before starting the next round to ensure synchronization
                        time.sleep(5)

def run_client(node, dataset, poisoned_data, strategy, log_file):
    client_ip = node_ips[node]
    
    reg_flag = "--reg" if strategy in ["regularization", "all"] else ""
    dp_flag = "--dp" if strategy in ["differential_privacy", "all"] else ""
    prune_flag = "--prune" if strategy in ["model_pruning", "all"] else ""
    adv_flag = "--adversarial" if strategy in ["adversarial_training", "all"] else ""

    p_data_flag = f"--pData {poisoned_data}" if poisoned_data else ""

    command = (
        f"cd FLVision/X_REUSummer2024Programs && python3 clientExperiment.py --dataset {dataset} "
        f"--node {node} {p_data_flag} --evalLog eval_{log_file} --trainLog train_{log_file} "
        f"{reg_flag} {dp_flag} {prune_flag} {adv_flag}"
    )
    
    run_command_on_host(client_ip, command)

def main():
    parser = argparse.ArgumentParser(description="Federated Learning Training Script Automation")
    parser.add_argument("--datasets", type=str, nargs='+', choices=datasets, required=True, help="List of datasets to use")
    parser.add_argument("--pvar", type=str, nargs='+', choices=poisoned_variants, required=True, help="List of poisoned variants to use")
    parser.add_argument("--defense_strat", type=str, nargs='+', choices=defense_strategies, required=True, help="List of defense strategies to use")
    parser.add_argument("--cleannodes", type=int, nargs='+', required=True, help="List of numbers of clean nodes to use in each experiment")

    args = parser.parse_args()

    compromised_node = 1  # Always set node 1 as the compromised node
    num_clean_nodes_list = args.cleannodes
    selected_datasets = args.datasets
    selected_poisoned_variants = args.pvar
    selected_defense_strategies = args.defense_strat

    # Create a thread for the server
    server_thread = threading.Thread(target=run_server)
    server_thread.start()

    # Give the server some time to start up
    time.sleep(10)

    # Create and start client threads
    client_threads = []

    for node in range(1, 7):
        client_thread = threading.Thread(target=run_client_experiments_on_node, args=(node, selected_datasets, selected_poisoned_variants, selected_defense_strategies, num_clean_nodes_list))
        client_threads.append(client_thread)
        client_thread.start()

    # Wait for all client threads to finish
    for thread in client_threads:
        thread.join()

    # Wait for the server thread to finish
    server_thread.join()

if __name__ == "__main__":
    main()

# Example usage:
# python3 experimentAutomation3.py --datasets IOTBOTNET CICIOT --pvar LF33 LF66 FN33 FN66 --defense_strat none differential_privacy adversarial_training --cleannodes 1 2 4
