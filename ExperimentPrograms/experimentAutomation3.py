import subprocess
import threading
import argparse
import socket

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

# Define the IP addresses for each node
node_ips = {
    1: "192.168.129.1",
    2: "192.168.129.3",
    3: "192.168.129.4",
    4: "192.168.129.5",
    5: "192.168.129.9",
    6: "192.168.129.10",
    "server": "192.168.129.2"
}

def run_command(command):
    print(f"Running command: {command}")
    result = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode()
    error = result.stderr.decode()
    print(f"Output: {output}")
    print(f"Error: {error}")
    if result.returncode != 0:
        print(f"Command failed with error code {result.returncode}.")
    else:
        print(f"Command completed successfully.")

def run_server():
    print("Starting the server node")
    command = "python3 server.py"
    # Using subprocess.Popen to run the server process in the background
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    while True:
        output = process.stdout.readline()
        if output:
            print(output.decode().strip())
        error = process.stderr.readline()
        if error:
            print(error.decode().strip())
        if output == b'' and error == b'' and process.poll() is not None:
            break

def run_client(node, dataset, poisoned_data, strategy, log_file):
    reg_flag = "--reg" if strategy in ["regularization", "all"] else ""
    dp_flag = "--dp 1" if strategy in ["differential_privacy", "all"] else "--dp 0"
    prune_flag = "--prune" if strategy in ["model_pruning", "all"] else ""
    adv_flag = "--adversarial" if strategy in ["adversarial_training", "all"] else ""

    command = (
        f"python3 clientExperiment.py --dataset {dataset} --node {node} --pData {poisoned_data} --evalLog eval_{log_file} --trainLog train_{log_file} {reg_flag} {dp_flag} {prune_flag} {adv_flag}"
    )
    
    run_command(command)

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

    # Determine the current node's IP address
    current_ip = socket.gethostbyname(socket.gethostname())

    # Identify the current node based on its IP address
    current_node = None
    for node, ip in node_ips.items():
        if ip == current_ip:
            current_node = node
            break

    if current_node is None:
        print("Current node IP address not found in node_ips dictionary.")
        return

    if current_node == "server":
        run_server()
    else:
        threads = []
        for dataset in selected_datasets:
            for poisoned_variant in selected_poisoned_variants:
                for strategy in selected_defense_strategies:
                    for num_nodes in num_clean_nodes_list:
                        nodes_to_use = [compromised_node] + [i for i in range(2, 7)][:num_nodes]  # Select clean nodes from 2 to 6
                        if current_node in nodes_to_use:
                            log_file = f"log_node{current_node}_dataset{dataset}_poisoned{poisoned_variant}_strategy{strategy}_clean{num_nodes}.txt"
                            client_thread = threading.Thread(target=run_client, args=(current_node, dataset, poisoned_variant, strategy, log_file))
                            threads.append(client_thread)
                            client_thread.start()

        for thread in threads:
            thread.join()

if __name__ == "__main__":
    main()

#exmaple usage
#python experimentAutomation3.py --datasets IOTBOTNET CICIOT --pvar LF33 LF66 FN33 FN66 --defense_strat none differential_privacy adversarial_training --cleannodes 1 2 4

