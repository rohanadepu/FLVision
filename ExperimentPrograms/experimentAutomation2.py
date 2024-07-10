#########################################################
#    Imports and env configs                           #
#########################################################
import argparse
import os
import subprocess
import threading
import time

#########################################################
#    Experiment Configurations                          #
#########################################################
experiments = [
    {"dataset": "CICIOT", "reg": True, "dp": 0, "prune": False, "adversarial": False, "eS": True, "lrSched": True, "mChkpnt": True, "node": 1, "pData": None},
    {"dataset": "IOTBOTNET", "reg": False, "dp": 1, "prune": True, "adversarial": True, "eS": False, "lrSched": False, "mChkpnt": False, "node": 2, "pData": "LF33"},
    {"dataset": "CIFAR", "reg": True, "dp": 2, "prune": False, "adversarial": True, "eS": True, "lrSched": True, "mChkpnt": False, "node": 3, "pData": "FN66"}
    # Add more experiments as needed
]

#########################################################
#    Node/process/thread manager                        #
#########################################################

def run_experiment_on_node(node, command):
    print(f"Starting experiment on node {node} with command: {command}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Experiment on node {node} failed.")
    else:
        print(f"Experiment on node {node} completed successfully.")

def run_server(experiments):
    for experiment in experiments:
        command = f"ssh user@server-node 'python3 serverExperiment.py --dataset {experiment['dataset']}'"
        run_experiment_on_node("server", command)

def run_client(experiment):
    node = experiment["node"]
    command = (
        f"ssh user@client-node-{node} 'python3 clientExperiment.py --dataset {experiment['dataset']} "
        f"--reg {'--reg' if experiment['reg'] else ''} "
        f"--dp {experiment['dp']} "
        f"--prune {'--prune' if experiment['prune'] else ''} "
        f"--adversarial {'--adversarial' if experiment['adversarial'] else ''} "
        f"--eS {'--eS' if experiment['eS'] else ''} "
        f"--lrSched {'--lrSched' if experiment['lrSched'] else ''} "
        f"--mChkpnt {'--mChkpnt' if experiment['mChkpnt'] else ''} "
        f"--node {experiment['node']} "
        f"--pData {experiment['pData']} "
        f"--evalLog {experiment['evalLog']} "
        f"--trainLog {experiment['trainLog']}'"
    )
    run_experiment_on_node(node, command)

#########################################################
#    Execution                                          #
#########################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Automate Federated Learning Experiments")
    parser.add_argument('--experiment', type=int, default=None, help="Index of the experiment to run (default: run all experiments)")
    args = parser.parse_args()

    if args.experiment is not None:
        experiments = [experiments[args.experiment]]

    # Start server
    server_thread = threading.Thread(target=run_server, args=(experiments,))
    server_thread.start()

    # Start clients
    client_threads = []
    for experiment in experiments:
        client_thread = threading.Thread(target=run_client, args=(experiment,))
        client_threads.append(client_thread)
        client_thread.start()

    # Wait for all threads to complete
    server_thread.join()
    for client_thread in client_threads:
        client_thread.join()

    print("All experiments completed.")
