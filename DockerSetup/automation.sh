#!/bin/bash

# Usage:
# ./run_training_node.sh --node NODE_NUMBER

# Define the datasets and their paths
declare -A datasets=(
    ["IOTBOTNET_33"]="/root/attacks/IOTBOTNET2020_POISONED33/IOTBOTNET2020/iotbotnet2020_archive"
    ["IOTBOTNET_66"]="/root/attacks/IOTBOTNET2020_POISONED66/IOTBOTNET2020/iotbotnet2020_archive"
    ["CICIOT_33"]="/root/attacks/CICIOT2023_POISONED33/CICIOT2023/ciciot2023_archive"
    ["CICIOT_66"]="/root/attacks/CICIOT2023_POISONED66/CICIOT2023/ciciot2023_archive"
)

# Clean datasets paths for node 2
declare -A clean_datasets=(
    ["IOTBOTNET"]="/root/attacks/IOTBOTNET2020/IOTBOTNET2020/iotbotnet2020_archive"
    ["CICIOT"]="/root/attacks/CICIOT2023/CICIOT2023/ciciot2023_archive"
)

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --node) node_number="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [[ -z "$node_number" ]]; then
    echo "Missing required parameter: --node"
    exit 1
fi

run_training_node1() {
    local dataset_name=$1
    local dataset_path=${datasets[$dataset_name]}
    local evaluation_log="evaluation_metrics_${dataset_name}.txt"
    local training_log="training_metrics_${dataset_name}.txt"
    local flag_file="/tmp/node_1_completed.flag"

    local command="python3 clientPoisoned.py \
        --node 1 \
        --dataset_path $dataset_path \
        --evaluation_log $evaluation_log \
        --training_log $training_log"

    echo "Running training on node 1 with dataset $dataset_name"
    $command

    # Wait for the flag file to indicate completion
    while [ ! -f $flag_file ]; do
        sleep 10
    done
    echo "Node 1 completed training with dataset $dataset_name"
}

run_training_node2() {
    local dataset_name=$1
    local clean_dataset_name
    if [[ $dataset_name == IOTBOTNET* ]]; then
        clean_dataset_name="IOTBOTNET"
    else
        clean_dataset_name="CICIOT"
    fi
    local dataset_path=${clean_datasets[$clean_dataset_name]}
    local evaluation_log="evaluation_metrics_${clean_dataset_name}_CLEAN.log"
    local training_log="training_metrics_${clean_dataset_name}_CLEAN.log"
    local flag_file="/tmp/node_2_completed.flag"

    local command="python3 /path/to/clientPoisoned.py \
        --node 2 \
        --dataset_path $dataset_path \
        --evaluation_log $evaluation_log \
        --training_log $training_log"

    echo "Running training on node 2 with dataset $clean_dataset_name (clean)"
    $command

    # Wait for the flag file to indicate completion
    while [ ! -f $flag_file ]; do
        sleep 10
    done
    echo "Node 2 completed training with dataset $clean_dataset_name"
}

run_server() {
    echo "Starting the server node"
    python3 server.py
    echo "Server node completed"
}

# Main function to run the training process
main() {
    if [[ $node_number == 1 ]]; then
        for dataset_name in "IOTBOTNET_33" "IOTBOTNET_66" "CICIOT_33" "CICIOT_66"; do
            run_training_node1 $dataset_name
        done
    elif [[ $node_number == 2 ]]; then
        for dataset_name in "IOTBOTNET" "IOTBOTNET" "CICIOT" "CICIOT"; do
            run_training_node2 $dataset_name
        done
    elif [[ $node_number == "server" ]]; then
        run_server
    else
        echo "Unknown node number: $node_number"
        exit 1
    fi
}

# Run the main function
main
