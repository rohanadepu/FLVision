import matplotlib.pyplot as plt
import re
import itertools
import numpy as np

log_files = {
    'No Defense - 1 Node - Baseline': 'train_log_node1_datasetIOTBOTNET_baseline_strategynone_clean1.txt',
    'No Defense - 1 Node - FN66': 'train_log_node2_datasetIOTBOTNET_poisonedFN33_strategynone_clean2.txt',
    'No Defense - 3 Nodes - FN66': 'train_log_node1_datasetIOTBOTNET_poisonedFN66_strategynone_clean3.txt',
    'Adversarial Training - 1 Node - FN66': 'train_log_node1_datasetIOTBOTNET_poisonedFN66_strategyadversarial_training_clean1.txt',
    'Adversarial Training - 3 Nodes - FN66': 'train_log_node1_datasetIOTBOTNET_poisonedFN66_strategyadversarial_training_clean3.txt',
    'Differential Privacy - 1 Node - FN66': 'train_log_node1_datasetIOTBOTNET_poisonedFN66_strategydifferential_privacy_clean1.txt',
    'Differential Privacy - 3 Nodes - FN66': 'train_log_node1_datasetIOTBOTNET_poisonedFN66_strategydp_clean3.txt',
    'All Defenses - 1 Node - FN66': 'train_log_node1_datasetIOTBOTNET_poisonedFN66_strategyall_clean1.txt',
    'All Defenses - 3 Nodes - FN66': 'train_log_node1_datasetIOTBOTNET_poisonedFN66_strategyall_clean3.txt',
}


def parse_log(file_path):
    epochs = []
    accuracy = []
    precision = []
    recall = []
    val_accuracy = []
    val_precision = []
    val_recall = []

    with open(file_path, 'r') as file:
        epoch_num = None
        for line in file:
            line = line.strip()
            if not line:  # Skip empty lines
                continue

            epoch_match = re.match(r'Epoch (\d+)/\d+', line)
            if epoch_match:
                epoch_num = int(epoch_match.group(1))
                epochs.append(epoch_num)
            else:
                metric_match = re.match(r'val_accuracy: ([\d\.]+)', line)
                if metric_match:
                    val_accuracy.append(float(metric_match.group(1)))
                    continue
                metric_match = re.match(r'val_precision: ([\d\.]+)', line)
                if metric_match:
                    val_precision.append(float(metric_match.group(1)))
                    continue
                metric_match = re.match(r'val_recall: ([\d\.]+)', line)
                if metric_match:
                    val_recall.append(float(metric_match.group(1)))
                    continue
                metric_match = re.match(r'accuracy: ([\d\.]+)', line)
                if metric_match:
                    accuracy.append(float(metric_match.group(1)))
                    continue
                metric_match = re.match(r'precision: ([\d\.]+)', line)
                if metric_match:
                    precision.append(float(metric_match.group(1)))
                    continue
                metric_match = re.match(r'recall: ([\d\.]+)', line)
                if metric_match:
                    recall.append(float(metric_match.group(1)))
                    continue

    if val_accuracy and val_precision and val_recall:
        return epochs, val_accuracy, val_precision, val_recall
    else:
        return epochs, accuracy, precision, recall


def compute_average_metrics(metrics, epochs_per_round=5, max_rounds=10):
    rounds = min(len(metrics) // epochs_per_round, max_rounds)
    avg_metrics = []
    for i in range(rounds):
        start_idx = i * epochs_per_round
        end_idx = start_idx + epochs_per_round
        avg_metrics.append(np.mean(metrics[start_idx:end_idx]))
    return avg_metrics


def adjust_lengths(x, y):
    min_length = min(len(x), len(y))
    return x[:min_length], y[:min_length]


# Collect metrics from all log files
grouped_data = {
    'Baseline': [],
    'No Defense': [],
    'Adversarial Training': [],
    'Differential Privacy': [],
    'All Defenses': []
}

for experiment_name, file_path in log_files.items():
    try:
        epochs, accuracy, precision, recall = parse_log(file_path)
        avg_accuracy = compute_average_metrics(accuracy)
        avg_precision = compute_average_metrics(precision)
        avg_recall = compute_average_metrics(recall)

        if 'Baseline' in experiment_name:
            group = 'Baseline'
        elif 'No Defense' in experiment_name:
            group = 'No Defense'
        elif 'Adversarial Training' in experiment_name:
            group = 'Adversarial Training'
        elif 'Differential Privacy' in experiment_name:
            group = 'Differential Privacy'
        elif 'All Defenses' in experiment_name:
            group = 'All Defenses'

        grouped_data[group].append((experiment_name, avg_accuracy, avg_precision, avg_recall))
    except FileNotFoundError:
        print(f"File {file_path} not found.")
    except Exception as e:
        print(f"An error occurred while processing {file_path}: {e}")

# Define markers
markers = itertools.cycle(('o', 's', 'D', '^', 'v', '<', '>', 'p', 'h', '8'))

# Plotting
plt.figure(figsize=(20, 8))


# Helper function to plot metrics
def plot_metric(ax, x, y, label, linestyle, marker):
    x, y = adjust_lengths(x, y)
    ax.plot(x, y, label=label, linestyle=linestyle, marker=marker, linewidth=3, markersize=8)


left_epochs = list(range(1, 11))
right_epochs = list(range(11, 21))

# Plot Accuracy
ax = plt.gca()
for group, experiments in grouped_data.items():
    for experiment_name, avg_accuracy, _, _ in experiments:
        linestyle = '-' if '1 Node' in experiment_name else '--'
        marker = next(markers)
        if group in ['Baseline', 'No Defense']:
            plot_metric(ax, left_epochs, avg_accuracy[:10], f'{experiment_name} Accuracy', linestyle, marker)
        else:
            plot_metric(ax, right_epochs, avg_accuracy[:10], f'{experiment_name} Accuracy', linestyle, marker)

# Plot Precision
for group, experiments in grouped_data.items():
    for experiment_name, _, avg_precision, _ in experiments:
        linestyle = '-' if '1 Node' in experiment_name else '--'
        marker = next(markers)
        if group in ['Baseline', 'No Defense']:
            plot_metric(ax, [e + 10 for e in left_epochs], avg_precision[:10], f'{experiment_name} Precision',
                        linestyle, marker)
        else:
            plot_metric(ax, [e + 10 for e in right_epochs], avg_precision[:10], f'{experiment_name} Precision',
                        linestyle, marker)

# Plot Recall
for group, experiments in grouped_data.items():
    for experiment_name, _, _, avg_recall in experiments:
        linestyle = '-' if '1 Node' in experiment_name else '--'
        marker = next(markers)
        if group in ['Baseline', 'No Defense']:
            plot_metric(ax, [e + 20 for e in left_epochs], avg_recall[:10], f'{experiment_name} Recall', linestyle,
                        marker)
        else:
            plot_metric(ax, [e + 20 for e in right_epochs], avg_recall[:10], f'{experiment_name} Recall', linestyle,
                        marker)

# Adding vertical lines to separate the sections
plt.axvline(x=10, color='black', linestyle='--')
plt.axvline(x=20, color='black', linestyle='--')

# Custom x-tick labels
custom_labels = [str((i % 10) if (i % 10) != 0 else 10) for i in np.arange(1, 31)]
plt.xticks(ticks=np.arange(1, 31), labels=custom_labels)

# Labels and title
plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.title('Accuracy, Precision, and Recall Over Epochs')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

# Display the plot
plt.tight_layout()
plt.show()
