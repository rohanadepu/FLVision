import matplotlib.pyplot as plt
import numpy as np


def plot_attack_detection_performance(true_positives=85, true_negatives=90, false_positives=10, false_negatives=15,
                                      model_name="NIDS Model", dataset="CICIOT",
                                      attack_types=None, attack_detection_rates=None):
    """
    Create a bar chart showing the performance metrics for network attack detection.

    Parameters:
    -----------
    true_positives : int
        Number of correctly identified attacks
    true_negatives : int
        Number of correctly identified normal traffic
    false_positives : int
        Number of normal traffic incorrectly labeled as attacks
    false_negatives : int
        Number of attacks incorrectly labeled as normal traffic
    model_name : str
        Name of the model used for detection
    dataset : str
        Name of the dataset used (e.g., "CICIOT", "IOTBOTNET", "IOT")
    attack_types : list of str, optional
        List of attack types to show detection rates for
    attack_detection_rates : list of float, optional
        List of detection rates corresponding to attack_types
    """

    # Calculate metrics
    total = true_positives + true_negatives + false_positives + false_negatives
    accuracy = 100 * (true_positives + true_negatives) / total
    precision = 100 * true_positives / (true_positives + false_positives) if (
                                                                                         true_positives + false_positives) > 0 else 0
    recall = 100 * true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    # Set up figure and primary axis for main metrics
    fig = plt.figure(figsize=(14, 10))

    # First subplot for confusion matrix data
    ax1 = plt.subplot2grid((2, 2), (0, 0))

    # Create grouped bar chart for TP, TN, FP, FN
    categories = ['True\nPositives', 'True\nNegatives', 'False\nPositives', 'False\nNegatives']
    values = [true_positives, true_negatives, false_positives, false_negatives]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']

    bars1 = ax1.bar(categories, values, color=colors, width=0.6)

    # Add data labels to the bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # Set labels and title
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)

    # Second subplot for performance metrics
    ax2 = plt.subplot2grid((2, 2), (0, 1))

    # Create bar chart for accuracy, precision, recall, F1
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    performance = [accuracy, precision, recall, f1_score]
    colors = ['#9b59b6', '#1abc9c', '#34495e', '#2980b9']

    bars2 = ax2.bar(metrics, performance, color=colors, width=0.6)

    # Add data labels
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Set labels and title
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_ylim(0, 110)  # Set y-axis to go from 0 to 110 to make room for text
    ax2.grid(axis='y', linestyle='--', alpha=0.7)

    # Third subplot for attack type detection rates (if provided)
    if attack_types and attack_detection_rates:
        ax3 = plt.subplot2grid((2, 2), (1, 0), colspan=2)

        # Sort attack types by detection rate for better visualization
        sorted_indices = np.argsort(attack_detection_rates)
        sorted_attack_types = [attack_types[i] for i in sorted_indices]
        sorted_rates = [attack_detection_rates[i] for i in sorted_indices]

        # Create color gradient based on detection rates
        colors = plt.cm.RdYlGn(np.array(sorted_rates) / 100)

        bars3 = ax3.bar(sorted_attack_types, sorted_rates, color=colors, width=0.6)

        # Add data labels
        for bar in bars3:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')

        # Set labels and title
        ax3.set_title('Detection Rate by Attack Type', fontsize=14, pad=20)
        ax3.set_xlabel('Attack Type', fontsize=12)
        ax3.set_ylabel('Detection Rate (%)', fontsize=12)
        ax3.set_ylim(0, 110)  # Set y-axis to go from 0 to 110 to make room for text
        ax3.grid(axis='y', linestyle='--', alpha=0.7)

        # Rotate x-axis labels if there are many attack types
        if len(attack_types) > 5:
            plt.setp(ax3.get_xticklabels(), rotation=45, ha='right')



    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    return fig


# Example usage with default values
fig = plot_attack_detection_performance()

# Example with custom values
# fig = plot_attack_detection_performance(
#     true_positives=120,
#     true_negatives=150,
#     false_positives=8,
#     false_negatives=12,
#     model_name="WGAN-GP Federated Model",
#     dataset="IOTBOTNET",
#     attack_types=["DDoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce"],
#     attack_detection_rates=[96.5, 88.2, 92.7, 75.4, 89.1, 94.3]
# )

plt.show()