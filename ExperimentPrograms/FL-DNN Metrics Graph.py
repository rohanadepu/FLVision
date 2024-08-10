import matplotlib.pyplot as plt
import numpy as np

# Sample data
epochs = np.arange(1, 4)


acc_baseline_node = [0.9, 0.9, 0.9]
acc_no_defense_1_node = [0.7, 0.75, 0.78]
acc_no_defense_3_nodes = [0.5, 0.55, 0.6]
acc_adv_training_1_node = [0.6, 0.65, 0.7]
acc_adv_training_3_nodes = [0.55, 0.6, 0.65]
# acc_diff_privacy_1_node = [0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.79, 0.8, 0.81, 0.82]
# acc_diff_privacy_3_nodes = [0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.73, 0.74, 0.75, 0.76]
# acc_all_defenses_1_node = [0.66, 0.68, 0.7, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79]
# acc_all_defenses_3_nodes = [0.61, 0.63, 0.65, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73]

prec_baseline_node = [0.9, 0.9, 0.9]
prec_no_defense_1_node = [0.7, 0.75, 0.78]
prec_no_defense_3_nodes = [0.5, 0.55, 0.6]
prec_adv_training_1_node = [0.6, 0.65, 0.7]
prec_adv_training_3_nodes = [0.55, 0.6, 0.65]
# prec_diff_privacy_1_node = [0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.79, 0.8, 0.81, 0.82]
# prec_diff_privacy_3_nodes = [0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.73, 0.74, 0.75, 0.76]
# prec_all_defenses_1_node = [0.66, 0.68, 0.7, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79]
# prec_all_defenses_3_nodes = [0.61, 0.63, 0.65, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73]

recall_baseline_node = [0.9, 0.9, 0.9]
recall_no_defense_1_node = [0.7, 0.75, 0.78]
recall_no_defense_3_nodes = [0.5, 0.55, 0.6]
recall_adv_training_1_node = [0.6, 0.65, 0.7]
recall_adv_training_3_nodes = [0.55, 0.6, 0.65]
# recall_diff_privacy_1_node = [0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.79, 0.8, 0.81, 0.82]
# recall_diff_privacy_3_nodes = [0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.73, 0.74, 0.75, 0.76]
# recall_all_defenses_1_node = [0.66, 0.68, 0.7, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79]
# recall_all_defenses_3_nodes = [0.61, 0.63, 0.65, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73]

# Create new x values for each section
epochs_acc = np.arange(1, 4)
epochs_prec = np.arange(1, 4)
epochs_recall = np.arange(1, 4)

# Plotting
plt.figure(figsize=(9, 6.3))

# Accuracy
plt.plot(epochs_acc, acc_baseline_node, label='No Defense - Baseline Accuracy', color='blue', marker='x')
plt.plot(epochs_acc, acc_no_defense_1_node, label='No Defense - 1 Node - Baseline Accuracy', color='green', marker='o')
plt.plot(epochs_acc, acc_no_defense_3_nodes, label='No Defense - 3 Nodes - FN66 Accuracy', color='green', marker='s')
plt.plot(epochs_acc, acc_adv_training_1_node, label='Adversarial Training - 1 Node - FN66 Accuracy', color='olive', marker='o')
plt.plot(epochs_acc, acc_adv_training_3_nodes, label='Adversarial Training - 3 Nodes - FN66 Accuracy', color='olive', marker='s')

# Precision
plt.plot(epochs_prec + 3, prec_baseline_node, label='No Defense - Baseline Precision', color='blue', marker='x')
plt.plot(epochs_prec + 3, prec_no_defense_1_node, label='No Defense - 1 Node - Baseline Precision', color='green', marker='o')
plt.plot(epochs_prec + 3, prec_no_defense_3_nodes, label='No Defense - 3 Nodes - FN66 Precision', color='green', marker='s')
plt.plot(epochs_prec + 3, prec_adv_training_1_node, label='Adversarial Training - 1 Node - FN66 Precision', color='olive', marker='o')
plt.plot(epochs_prec + 3, prec_adv_training_3_nodes, label='Adversarial Training - 3 Nodes - FN66 Precision', color='olive', marker='s')

# Recall
plt.plot(epochs_recall + 6, recall_baseline_node, label='No Defense - Baseline Recall', color='blue', marker='x')
plt.plot(epochs_recall + 6, recall_no_defense_1_node, label='No Defense - 1 Node - Baseline Recall', color='green', marker='o')
plt.plot(epochs_recall + 6, recall_no_defense_3_nodes, label='No Defense - 3 Nodes - FN66 Recall', color='green', marker='s')
plt.plot(epochs_recall + 6, recall_adv_training_1_node, label='Adversarial Training - 1 Node - FN66 Recall', color='olive', marker='o')
plt.plot(epochs_recall + 6, recall_adv_training_3_nodes, label='Adversarial Training - 3 Nodes - FN66 Recall', color='olive', marker='s')

# Adding vertical lines to separate the sections
plt.axvline(x=3, color='black', linestyle='--')
plt.axvline(x=6, color='black', linestyle='--')

# Custom x-tick labels
custom_labels = [str((i % 3) if (i % 3) != 0 else 3) for i in np.arange(1, 10)]
plt.xticks(ticks=np.arange(1, 10), labels=custom_labels)

# Adjusting the x-axis limits to be closer to the vertical lines
plt.xlim(0.8, 9.1)

# Labels and title
plt.xlabel('Rounds')
plt.ylabel('Metric')
plt.title('Accuracy, Precision, and Recall Over Epochs')
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)

# Display the plot
plt.tight_layout()
plt.show()
