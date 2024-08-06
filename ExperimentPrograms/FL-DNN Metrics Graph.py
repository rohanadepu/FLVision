import matplotlib.pyplot as plt
import numpy as np

# Sample data
epochs = np.arange(1, 11)
recall_no_defense_1_node = [0.7, 0.75, 0.78, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89]
recall_no_defense_3_nodes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.83, 0.84]
recall_adv_training_1_node = [0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.81, 0.83, 0.85]
recall_adv_training_3_nodes = [0.55, 0.6, 0.65, 0.7, 0.73, 0.75, 0.77, 0.78, 0.79, 0.8]
# recall_diff_privacy_1_node = [0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.79, 0.8, 0.81, 0.82]
# recall_diff_privacy_3_nodes = [0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.73, 0.74, 0.75, 0.76]
# recall_all_defenses_1_node = [0.66, 0.68, 0.7, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79]
# recall_all_defenses_3_nodes = [0.61, 0.63, 0.65, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73]

acc_no_defense_1_node = [0.7, 0.75, 0.78, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89]
acc_no_defense_3_nodes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.83, 0.84]
acc_adv_training_1_node = [0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.81, 0.83, 0.85]
acc_adv_training_3_nodes = [0.55, 0.6, 0.65, 0.7, 0.73, 0.75, 0.77, 0.78, 0.79, 0.8]
# acc_diff_privacy_1_node = [0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.79, 0.8, 0.81, 0.82]
# acc_diff_privacy_3_nodes = [0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.73, 0.74, 0.75, 0.76]
# acc_all_defenses_1_node = [0.66, 0.68, 0.7, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79]
# acc_all_defenses_3_nodes = [0.61, 0.63, 0.65, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73]

prec_no_defense_1_node = [0.7, 0.75, 0.78, 0.82, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89]
prec_no_defense_3_nodes = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.82, 0.83, 0.84]
prec_adv_training_1_node = [0.6, 0.65, 0.7, 0.72, 0.75, 0.78, 0.8, 0.81, 0.83, 0.85]
prec_adv_training_3_nodes = [0.55, 0.6, 0.65, 0.7, 0.73, 0.75, 0.77, 0.78, 0.79, 0.8]
# prec_diff_privacy_1_node = [0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.79, 0.8, 0.81, 0.82]
# prec_diff_privacy_3_nodes = [0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.73, 0.74, 0.75, 0.76]
# prec_all_defenses_1_node = [0.66, 0.68, 0.7, 0.72, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79]
# prec_all_defenses_3_nodes = [0.61, 0.63, 0.65, 0.67, 0.68, 0.69, 0.7, 0.71, 0.72, 0.73]

# Create new x values for each section
epochs_acc = np.arange(1, 11)
epochs_prec = np.arange(1, 11)
epochs_recall = np.arange(1, 11)

# Plotting
plt.figure(figsize=(20, 8))

# Accuracy
plt.plot(epochs_acc, acc_no_defense_1_node, label='No Defense - 1 Node - Baseline Accuracy', color='blue', marker='o')
plt.plot(epochs_acc, acc_no_defense_3_nodes, label='No Defense - 3 Nodes - FN66 Accuracy', color='orange', marker='o')
plt.plot(epochs_acc, acc_adv_training_1_node, label='Adversarial Training - 1 Node - FN66 Accuracy', color='green', marker='o')
plt.plot(epochs_acc, acc_adv_training_3_nodes, label='Adversarial Training - 3 Nodes - FN66 Accuracy', color='red', marker='o')
# plt.plot(epochs_acc, acc_diff_privacy_1_node, label='Differential Privacy - 1 Node - FN66 Accuracy', color='purple', marker='o')
# plt.plot(epochs_acc, acc_diff_privacy_3_nodes, label='Differential Privacy - 3 Nodes - FN66 Accuracy', color='pink', marker='o')
# plt.plot(epochs_acc, acc_all_defenses_1_node, label='All Defenses - 1 Node - FN66 Accuracy', color='brown', marker='o')
# plt.plot(epochs_acc, acc_all_defenses_3_nodes, label='All Defenses - 3 Nodes - FN66 Accuracy', color='yellow', marker='o')

# Precision
plt.plot(epochs_prec + 10, prec_no_defense_1_node, label='No Defense - 1 Node - Baseline Precision', color='blue', marker='x')
plt.plot(epochs_prec + 10, prec_no_defense_3_nodes, label='No Defense - 3 Nodes - FN66 Precision', color='orange', marker='x')
plt.plot(epochs_prec + 10, prec_adv_training_1_node, label='Adversarial Training - 1 Node - FN66 Precision', color='green', marker='x')
plt.plot(epochs_prec + 10, prec_adv_training_3_nodes, label='Adversarial Training - 3 Nodes - FN66 Precision', color='red', marker='x')
# plt.plot(epochs_prec + 10, prec_diff_privacy_1_node, label='Differential Privacy - 1 Node - FN66 Precision', color='purple', marker='x')
# plt.plot(epochs_prec + 10, prec_diff_privacy_3_nodes, label='Differential Privacy - 3 Nodes - FN66 Precision', color='pink', marker='x')
# plt.plot(epochs_prec + 10, prec_all_defenses_1_node, label='All Defenses - 1 Node - FN66 Precision', color='brown', marker='x')
# plt.plot(epochs_prec + 10, prec_all_defenses_3_nodes, label='All Defenses - 3 Nodes - FN66 Precision', color='yellow', marker='x')

# Recall
plt.plot(epochs_recall + 20, recall_no_defense_1_node, label='No Defense - 1 Node - Baseline Recall', color='blue', marker='s')
plt.plot(epochs_recall + 20, recall_no_defense_3_nodes, label='No Defense - 3 Nodes - FN66 Recall', color='orange', marker='s')
plt.plot(epochs_recall + 20, recall_adv_training_1_node, label='Adversarial Training - 1 Node - FN66 Recall', color='green', marker='s')
plt.plot(epochs_recall + 20, recall_adv_training_3_nodes, label='Adversarial Training - 3 Nodes - FN66 Recall', color='red', marker='s')
# plt.plot(epochs_recall + 20, recall_diff_privacy_1_node, label='Differential Privacy - 1 Node - FN66 Recall', color='purple', marker='s')
# plt.plot(epochs_recall + 20, recall_diff_privacy_3_nodes, label='Differential Privacy - 3 Nodes - FN66 Recall', color='pink', marker='s')
# plt.plot(epochs_recall + 20, recall_all_defenses_1_node, label='All Defenses - 1 Node - FN66 Recall', color='brown', marker='s')
# plt.plot(epochs_recall + 20, recall_all_defenses_3_nodes, label='All Defenses - 3 Nodes - FN66 Recall', color='yellow', marker='s')

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
