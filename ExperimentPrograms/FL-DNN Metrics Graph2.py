import matplotlib.pyplot as plt
import numpy as np

# Sample data
rounds = np.arange(1, 4)

acc_baseline_node = [0.9, 0.9, 0.9]
acc_no_defense_1_node = [0.7, 0.75, 0.78]
acc_no_defense_3_nodes = [0.5, 0.55, 0.6]
acc_adv_training_1_node = [0.6, 0.65, 0.7]
acc_adv_training_3_nodes = [0.55, 0.6, 0.65]

prec_baseline_node = [0.9, 0.9, 0.9]
prec_no_defense_1_node = [0.7, 0.75, 0.78]
prec_no_defense_3_nodes = [0.5, 0.55, 0.6]
prec_adv_training_1_node = [0.6, 0.65, 0.7]
prec_adv_training_3_nodes = [0.55, 0.6, 0.65]

recall_baseline_node = [0.9, 0.9, 0.9]
recall_no_defense_1_node = [0.7, 0.75, 0.78]
recall_no_defense_3_nodes = [0.5, 0.55, 0.6]
recall_adv_training_1_node = [0.6, 0.65, 0.7]
recall_adv_training_3_nodes = [0.55, 0.6, 0.65]

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(9, 5))

# Plot Accuracy
axs[0].plot(rounds, acc_baseline_node, label='No Defense - Baseline', color='blue', marker='x')
axs[0].plot(rounds, acc_no_defense_1_node, label='No Defense - 1 Node - FN66', color='green', marker='o')
axs[0].plot(rounds, acc_no_defense_3_nodes, label='No Defense - 3 Nodes - FN66', color='green', marker='s')
axs[0].plot(rounds, acc_adv_training_1_node, label='Adversarial Training - 1 Node - FN66', color='olive', marker='o')
axs[0].plot(rounds, acc_adv_training_3_nodes, label='Adversarial Training - 3 Nodes - FN66', color='olive', marker='s')
axs[0].set_title('Accuracy', fontsize=16)
axs[0].set_ylabel('Metrics', fontweight='bold', fontsize=12)
axs[0].set_xticks(rounds)
axs[0].set_xticklabels(rounds)

# Plot Precision
axs[1].plot(rounds, prec_baseline_node, color='blue', marker='x')
axs[1].plot(rounds, prec_no_defense_1_node, color='green', marker='o')
axs[1].plot(rounds, prec_no_defense_3_nodes, color='green', marker='s')
axs[1].plot(rounds, prec_adv_training_1_node, color='olive', marker='o')
axs[1].plot(rounds, prec_adv_training_3_nodes, color='olive', marker='s')
axs[1].set_title('Precision', fontsize=16)
axs[1].set_xlabel('Rounds', fontweight='bold', fontsize=12)
axs[1].set_xticks(rounds)
axs[1].set_xticklabels(rounds)
axs[1].set_yticklabels([])  # Remove y-tick labels on the second graph

# Plot Recall
axs[2].plot(rounds, recall_baseline_node, color='blue', marker='x')
axs[2].plot(rounds, recall_no_defense_1_node, color='green', marker='o')
axs[2].plot(rounds, recall_no_defense_3_nodes, color='green', marker='s')
axs[2].plot(rounds, recall_adv_training_1_node, color='olive', marker='o')
axs[2].plot(rounds, recall_adv_training_3_nodes, color='olive', marker='s')
axs[2].set_title('Recall', fontsize=16)
axs[2].set_xticks(rounds)
axs[2].set_xticklabels(rounds)
axs[2].set_yticklabels([])  # Remove y-tick labels on the second graph


# Adding a single legend outside the subplots
fig.legend(['No Defense - Baseline', 'No Defense - 1 Node - FN66', 'No Defense - 3 Nodes - FN66',
            'Adversarial Training - 1 Node - FN66', 'Adversarial Training - 3 Nodes - FN66'],
           loc='upper center', bbox_to_anchor=(0.5, 0.16), ncol=2, frameon=True, fontsize=12, markerscale=1.5)

# Adjust layout to give more space for the legend
plt.subplots_adjust(left=0.07, right=0.99, bottom=0.25, wspace=0.0)

# Ensure the entire figure, including the legend, is displayed properly
plt.show()
