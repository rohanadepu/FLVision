import matplotlib.pyplot as plt
import numpy as np

# Sample data
rounds = np.arange(1, 4)

acc_baseline_node = [0.7054747343063354, 0.8180873394012451, 0.8508316278457642]
acc_no_defense_1_node = [0.2204739898443222, 0.7835087180137634, 0.7836562395095825]
acc_no_defense_3_nodes = [0.7999090552330017, 0.800010085105896, 0.800010085105896]
acc_adv_training_1_node = [0.7572298645973206, 0.7562969923019409, 0.7579432725906372]
acc_adv_training_3_nodes = [0.800000011920929, 0.800000011920929, 0.800000011920929]
acc_dp_training_1_node = [0.7626624703407288, 0.7825270295143127, 0.7820329070091248]
acc_dp_training_3_nodes = [0.7997569441795349, 0.7998148202896118, 0.7999884486198425]
acc_all_training_1_node = [0.23777511715888977, 0.43382132053375244, 0.6740862727165222]
acc_all_training_3_nodes = [0.745245635509491, 0.7671358585357666, 0.7767882943153381]


prec_baseline_node = [0.9982923269271851, 0.9991605877876282, 0.999468207359314]
prec_no_defense_1_node = [0.5853658318519592, 0.7835087180137634, 0.7836562395095825]
prec_no_defense_3_nodes = [0.7999898791313171, 0.800010085105896, 0.800010085105896]
prec_adv_training_1_node = [0.7582568526268005, 0.7562969923019409, 0.7584010362625122]
prec_adv_training_3_nodes = [0.800000011920929, 0.800000011920929, 0.800000011920929]
prec_dp_training_1_node = [0.7798234820365906, 0.782565712928772, 0.7824582457542419]
prec_dp_training_3_nodes = [0.799942135810852, 0.7999536991119385, 0.7999884486198425]
prec_all_training_1_node = [0.9820051193237305, 0.9997681975364685, 0.9996718764305115]
prec_all_training_3_nodes = [0.7886252999305725, 0.7935177087783813, 0.7955602407455444]

recall_baseline_node = [0.6329579949378967, 0.7732784748077393, 0.8139887452125549]
recall_no_defense_1_node = [0.018070021644234657, 0.9998117685317993, 1.0]
recall_no_defense_3_nodes = [1.0, 1.0, 1.0]
recall_adv_training_1_node = [0.9981914162635803, 0.9969615936279297, 0.9992042183876038]
recall_adv_training_3_nodes = [1.0, 1.0, 1.0]
recall_dp_training_1_node = [0.9708278179168701, 0.9999368786811829, 0.9993054270744324]
recall_dp_training_3_nodes = [0.9997106194496155, 0.9997829794883728, 1.0]
recall_all_training_1_node = [0.024359138682484627, 0.27502870559692383, 0.5828338265419006]
recall_all_training_3_nodes = [0.9311261177062988, 0.9582734704017639, 0.9703390002250671]

loss_baseline_node = [0.6034379005432129, 0.4495866596698761, 0.3253514766693115]
loss_no_defense_1_node = [0.7151358127593994, 0.5569000244140625, 0.7246222496032715]
loss_no_defense_3_nodes = [0.4836266338825226, 0.5258604288101196, 0.7234001159667969]
loss_adv_training_1_node = [0.4903354048728943, 0.6697665452957153, 0.7958751320838928]
loss_adv_training_3_nodes = [0.7518763542175293, 0.5796926617622375, 0.5931104421615601]
loss_dp_training_1_node = [0.42342233657836914, 0.47095540165901184, 0.5522009134292603]
loss_dp_training_3_nodes = [0.5073424577713013, 0.4574194848537445, 0.4639563262462616]
loss_all_training_1_node = [0.8245664238929749, 0.7991995811462402, 0.6560931205749512]
loss_all_training_3_nodes = [0.6677418351173401, 0.6529982089996338, 0.6371238827705383]



# Create subplots
fig, axs = plt.subplots(1, 4, figsize=(9, 5))

# Plot Accuracy
axs[0].plot(rounds, acc_no_defense_1_node, label='No Defense - 1 Node - FN66', color='green', marker='o')
axs[0].plot(rounds, acc_no_defense_3_nodes, label='No Defense - 3 Nodes - FN66', color='green', marker='s')
axs[0].plot(rounds, acc_adv_training_1_node, label='Adversarial Training - 1 Node - FN66', color='olive', marker='o')
axs[0].plot(rounds, acc_adv_training_3_nodes, label='Adversarial Training - 3 Nodes - FN66', color='olive', marker='s')
axs[0].set_title('Accuracy', fontsize=16)
axs[0].set_ylabel('Metrics', fontweight='bold', fontsize=14)
axs[0].set_xticks(rounds)
axs[0].set_xticklabels(rounds, fontsize=12, rotation=0)  # Rotating labels for better separation
axs[0].tick_params(axis='y', labelsize=12)  # Enlarging y-axis tick labels

# Plot Precision

axs[1].plot(rounds, prec_no_defense_1_node, color='green', marker='o')
axs[1].plot(rounds, prec_no_defense_3_nodes, color='green', marker='s')
axs[1].plot(rounds, prec_adv_training_1_node, color='olive', marker='o')
axs[1].plot(rounds, prec_adv_training_3_nodes, color='olive', marker='s')
axs[1].set_title('Precision', fontsize=16)
axs[1].set_xlabel('Rounds', fontweight='bold', fontsize=14)
axs[1].set_xticks(rounds)
axs[1].set_xticklabels(rounds, fontsize=12, rotation=0)  # Rotating labels for better separation
axs[1].set_yticklabels([])  # Remove y-tick labels on the second graph

# Plot Recall

axs[2].plot(rounds, recall_no_defense_1_node, color='green', marker='o')
axs[2].plot(rounds, recall_no_defense_3_nodes, color='green', marker='s')
axs[2].plot(rounds, recall_adv_training_1_node, color='olive', marker='o')
axs[2].plot(rounds, recall_adv_training_3_nodes, color='olive', marker='s')
axs[2].set_title('Recall', fontsize=16)
axs[2].set_xticks(rounds)
axs[2].set_xticklabels(rounds, fontsize=12, rotation=0)  # Rotating labels for better separation
axs[2].set_yticklabels([])  # Remove y-tick labels on the second graph

# Plot Recall
axs[3].plot(rounds, loss_no_defense_1_node, color='green', marker='o')
axs[3].plot(rounds, loss_no_defense_3_nodes, color='green', marker='s')
axs[3].plot(rounds, loss_adv_training_1_node, color='olive', marker='o')
axs[3].plot(rounds, loss_adv_training_3_nodes, color='olive', marker='s')
axs[3].set_title('Loss', fontsize=16)
axs[3].set_xticks(rounds)
axs[3].set_xticklabels(rounds, fontsize=12, rotation=0)  # Rotating labels for better separation
axs[3].set_yticklabels([])  # Remove y-tick labels on the second graph


# Adding a single legend outside the subplots
fig.legend(['No Defense - Baseline', 'No Defense - 1 Node - FN66', 'No Defense - 3 Nodes - FN66',
            'Adversarial Training - 1 Node - FN66', 'Adversarial Training - 3 Nodes - FN66'],
           loc='upper center', bbox_to_anchor=(0.5, 0.16), ncol=2, frameon=True, fontsize=12, markerscale=1.5)

# Adjust layout to give more space for the legend
plt.subplots_adjust(left=0.08, right=0.99, bottom=0.25, wspace=0.00)

# Ensure the entire figure, including the legend, is displayed properly
plt.show()
