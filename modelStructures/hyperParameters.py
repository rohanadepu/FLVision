def nids_hyperparam_full(dataset_used, X_train_data, regularizationEnabled):
    print("\n /////////////////////////////////////////////// \n")

    # base hyperparameters for most models
    model_name = dataset_used  # name for file

    input_dim = X_train_data.shape[1]  # dependant for feature size

    batch_size = 64  # 32 - 128; try 64, 96, 128; maybe intervals of 16, maybe even 256

    epochs = 5  # 1, 2 , 3 or 5 epochs

    # steps_per_epoch = (len(X_train_data) // batch_size) // epochs  # dependant  # debug
    # dependant between sample size of the dataset and the batch size chosen
    steps_per_epoch = len(X_train_data) // batch_size

    learning_rate = 0.0001  # 0.001 or .0001
    betas = [0.9, 0.999]  # Stable

    # initiate optional variables
    l2_alpha = None
    l2_norm_clip = None
    noise_multiplier = None
    num_microbatches = None
    adv_portion = None
    metric_to_monitor_es = None
    es_patience = None
    restor_best_w = None
    metric_to_monitor_l2lr = None
    l2lr_patience = None
    save_best_only = None
    metric_to_monitor_mc = None
    checkpoint_mode = None

    # regularization param
    if regularizationEnabled:
        l2_alpha = 0.0001  # Increase if overfitting, decrease if underfitting

        if DP_enabled:
            l2_alpha = 0.001  # Increase if overfitting, decrease if underfitting

        print("\nRegularization Parameter:")
        print("L2_alpha:", l2_alpha)

    if DP_enabled:
        num_microbatches = 1  # this is bugged keep at 1

        noise_multiplier = 0.3  # need to optimize noise budget and determine if noise is properly added
        l2_norm_clip = 1.5  # determine if l2 needs to be tuned as well 1.0 - 2.0

        epochs = 10
        learning_rate = 0.0007  # will be optimized

        print("\nDifferential Privacy Parameters:")
        print("L2_norm clip:", l2_norm_clip)
        print("Noise Multiplier:", noise_multiplier)
        print("MicroBatches", num_microbatches)

    if adversarialTrainingEnabled:
        adv_portion = 0.05  # in intervals of 0.05 until to 0.20
        # adv_portion = 0.1
        learning_rate = 0.0001  # will be optimized

        print("\nAdversarial Training Parameter:")
        print("Adversarial Sample %:", adv_portion * 100, "%")

    # set hyperparameters for callback

    # early stop
    if earlyStopEnabled:
        es_patience = 5  # 3 -10 epochs
        restor_best_w = True
        metric_to_monitor_es = 'val_loss'

        print("\nEarly Stop Callback Parameters:")
        print("Early Stop Patience:", es_patience)
        print("Early Stop Restore best weights?", restor_best_w)
        print("Early Stop Metric Monitored:", metric_to_monitor_es)

    # lr sched
    if lrSchedRedEnabled:
        l2lr_patience = 3  # eppoch when metric stops imporving
        l2lr_factor = 0.1  # Reduce lr to 10%
        metric_to_monitor_l2lr = 'val_auc'
        if DP_enabled:
            metric_to_monitor_l2lr = 'val_loss'

        print("\nLR sched Callback Parameters:")
        print("LR sched Patience:", l2lr_patience)
        print("LR sched Factor:", l2lr_factor)
        print("LR sched Metric Monitored:", metric_to_monitor_l2lr)

    # save best model
    if modelCheckpointEnabled:
        save_best_only = True
        checkpoint_mode = "min"
        metric_to_monitor_mc = 'val_loss'

        print("\nModel Checkpoint Callback Parameters:")
        print("Model Checkpoint Save Best only?", save_best_only)
        print("Model Checkpoint mode:", checkpoint_mode)
        print("Model Checkpoint Metric Monitored:", metric_to_monitor_mc)

    # 'val_loss' for general error, 'val_auc' for eval trade off for TP and TF rate for BC problems, "precision", "recall", ""F1-Score for imbalanced data

    print("\nBase Hyperparameters:")
    print("Input Dim (Feature Size):", input_dim)
    print("Epochs:", epochs)
    print("Batch Size:", batch_size)
    print(f"Steps per epoch (({len(X_train_data)} // {batch_size})):", steps_per_epoch)
    # print(f"Steps per epoch (({len(X_train_data)} // {batch_size}) // {epochs}):", steps_per_epoch)  ## Debug
    print("Betas:", betas)
    print("Learning Rate:", learning_rate)