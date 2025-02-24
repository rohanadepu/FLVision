def hyperparameterLoading(model_type, X_train_data, regularizationEnabled, DP_enabled, earlyStopEnabled,
                          lrSchedRedEnabled, modelCheckpointEnabled):

    # initiate optional variables
    l2_alpha = None
    l2_norm_clip = None
    noise_multiplier = None
    num_microbatches = None
    metric_to_monitor_es = None
    es_patience = None
    restor_best_w = None
    metric_to_monitor_l2lr = None
    l2lr_patience = None
    save_best_only = None
    metric_to_monitor_mc = None
    checkpoint_mode = None
    regularizationEnabled = True
    noise_dim = None
    num_classes = None
    latent_dim = None
    betas = None
    learning_rate = None


    if model_type == 'NIDS':
        input_dim = X_train_data.shape[1]  # dependant for feature size

        BATCH_SIZE = 64  # 32 - 128; try 64, 96, 128; maybe intervals of 16, maybe even 256

        epochs = 5  # 1, 2 , 3 or 5 epochs

        # steps_per_epoch = (len(X_train_data) // batch_size) // epochs  # dependant  # debug
        # dependant between sample size of the dataset and the batch size chosen
        steps_per_epoch = len(X_train_data) // BATCH_SIZE

        learning_rate = 0.0001  # 0.001 or .0001
        betas = [0.9, 0.999]  # Stable

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

        # -- set hyperparameters for callback -- #

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
        print("Batch Size:", BATCH_SIZE)
        print(f"Steps per epoch (({len(X_train_data)} // {BATCH_SIZE})):", steps_per_epoch)
        # print(f"Steps per epoch (({len(X_train_data)} // {batch_size}) // {epochs}):", steps_per_epoch)  ## Debug
        print("Betas:", betas)
        print("Learning Rate:", learning_rate)

    elif model_type == 'GAN':
        BATCH_SIZE = 256
        noise_dim = 100
        steps_per_epoch = len(X_train_data) // BATCH_SIZE
        input_dim = X_train_data.shape[1]

        learning_rate = 0.0001
    elif model_type == 'WGAN-GP':
        BATCH_SIZE = 256
        noise_dim = 100
        steps_per_epoch = len(X_train_data) // BATCH_SIZE
        input_dim = X_train_data.shape[1]

        learning_rate = 0.0001

    elif model_type == 'AC-GAN':
        BATCH_SIZE = 256
        noise_dim = 100
        latent_dim = 100
        steps_per_epoch = len(X_train_data) // BATCH_SIZE
        input_dim = X_train_data.shape[1]
        # num_classes = len(np.unique(y_train_categorical))
        num_classes = 3

        if regularizationEnabled:
            l2_alpha = 0.01  # Increase if overfitting, decrease if underfitting

        betas = [0.9, 0.999]  # Stable
        learning_rate = 0.0001

    return (BATCH_SIZE, noise_dim, steps_per_epoch, input_dim, num_classes, latent_dim, betas, learning_rate, l2_alpha,
            l2_norm_clip, noise_multiplier, num_microbatches, metric_to_monitor_es, es_patience, restor_best_w,
            metric_to_monitor_l2lr, l2lr_patience, save_best_only, metric_to_monitor_mc, checkpoint_mode)