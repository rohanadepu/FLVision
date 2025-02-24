# Load the pretrained discriminator model
if args.pretrained_discriminator:
    print(f"Loading pretrained discriminator from {args.pretrained_discriminator}")
    discriminator = tf.keras.models.load_model(args.pretrained_discriminator)
else:
    print("No pretrained discriminator provided. Creating a new discriminator.")
    discriminator = create_discriminator(input_dim)  # Define this function

# Start the federated server
fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=roundInput),
    strategy=DiscriminatorFullStrategy(
        discriminator,
        X_train_data, X_val_data, y_train_data, y_val_data, X_test_data, y_test_data,
        BATCH_SIZE, noise_dim, epochs, steps_per_epoch, dataset_used, input_dim,
        min_fit_clients=minClients,
        min_evaluate_clients=minClients,
        min_available_clients=minClients
    )
)
