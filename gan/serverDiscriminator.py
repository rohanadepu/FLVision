import flwr as fl
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# Custom FedAvg strategy with server-side model training and saving
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, server_data, server_labels, epochs=5, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.server_data = server_data
        self.server_labels = server_labels
        self.epochs = epochs
        self.batch_size = batch_size

    def on_fit_end(self, server_round, aggregated_weights, failures):
        # Create model and set aggregated weights
        model = create_model()
        model.set_weights(aggregated_weights)

        # Compile model for server-side training
        model.compile(optimizer=Adam(learning_rate=0.0001), loss="categorical_crossentropy", metrics=["accuracy"])

        # Further train model on server-side data
        if self.server_data is not None and self.server_labels is not None:
            print(f"Training aggregated model on server-side data for {self.epochs} epochs...")
            model.fit(self.server_data, self.server_labels, epochs=self.epochs, batch_size=self.batch_size, verbose=1)

        # Save the fine-tuned model
        model.save("federated_model_fine_tuned.h5")
        print(f"Model fine-tuned and saved after round {server_round}.")

        # Send updated weights back to clients
        return model.get_weights(), {}


# Assuming a function that creates the model architecture
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(512, activation='relu', input_shape=(input_dim,)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')  # Assuming 3 classes
    ])
    return model


# Load or define server-side data
# Replace these lines with your actual server data loading process
server_data = ...  # Server-side data for further training (e.g., representative or synthetic data)
server_labels = ...  # Corresponding labels for server-side data

# Parsing command line arguments
parser = argparse.ArgumentParser(description="Federated Learning Server with Model Saving")
parser.add_argument("--rounds", type=int, choices=range(1, 11), default=8, help="Rounds of training 1-10")
parser.add_argument("--min_clients", type=int, choices=range(1, 7), default=2,
                    help="Minimum number of clients required for training")

args = parser.parse_args()
roundInput = args.rounds
minClients = args.min_clients

# Start the federated server with custom strategy
fl.server.start_server(
    config=fl.server.ServerConfig(num_rounds=roundInput),
    strategy=SaveModelStrategy(
        server_data=server_data,
        server_labels=server_labels,
        epochs=3,  # Set the number of server-side fine-tuning epochs
        batch_size=32,
        min_fit_clients=minClients,
        min_evaluate_clients=minClients,
        min_available_clients=minClients
    )
)
