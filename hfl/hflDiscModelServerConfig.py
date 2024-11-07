import flwr as fl
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from hflDiscModelConfig import create_discriminator

# Custom FedAvg strategy with server-side model training and saving
class SaveModelStrategy(fl.server.strategy.FedAvg):
    def __init__(self, server_data, server_labels, epochs=5, batch_size=32, **kwargs):
        super().__init__(**kwargs)
        self.server_data = server_data
        self.server_labels = server_labels
        self.epochs = epochs
        self.batch_size = batch_size

    def on_fit_end(self, server_round, aggregated_weights, failures, input_dim):
        # Create model and set aggregated weights
        model = create_discriminator(input_dim=input_dim)
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