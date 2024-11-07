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

        # initiate optimizers
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        for epoch in range(self.epochs):
            for step, real_data in enumerate(self.x_train_ds.take(self.steps_per_epoch)):
                # Assume real_data contains both normal and intrusive traffic
                # Split the real_data into normal and intrusive samples
                normal_data = real_data[
                    real_data['Label' if self.dataset_used == "IOTBOTNET" else 'label'] == 1]  # Real normal traffic
                intrusive_data = real_data[
                    real_data['Label' if self.dataset_used == "IOTBOTNET" else 'label'] == 0]  # Real malicious traffic

                # Generate fake data using the generator
                noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
                generated_data = self.generator(noise, training=False)

                # captures the discriminator’s operations to compute the gradients for adjusting its weights based on how well it classified real vs. fake data.
                # using tape to track trainable variables during discriminator classification and loss calculations
                with tf.GradientTape() as tape:
                    # Discriminator outputs based on its classifications from inputted data in parameters
                    real_normal_output = self.discriminator(normal_data, training=True)
                    real_intrusive_output = self.discriminator(intrusive_data, training=True)
                    fake_output = self.discriminator(generated_data, training=True)

                    # Loss calculation for normal, intrusive, and fake data
                    loss = discriminator_server_loss(real_normal_output, real_intrusive_output, fake_output)

                # calculate the gradient based on the loss respect to the weights of the model
                gradients = tape.gradient(loss, self.discriminator.trainable_variables)

                # Update the model based on the gradient of the loss respect to the weights of the model
                optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

                if step % 100 == 0:
                    print(f"Epoch {epoch + 1}, Step {step}, D Loss: {loss.numpy()}")

            # After each epoch, evaluate on the validation set
            val_disc_loss = self.evaluate_validation()
            print(f'Epoch {epoch + 1}, Validation D Loss: {val_disc_loss}')

        # Save the fine-tuned model
        model.save("federated_model_fine_tuned.h5")
        print(f"Model fine-tuned and saved after round {server_round}.")

        # Send updated weights back to clients
        return model.get_weights(), {}