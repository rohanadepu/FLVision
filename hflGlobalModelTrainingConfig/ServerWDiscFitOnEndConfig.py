import flwr as fl
import argparse
import random
import tensorflow as tf
from tensorflow.keras.optimizers import Adam


# Custom FedAvg strategy with server-side model training and saving
class DiscriminatorSyntheticStrategy(fl.server.strategy.FedAvg):
    def __init__(self, discriminator, generator, x_train, x_val, y_train, y_val, x_test, y_test, BATCH_SIZE, noise_dim, epochs, steps_per_epoch,
                 dataset_used, input_dim, **kwargs):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.generator = generator  # Generator is fixed during discriminator training
        # create model
        self.discriminator = discriminator

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val  # Add validation data
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test

        self.BATCH_SIZE = BATCH_SIZE
        self.noise_dim = noise_dim
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.dataset_used = dataset_used3

        self.x_train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(self.BATCH_SIZE)
        self.x_val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(self.BATCH_SIZE)

        self.disc_optimizer = Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)

        self.precision = Precision()
        self.recall = Recall()
        self.accuracy = BinaryAccuracy()


    def update_critic_metrics(self, real_output, fake_output, threshold=0.0):
        """
        Update classification metrics by thresholding raw scores.
        Real samples are expected to have scores > threshold (label 1),
        while fake samples should be below threshold (label 0).
        """
        # Convert scores to binary predictions based on the threshold.
        real_preds = tf.cast(real_output > threshold, tf.int32)
        fake_preds = tf.cast(fake_output > threshold, tf.int32)

        # Create corresponding labels.
        real_labels = tf.zeros_like(real_preds)
        fake_labels = tf.ones_like(fake_preds)

        # Concatenate predictions and labels.
        all_preds = tf.concat([real_preds, fake_preds], axis=0)
        all_labels = tf.concat([real_labels, fake_labels], axis=0)

        # Update metrics.
        self.accuracy.update_state(all_labels, all_preds)
        self.precision.update_state(all_labels, all_preds)
        self.recall.update_state(all_labels, all_preds)

    def log_metrics(self, step, disc_loss):
        # Retrieve critic (discriminator) metrics.
        acc = self.accuracy.result().numpy()
        prec = self.precision.result().numpy()
        rec = self.recall.result().numpy()

        print(f"Step {step}, D Loss: {disc_loss.numpy():.4f}")
        print(f"Critic Metrics -- Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

    def reset_metrics(self):
        self.accuracy.reset_states()
        self.precision.reset_states()
        self.recall.reset_states()

    # Sending training params to host
    def get_parameters(self, config):
        # Combine generator and discriminator weights into a single list
        return self.discriminator.get_weights()

    # Loss Function
    def discriminator_loss(self, real_output, fake_output, gradient_penalty):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(
            real_output) + 15.0 * gradient_penalty  # Increased from 10.0 to 15.0

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def gradient_penalty(self, discriminator, real_data, fake_data):
        feature_dim = tf.shape(real_data)[1]

        # Generate alpha and broadcast it
        alpha = tf.random.uniform([tf.shape(real_data)[0], 1], 0., 1., dtype=tf.float32)
        alpha = tf.broadcast_to(alpha, [tf.shape(real_data)[0], feature_dim])

        real_data = tf.cast(real_data, tf.float32)
        fake_data = tf.cast(fake_data, tf.float32)

        interpolated = alpha * real_data + (1 - alpha) * fake_data

        if random.random() < 0.01:  # Log occasionally
            print(
                f"Interpolated Mean: {tf.reduce_mean(interpolated).numpy()}, Std: {tf.math.reduce_std(interpolated).numpy()}")

        with tf.GradientTape() as tape:
            tape.watch(interpolated)
            pred = discriminator(interpolated, training=True)

        grads = tape.gradient(pred, [interpolated])[0]
        grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1]))

        if random.random() < 0.01:
            print(
                f"Gradient Norm Mean: {tf.reduce_mean(grad_norm).numpy()}, Min: {tf.reduce_min(grad_norm).numpy()}, Max: {tf.reduce_max(grad_norm).numpy()}")

        return tf.reduce_mean((grad_norm - 1.0) ** 2)

    def aggregate_fit(self, server_round, results, failures):
        # -- Set the model with global weights, Bring in the parameters for the global model --#
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving global model after round {server_round}...")
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters[0])
            if len(aggregated_weights) == len(self.nids.get_weights()):
                self.nids.set_weights(aggregated_weights)
        # EoF Set global weights

        for epoch in range(self.epochs):
            for step, (real_data, real_labels) in enumerate(train_data.take(self.steps_per_epoch)):
                # Create masks for normal and intrusive traffic based on labels
                normal_mask = tf.equal(real_labels, 1)  # Assuming label 1 for normal
                # Filter data based on these masks
                normal_data = tf.boolean_mask(real_data, normal_mask)

                # Generate fake data using the generator
                noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
                generated_data = self.generator(noise, training=False)

                # captures the discriminator’s operations to compute the gradients for adjusting its weights based on how well it classified real vs. fake data.
                # using tape to track trainable variables during discriminator classification and loss calculations
                with tf.GradientTape() as tape:
                    # Discriminator outputs based on its classifications from inputted data in parameters
                    real_normal_output = self.discriminator(normal_data, training=True)
                    fake_output = self.discriminator(generated_data, training=True)

                    # Loss calculation for normal, intrusive, and fake data
                    loss = discriminator_loss_synthetic(real_normal_output, fake_output)

                # calculate the gradient based on the loss respect to the weights of the model
                gradients = tape.gradient(loss, self.discriminator.trainable_variables)

                # Update the model based on the gradient of the loss respect to the weights of the model
                self.optimizer.apply_gradients(zip(gradients, self.discriminator.trainable_variables))

                if step % 100 == 0:
                    print(f"Epoch {epoch + 1}, Step {step}, D Loss: {loss.numpy()}")

            # After each epoch, evaluate on the validation set
            val_disc_loss = self.evaluate_validation()
            print(f'Epoch {epoch + 1}, Validation D Loss: {val_disc_loss}')

        # Save the fine-tuned model
        self.discriminator.save("disc_model_fine_tuned.h5")
        print(f"Model fine-tuned and saved after round {server_round}.")

        # Send updated weights back to clients
        return self.discriminator.get_weights(), {}

    # Function to evaluate the discriminator on validation data
    def evaluate_validation(self):
        # Generate fake samples
        noise = tf.random.normal([self.BATCH_SIZE, self.noise_dim])
        generated_samples = self.generator(noise, training=False)

        # Create a TensorFlow dataset for validation data
        val_data = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val)).batch(self.BATCH_SIZE)

        total_loss = 0
        for instances, labels in val_data:
            # Filter normal and intrusive instances
            normal_data = tf.boolean_mask(instances, tf.equal(labels, 1))  # Assuming label 1 for normal

            # Discriminator predictions
            real_normal_output = self.discriminator(normal_data, training=False)
            fake_output = self.discriminator(generated_samples, training=False)

            # Compute the loss for this batch
            batch_loss = discriminator_loss_synthetic(real_normal_output, fake_output)
            total_loss += batch_loss

        return float(total_loss.numpy())
