import flwr as fl
import argparse
import tensorflow as tf
import logging
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
import numpy as np


# loss based on correct classifications between normal, intrusive, and fake traffic
def discriminator_loss(self, real_normal_output, real_intrusive_output, fake_output):
    # Create labels matching the shape of the output logits
    real_normal_labels = tf.ones((tf.shape(real_normal_output)[0],), dtype=tf.int32)  # Label 1 for normal
    real_intrusive_labels = tf.zeros((tf.shape(real_intrusive_output)[0],), dtype=tf.int32)  # Label 0 for intrusive
    fake_labels = tf.fill([tf.shape(fake_output)[0]], 2)  # Label 2 for fake traffic

    # Calculate sparse categorical cross-entropy loss for each group separately
    real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(real_normal_labels, real_normal_output)
    real_intrusive_loss = tf.keras.losses.sparse_categorical_crossentropy(real_intrusive_labels,
                                                                          real_intrusive_output)
    fake_loss = tf.keras.losses.sparse_categorical_crossentropy(fake_labels, fake_output)

    # Compute the mean for each loss group independently
    mean_real_normal_loss = tf.reduce_mean(real_normal_loss)
    mean_real_intrusive_loss = tf.reduce_mean(real_intrusive_loss)
    mean_fake_loss = tf.reduce_mean(fake_loss)

    # Total loss as the average of mean losses for each group
    total_loss = (mean_real_normal_loss + mean_real_intrusive_loss + mean_fake_loss) / 3
    return total_loss


# Loss for intrusion training (normal and intrusive)
def discriminator_loss_intrusion(real_normal_output, real_intrusive_output):
    # Create labels matching the shape of the output logits
    real_normal_labels = tf.ones((tf.shape(real_normal_output)[0],), dtype=tf.int32)  # Label 0 for normal
    real_intrusive_labels = tf.zeros((tf.shape(real_intrusive_output)[0],), dtype=tf.int32)  # Label 1 for intrusive

    # Calculate sparse categorical cross-entropy loss for each group separately
    real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(real_normal_labels, real_normal_output)
    real_intrusive_loss = tf.keras.losses.sparse_categorical_crossentropy(real_intrusive_labels,
                                                                          real_intrusive_output)

    # Compute the mean for each loss group independently
    mean_real_normal_loss = tf.reduce_mean(real_normal_loss)
    mean_real_intrusive_loss = tf.reduce_mean(real_intrusive_loss)

    # Total loss as the average of mean losses for each group
    total_loss = (mean_real_normal_loss + mean_real_intrusive_loss) / 2

    # real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.zeros_like(real_normal_output), real_normal_output)  # Label 0 for normal
    # real_intrusive_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.ones_like(real_intrusive_output), real_intrusive_output)  # Label 1 for intrusive
    # total_loss = real_normal_loss + real_intrusive_loss

    return total_loss


# Loss for synthetic training (normal and fake)
def discriminator_loss_synthetic(real_normal_output, fake_output):
    # Create labels matching the shape of the output logits
    real_normal_labels = tf.ones((tf.shape(real_normal_output)[0],), dtype=tf.int32)  # Label 1 for normal
    fake_labels = tf.fill([tf.shape(fake_output)[0]], 2)  # Label 2 for fake traffic

    # Calculate sparse categorical cross-entropy loss for each group separately
    real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(real_normal_labels, real_normal_output)

    fake_loss = tf.keras.losses.sparse_categorical_crossentropy(fake_labels, fake_output)

    # Compute the mean for each loss group independently
    mean_real_normal_loss = tf.reduce_mean(real_normal_loss)
    mean_fake_loss = tf.reduce_mean(fake_loss)

    # Total loss as the average of mean losses for each group
    total_loss = (mean_real_normal_loss + mean_fake_loss) / 2

    # real_normal_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.zeros_like(real_normal_output), real_normal_output)  # Label 0 for normal
    # fake_loss = tf.keras.losses.sparse_categorical_crossentropy(tf.fill(tf.shape(fake_output), 2), fake_output)  # Label 2 for fake
    # total_loss = real_normal_loss + fake_loss

    return total_loss


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
        self.dataset_used = dataset_used

        self.x_train_ds = tf.data.Dataset.from_tensor_slices(self.x_train).batch(self.BATCH_SIZE)
        self.x_test_ds = tf.data.Dataset.from_tensor_slices(self.x_test).batch(self.BATCH_SIZE)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # -- logging Functions -- #
    def setup_logger(self, log_file):
        """Set up a logger that records both to a file and to the console."""
        self.logger = logging.getLogger("CentralACGan")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)

        # Avoid adding duplicate handlers if logger already has them.
        if not self.logger.handlers:
            self.logger.addHandler(fh)
            self.logger.addHandler(ch)

    def log_model_settings(self):
        """Logs model names, structures, and hyperparameters."""
        self.logger.info("=== Model Settings ===")

        self.logger.info("Generator Model Summary:")
        generator_summary = []
        self.generator.summary(print_fn=lambda x: generator_summary.append(x))
        for line in generator_summary:
            self.logger.info(line)

        self.logger.info("Discriminator Model Summary:")
        discriminator_summary = []
        self.discriminator.summary(print_fn=lambda x: discriminator_summary.append(x))
        for line in discriminator_summary:
            self.logger.info(line)

        if self.nids is not None:
            self.logger.info("NIDS Model Summary:")
            nids_summary = []
            self.nids.summary(print_fn=lambda x: nids_summary.append(x))
            for line in nids_summary:
                self.logger.info(line)
        else:
            self.logger.info("NIDS Model is not defined.")

        self.logger.info("--- Hyperparameters ---")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Noise Dimension: {self.noise_dim}")
        self.logger.info(f"Latent Dimension: {self.latent_dim}")
        self.logger.info(f"Number of Classes: {self.num_classes}")
        self.logger.info(f"Input Dimension: {self.input_dim}")
        self.logger.info(f"Epochs: {self.epochs}")
        self.logger.info(f"Steps per Epoch: {self.steps_per_epoch}")
        self.logger.info(f"Learning Rate (Generator): {self.gen_optimizer.learning_rate}")
        self.logger.info(f"Learning Rate (Discriminator): {self.disc_optimizer.learning_rate}")
        self.logger.info("=" * 50)

    def log_epoch_metrics(self, epoch, d_metrics, g_metrics, nids_metrics=None):
        """Logs a formatted summary of the metrics for this epoch."""
        self.logger.info(f"=== Epoch {epoch} Metrics Summary ===")
        self.logger.info("Discriminator Metrics:")
        for key, value in d_metrics.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("Generator Metrics:")
        for key, value in g_metrics.items():
            self.logger.info(f"  {key}: {value}")
        if nids_metrics is not None:
            self.logger.info("NIDS Metrics:")
            for key, value in nids_metrics.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    def log_evaluation_metrics(self, d_eval, g_eval, nids_eval=None):
        """Logs a formatted summary of evaluation metrics."""
        self.logger.info("=== Evaluation Metrics Summary ===")
        self.logger.info("Discriminator Evaluation:")
        for key, value in d_eval.items():
            self.logger.info(f"  {key}: {value}")
        self.logger.info("Generator Evaluation:")
        for key, value in g_eval.items():
            self.logger.info(f"  {key}: {value}")
        if nids_eval is not None:
            self.logger.info("NIDS Evaluation:")
            for key, value in nids_eval.items():
                self.logger.info(f"  {key}: {value}")
        self.logger.info("=" * 50)

    def on_fit_end(self, server_round, results, failures):
        # -- Set the model with global weights, Bring in the parameters for the global model --#
        aggregated_parameters = super().aggregate_fit(server_round, results, failures)

        if aggregated_parameters is not None:
            print(f"Saving global model after round {server_round}...")
            aggregated_weights = parameters_to_ndarrays(aggregated_parameters[0])
            if len(aggregated_weights) == len(self.discriminator.get_weights()):
                self.discriminator.set_weights(aggregated_weights)
        # EoF Set global weights

        print("Discriminator Output:", self.discriminator.output_names)

        # -- Model Compilations
        # Compile Discriminator separately (before freezing)
        self.discriminator.compile(
            loss={'validity': 'binary_crossentropy', 'class': 'categorical_crossentropy'},
            optimizer=self.disc_optimizer,
            metrics={
                'validity': ['accuracy', 'binary_accuracy', 'AUC'],
                'class': ['accuracy', 'categorical_accuracy']
            }
        )

        # Freeze Discriminator only for AC-GAN training
        self.discriminator.trainable = False

        # Define AC-GAN (Generator + Frozen Discriminator)
        # I/O
        noise_input = tf.keras.Input(shape=(self.latent_dim,))
        label_input = tf.keras.Input(shape=(1,), dtype='int32')
        generated_data = self.generator([noise_input, label_input])
        validity, class_pred = self.discriminator(generated_data)
        # Compile Combined Model
        self.ACGAN = Model([noise_input, label_input], [validity, class_pred])

        print("ACGAN Output:", self.ACGAN.output_names)

        self.ACGAN.compile(
            loss={'Discriminator': 'binary_crossentropy', 'Discriminator_1': 'categorical_crossentropy'},
            optimizer=self.gen_optimizer,
            metrics={
                'Discriminator': ['accuracy', 'binary_accuracy', 'AUC'],
                'Discriminator_1': ['accuracy', 'categorical_accuracy']
            }
        )

        # -- Set the Data --#
        X_train = self.x_train
        y_train = self.y_train

        print("Xtrain Data", X_train.head())

        # Log model settings at the start
        self.log_model_settings()

        valid = tf.ones((self.batch_size, 1))
        fake = tf.zeros((self.batch_size, 1))

        # -- Training loop --#
        for epoch in range(self.epochs):
            print("Discriminator Metrics:", self.discriminator.metrics_names)
            print("ACGAN Metrics:", self.ACGAN.metrics_names)

            print(f'\n=== Epoch {epoch}/{self.epochs} ===\n')
            self.logger.info(f'=== Epoch {epoch}/{self.epochs} ===')
            # --------------------------
            # Train Discriminator
            # --------------------------

            # Sample a batch of real data
            X_train = np.array(X_train)
            y_train = np.array(y_train)

            idx = tf.random.shuffle(tf.range(len(X_train)))[:self.batch_size]
            real_data = tf.gather(X_train, idx)
            real_labels = tf.gather(y_train, idx)

            # Ensure labels are one-hot encoded
            if len(real_labels.shape) == 1:
                real_labels_onehot = tf.one_hot(tf.cast(real_labels, tf.int32), depth=self.num_classes)
            else:
                real_labels_onehot = real_labels

            # Sample the noise data
            noise = tf.random.normal((self.batch_size, self.latent_dim))
            fake_labels = tf.random.uniform((self.batch_size,), minval=0, maxval=self.num_classes, dtype=tf.int32)
            fake_labels_onehot = tf.one_hot(fake_labels, depth=self.num_classes)

            # Generate fake data
            generated_data = self.generator.predict([noise, fake_labels])

            # Train discriminator on real and fake data
            d_loss_real = self.discriminator.train_on_batch(real_data, [valid, real_labels_onehot])
            d_loss_fake = self.discriminator.train_on_batch(generated_data, [fake, fake_labels_onehot])
            d_loss = 0.5 * tf.add(d_loss_real, d_loss_fake)

            # Collect discriminator metrics
            d_metrics = {
                "Total Loss": f"{d_loss[0]:.4f}",
                "Validity Loss": f"{d_loss[1]:.4f}",
                "Class Loss": f"{d_loss[2]:.4f}",
                "Validity Accuracy": f"{d_loss[3] * 100:.2f}%",
                "Validity Binary Accuracy": f"{d_loss[4] * 100:.2f}%",
                "Validity AUC": f"{d_loss[5] * 100:.2f}%",
                "Class Accuracy": f"{d_loss[6] * 100:.2f}%",
                "Class Categorical Accuracy": f"{d_loss[7] * 100:.2f}%"
            }
            self.logger.info("Training Discriminator")
            self.logger.info(
                f"Discriminator Total Loss: {d_loss[0]:.4f} | Validity Loss: {d_loss[1]:.4f} | Class Loss: {d_loss[2]:.4f}")
            self.logger.info(
                f"Validity Accuracy: {d_loss[3] * 100:.2f}%, Binary Accuracy: {d_loss[4] * 100:.2f}%, AUC: {d_loss[5] * 100:.2f}%")
            self.logger.info(
                f"Class Accuracy: {d_loss[6] * 100:.2f}%, Categorical Accuracy: {d_loss[7] * 100:.2f}%")

            # --------------------------
            # Train Generator (AC-GAN)
            # --------------------------

            # Generate noise and label inputs for ACGAN
            noise = tf.random.normal((self.batch_size, self.latent_dim))
            sampled_labels = tf.random.uniform((self.batch_size,), minval=0, maxval=self.num_classes,
                                               dtype=tf.int32)
            sampled_labels_onehot = tf.one_hot(sampled_labels, depth=self.num_classes)

            # Train ACGAN with sampled noise data
            g_loss = self.ACGAN.train_on_batch([noise, sampled_labels], [valid, sampled_labels_onehot])

            # Collect generator metrics
            g_metrics = {
                "Total Loss": f"{g_loss[0]:.4f}",
                "Validity Loss": f"{g_loss[1]:.4f}",  # This is Discriminator_loss
                "Class Loss": f"{g_loss[2]:.4f}",  # This is Discriminator_1_loss
                "Validity Accuracy": f"{g_loss[3] * 100:.2f}%",  # Discriminator_accuracy
                "Validity Binary Accuracy": f"{g_loss[4] * 100:.2f}%",  # Discriminator_binary_accuracy
                "Validity AUC": f"{g_loss[5] * 100:.2f}%",  # Discriminator_auc
                "Class Accuracy": f"{g_loss[6] * 100:.2f}%",  # Discriminator_1_accuracy
                "Class Categorical Accuracy": f"{g_loss[7] * 100:.2f}%"  # Discriminator_1_categorical_accuracy
            }
            self.logger.info("Training Generator with ACGAN FLOW")
            self.logger.info(
                f"AC-GAN Generator Total Loss: {g_loss[0]:.4f} | Validity Loss: {g_loss[1]:.4f} | Class Loss: {g_loss[2]:.4f}")
            self.logger.info(
                f"Validity Accuracy: {g_loss[3] * 100:.2f}%, Binary Accuracy: {g_loss[4] * 100:.2f}%, AUC: {g_loss[5] * 100:.2f}%")
            self.logger.info(
                f"Class Accuracy: {g_loss[6] * 100:.2f}%, Categorical Accuracy: {g_loss[7] * 100:.2f}%")

            # --------------------------
            # Validation every 1 epochs
            # --------------------------
            if epoch % 1 == 0:
                self.logger.info(f"=== Epoch {epoch} Validation ===")
                d_val_loss, d_val_metrics = self.validation_disc()
                g_val_loss, g_val_metrics = self.validation_gen()
                nids_val_metrics = None
                if self.nids is not None:
                    nids_custom_loss, nids_val_metrics = self.validation_NIDS()
                    self.logger.info(f"Validation NIDS Custom Loss: {nids_custom_loss:.4f}")

                # Log the metrics for this epoch using our new logging method
                self.log_epoch_metrics(epoch, d_val_metrics, g_val_metrics, nids_val_metrics)
                self.logger.info(
                    f"Epoch {epoch}: D Loss: {d_loss[0]:.4f}, G Loss: {g_loss[0]:.4f}, D Acc: {d_loss[3] * 100:.2f}%")

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
