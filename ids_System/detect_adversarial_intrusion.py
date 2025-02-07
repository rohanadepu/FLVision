import sys
import os
import argparse
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import logging

sys.path.append(os.path.abspath('..'))

# Import dataset loading and preprocessing
from datasetLoadProcess.loadCiciotOptimized import loadCICIOT
from datasetLoadProcess.iotbotnetDatasetLoad import loadIOTBOTNET
from datasetLoadProcess.datasetPreprocess import preprocess_dataset
from detectIntrusions import predict_and_act

# Import GAN structure for synthetic data detection
from centralTrainingConfig.GANBinaryCentralTrainingConfig import CentralBinaryGan
from modelStructures.discriminatorStruct import create_discriminator_binary, create_discriminator_binary_optimized, create_discriminator_binary
from modelStructures.generatorStruct import create_generator, create_generator_optimized
from modelStructures.ganStruct import create_model, load_GAN_model, create_model_binary, create_model_binary_optimized

# Logging setup
logging.basicConfig(filename="detection_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")


def load_model(pretrained_model):
    """ Loads the pre-trained NIDS model. """
    try:
        if pretrained_model:
            print(f"📥 Loading pretrained model from {pretrained_model}...")
            with tf.keras.utils.custom_object_scope({'LogCosh': tf.keras.losses.LogCosh}):
                model = tf.keras.models.load_model(pretrained_model)
            print("✅ Model successfully loaded!")
            return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        sys.exit(1)


def generate_synthetic_data(generator, num_samples, noise_dim):
    """ Generate synthetic network traffic using a trained GAN generator. """
    print(f"🎭 Generating {num_samples} synthetic network traffic samples...")
    noise = tf.random.normal([num_samples, noise_dim])
    synthetic_data = generator(noise, training=False)
    return synthetic_data.numpy()


def detect_adversarial_or_synthetic(discriminator, X_test, synthetic_data, batch_size=256):
    """
    Uses the GAN Discriminator to classify network traffic as real or synthetic/adversarial.

    Args:
        discriminator: The trained GAN Discriminator model.
        X_test: The dataset to evaluate.
        synthetic_data: Generated fake network traffic from the GAN generator.
        batch_size: The batch size for inference.

    Returns:
        A boolean mask array where True = Real and False = Synthetic/Adversarial.
    """
    print("🎭 Running Discriminator for Synthetic & Adversarial Detection...")

    # Convert input data into TensorFlow dataset
    X_test_ds = tf.data.Dataset.from_tensor_slices(X_test).batch(batch_size)
    synthetic_ds = tf.data.Dataset.from_tensor_slices(synthetic_data).batch(batch_size)

    # Initialize prediction lists
    real_pred_probs = []
    synthetic_pred_probs = []

    # Process real data in batches
    for batch in X_test_ds:
        batch_pred = discriminator(batch, training=False).numpy()
        real_pred_probs.append(batch_pred)

    # Process synthetic data in batches
    for batch in synthetic_ds:
        batch_pred = discriminator(batch, training=False).numpy()
        synthetic_pred_probs.append(batch_pred)

    # Convert lists to numpy arrays
    real_pred_probs = np.concatenate(real_pred_probs, axis=0).flatten()
    synthetic_pred_probs = np.concatenate(synthetic_pred_probs, axis=0).flatten()

    # Debugging: Check shape consistency
    print(f"🔍 X_test shape: {X_test.shape}, real_pred_probs shape: {real_pred_probs.shape}")

    if real_pred_probs.shape[0] != X_test.shape[0]:
        raise ValueError(f"❌ Shape mismatch! Expected {X_test.shape[0]}, got {real_pred_probs.shape[0]}")

    # Convert probabilities to binary labels
    real_pred = (real_pred_probs > 0.5).astype("int32")
    synthetic_pred = (synthetic_pred_probs > 0.5).astype("int32")

    # Print and log results
    print(f"✅ Real Data Detected: {np.sum(real_pred == 1)}")
    print(f"🛑 Synthetic/Adversarial Data Detected: {np.sum(real_pred == 0)}")
    print(f"🎭 Fake Data Incorrectly Classified as Real: {np.sum(synthetic_pred == 1)}")
    print(f"✅ Fake Data Correctly Classified as Fake: {np.sum(synthetic_pred == 0)}")

    logging.info(f"Real Data: {np.sum(real_pred == 1)}")
    logging.info(f"Synthetic/Adversarial Data: {np.sum(real_pred == 0)}")
    logging.info(f"Fake Data as Real: {np.sum(synthetic_pred == 1)}")
    logging.info(f"Fake Data as Fake: {np.sum(synthetic_pred == 0)}")

    return real_pred





def detect_intrusions(nids_model, X_real, y_real, threshold=0.5):
    """
    Runs the NIDS model on filtered real data to classify it as benign or an attack.

    Args:
        nids_model: The trained NIDS model.
        X_real: Only real network traffic data (filtered by the discriminator).
        y_real: Ground truth labels corresponding to real data.
        threshold: Probability threshold for intrusion classification.

    Returns:
        None (Prints and logs results)
    """
    print("\n🔍 Running NIDS Intrusion Detection...")
    y_pred_probs = nids_model.predict(X_real)
    y_pred = (y_pred_probs < threshold).astype("int32")

    # Evaluate NIDS Performance
    nids_report = classification_report(y_real, y_pred, target_names=["Attack", "Benign"])
    print("\n📊 NIDS Model Performance:\n", nids_report)
    logging.info("\nNIDS Performance:\n" + nids_report)

    # Confusion Matrix
    cm = confusion_matrix(y_real, y_pred)
    print("\n📊 Confusion Matrix (NIDS Detection):")
    print(cm)
    logging.info(f"Confusion Matrix: \n{cm}")


def main():
    print("\n============================")
    print("🔍 Adversarial & Synthetic Data Detection")
    print("============================\n")

    # Argument Parsing
    parser = argparse.ArgumentParser(description='Detect adversarial and synthetic traffic.')
    parser.add_argument('--dataset', type=str, choices=["CICIOT", "IOTBOTNET"], default="CICIOT",
                        help='Dataset to use: CICIOT or IOTBOTNET')
    parser.add_argument('--pretrained_generator', type=str, help="Path to pretrained generator model (optional)",
                        default=None)
    parser.add_argument('--pretrained_discriminator', type=str,
                        help="Path to pretrained discriminator model (optional)", default=None)
    parser.add_argument('--pretrained_nids', type=str,
                        help="Path to pretrained nids model (optional)", default=None)
    parser.add_argument('--num_synthetic', type=int, default=1000,
                        help="Number of synthetic samples to generate.")
    parser.add_argument('--threshold', type=float, default=0.5, help="Threshold for classifying intrusions.")


    args = parser.parse_args()
    pretrainedGan = None
    pretrainedGenerator = args.pretrained_generator
    pretrainedDiscriminator = args.pretrained_discriminator
    pretrainedNids = args.pretrained_nids

    # Load Dataset
    print(f"📡 Loading {args.dataset} dataset...")
    if args.dataset == "CICIOT":
        train_data, test_data, irrelevant_features = loadCICIOT()
    elif args.dataset == "IOTBOTNET":
        train_data, test_data, irrelevant_features = loadIOTBOTNET()

    # Preprocess Dataset
    print("🔄 Preprocessing dataset...")
    X_train, X_val, y_train, y_val, X_test, y_test = preprocess_dataset(
        args.dataset, train_data, test_data, None, None, irrelevant_features, None
    )

    # --- Load or Create model ----#

    # --- Model setup --- #

    # --- Hyperparameters ---#
    BATCH_SIZE = 256
    noise_dim = 100

    input_dim = X_train.shape[1]

    # Load or create the discriminator, generator, or whole ganLegacy model
    if pretrainedGan:
        print(f"Loading pretrained GAN Model from {pretrainedGan}")
        model = tf.keras.models.load_model(pretrainedGan)

    elif pretrainedGenerator and not pretrainedDiscriminator:

        print(f"Pretrained Generator provided from {pretrainedGenerator}. Creating a new Discriminator model.")
        generator = tf.keras.models.load_model(args.pretrained_generator)

        discriminator = create_discriminator_binary(input_dim)

        model = load_GAN_model(generator, discriminator)

    elif pretrainedDiscriminator and not pretrainedGenerator:
        print(f"Pretrained Discriminator provided from {pretrainedDiscriminator}. Creating a new Generator model.")
        discriminator = tf.keras.models.load_model(args.pretrained_discriminator)

        generator = create_generator(input_dim, noise_dim)

        model = load_GAN_model(generator, discriminator)

    elif pretrainedDiscriminator and pretrainedGenerator:
        print(f"Pretrained Generator and Discriminator provided from {pretrainedGenerator} , {pretrainedDiscriminator}")
        discriminator = tf.keras.models.load_model(args.pretrained_discriminator)
        generator = tf.keras.models.load_model(args.pretrained_generator)

        model = load_GAN_model(generator, discriminator)

    else:
        print("No pretrained GAN provided. Creating a new GAN model.")
        model = create_model_binary(input_dim, noise_dim)

    # load NIDS model
    print(f"Loading pretrained NIDS from {pretrainedNids}")
    nids = load_model(pretrainedNids)

    # Split the gan model into its submodels
    generator = model.layers[0]
    discriminator = model.layers[1]

    # Step 1: Generate Synthetic Data
    synthetic_data = generate_synthetic_data(generator, BATCH_SIZE, noise_dim=100)

    # Step 2: Detect Adversarial or Synthetic Data
    is_real_mask = detect_adversarial_or_synthetic(discriminator, X_test, synthetic_data)

    # Predict and Take Action
    predict_and_act(model, X_test, y_test, threshold=args.threshold)


if __name__ == "__main__":
    main()
