import os
import numpy as np
import tensorflow as tf
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(filename='intrusion_alerts.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')


# Load the trained model
def load_model(model_path, input_dim, regularizationEnabled=False, DP_enabled=False, l2_alpha=0.01):
    # Ensure the model structure matches the one used in training
    model = create_CICIOT_Model(input_dim, regularizationEnabled, DP_enabled, l2_alpha)
    model.load_weights(model_path)
    return model


# Preprocess the input data
def preprocess_data(data, scaler):
    # Apply the same preprocessing used in training
    data = scaler.transform(data)
    return data


# Predict and take action
def predict_and_act(model, data, threshold=0.5):
    # Get prediction probabilities
    predictions = model.predict(data)
    intrusive_indices = np.where(predictions >= threshold)[0]

    if len(intrusive_indices) > 0:
        logging.info(f"Intrusion detected in {len(intrusive_indices)} instances.")
        print("Intrusion detected! Taking action...")
        # Perform actions to secure the system
        take_action(intrusive_indices)
    else:
        print("No intrusion detected.")


# Action to secure the system
def take_action(intrusive_indices):
    # Example: block traffic or send an alert
    for idx in intrusive_indices:
        print(f"Securing system from intrusion at index {idx}...")
        # Add your specific actions here (e.g., call a firewall API, send alert)
    logging.info(f"Secured the system from {len(intrusive_indices)} intrusions.")


# Example usage
if __name__ == "__main__":
    # Path to the saved model
    model_path = "best_model.h5"

    # Input dimensions and preprocessing parameters
    input_dim = 64  # Adjust to your dataset feature count
    regularizationEnabled = True
    l2_alpha = 0.01
    threshold = 0.5

    # Load the trained model
    model = load_model(model_path, input_dim, regularizationEnabled, False, l2_alpha)

    # Simulated data (replace with live network data)
    test_data = np.random.rand(10, input_dim)  # Mock test data
    test_data = preprocess_data(test_data, scaler)

    # Predict and act
    predict_and_act(model, test_data, threshold)
