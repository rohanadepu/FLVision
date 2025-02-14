import flwr as fl
import tensorflow as tf
import numpy as np



def main():
    # Load dataset
    X_train, y_train, _, _, _, _ = preprocess_dataset("CICIOT")  # Example dataset


    # Start the client
    client = FLClient(X_train, y_train)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client)

if __name__ == "__main__":
    main()
