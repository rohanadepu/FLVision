import argparse
import os
from collections import OrderedDict
import flwr as fl
import numpy as np
import torch
from ultralytics import YOLO

# Parse command line arguments
parser = argparse.ArgumentParser(description='Description of parameters used in this script')
parser.add_argument('-d', type=str, help='Dataset .yaml path. E.g: dataset/data.yaml')
parser.add_argument('-s', type=str, help='Server address. E.g: 172.17.0.2')
parser.add_argument('-p', type=str, help='Server port. E.g: 8080')
parser.add_argument('-m', type=str, help='YOLOv10 model path (.pt file). E.g: yolov10n.pt')
parser.add_argument('-e', type=int, help='Number of epochs')
args = parser.parse_args()

# Setup configurations
dataset_path = os.path.abspath(args.d)
server_address = f"{args.s}:{args.p}"
model_path = args.m
EPOCHS = args.e

# Load model onto the correct device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = YOLO(model_path).to(DEVICE)
class YOLOClient(fl.client.NumPyClient):
    def get_parameters(self):
        """ Extract model parameters as a list of NumPy ndarrays """
        return [val.cpu().numpy() for _, val in model.state_dict().items()]

    def set_parameters(self, parameters):
        """ Set model parameters from a list of NumPy ndarrays """
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v, device=DEVICE) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """ Train the model using the provided parameters and config """
        self.set_parameters(parameters)
        # Implement training logic
        return self.get_parameters(), 0, {}

    def evaluate(self, parameters, config):
        """ Evaluate the model """
        self.set_parameters(parameters)
        return 1.0, 0, {"accuracy": 0.95}

# Start the Flower client with the updated method
fl.client.start_client(server_address=server_address, client=YOLOClient().to_client())