import flwr as fl
from ultralytics import YOLO
import torch
from collections import OrderedDict

import logging

# Configure logging to show the INFO level messages
logging.basicConfig(level=logging.INFO)
# Device configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Flower client that integrates with YOLOv8
class YOLOClient(fl.client.NumPyClient):
    def __init__(self, model_name, data_yaml):
        self.model = YOLO(model_name).to(DEVICE)
        self.data_yaml = data_yaml

    def get_parameters(self, config):
        # Retrieve model parameters as a list of NumPy ndarrays
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v).clone() for k, v in params_dict})
        new_state_dict = OrderedDict()
        for key, param in state_dict.items():
            new_state_dict[key] = param.clone()  # Clone the parameters
        self.model.load_state_dict(new_state_dict, strict=True)


    def fit(self, parameters, config):
        self.set_parameters(parameters)
        results = self.model.train(data=self.data_yaml, epochs=2, name='run_results', device=DEVICE ,workers=0 ,imgsz=640,  plots=False )
        #self.model.save('runs/detect/run_results/weights/last.pt')  # Overwrite current model with the new one
        return self.get_parameters(config={}), len(results.maps), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        metrics = self.model.val()
        return float(metrics.results_dict['metrics/precision(B)']), len(metrics.curves_results), \
            {'mAP': metrics.results_dict['metrics/mAP50(B)']}

model_name = " /root/FLVision/runs/detect/run_results/weights/last.pt" # Using pre-initizalized weights for 1 epoch
data_yaml = "VisDrone.yaml"  # Name of the dataset configuration file

fl.client.start_client(server_address="192.168.129.8:8080", client=YOLOClient(model_name, data_yaml))
