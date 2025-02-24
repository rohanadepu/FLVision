from flwr.server.strategy import FedAvg

class DiscriminatorFullStrategy(FedAvg):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model

    def get_parameters(self, server_round: int):
        """Send initial model parameters to clients."""
        return [val.numpy() for val in self.model.get_weights()]
