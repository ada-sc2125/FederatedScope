import torch
from collections import OrderedDict

class FedAvgAggregator:
    """
    Performs FedAvg aggregation on model state dictionaries.
    """
    def __init__(self, model):
        """
        Initializes the aggregator with the client's own model.

        Args:
            model (torch.nn.Module): The client's model, which is expected to be on the target device (e.g., GPU).
        """
        self.model = model
        self.device = next(model.parameters()).device

    def aggregate(self, neighbor_state_dicts):
        """
        Aggregates the state_dicts from neighbors with the client's own model's state_dict.

        Args:
            neighbor_state_dicts (list): A list of neighbor state_dicts, with tensors expected to be on the CPU.

        Returns:
            OrderedDict: The new aggregated model state dictionary, with tensors on the same device as the client's model.
        """
        # The local model's state_dict is already on the target device (GPU).
        local_state_dict = self.model.state_dict()
        
        # Initialize the aggregated_state_dict with the values from the local model.
        aggregated_state_dict = OrderedDict()
        for key, value in local_state_dict.items():
            aggregated_state_dict[key] = value.clone()

        # Add the parameters from each neighbor.
        # The neighbor state_dicts are on CPU, so we move them to the target device.
        for state_dict in neighbor_state_dicts:
            for key, value in state_dict.items():
                aggregated_state_dict[key] += value.to(self.device)

        # Average the parameters. The count includes the client's own model.
        num_models = 1 + len(neighbor_state_dicts)
        if num_models > 1:
            for key in aggregated_state_dict:
                aggregated_state_dict[key] /= num_models

        return aggregated_state_dict
