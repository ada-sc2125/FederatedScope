import torch
from collections import OrderedDict

def aggregate_state_dicts(state_dicts, weights=None, device='cpu'):
    """
    Aggregates a list of state_dicts.
    
    Args:
        state_dicts (list): List of state_dicts (OrderedDict).
        weights (list, optional): List of weights for each state_dict. Defaults to equal weighting.
        device (str or torch.device): Device to perform aggregation on. Defaults to 'cpu'.
        
    Returns:
        OrderedDict: The aggregated state_dict.
    """
    if not state_dicts:
        return None
    
    num_models = len(state_dicts)
    if weights is None:
        weights = [1.0 / num_models] * num_models
    else:
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
    # Initialize aggregated_state_dict with zeros based on the first model structure
    first_state = state_dicts[0]
    aggregated_state_dict = OrderedDict()
    
    for key in first_state.keys():
        # Check if the value is a tensor (to avoid aggregating non-tensor metadata if any)
        if isinstance(first_state[key], torch.Tensor):
            dtype = first_state[key].dtype
            # Determine accumulation dtype (float32 is safer for accumulation)
            acc_dtype = torch.float32 if dtype in [torch.float16, torch.bfloat16] else dtype
            aggregated_state_dict[key] = torch.zeros_like(first_state[key], dtype=acc_dtype, device=device)
        else:
             # For non-tensor data, just take the first one (or handle differently?)
             # Usually state_dict only has tensors.
            aggregated_state_dict[key] = first_state[key]

    # Accumulate
    for i, state_dict in enumerate(state_dicts):
        w = weights[i]
        for key in aggregated_state_dict:
             if isinstance(state_dict[key], torch.Tensor):
                aggregated_state_dict[key] += state_dict[key].to(device) * w
    
    # Cast back to original dtype if needed
    for key in aggregated_state_dict:
        if isinstance(aggregated_state_dict[key], torch.Tensor):
             target_dtype = first_state[key].dtype
             if aggregated_state_dict[key].dtype != target_dtype:
                 aggregated_state_dict[key] = aggregated_state_dict[key].to(target_dtype)

    return aggregated_state_dict

def aggregate_optimizer_states(optimizer_states, weights=None, device='cpu'):
    """
    Aggregates a list of optimizer states.
    Assumes optimizer state structure matches standard PyTorch optimizer state dicts
    or the custom structure used in MeZO optimizers.
    
    Args:
        optimizer_states (list): List of optimizer state dicts.
        weights (list, optional): Weights.
        device: Device.
        
    Returns:
        dict: Aggregated optimizer state.
    """
    if not optimizer_states:
        return {}
    
    # If empty dicts
    if not optimizer_states[0]:
        return {}

    num_states = len(optimizer_states)
    if weights is None:
        weights = [1.0 / num_states] * num_states
    else:
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

    first_opt = optimizer_states[0]
    agg_opt = {}
    
    # Keys in optimizer state are usually parameter names (in MeZO implementation)
    # Check structure: MeZO store state[name] = {'exp_avg': ..., 'step': ...}
    
    for param_name in first_opt:
        agg_opt[param_name] = {}
        for key in first_opt[param_name]:
            val = first_opt[param_name][key]
            if isinstance(val, torch.Tensor):
                # Aggregate tensor states (exp_avg, exp_avg_sq, etc.)
                 dtype = val.dtype
                 acc_dtype = torch.float32 if dtype in [torch.float16, torch.bfloat16] else dtype
                 agg_val = torch.zeros_like(val, dtype=acc_dtype, device=device)
                 
                 for i, opt_state in enumerate(optimizer_states):
                     if param_name in opt_state and key in opt_state[param_name]:
                         agg_val += opt_state[param_name][key].to(device) * weights[i]
                 
                 if agg_val.dtype != dtype:
                     agg_val = agg_val.to(dtype)
                 agg_opt[param_name][key] = agg_val
            
            elif isinstance(val, (int, float)):
                # Aggregate scalars (like 'step')? 
                # For 'step', usually we take the max or average? 
                # Taking average seems reasonable for synchronized steps, or max.
                # Let's take the first one or average. MeZO uses step for bias correction.
                # If we average steps, it might be fractional.
                # Let's just average and cast to int if it was int.
                agg_scalar = 0.0
                for i, opt_state in enumerate(optimizer_states):
                     if param_name in opt_state and key in opt_state[param_name]:
                         agg_scalar += opt_state[param_name][key] * weights[i]
                
                if isinstance(val, int):
                    agg_opt[param_name][key] = int(agg_scalar)
                else:
                    agg_opt[param_name][key] = agg_scalar
            else:
                # Copy other types
                agg_opt[param_name][key] = val
                
    return agg_opt

class FedAvgAggregator:
    """
    Legacy class wrapper if needed, or can be removed if not used.
    """
    def __init__(self, model):
        self.model = model
        self.device = next(model.parameters()).device

    def aggregate(self, neighbor_state_dicts):
        # Use the new functional implementation
        # Include self.model.state_dict() in the list if logic requires it, 
        # but the caller usually handles list construction.
        # This legacy method assumed "add neighbors to self".
        
        # We will not use this class in the new implementation, 
        # but keep it compatible if needed or just replace functionality.
        pass