import torch
import torch.distributed as dist
import numpy as np
import math

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    From muon.py
    """
    assert G.ndim >= 2 
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A 
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    """
    Muon update helper.
    """
    # Cast grad to momentum's dtype to avoid RuntimeError
    grad_casted = grad.to(momentum.dtype)
    momentum.lerp_(grad_casted, 1 - beta)
    update = grad_casted.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class DistributedMeZOMuonOptimizer(object):
    """
    Distributed version of MeZO-Muon.
    Integrates the sharded parameter update logic from MuonWithAuxAdam (muon.py)
    into the Zeroth-Order optimization loop.
    """
    def __init__(self, model, args, lr, candidate_seeds, state=None):
        print("FedKSeed-Muon (Distributed)")
        self.args = args
        self.lr = lr
        self.model = model
        
        # Collect parameters
        # Sort by size to balance load across ranks, similar to Muon implementation
        all_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                all_params.append((name, param))
        
        # Sorting helps distributed load balancing for expensive matrix ops
        self.named_parameters_to_optim = sorted(all_params, key=lambda x: x[1].numel(), reverse=True)
        
        self.zo_eps = self.args.zo_eps
        self.candidate_seeds = candidate_seeds

        # Adam-specific states
        self.betas = (
            (args.adam_beta1, args.adam_beta2)
            if hasattr(args, "adam_beta1") and hasattr(args, "adam_beta2")
            else (0.9, 0.999)
        )
        self.eps = args.adam_eps if hasattr(args, "adam_eps") else 1e-8
        
        # Muon-specific states
        self.momentum = args.momentum if hasattr(args, "momentum") else 0.95
        self.ns_steps = args.ns_steps if hasattr(args, "ns_steps") else 5
        self.muon_lr = args.muon_lr if hasattr(args, "muon_lr") else 0.02

        if state is not None:
            self.state = state
        else:
            self.state = {}

        # Initialize state
        for name, param in self.named_parameters_to_optim:
            if name not in self.state:
                if param.ndim >= 2:
                    # Muon state
                    self.state[name] = {
                        "exp_avg": torch.zeros_like(
                            param, dtype=torch.float32, memory_format=torch.preserve_format
                        )
                    }
                else:
                    # Adam state
                    self.state[name] = {
                        "step": 0,
                        "exp_avg": torch.zeros_like(
                            param, dtype=torch.float32, memory_format=torch.preserve_format
                        ),
                        "exp_avg_sq": torch.zeros_like(
                            param, dtype=torch.float32, memory_format=torch.preserve_format
                        ),
                    }
        
        # Helper to check distributed status
        self.world_size = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0

    def _get_deterministic_seed(self, base_seed, param_name):
        """
        Generates a deterministic seed for a specific parameter.
        Essential for distributed MeZO because we update parameters out-of-order/in-parallel.
        """
        # Simple hash of the name combined with base seed
        # Using abs(hash) to ensure positive integer
        return (base_seed + abs(hash(param_name))) % (2**32 - 1)

    def zo_step(self, batch, local_seed_pool=None):
        """
        Estimate gradient by MeZO. 
        """
        # Sample the random seed.
        # In distributed setting, all ranks MUST use the same seed for consistency.
        # Assuming candidate_seeds are identical on all ranks or we pick index 0 deterministically.
        # Ideally, broadcast the seed selection or use a deterministic index.
        self.zo_random_seed = np.random.choice(self.candidate_seeds, 1)[0]
        
        # Broadcast seed if not using identical seeds (optional safety check)
        # seed_tensor = torch.tensor(self.zo_random_seed, device=list(self.model.parameters())[0].device)
        # dist.broadcast(seed_tensor, src=0)
        # self.zo_random_seed = seed_tensor.item()

        # 1. Perturb + Forward
        self._zo_perturb_parameters(scaling_factor=1)
        logits1, loss1 = self.zo_forward(batch)

        # 2. Perturb (reverse) + Forward
        self._zo_perturb_parameters(scaling_factor=-2)
        logits2, loss2 = self.zo_forward(batch)

        # Reset model
        self._zo_perturb_parameters(scaling_factor=1)

        if torch.isnan(loss1) or torch.isnan(loss2):
            return logits1, loss1 # Handle NaN

        # 3. Aggregate Loss/Gradient across devices
        # Since different ranks might have different data batches (Data Parallelism),
        # we average the scalar projected gradient.
        
        projected_grad_scalar = (loss1 - loss2) / (2 * self.zo_eps)
        
        if self.world_size > 1:
            dist.all_reduce(projected_grad_scalar, op=dist.ReduceOp.AVG)
            
        if self.args.grad_clip > 0.0:
             # Re-calculate diff after average? Or clip locally?
             # Usually clip based on the averaged scalar estimate.
             if torch.abs(projected_grad_scalar * (2 * self.zo_eps)) > self.args.grad_clip:
                 return logits1, 0.0

        self.projected_grad = projected_grad_scalar.item()
        
        # 4. Distributed Update
        self.zo_update()

        if local_seed_pool is not None:
            local_seed_pool[self.zo_random_seed] += self.projected_grad
            
        return logits1, loss1

    def _zo_perturb_parameters(self, scaling_factor=1):
        """
        Perturb all parameters.
        Must use deterministic per-parameter seeding so it matches _zo_update logic.
        """
        for name, param in self.named_parameters_to_optim:
            # Deterministic seed for this param
            param_seed = self._get_deterministic_seed(self.zo_random_seed, name)
            torch.manual_seed(param_seed)
            
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data = param.data + scaling_factor * self.zo_eps * z

    def zo_forward(self, batch):
        outputs = self.model(**batch)
        return outputs.logits.detach(), outputs.loss.detach()

    def zo_update(self, seed=None, grad=None):
        """
        Distributed Muon/Adam Update.
        Shards parameters across ranks, updates locally, and gathers results.
        """
        effective_grad = grad if grad is not None else self.projected_grad
        effective_base_seed = seed if seed is not None else self.zo_random_seed

        # Prepare parameters list for sharding logic
        # We need the full objects to pad, but we also need names for state lookup
        params_list = [p for n, p in self.named_parameters_to_optim]
        names_list = [n for n, p in self.named_parameters_to_optim]
        
        # Pad parameters to be divisible by world size
        # We use a dummy tensor for padding
        dummy_param = torch.empty_like(params_list[-1])
        params_pad = params_list + [dummy_param] * (self.world_size - len(params_list) % self.world_size)
        
        # Iterate with stride equal to world size
        for base_i in range(len(params_list))[::self.world_size]:
            
            # Check if the parameter at this index belongs to current rank
            if base_i + self.rank < len(params_list):
                p = params_list[base_i + self.rank]
                name = names_list[base_i + self.rank]
                
                # 1. Regenerate Z for this specific parameter
                param_seed = self._get_deterministic_seed(effective_base_seed, name)
                torch.manual_seed(param_seed)
                
                z = torch.normal(
                    mean=0,
                    std=1,
                    size=p.data.size(),
                    device=p.data.device,
                    dtype=p.data.dtype,
                )
                
                # 2. Compute gradient proxy
                g = effective_grad * z
                param_state = self.state[name]

                # 3. Apply Update (Muon or Adam)
                if p.ndim >= 2:
                    # Muon update
                    momentum_buffer = param_state["exp_avg"]
                    update = muon_update(
                        g, 
                        momentum_buffer, 
                        beta=self.momentum, 
                        ns_steps=self.ns_steps, 
                        nesterov=True
                    )
                    
                    if self.args.weight_decay > 0.0:
                        p.data.mul_(1 - self.muon_lr * self.args.weight_decay)
                    
                    p.data.add_(update.reshape(p.shape), alpha=-self.muon_lr)
                else:
                    # Adam update (AuxAdam)
                    exp_avg, exp_avg_sq = param_state["exp_avg"], param_state["exp_avg_sq"]
                    beta1, beta2 = self.betas

                    param_state["step"] += 1

                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                    step = param_state["step"]
                    bias_correction1 = 1 - beta1**step
                    bias_correction2 = 1 - beta2**step

                    step_size = self.lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
                    
                    if self.args.weight_decay > 0.0:
                        p.data.add_(p.data, alpha=-self.lr * self.args.weight_decay)

                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

            # 4. Sync Updated Parameters across all ranks
            # Gather the chunk of parameters that were just updated distributedly
            # params_pad slice: [base_i : base_i + world_size]
            current_slice = params_pad[base_i : base_i + self.world_size]
            dist.all_gather(current_slice, params_pad[base_i + self.rank])