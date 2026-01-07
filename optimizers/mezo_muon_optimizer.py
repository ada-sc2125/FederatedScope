"""
The implementations of MeZO optimizer is
adapted from https://github.com/princeton-nlp/MeZO (MIT License)

Copyright (c) 2021 Princeton Natural Language Processing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import numpy as np
import math

ZO_RANDOM_SEED = 12345


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def muon_update(grad, momentum, beta=0.95, ns_steps=5, nesterov=True):
    # Cast grad to momentum's dtype to avoid RuntimeError
    grad_casted = grad.to(momentum.dtype)
    momentum.lerp_(grad_casted, 1 - beta)
    update = grad_casted.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4: # for the case of conv filters
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


class MeZOMuonOptimizer(object):
    def __init__(self, model, args, lr, state=None): # Removed candidate_seeds
        print("FedKSeed-Muon")
        # determine which parameters to optimizes
        self.args = args
        self.lr = lr
        self.model = model
        self.named_parameters_to_optim = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        self.zo_eps = self.args.zo_eps
        # self.candidate_seeds = candidate_seeds # Removed

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
        self.rng = np.random.RandomState(ZO_RANDOM_SEED)

        if state is not None:
            self.state = state
        else:
            self.state = {}

        for name, param in self.named_parameters_to_optim:
            if name not in self.state:
                if param.ndim >= 2:
                    # Muon state: reuse 'exp_avg' as momentum buffer to be compatible with client.py data moving
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

    def zo_step(self, batch): # Removed local_seed_pool
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        # Sample the random seed for sampling z
        # self.zo_random_seed = np.random.choice(self.candidate_seeds, 1)[0] # Changed
        self.zo_random_seed = int(self.rng.randint(1000000000))

        self._zo_perturb_parameters(scaling_factor=1)
        logits1, loss1 = self.zo_forward(batch)

        # Second function evaluation
        self._zo_perturb_parameters(scaling_factor=-2)
        logits2, loss2 = self.zo_forward(batch)

        # Reset model back to its parameters at start of step
        self._zo_perturb_parameters(scaling_factor=1)

        if torch.isnan(loss1):
            return logits1, loss1
        if torch.isnan(loss2):
            return logits2, loss2
        if self.args.grad_clip > 0.0:
            if torch.abs(loss1 - loss2) > self.args.grad_clip:
                return logits1, 0.0

        self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()
        self.zo_update()

        # if local_seed_pool is not None: # Removed
        #     local_seed_pool[self.zo_random_seed] += self.projected_grad # Removed
        return logits1, loss1

    def _zo_perturb_parameters(self, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """
        torch.manual_seed(self.zo_random_seed)

        for _, param in self.named_parameters_to_optim:
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            param.data = param.data + scaling_factor * self.zo_eps * z

    def zo_forward(self, batch):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        outputs = self.model(**batch)
        logits = outputs.logits
        loss = outputs.loss
        return logits.detach(), loss.detach()

    def zo_update(self): # Removed seed=None, grad=None
        """
        Update the parameters with the estimated gradients using Muon (for >=2D) or Adam (for <2D).
        """

        effective_grad = self.projected_grad # Simplified
        effective_seed = self.zo_random_seed # Simplified

        torch.manual_seed(effective_seed)

        for name, param in self.named_parameters_to_optim:
            # Resample the same perturbation vector z
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )

            # Gradient for this parameter is projected_grad * z
            g = effective_grad * z
            param_state = self.state[name]

            if param.ndim >= 2:
                # Muon update for >= 2D parameters
                # We use 'exp_avg' to store the momentum buffer
                momentum_buffer = param_state["exp_avg"]
                
                # Apply Muon update
                # Note: muon_update modifies momentum_buffer in-place if using lerp_
                update = muon_update(
                    g, 
                    momentum_buffer, 
                    beta=self.momentum, 
                    ns_steps=self.ns_steps, 
                    nesterov=True
                )
                
                # Weight decay
                if self.args.weight_decay > 0.0:
                    param.data.mul_(1 - self.muon_lr * self.args.weight_decay)
                
                # Apply update
                param.data.add_(update.reshape(param.shape), alpha=-self.muon_lr)
            
            else:
                # Adam update for < 2D parameters (AuxAdam)
                exp_avg, exp_avg_sq = param_state["exp_avg"], param_state["exp_avg_sq"]
                beta1, beta2 = self.betas

                param_state["step"] += 1

                # Adam state update
                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                step = param_state["step"]
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = self.lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
                
                # Decoupled weight decay (AdamW style)
                if self.args.weight_decay > 0.0:
                    param.data.add_(param.data, alpha=-self.lr * self.args.weight_decay)

                param.data.addcdiv_(exp_avg, denom, value=-step_size)
