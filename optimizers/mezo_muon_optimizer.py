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

def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
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


class MeZOMuonOptimizer(object):
    def __init__(self, model, args, lr, candidate_seeds, state=None):
        print('FedKSeed-Muon')
        # determine which parameters to optimizes
        self.args = args
        self.lr = lr
        self.model = model
        self.named_parameters_to_optim = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))
        self.zo_eps = self.args.zo_eps
        self.candidate_seeds = candidate_seeds

        # Adam-specific states
        self.betas = (args.adam_beta1, args.adam_beta2) if hasattr(args, 'adam_beta1') and hasattr(args, 'adam_beta2') else (0.9, 0.999)
        self.eps = args.adam_eps if hasattr(args, 'adam_eps') else 1e-8
        self.ns_steps = args.ns_steps if hasattr(args, 'ns_steps') else 5
        
        if state is not None:
            self.state = state
        else:
            self.state = {}

        for name, param in self.named_parameters_to_optim:
            if name not in self.state:
                self.state[name] = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(param, memory_format=torch.preserve_format),
                    "exp_avg_sq": torch.zeros_like(param, memory_format=torch.preserve_format)
                }
        
    def zo_step(self, batch, local_seed_pool=None):
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        # Sample the random seed for sampling z
        self.zo_random_seed = np.random.choice(self.candidate_seeds, 1)[0]
        
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
        
        if local_seed_pool is not None:
            local_seed_pool[self.zo_random_seed] += self.projected_grad
        return logits1, loss1

    def _zo_perturb_parameters(self, scaling_factor=1):
        """
        Perturb the parameters with random vector z.
        Input:
        - scaling_factor: theta = theta + scaling_factor * z * eps
        """
        torch.manual_seed(self.zo_random_seed)

        for _, param in self.named_parameters_to_optim:
            z = torch.normal(mean=0,
                             std=1,
                             size=param.data.size(),
                             device=param.data.device,
                             dtype=param.data.dtype)
            param.data = param.data + scaling_factor * self.zo_eps * z

    def zo_forward(self, batch):
        """
        Get (no gradient) loss from the model. Dropout is turned off too.
        """
        outputs = self.model(**batch)
        logits = outputs.logits
        loss = outputs.loss
        return logits.detach(), loss.detach()
    
    def zo_update(self, seed=None, grad=None):
        """
        Update the parameters with the estimated gradients using Muon-Adam-like update.
        """
        
        effective_grad = grad if grad is not None else self.projected_grad
        effective_seed = seed if seed is not None else self.zo_random_seed

        torch.manual_seed(effective_seed)
        
        for name, param in self.named_parameters_to_optim:
            # Resample the same perturbation vector z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            
            # Gradient for this parameter is projected_grad * z
            g = effective_grad * z
            
            param_state = self.state[name]
            exp_avg, exp_avg_sq = param_state["exp_avg"], param_state["exp_avg_sq"]
            beta1, beta2 = self.betas

            param_state["step"] += 1
            
            # Adam state update
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            
            step = param_state["step"]
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            # For 1D params (biases, etc.), just do the normal Adam update
            if param.ndim < 2:
                step_size = self.lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
                param.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Decoupled weight decay (AdamW style)
                if self.args.weight_decay > 0.0:
                    param.data.add_(param.data, alpha=-self.lr * self.args.weight_decay)
                continue

            # For 2D+ params, apply Newton-Schulz orthogonalization
            
            # This is the Adam-calculated update direction
            update_direction = exp_avg / (exp_avg_sq.sqrt() / math.sqrt(bias_correction2) + self.eps)

            original_shape = update_direction.shape
            if update_direction.ndim > 2:
                update_direction_2d = update_direction.reshape(original_shape[0], -1)
            else:
                update_direction_2d = update_direction
            
            # Apply Newton-Schulz
            orthogonalized_update = zeropower_via_newtonschulz5(update_direction_2d, steps=self.ns_steps)
            
            # Rescale
            orthogonalized_update *= max(1, update_direction_2d.size(-2) / update_direction_2d.size(-1))**0.5
            
            final_update = orthogonalized_update.reshape(original_shape)
            
            step_size = self.lr / bias_correction1
            param.data.add_(final_update, alpha=-step_size)

            # Decoupled weight decay (AdamW style)
            if self.args.weight_decay > 0.0:
                param.data.add_(param.data, alpha=-self.lr * self.args.weight_decay)
