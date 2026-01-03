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


class MeZOAdamOptimizer(object):
    def __init__(self, model, args, lr, state=None): # Removed candidate_seeds
        print('FedKSeed-Adam')
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
        self.betas = (args.adam_beta1, args.adam_beta2) if hasattr(args, 'adam_beta1') and hasattr(args, 'adam_beta2') else (0.9, 0.999)
        self.eps = args.adam_eps if hasattr(args, 'adam_eps') else 1e-8
        
        if state is not None:
            self.state = state
        else:
            self.state = {}

        for name, param in self.named_parameters_to_optim:
            if name not in self.state:
                self.state[name] = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(param, dtype=torch.float32, memory_format=torch.preserve_format),
                    "exp_avg_sq": torch.zeros_like(param, dtype=torch.float32, memory_format=torch.preserve_format)
                }
        
    def zo_step(self, batch): # Removed local_seed_pool
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        # Sample the random seed for sampling z
        # self.zo_random_seed = np.random.choice(self.candidate_seeds, 1)[0] # Changed
        self.zo_random_seed = np.random.randint(1000000000)
        
        self._zo_perturb_parameters(scaling_factor=1)
        logits1, loss1 = self.zo_forward(batch)

        # Second function evaluation
        self._zo_perturb_parameters(scaling_factor=-2)
        logits2, loss2 = self.zo_forward(batch)
        
        # Reset model back to its parameters at start of step
        self._zo_perturb_parameters(scaling_factor=1)
        
        if torch.isnan(loss1):
            # print(f"Debug: loss1 is NaN. Returning early.")
            return logits1, loss1
        if torch.isnan(loss2):
            # print(f"Debug: loss2 is NaN. Returning early.")
            return logits2, loss2
        if self.args.grad_clip > 0.0:
            if torch.abs(loss1 - loss2) > self.args.grad_clip:
                # print(f"Debug: Grad clipped. loss1={loss1.item()}, loss2={loss2.item()}")
                return logits1, 0.0

        self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()
        # print(f"Debug: loss1={loss1.item()}, loss2={loss2.item()}, projected_grad={self.projected_grad}")
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

        for name, param in self.named_parameters_to_optim:
            # print(f"Debug: {name} param.data.norm() before perturb: {param.data.norm().item()}")
            z = torch.normal(mean=0,
                             std=1,
                             size=param.data.size(),
                             device=param.data.device,
                             dtype=param.data.dtype)
            param.data = param.data + scaling_factor * self.zo_eps * z
            # print(f"Debug: {name} param.data.norm() after perturb: {param.data.norm().item()}")

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
        Update the parameters with the estimated gradients using Adam-like update.
        """
        
        effective_grad = self.projected_grad # Simplified
        effective_seed = self.zo_random_seed # Simplified

        torch.manual_seed(effective_seed)
        
        for name, param in self.named_parameters_to_optim:
            # print(f"Debug: {name} param.data.norm() before update: {param.data.norm().item()}")
            param_state = self.state[name]
            # print(f"Debug: {name} exp_avg.norm() before update: {param_state['exp_avg'].norm().item()}")
            # print(f"Debug: {name} exp_avg_sq.norm() before update: {param_state['exp_avg_sq'].norm().item()}")

            # Resample the same perturbation vector z
            z = torch.normal(mean=0, std=1, size=param.data.size(), device=param.data.device, dtype=param.data.dtype)
            
            # Gradient for this parameter is projected_grad * z
            g = effective_grad * z
            # print(f"Debug: {name} g.norm(): {g.norm().item()}")
            
            exp_avg, exp_avg_sq = param_state["exp_avg"], param_state["exp_avg_sq"]
            beta1, beta2 = self.betas

            param_state["step"] += 1
            
            # Adam update logic
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            
            step = param_state["step"]
            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step
            
            step_size = self.lr / bias_correction1
            
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
            
            # print(f"Debug: {name} denom.norm(): {denom.norm().item()}, step_size: {step_size}")
            param.data.addcdiv_(exp_avg, denom, value=-step_size)

            # Decoupled weight decay (AdamW style)
            if self.args.weight_decay > 0.0:
                param.data.add_(param.data, alpha=-self.lr * self.args.weight_decay)
            # print(f"Debug: {name} param.data.norm() after update: {param.data.norm().item()}")
