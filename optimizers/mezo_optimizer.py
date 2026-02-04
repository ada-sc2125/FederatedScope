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
from optimizers.mezo_muon_optimizer import zeropower_via_newtonschulz5


class MeZOFramework(object):
    def __init__(self, model, args, lr, subspace_bases=None): # Removed candidate_seeds
        print("FedKSeed")
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
        self.rng = np.random.default_rng()
        self.subspace_enabled = bool(getattr(args, "subspace", False))
        self.subspace_bases = subspace_bases or {}
        self._subspace_bases_device = {}
        if self.subspace_enabled and not self.subspace_bases:
            raise ValueError("subspace is enabled but subspace_bases was not provided.")

    def _get_subspace_bases(self, name, param):
        cached = self._subspace_bases_device.get(name)
        if cached is not None:
            return cached
        bases = self.subspace_bases.get(name)
        if not bases:
            return None, None
        U, V = bases
        if U is None or V is None:
            return None, None
        if U.device != param.device or U.dtype != param.dtype:
            U = U.to(device=param.device, dtype=param.dtype)
        if V.device != param.device or V.dtype != param.dtype:
            V = V.to(device=param.device, dtype=param.dtype)
        self._subspace_bases_device[name] = (U, V)
        return U, V

    def _sample_z(self, name, param):
        if not self.subspace_enabled:
            return torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
        U, V = self._get_subspace_bases(name, param)
        if U is None or V is None or U.ndim < 2 or V.ndim < 2:
            z = torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
        else:
            z0 = torch.normal(
                mean=0,
                std=1,
                size=(U.shape[1], V.shape[0]),
                device=param.data.device,
                dtype=param.data.dtype,
            )
            if getattr(self.args, "subspace_orthogonalize_z", False):
                z0 = zeropower_via_newtonschulz5(
                    z0, steps=getattr(self.args, "ns_steps", 5)
                )
            z = (U @ z0 @ V) * math.sqrt(param.data.numel() / z0.numel())
            z = z.view(param.data.shape)

        return z

    def zo_step(self, batch): # Removed local_seed_pool
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        # Sample the random seed for sampling z
        # self.zo_random_seed = np.random.choice(self.candidate_seeds, 1)[0] # Changed
        self.zo_random_seed = int(self.rng.integers(0, 1_000_000_000))

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
                print(
                    f"Debug: Grad clipped. loss1={loss1.item()}, loss2={loss2.item()}"
                )
                return logits1, 0.0

        self.projected_grad = ((loss1 - loss2) / (2 * self.zo_eps)).item()
        # print(
        #     f"Debug: loss1={loss1.item()}, loss2={loss2.item()}, projected_grad={self.projected_grad}"
        # )
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
            z = self._sample_z(name, param)
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
        Update the parameters with the estimated gradients.
        """

        # Reset the random seed for sampling zs
        # if seed is None: # Removed
        torch.manual_seed(self.zo_random_seed)
        for name, param in self.named_parameters_to_optim:
            # Resample z
            z = self._sample_z(name, param)
            param.data = param.data - (self.lr * self.projected_grad) * z
        # else: # Removed
        #     torch.manual_seed(seed) # Removed
        #     for name, param in self.named_parameters_to_optim: # Removed
        #         # Resample z # Removed
        #         z = torch.normal( # Removed
        #             mean=0, # Removed
        #             std=1, # Removed
        #             size=param.data.size(), # Removed
        #             device=param.data.device, # Removed
        #             dtype=param.data.dtype, # Removed
        #         ) # Removed
        #         param.data = param.data - (self.lr * grad) * z # Removed
