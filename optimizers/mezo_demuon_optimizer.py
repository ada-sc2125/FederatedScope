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

import math
import numpy as np
import torch


def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


def matrix_sign(update, ns_steps):
    if update.ndim == 4:
        update = update.view(len(update), -1)
    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
    update *= max(1, update.size(-2) / update.size(-1)) ** 0.5
    return update


class MeZODEMuonOptimizer(object):
    def __init__(self, model, args, lr, state=None, subspace_bases=None):  # Removed candidate_seeds
        print("FedKSeed-DeMuon")
        self.args = args
        self.lr = lr
        self.model = model
        self.theta = getattr(args, "demuon_theta", 0.1)
        self.ns_steps = getattr(args, "ns_steps", 5)
        self.betas = (
            (args.adam_beta1, args.adam_beta2)
            if hasattr(args, "adam_beta1") and hasattr(args, "adam_beta2")
            else (0.9, 0.999)
        )
        self.eps = args.adam_eps if hasattr(args, "adam_eps") else 1e-8
        self.weight_decay = getattr(args, "weight_decay", 0.0)

        self.named_parameters_to_optim = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.named_parameters_to_optim.append((name, param))

        self.zo_eps = self.args.zo_eps
        self.rng = np.random.default_rng()
        self.subspace_enabled = bool(getattr(args, "subspace", False))
        self.subspace_bases = subspace_bases or {}
        self._subspace_bases_device = {}
        if self.subspace_enabled and not self.subspace_bases:
            raise ValueError("subspace is enabled but subspace_bases was not provided.")

        if state is not None:
            self.state = state
        else:
            self.state = {}

        for name, param in self.named_parameters_to_optim:
            if name in self.state:
                continue
            if param.ndim >= 2:
                self.state[name] = {
                    "momentum": torch.zeros_like(
                        param, dtype=torch.float32, memory_format=torch.preserve_format
                    ),
                    "v": torch.zeros_like(
                        param, dtype=torch.float32, memory_format=torch.preserve_format
                    ),
                }
            else:
                self.state[name] = {
                    "step": 0,
                    "exp_avg": torch.zeros_like(
                        param, dtype=torch.float32, memory_format=torch.preserve_format
                    ),
                    "exp_avg_sq": torch.zeros_like(
                        param, dtype=torch.float32, memory_format=torch.preserve_format
                    ),
                }

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
            return torch.normal(
                mean=0,
                std=1,
                size=param.data.size(),
                device=param.data.device,
                dtype=param.data.dtype,
            )
        z0 = torch.normal(
            mean=0,
            std=1,
            size=(U.shape[1], V.shape[0]),
            device=param.data.device,
            dtype=param.data.dtype,
        )
        z = (U @ z0 @ V) * math.sqrt(param.data.numel() / z0.numel())
        return z.view(param.data.shape)

    def zo_step(self, batch):  # Removed local_seed_pool
        """
        Estimate gradient by MeZO. Return the loss from f(theta + z)
        """
        self.zo_random_seed = int(self.rng.integers(0, 1_000_000_000))

        self._zo_perturb_parameters(scaling_factor=1)
        logits1, loss1 = self.zo_forward(batch)

        self._zo_perturb_parameters(scaling_factor=-2)
        logits2, loss2 = self.zo_forward(batch)

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

    def zo_update(self):
        """
        Update the parameters with the estimated gradients using DeMuon (>=2D) or Adam (<2D).
        """
        torch.manual_seed(self.zo_random_seed)

        for name, param in self.named_parameters_to_optim:
            z = self._sample_z(name, param)
            g = self.projected_grad * z

            if param.ndim >= 2:
                state = self.state[name]
                momentum = state["momentum"]
                v_state = state["v"]

                g_f = g.float()
                prev_m = momentum.clone()
                momentum.mul_(1 - self.theta).add_(g_f, alpha=self.theta)
                v_state.add_(momentum - prev_m)

                update = matrix_sign(v_state.to(param.dtype), self.ns_steps)
                param.data.add_(update.reshape(param.shape), alpha=-self.lr)
            else:
                state = self.state[name]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = self.betas

                state["step"] += 1
                g_f = g.float()
                exp_avg.mul_(beta1).add_(g_f, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g_f, g_f, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = self.lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
                param.data.addcdiv_(exp_avg, denom, value=-step_size)

            if self.weight_decay > 0.0:
                param.data.add_(param.data, alpha=-self.lr * self.weight_decay)
