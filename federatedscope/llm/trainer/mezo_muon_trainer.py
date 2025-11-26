'''
The implementations of
MeZO optimizer in federatedscope/llm/trainer/mezo_trainer.py
is adapted from https://github.com/princeton-nlp/MeZO (MIT License)

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
'''

import numpy as np
import torch
import logging
from federatedscope.register import register_trainer
from federatedscope.llm.trainer.trainer import LLMTrainer
from federatedscope.core.auxiliaries.scheduler_builder import get_scheduler
from federatedscope.core.trainers.context import CtxVar
from federatedscope.core.trainers.enums import LIFECYCLE, MODE

logger = logging.getLogger(__name__)


# Muon specific functions, copied from muon.py
def zeropower_via_newtonschulz5(G, steps: int):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G.
    """
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
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
    momentum.lerp_(grad, 1 - beta)
    update = grad.lerp_(momentum, beta) if nesterov else momentum
    if update.ndim == 4:  # for the case of conv filters
        update = update.view(len(update), -1)
    # Skip orthogonalization for vectors (e.g., biases)
    if update.ndim >= 2:
        update = zeropower_via_newtonschulz5(update, steps=ns_steps)
        update *= max(1, grad.size(-2) / grad.size(-1))**0.5
    return update


# MeZO specific functions
def zo_step(ctx, zo_eps=1e-3):
    """
    Estimate gradient by MeZO. Return the loss from f(theta + z)
    """
    # determine which parameters to optimize
    ctx.named_parameters_to_optim = []
    for name, param in ctx.model.named_parameters():
        if param.requires_grad:
            ctx.named_parameters_to_optim.append((name, param))

    # Sample the random seed for sampling z
    ctx.zo_random_seed = np.random.randint(1000000000)

    # First function evaluation
    zo_perturb_parameters(ctx=ctx, scaling_factor=1, zo_eps=zo_eps)
    logits1, loss1 = zo_forward(ctx)

    # Second function evaluation
    zo_perturb_parameters(ctx=ctx, scaling_factor=-2, zo_eps=zo_eps)
    _, loss2 = zo_forward(ctx)

    ctx.projected_grad = ((loss1 - loss2) / (2 * zo_eps)).item()

    # Reset model back to its parameters at start of step
    zo_perturb_parameters(ctx=ctx, scaling_factor=1, zo_eps=zo_eps)
    return logits1, loss1


def zo_perturb_parameters(ctx, scaling_factor=1, zo_eps=1e-3):
    """
    Perturb the parameters with random vector z.
    """
    torch.manual_seed(ctx.zo_random_seed)

    for _, param in ctx.named_parameters_to_optim:
        z = torch.normal(mean=0,
                         std=1,
                         size=param.data.size(),
                         device=param.data.device,
                         dtype=param.data.dtype)
        param.data = param.data + scaling_factor * z * zo_eps


def zo_forward(ctx):
    """
    Get (no gradient) loss from the model. Dropout is turned off too.
    """
    ctx.model.eval()
    with torch.inference_mode():
        input_ids = ctx.data_batch['input_ids'].to(ctx.device)
        labels = ctx.data_batch['labels'].to(ctx.device)
        attention_mask = ctx.data_batch['attention_mask'].to(ctx.device)

        outputs = ctx.model(input_ids=input_ids,
                            labels=labels,
                            attention_mask=attention_mask)
        logits = outputs.logits
        loss = outputs.loss
    return logits.detach(), loss.detach()


class MeZOMuonTrainer(LLMTrainer):
    def train(self, *args, **kwargs):
        # Force-initialize counters at the entry point of training
        self.ctx.num_samples = 0
        self.ctx.loss_batch_total = 0
        self.ctx.loss_regular_total = 0
        if self.ctx.cfg.grad.grad_accum_count > 1:
            self.ctx.loss_task_total = 0

        return super().train(*args, **kwargs)

    def _hook_on_fit_start_init(self, ctx):
        """
        Custom initialization for MeZO-Muon.
        """
        ctx.model.to(ctx.device)

        # Create a dummy optimizer for the scheduler to wrap
        dummy_optimizer = torch.optim.SGD(ctx.model.parameters(), lr=0.0)
        ctx.scheduler = get_scheduler(dummy_optimizer,
                                      **ctx.cfg.train.scheduler)
        ctx.optimizer = None  # Explicitly set to None

        # Initialize states for MeZO-Muon (momentum)
        ctx.momentum_buffer = {}
        named_parameters_to_optim = []
        for name, param in ctx.model.named_parameters():
            if param.requires_grad:
                named_parameters_to_optim.append((name, param))

        for name, param in named_parameters_to_optim:
            ctx.momentum_buffer[name] = torch.zeros_like(
                param, memory_format=torch.preserve_format)

    def _hook_on_batch_forward(self, ctx):
        if ctx.cur_mode in [MODE.TRAIN, MODE.FINETUNE]:

            logits, loss = zo_step(ctx)
            labels = ctx.data_batch['labels'].to(ctx.device)
            if torch.isnan(loss):
                ctx.skip_this_batch = CtxVar(True, LIFECYCLE.BATCH)
                logger.warning(
                    'Skip the batch due to the loss is NaN, '
                    'it may be caused by exceeding the precision or '
                    'invalid labels.')
            else:
                ctx.skip_this_batch = CtxVar(False, LIFECYCLE.BATCH)

            ctx.y_true = CtxVar(labels, LIFECYCLE.BATCH)
            ctx.y_prob = CtxVar(logits, LIFECYCLE.BATCH)

            ctx.loss_batch = CtxVar(loss, LIFECYCLE.BATCH)
            ctx.batch_size = CtxVar(len(labels), LIFECYCLE.BATCH)

        else:
            return super()._hook_on_batch_forward(ctx)

    def _hook_on_batch_backward(self, ctx):

        if ctx.skip_this_batch:
            return

        torch.manual_seed(ctx.zo_random_seed)

        if ctx.scheduler is not None:
            lr = ctx.scheduler.get_lr()[0]
        else:
            lr = ctx.cfg.train.optimizer.lr

        # Get Muon hyperparameters from config
        beta = ctx.cfg.train.optimizer.momentum
        weight_decay = ctx.cfg.train.optimizer.weight_decay
        ns_steps = ctx.cfg.train.optimizer.ns_steps
        nesterov = ctx.cfg.train.optimizer.nesterov

        for name, param in ctx.named_parameters_to_optim:
            if not param.requires_grad:
                continue

            # Re-generate the random vector z with the same seed
            z = torch.normal(mean=0,
                             std=1,
                             size=param.data.size(),
                             device=param.data.device,
                             dtype=param.data.dtype)

            # Estimated gradient from MeZO
            g = ctx.projected_grad * z

            # Muon update logic
            momentum = ctx.momentum_buffer[name]
            update = muon_update(g,
                                 momentum,
                                 beta=beta,
                                 ns_steps=ns_steps,
                                 nesterov=nesterov)

            # AdamW-style weight decay
            if weight_decay != 0 and "bias" not in name and "layer_norm" \
                    not in name and "layernorm" not in name:
                param.data.mul_(1 - lr * weight_decay)

            # Parameter update
            param.data.add_(update.reshape(param.shape), alpha=-lr)

        if ctx.scheduler is not None:
            ctx.scheduler.step()


def call_mezo_muon_trainer(trainer_type):
    if trainer_type == 'mezo_muon_trainer':
        trainer_builder = MeZOMuonTrainer
        return trainer_builder


register_trainer('mezo_muon_trainer', call_mezo_muon_trainer)