import math
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


class DeMuonOptimizer(object):
    def __init__(self, model, args, lr, state=None):
        print("DeMuon")
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

    def zero_grad(self):
        for _, param in self.named_parameters_to_optim:
            if param.grad is not None:
                param.grad.zero_()

    @torch.no_grad()
    def step(self):
        for name, param in self.named_parameters_to_optim:
            if param.grad is None:
                continue
            grad = param.grad.detach()
            grad_f = grad.float()

            if param.ndim >= 2:
                state = self.state[name]
                momentum = state["momentum"]
                v_state = state["v"]

                prev_m = momentum.clone()
                momentum.mul_(1 - self.theta).add_(grad_f, alpha=self.theta)
                v_state.add_(momentum - prev_m)

                update = matrix_sign(v_state.to(param.dtype), self.ns_steps)
                param.data.add_(update.reshape(param.shape), alpha=-self.lr)
            else:
                state = self.state[name]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                beta1, beta2 = self.betas

                state["step"] += 1
                exp_avg.mul_(beta1).add_(grad_f, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad_f, grad_f, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = self.lr / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(self.eps)
                param.data.addcdiv_(exp_avg, denom, value=-step_size)

            if self.weight_decay > 0.0:
                param.data.add_(param.data, alpha=-self.lr * self.weight_decay)
