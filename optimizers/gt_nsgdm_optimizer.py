import torch


class GTNSGDMOptimizer(object):
    def __init__(self, model, args, lr, state=None):
        print("GT-NSGDm")
        self.args = args
        self.lr = lr
        self.beta = getattr(args, "gt_beta", 0.9)
        self.eps = getattr(args, "gt_eps", 1e-12)
        self.weight_decay = getattr(args, "weight_decay", 0.0)
        self.model = model

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
            self.state[name] = {
                "v": torch.zeros_like(
                    param, dtype=torch.float32, memory_format=torch.preserve_format
                ),
                "y": torch.zeros_like(
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
            grad = param.grad.detach().float()
            state = self.state[name]
            v = state["v"]
            y = state["y"]

            prev_v = v.clone()
            v.mul_(self.beta).add_(grad, alpha=1 - self.beta)
            y.add_(v - prev_v)

            update = y.to(param.dtype)
            norm = update.norm()
            update = update / (norm + self.eps)

            param.data.add_(update.reshape(param.shape), alpha=-self.lr)

            if self.weight_decay > 0.0:
                param.data.add_(param.data, alpha=-self.lr * self.weight_decay)
