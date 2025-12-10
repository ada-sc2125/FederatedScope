from optimizers.mezo_optimizer import MeZOFramework
from optimizers.mezo_adam_optimizer import MeZOAdamOptimizer
from optimizers.mezo_muon_optimizer import MeZOMuonOptimizer
from optimizers.mezo_bias_optimizer import *
from tqdm import tqdm
import torch
from aggregator import FedAvgAggregator
from copy import deepcopy


class Client(object):
    def __init__(self, idx, args, candidate_seeds, train_loader):
        self.idx = idx
        self.args = args
        self.train_loader = train_loader
        self.train_iterator = iter(self.train_loader)
        self.model = None
        self.aggregator = None
        self.optimizer_state = {}

        self.device = torch.device(f"cuda:{args.device}")
        self.candidate_seeds = candidate_seeds
        self.local_seed_pool = {seed: 0.0 for seed in self.candidate_seeds}

    def local_train(
        self,
        cur_round,
        memory_record_dic=None,
        probabilities=None,
        gradient_history=None,
    ):
        self.model.to(self.device)

        # Move optimizer state to GPU
        for name in self.optimizer_state:
            if "exp_avg" in self.optimizer_state[name]:
                self.optimizer_state[name]["exp_avg"] = self.optimizer_state[name][
                    "exp_avg"
                ].to(self.device)
            if "exp_avg_sq" in self.optimizer_state[name]:
                self.optimizer_state[name]["exp_avg_sq"] = self.optimizer_state[name][
                    "exp_avg_sq"
                ].to(self.device)

        if memory_record_dic is not None:
            torch.cuda.empty_cache()

        lr = self.args.lr

        if self.args.batch_or_epoch == "epoch":
            iter_steps = self.args.local_step * len(self.train_loader)
        else:
            iter_steps = self.args.local_step

        if self.args.bias_sampling:
            assert probabilities is not None
            framework = MeZOBiasOptimizer(
                self.model,
                args=self.args,
                lr=lr,
                candidate_seeds=self.candidate_seeds,
                probabilities=probabilities,
                gradient_history=gradient_history,
            )
        else:
            if self.args.mezo_optimizer == "adam":
                framework = MeZOAdamOptimizer(
                    self.model,
                    args=self.args,
                    lr=lr,
                    candidate_seeds=self.candidate_seeds,
                    state=self.optimizer_state,
                )
            elif self.args.mezo_optimizer == "muon":
                framework = MeZOMuonOptimizer(
                    self.model,
                    args=self.args,
                    lr=lr,
                    candidate_seeds=self.candidate_seeds,
                    state=self.optimizer_state,
                )
            else:  # 'sgd'
                framework = MeZOFramework(
                    self.model,
                    args=self.args,
                    lr=lr,
                    candidate_seeds=self.candidate_seeds,
                )
        self.model.eval()
        with torch.inference_mode():
            if self.args.batch_or_epoch == "batch":
                loss_total_train = 0.0
                num_trained = 0
                progress_bar = tqdm(range(iter_steps))

            for cur_step in range(iter_steps):
                # init epoch progress bar
                if self.args.batch_or_epoch == "epoch":
                    if cur_step % len(self.train_loader) == 0:
                        loss_total_train = 0.0
                        num_trained = 0
                        progress_bar = tqdm(range(len(self.train_loader)))
                try:
                    batch = next(self.train_iterator)
                except StopIteration:
                    self.train_iterator = iter(self.train_loader)
                    batch = next(self.train_iterator)
                batch = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "labels": batch["labels"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }
                logits, loss = framework.zo_step(
                    batch, local_seed_pool=self.local_seed_pool
                )
                progress_bar.update(1)
                if (not torch.isnan(loss)) and (
                    self.args.grad_clip <= 0 or loss != 0.0
                ):
                    loss_total_train += loss
                    num_trained += len(batch["input_ids"])
                if self.args.batch_or_epoch == "epoch":
                    progress_bar.set_description(
                        f"client {self.idx} train at epoch {int(cur_step / len(self.train_loader)) + 1}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}"
                    )
                else:
                    progress_bar.set_description(
                        f"client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}"
                    )

        # Move optimizer state to CPU
        for name in self.optimizer_state:
            if "exp_avg" in self.optimizer_state[name]:
                self.optimizer_state[name]["exp_avg"] = self.optimizer_state[name][
                    "exp_avg"
                ].cpu()
            if "exp_avg_sq" in self.optimizer_state[name]:
                self.optimizer_state[name]["exp_avg_sq"] = self.optimizer_state[name][
                    "exp_avg_sq"
                ].cpu()

        if memory_record_dic is not None:
            memory_record_dic[self.device.index] = {}
            memory_record_dic[self.device.index]["max_memory_allocated"] = (
                torch.cuda.max_memory_allocated(self.device)
            )
            memory_record_dic[self.device.index]["max_memory_reserved"] = (
                torch.cuda.max_memory_reserved(self.device)
            )

        self.model = None

    def update_model_by_seed_pool(self, pulled_model):
        """
        Resets the model to a pulled state and then updates it using the current local_seed_pool.
        """
        self.model = pulled_model
        self.model.to(self.device)

        self.optimizer_state = {}

        if self.args.mezo_optimizer == "adam":
            framework = MeZOAdamOptimizer(
                self.model,
                args=self.args,
                lr=self.args.lr,
                candidate_seeds=self.candidate_seeds,
                state=self.optimizer_state,
            )
        elif self.args.mezo_optimizer == "muon":
            framework = MeZOMuonOptimizer(
                self.model,
                args=self.args,
                lr=self.args.lr,
                candidate_seeds=self.candidate_seeds,
                state=self.optimizer_state,
            )
        else:  # 'sgd'
            framework = MeZOFramework(
                self.model,
                args=self.args,
                lr=self.args.lr,
                candidate_seeds=self.candidate_seeds,
            )

        progress_bar = tqdm(range(len(self.local_seed_pool)))
        for i, (seed, grad) in enumerate(self.local_seed_pool.items()):
            if grad != 0.0:
                framework.zo_update(seed=seed, grad=grad)
            progress_bar.update(1)
            progress_bar.set_description(f"Client {self.idx} updating model from seed pool")
