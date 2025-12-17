from optimizers.mezo_optimizer import MeZOFramework
from optimizers.mezo_adam_optimizer import MeZOAdamOptimizer
from optimizers.mezo_muon_optimizer import MeZOMuonOptimizer
from optimizers.mezo_bias_optimizer import *
from tqdm import tqdm
import torch
from aggregator import FedAvgAggregator
from copy import deepcopy
from evaluations import *
from utils_data.default_tokens import DefaultToken
from transformers import AutoTokenizer


class Client(object):
    def __init__(self, idx, args, candidate_seeds, train_loader, eval_loader):
        self.idx = idx
        self.args = args
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.train_iterator = iter(self.train_loader)
        self.model = None
        self.aggregator = None
        self.optimizer_state = {}

        self.device = torch.device(f"cuda:{args.device}")
        self.candidate_seeds = candidate_seeds
        self.local_seed_pool = {seed: 0.0 for seed in self.candidate_seeds}

        # Initialize tokenizer for evaluation
        self.tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        self.tokenizer.model_max_length = self.args.max_length
        special_tokens = dict()
        if self.tokenizer.pad_token is None:
            special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
        if self.tokenizer.eos_token is None:
            special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
        if self.tokenizer.bos_token is None:
            special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
        if self.tokenizer.unk_token is None:
            special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
        self.tokenizer.add_special_tokens(special_tokens)

    def __getstate__(self):
        state = self.__dict__.copy()
        if "train_iterator" in state:
            del state["train_iterator"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.train_iterator = None

    def local_train(
        self,
        cur_round,
        memory_record_dic=None,
        probabilities=None,
        gradient_history=None,
    ):
        if self.train_iterator is None:
            self.train_iterator = iter(self.train_loader)

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
                # Use position based on client idx to avoid conflict (max 5 concurrent)
                progress_bar = tqdm(
                    range(iter_steps),
                    position=self.idx % 5,
                    leave=False,
                    desc=f"Client {self.idx} Train",
                )

            for cur_step in range(iter_steps):
                # init epoch progress bar
                if self.args.batch_or_epoch == "epoch":
                    if cur_step % len(self.train_loader) == 0:
                        loss_total_train = 0.0
                        num_trained = 0
                        progress_bar = tqdm(
                            range(len(self.train_loader)),
                            position=self.idx % 5,
                            leave=False,
                            desc=f"Client {self.idx} Train",
                        )
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

        progress_bar = tqdm(
            range(len(self.local_seed_pool)),
            position=self.idx % 5,
            leave=False,
            desc=f"Client {self.idx} Update",
        )
        for i, (seed, grad) in enumerate(self.local_seed_pool.items()):
            if grad != 0.0:
                framework.zo_update(seed=seed, grad=grad)
            progress_bar.update(1)
            progress_bar.set_description(
                f"Client {self.idx} updating model from seed pool"
            )
        progress_bar.close() # Added this line

    def eval(self, cur_round):
        if self.args.eval_metric == "loss":
            return self.eval_loss(cur_round)
        else:
            return self.eval_generate(cur_round)

    def eval_loss(self, cur_round):
        self.model = self.model.to(self.device)
        self.model.eval()

        loss_total_eval = 0.0
        num_eval = 0

        position = self.idx % 5
        progress_bar = tqdm(
            total=len(self.eval_loader),
            position=position,
            leave=False,
            desc=f"Client {self.idx} Eval Loss",
        )

        with torch.inference_mode():
            for batch in self.eval_loader:
                batch = {
                    "input_ids": batch["input_ids"].to(self.device),
                    "labels": batch["labels"].to(self.device),
                    "attention_mask": batch["attention_mask"].to(self.device),
                }
                outputs = self.model(**batch)
                loss = outputs.loss
                progress_bar.update(1)
                if torch.isnan(loss):
                    continue
                loss_total_eval += loss
                num_eval += len(batch["input_ids"])
                if num_eval == 0:
                    num_eval = 1e-10
                progress_bar.set_description(
                    f"Client {self.idx} eval loss: {loss_total_eval / num_eval:.4f}"
                )

        progress_bar.close()
        if num_eval == 0:
            return float("inf")
        return (loss_total_eval / num_eval).item()

    def eval_generate(self, cur_round):
        self.model = self.model.to(self.device)
        self.model.eval()

        acc_total_eval = 0.0
        num_eval = 0

        position = self.idx % 5
        progress_bar = tqdm(
            total=len(self.eval_loader),
            position=position,
            leave=False,
            desc=f"Client {self.idx} Eval ROUGE",
        )

        with torch.inference_mode():
            for batch in self.eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                label_ids = batch["labels"].to(self.device)
                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=batch["attention_mask"].to(self.device),
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=128,
                    num_beams=1,
                )
                acc_total_eval += rouge_score(
                    output_ids[0][len(input_ids[0]) :], label_ids[0], self.tokenizer
                )
                progress_bar.update(1)
                num_eval += len(batch["input_ids"])
                if num_eval == 0:
                    num_eval = 1e-10
                progress_bar.set_description(
                    f"Client {self.idx} eval acc: {acc_total_eval / num_eval:.4f}"
                )

        progress_bar.close()
        return acc_total_eval / num_eval
