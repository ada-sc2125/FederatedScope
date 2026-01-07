from optimizers.mezo_optimizer import MeZOFramework
from optimizers.mezo_adam_optimizer import MeZOAdamOptimizer
from optimizers.mezo_muon_optimizer import MeZOMuonOptimizer
from optimizers.demuon_optimizer import DeMuonOptimizer
from optimizers.gt_nsgdm_optimizer import GTNSGDMOptimizer
from tqdm import tqdm
import torch
from aggregator import FedAvgAggregator
from copy import deepcopy
from evaluations import *
from utils_data.default_tokens import DefaultToken
from transformers import AutoTokenizer


class Client(object):
    def __init__(self, idx, args, train_loader, eval_loader): # Removed candidate_seeds
        self.idx = idx
        self.args = args
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.train_iterator = iter(self.train_loader)
        self.model = None
        self.aggregator = None
        self.optimizer_state = {}

        self.device = torch.device(f"cuda:{args.device}")
        # self.candidate_seeds = candidate_seeds # Removed
        # self.local_seed_pool = {seed: 0.0 for seed in self.candidate_seeds}

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
            for key, value in self.optimizer_state[name].items():
                if isinstance(value, torch.Tensor):
                    self.optimizer_state[name][key] = value.to(self.device)

        if memory_record_dic is not None:
            torch.cuda.empty_cache()

        lr = self.args.lr

        if self.args.batch_or_epoch == "epoch":
            iter_steps = self.args.local_step * len(self.train_loader)
        else:
            iter_steps = self.args.local_step

        if self.args.bias_sampling:
            assert probabilities is not None
            # MeZOBiasOptimizer is removed. This branch should ideally be removed
            # or refactored if bias sampling is still desired with a different mechanism.
            # For now, it will raise an error as MeZOBiasOptimizer is undefined.
            # Assuming bias_sampling is no longer used or will be handled differently.
            raise NotImplementedError("MeZOBiasOptimizer is not supported in this configuration.")
        else:
            if self.args.mezo_optimizer == "adam":
                framework = MeZOAdamOptimizer(
                    self.model,
                    args=self.args,
                    lr=lr,
                    state=self.optimizer_state,
                )
            elif self.args.mezo_optimizer == "muon":
                framework = MeZOMuonOptimizer(
                    self.model,
                    args=self.args,
                    lr=lr,
                    state=self.optimizer_state,
                )
            elif self.args.mezo_optimizer == "demuon":
                framework = DeMuonOptimizer(
                    self.model,
                    args=self.args,
                    lr=lr,
                    state=self.optimizer_state,
                )
            elif self.args.mezo_optimizer == "gt_nsgdm":
                framework = GTNSGDMOptimizer(
                    self.model,
                    args=self.args,
                    lr=lr,
                    state=self.optimizer_state,
                )
            else:  # 'sgd'
                framework = MeZOFramework(
                    self.model,
                    args=self.args,
                    lr=lr,
                )
        if self.args.mezo_optimizer in ["demuon", "gt_nsgdm"]:
            self.model.train()
            if self.args.batch_or_epoch == "batch":
                loss_total_train = 0.0
                num_trained = 0
                progress_bar = tqdm(
                    range(iter_steps),
                    leave=True,
                    desc=f"Client {self.idx} Train",
                )

            for cur_step in range(iter_steps):
                if self.args.batch_or_epoch == "epoch":
                    if cur_step % len(self.train_loader) == 0:
                        epoch_idx = cur_step // len(self.train_loader) + 1
                        print(
                            f"Client {self.idx} starting epoch {epoch_idx}/{self.args.local_step}"
                        )
                        loss_total_train = 0.0
                        num_trained = 0
                        progress_bar = tqdm(
                            range(len(self.train_loader)),
                            leave=True,
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
                outputs = self.model(**batch)
                loss = outputs.loss
                if torch.isnan(loss):
                    continue
                loss.backward()
                framework.step()
                framework.zero_grad()
                progress_bar.update(1)
                loss_total_train += loss.detach()
                num_trained += len(batch["input_ids"])
                if self.args.batch_or_epoch == "epoch":
                    progress_bar.set_description(
                        f"client {self.idx} train at epoch {int(cur_step / len(self.train_loader)) + 1}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}"
                    )
                else:
                    progress_bar.set_description(
                        f"client {self.idx} train at step {cur_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}"
                    )
                if (
                    self.args.batch_or_epoch == "epoch"
                    and (cur_step + 1) % len(self.train_loader) == 0
                ):
                    epoch_idx = (cur_step + 1) // len(self.train_loader)
                    print(
                        f"Client {self.idx} finished epoch {epoch_idx}/{self.args.local_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}"
                    )
            if self.args.batch_or_epoch == "epoch" and "progress_bar" in locals():
                progress_bar.close()
        else:
            self.model.eval()
            with torch.inference_mode():
                if self.args.batch_or_epoch == "batch":
                    loss_total_train = 0.0
                    num_trained = 0
                    progress_bar = tqdm(
                        range(iter_steps),
                        leave=True,
                        desc=f"Client {self.idx} Train",
                    )

                for cur_step in range(iter_steps):
                    # init epoch progress bar
                    if self.args.batch_or_epoch == "epoch":
                        if cur_step % len(self.train_loader) == 0:
                            epoch_idx = cur_step // len(self.train_loader) + 1
                            print(
                                f"Client {self.idx} starting epoch {epoch_idx}/{self.args.local_step}"
                            )
                            loss_total_train = 0.0
                            num_trained = 0
                            progress_bar = tqdm(
                                range(len(self.train_loader)),
                                leave=True,
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
                    logits, loss = framework.zo_step(batch) # Removed local_seed_pool
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
                    if (
                        self.args.batch_or_epoch == "epoch"
                        and (cur_step + 1) % len(self.train_loader) == 0
                    ):
                        epoch_idx = (cur_step + 1) // len(self.train_loader)
                        print(
                            f"Client {self.idx} finished epoch {epoch_idx}/{self.args.local_step}, loss: {loss_total_train / num_trained if num_trained != 0 else 0.0}"
                        )
                if self.args.batch_or_epoch == "epoch" and "progress_bar" in locals():
                    progress_bar.close()

        # Move optimizer state to CPU
        for name in self.optimizer_state:
            for key, value in self.optimizer_state[name].items():
                if isinstance(value, torch.Tensor):
                    self.optimizer_state[name][key] = value.cpu()

        if memory_record_dic is not None:
            memory_record_dic[self.device.index] = {}
            memory_record_dic[self.device.index]["max_memory_allocated"] = (
                torch.cuda.max_memory_allocated(self.device)
            )
            memory_record_dic[self.device.index]["max_memory_reserved"] = (
                torch.cuda.max_memory_reserved(self.device)
            )

        # Unload model to CPU, do not set to None
        self.model.cpu()
        model_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        return model_state, self.optimizer_state

    def set_parameters(self, model_state_dict, optimizer_state_dict):
        """
        Sets the model parameters and optimizer state.
        """
        # Kept for compatibility if needed, but load_model_and_optimizer is preferred
        pass

    def load_model_and_optimizer(self, model, model_state_dict, optimizer_state_dict):
        # 'model' arg is kept to match call signature in main.py, but we use self.model if set
        if self.model is None:
            self.model = model

        self.model.load_state_dict(model_state_dict)
        self.optimizer_state = optimizer_state_dict

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

        progress_bar = tqdm(
            total=len(self.eval_loader),
            leave=True,
            desc=f"Client {self.idx} Eval Loss",
        )

        with torch.inference_mode():
            for i, batch in enumerate(self.eval_loader):
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device) # Keep labels on GPU for now

                # # Print input and labels
                # print(f"\nClient {self.idx} Eval Input (Round {cur_round}):")
                # print(self.tokenizer.decode(input_ids[0], skip_special_tokens=True))
                # print(f"Client {self.idx} Eval Target (Labels, Round {cur_round}):")
                # print(self.tokenizer.decode(labels[0], skip_special_tokens=True))

                batch_on_device = {
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": batch["attention_mask"].to(self.device),
                }
                outputs = self.model(**batch_on_device)
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

        # Move optimizer state to CPU
        for name in self.optimizer_state:
            for key, value in self.optimizer_state[name].items():
                if isinstance(value, torch.Tensor):
                    self.optimizer_state[name][key] = value.cpu()
        self.model.cpu()
        if num_eval == 0:
            return float("inf")
        return (loss_total_eval / num_eval).item()

    def eval_generate(self, cur_round):
        self.model = self.model.to(self.device)
        self.model.eval()

        acc_total_eval = 0.0
        num_eval = 0

        progress_bar = tqdm(
            total=len(self.eval_loader),
            leave=True,
            desc=f"Client {self.idx} Eval ROUGE",
        )

        with torch.inference_mode():
            for batch in self.eval_loader:
                input_ids = batch["input_ids"].to(self.device)
                label_ids = batch["labels"].to(self.device)
                
                # # Print input
                # print(f"\nClient {self.idx} Eval Input (Round {cur_round}):")
                # print(self.tokenizer.decode(input_ids[0], skip_special_tokens=True))

                output_ids = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=batch["attention_mask"].to(self.device),
                    pad_token_id=self.tokenizer.pad_token_id,
                    max_new_tokens=128,
                    num_beams=1,
                )
                
                # # Print output
                # print(f"Client {self.idx} Eval Output (Round {cur_round}):")
                # print(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))
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
        # Move optimizer state to CPU
        for name in self.optimizer_state:
            for key, value in self.optimizer_state[name].items():
                if isinstance(value, torch.Tensor):
                    self.optimizer_state[name][key] = value.cpu()

        self.model.cpu()
        return acc_total_eval / num_eval
