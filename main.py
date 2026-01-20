import argparse
import os
import time
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
import resource
from tqdm import tqdm
from client import Client
from utils_data.load_data import get_loaders
from topologies import (
    create_ring_topology,
    create_full_topology,
    create_star_topology,
    create_grid_topology,
)
from aggregator import (
    aggregate_state_dicts,
    aggregate_optimizer_states,
    aggregate_state_dicts_streaming,
    aggregate_optimizer_states_streaming,
)
from transformers import AutoModelForCausalLM

import yaml
from copy import deepcopy
import json

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_model(args):
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    return model


def log_memory(tag, device):
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(device)
        max_allocated = torch.cuda.max_memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        max_reserved = torch.cuda.max_memory_reserved(device)
        print(
            f"[mem] {tag} | rss_kb={rss_kb} | cuda_alloc={allocated} "
            f"| cuda_max_alloc={max_allocated} | cuda_reserved={reserved} "
            f"| cuda_max_reserved={max_reserved}"
        )
    else:
        print(f"[mem] {tag} | rss_kb={rss_kb}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Federation
    parser.add_argument("--num_clients", type=int, default=200, help="N in our paper")
    parser.add_argument(
        "-m",
        type=float,
        default=0.05,
        help="ratio of activate clients in each round (only for centralized FL)",
    )
    parser.add_argument(
        "--rounds", type=int, default=40, help="the total number of rounds"
    )
    parser.add_argument(
        "--local_step", type=int, default=200, help=r"$ tau in our paper"
    )
    parser.add_argument(
        "--batch_or_epoch", type=str, default="batch", choices=["epoch", "batch"]
    )
    parser.add_argument(
        "--equal_weight",
        default=False,
        action="store_true",
        help="if `true`, the weights among clients for aggregation are the same",
    )

    # Gossip-related arguments
    parser.add_argument(
        "--topology",
        type=str,
        default="random",
        choices=["ring", "full", "star", "grid", "random"],
        help="Network topology for gossip communication",
    )
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=2,
        help="Number of neighbors for random gossip aggregation",
    )

    # Data
    # Arguments related to data on both datasets
    parser.add_argument(
        "--dataset",
        type=str,
        default="instruct",
        choices=["instruct", "dolly", "gsm8k", "code_contests", "sst2"],
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="batch size > 1 may cause error during running",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="evaluation batch size",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
        help="the max number of tokens of a data instance",
    )
    parser.add_argument(
        "--use_prompts",
        default=True,
        help="if `true`, the prompt template from alpaca is adopted",
    )

    # Arguments related to data only for Dolly-15K
    parser.add_argument(
        "--iid",
        type=str,
        default="dir0.5",
        help=r"`dir{alpha}` means that \alpha in Dirichlet distribution, `0` means IID split",
    )
    parser.add_argument(
        "--zerotask",
        default=7,
        type=int,
        help="the index of the task for evaluation in dolly-15K",
    )
    parser.add_argument(
        "--dataset_subsample",
        type=float,
        default=1.0,
        help="used for sampling a subset from the original dataset, only effective for dolly-15K, gsm8k, and sst2",
    )

    # Model
    parser.add_argument(
        "--model", type=str, default="datajuicer/LLaMA-1B-dj-refine-150B"
    )

    # Training
    parser.add_argument(
        "--lr", type=float, default=0.0000001, help=r"learning rate \eta"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-4, help="weight decay in MeZO"
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=-100.0,
        help="clip the over large loss value, if < 0, disable this feature",
    )

    # Training args only for `FedKSeed`
    parser.add_argument(
        "-K", type=int, default=4096, help="Number of candidate seeds for MeZO"
    )
    parser.add_argument("--zo_eps", type=float, default=0.0005, help=r"eps in MeZO")

    # MeZO Optimizer Arguments
    parser.add_argument(
        "--mezo_optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "muon", "demuon", "gt_nsgdm"],
        help="Which optimizer to use.",
    )
    parser.add_argument(
        "--adam_beta1", type=float, default=0.9, help="beta1 for MeZO-Adam"
    )
    parser.add_argument(
        "--adam_beta2", type=float, default=0.999, help="beta2 for MeZO-Adam"
    )
    parser.add_argument(
        "--adam_eps", type=float, default=1e-8, help="epsilon for MeZO-Adam"
    )
    parser.add_argument(
        "--muon_lr", type=float, default=0.0005, help="learning rate for Muon optimizer"
    )
    parser.add_argument("--mu", type=float, default=0.9, help="mu for MeZO-Muon")
    parser.add_argument(
        "--ns_steps", type=int, default=5, help="Number of Newton-Schulz steps for Muon"
    )
    parser.add_argument(
        "--demuon_theta",
        type=float,
        default=0.1,
        help="EMA factor for DeMuon momentum gradient estimator",
    )
    parser.add_argument(
        "--gt_beta",
        type=float,
        default=0.9,
        help="Momentum factor for GT-NSGDm",
    )
    parser.add_argument(
        "--gt_eps",
        type=float,
        default=1e-12,
        help="Normalization epsilon for GT-NSGDm",
    )

    # Training args only for `FedKSeed-Pro`
    parser.add_argument(
        "--bias_sampling",
        default=False,
        action="store_true",
        help="if `true`, the probabilities of candidate seeds to be sampled are not identical, i.e., FedKSeed-Pro",
    )
    parser.add_argument(
        "--bias_loss_clip",
        default=1000.0,
        type=float,
        help="scalar gradient whose abstract values exceeds this value will be cliped",
    )
    parser.add_argument(
        "--grad_initial",
        default=0.0,
        type=float,
        help="initial value of scalar gradient history corresponding to each candidate seed",
    )

    # Environment
    parser.add_argument(
        "--device", type=int, default=0, help="index of the targeted cuda device"
    )
    parser.add_argument(
        "--log",
        default=False,
        action="store_true",
        help="if `true`, running logs will be recorded in files",
    )
    parser.add_argument("--log_root", default="logs", help="root path of log files")
    parser.add_argument(
        "--seed", default=42, type=int, help="global seed, for reproducibility"
    )

    # Evaluation
    parser.add_argument(
        "--eval_metric",
        default="rouge",
        type=str,
        choices=["rouge", "loss"],
        help="metric to evaluate global model in the last round",
    )

    # Checkpoints
    parser.add_argument(
        "--save",
        default=False,
        action="store_true",
        help="if `true`, the checkpoint of tuned models will be stored",
    )
    parser.add_argument(
        "--eval_print_io",
        default=False,
        action="store_true",
        help="if `true`, print input/output during evaluation",
    )
    parser.add_argument(
        "--eval_print_n",
        type=int,
        default=2,
        help="number of samples to print per evaluation",
    )
    parser.add_argument(
        "--eval_rouge_limit",
        type=int,
        default=50,
        help="number of samples to evaluate for ROUGE each time",
    )

    time_stamp = str(time.time())
    args = parser.parse_args()

    eval_avg_acc = []
    eval_acc_every5 = []
    eval_rouge_every5 = []
    memory_record_dic = {}

    previous_metric = args.eval_metric
    # set CUDA visibility to targeted cuda device, to avoid the several hundred MB memory consumption of device 0
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    setup_seed(args.seed)
    list_train_loader, eval_loader, tokenizer = get_loaders(args)
    rouge_eval_loader = None
    if args.dataset == "dolly":
        prev_metric = args.eval_metric
        args.eval_metric = "rouge"
        _, rouge_eval_loader, _ = get_loaders(args, only_eval=True)
        args.eval_metric = prev_metric

    if args.dataset == "instruct":
        args.iid = "meta"
    log_dir = time_stamp

    if args.log_root != "":
        log_dir = os.path.join(args.log_root, log_dir)
    if args.log:
        os.makedirs(log_dir)
    config = yaml.dump(args, None)
    config = "\n".join(config.split("\n")[1:])
    print("Configs: ")
    print(config)
    print("=====================")
    if args.log:
        with open(os.path.join(log_dir, "config.yaml"), "w") as writer:
            writer.write(config)

    # since only CUDA device is available, load all models on device 0
    args.device = 0
    device = torch.device(f"cuda:{args.device}")

    client_list = []

    # Load base model (template)
    print("Loading base model...")
    base_model = get_model(args)
    base_model.resize_token_embeddings(len(tokenizer))
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = tokenizer.pad_token_id

    for idx in range(args.num_clients):
        client_list.append(
            Client(idx, args, list_train_loader[idx], eval_loader)
        )

    # --- Create network topology ---
    print(f"Creating '{args.topology}' topology...")
    client_adj = None
    if args.topology != "random":
        if args.topology == "ring":
            neighborhoods = create_ring_topology(client_list)
        elif args.topology == "full":
            neighborhoods = create_full_topology(client_list)
        elif args.topology == "star":
            neighborhoods = create_star_topology(client_list)
        elif args.topology == "grid":
            neighborhoods = create_grid_topology(client_list)
        # Convert to adjacency list of indices
        client_adj = {
            idx: [neighbor.idx for neighbor in neighbors]
            for idx, neighbors in neighborhoods.items()
        }
    print("Topology created.")

    # Initialize client models (Persistent in Memory)
    print("Initializing client models...")
    for client in client_list:
        client.model = deepcopy(base_model)
    print("Client models initialized.")

    # Initial evaluation (evaluating the initial model using the first client as a runner)
    print("Performing initial evaluation...")
    eval_result = client_list[0].eval(cur_round=0)
    torch.cuda.empty_cache()

    eval_avg_acc.append(eval_result)
    print(f"Initial evaluation result: {eval_result}")

    if args.log:
        with open(os.path.join(log_dir, "memory.json"), "w") as writer:
            json.dump(memory_record_dic, writer)
        with open(os.path.join(log_dir, "results.json"), "w") as writer:
            json.dump({"eval_avg_acc": eval_avg_acc}, writer)

    # Disable parallel workers for now as the logic has changed significantly
    use_parallel_workers = False

    for r in range(1, args.rounds + 1):
        print(f"--- Round {r}/{args.rounds} ---")

        # --- 1. Local Training Phase ---
        print("--- Kicking off client model updates and local training ---")

        trained_states = {}

        for client in client_list:
            # Local Train (updates client.model in-place, returns CPU tensors)
            model_state, optimizer_state = client.local_train(cur_round=r)
            log_memory(f"post-train client {client.idx}", device)

            # Store update in memory for aggregation buffer
            # We must deepcopy model_state because client.model will be overwritten in the next step
            trained_states[client.idx] = {
                "model": deepcopy(model_state),
                "optimizer": deepcopy(optimizer_state),
            }

            torch.cuda.empty_cache()

        print("--- Client updates and local training finished ---")

        # --- 2. Aggregation & Evaluation Phase ---
        print("--- Kicking off aggregation and evaluation ---")
        log_memory("pre-aggregation", device)

        # Determine current topology (random gossip changes per round)
        current_adj = client_adj
        if args.topology == "random":
            temp_neighborhoods = {}
            for client in client_list:
                other_clients = [c for c in client_list if c.idx != client.idx]
                num_neighbors = min(args.num_neighbors, len(other_clients))
                temp_neighborhoods[client] = random.sample(other_clients, num_neighbors)
            current_adj = {
                client.idx: [n.idx for n in temp_neighborhoods[client]]
                for client in client_list
            }

        round_eval_metrics = []

        for client in client_list:
            # Gather neighbors
            client_id = client.idx
            neighbor_ids = current_adj.get(client_id, [])
            ids_to_aggregate = [client_id] + neighbor_ids

            # Gather neighbor updates (CPU) from trained_states buffer
            neighbor_model_states = [
                trained_states[nid]["model"] for nid in ids_to_aggregate
            ]
            neighbor_opt_states = [
                trained_states[nid]["optimizer"] for nid in ids_to_aggregate
            ]

            # Aggregate on GPU with streaming to limit peak memory
            agg_model_state_gpu = aggregate_state_dicts_streaming(
                neighbor_model_states, device=device
            )
            agg_opt_state_gpu = aggregate_optimizer_states_streaming(
                neighbor_opt_states, device=device
            )
            log_memory(f"post-aggregation client {client.idx}", device)

            # # Move back to CPU for storage and client loading
            # agg_model_state_cpu = {k: v.cpu() for k, v in agg_model_state_gpu.items()}

            # agg_opt_state_cpu = {}
            # for k, v in agg_opt_state_gpu.items():
            #     agg_opt_state_cpu[k] = {}
            #     for sub_k, sub_v in v.items():
            #         if isinstance(sub_v, torch.Tensor):
            #             agg_opt_state_cpu[k][sub_k] = sub_v.cpu()
            #         else:
            #             agg_opt_state_cpu[k][sub_k] = sub_v

            # Load aggregated state into client for evaluation and next round
            # We use None for model arg because client.model is persistent
            client.load_model_and_optimizer(
                None, agg_model_state_gpu, agg_opt_state_gpu
            )

            # Evaluation
            eval_result = client.eval(cur_round=r)
            round_eval_metrics.append(eval_result)

            torch.cuda.empty_cache()

            # # Explicitly free GPU tensors from aggregation
            # del agg_model_state_gpu
            # del agg_opt_state_gpu

        # Average metric across all clients
        avg_metric = np.mean(round_eval_metrics) if round_eval_metrics else float("inf")
        eval_avg_acc.append(avg_metric)
        print(
            f"--- Round {r} evaluation finished. Average {args.eval_metric}: {avg_metric} ---"
        )

        if args.dataset == "sst2" and r % 5 == 0:
            print(f"--- Round {r} SST2 accuracy evaluation ---")
            prev_metric = args.eval_metric
            args.eval_metric = "rouge"
            acc_results = []
            from utils_data.llm_dataset import LLMDataset, LLMDataCollator
            acc_eval_dataset = LLMDataset(
                args.dataset, tokenizer=tokenizer, generation=True, split="validation"
            )
            acc_data_collator = LLMDataCollator(tokenizer=tokenizer)
            acc_eval_loader = DataLoader(
            acc_eval_dataset, batch_size=args.eval_batch_size, collate_fn=acc_data_collator
            )
            for client in tqdm(client_list, desc=f"Accuracy Eval (round {r})"):
                prev_loader = client.eval_loader
                client.eval_loader = acc_eval_loader
                acc_results.append(client.eval(cur_round=r))
                client.eval_loader = prev_loader
                torch.cuda.empty_cache()
            avg_acc = np.mean(acc_results) if acc_results else 0.0
            eval_acc_every5.append({"round": r, "accuracy": avg_acc})
            print(f"--- Round {r} SST2 Average Accuracy: {avg_acc} ---")
            args.eval_metric = prev_metric

        if args.dataset == "dolly" and r % 5 == 0 and rouge_eval_loader is not None:
            print(f"--- Round {r} ROUGE evaluation ---")
            prev_metric = args.eval_metric
            args.eval_metric = "rouge"
            prev_print = args.eval_print_io
            args.eval_print_io = True
            acc_results = []
            for client in tqdm(client_list, desc=f"ROUGE Eval (round {r})"):
                prev_loader = client.eval_loader
                if args.eval_rouge_limit > 0:
                    rouge_subset = torch.utils.data.Subset(
                        rouge_eval_loader.dataset,
                        list(
                            range(
                                min(
                                    args.eval_rouge_limit,
                                    len(rouge_eval_loader.dataset),
                                )
                            )
                        ),
                    )
                    rouge_eval_subset_loader = DataLoader(
                        rouge_subset,
                        batch_size=rouge_eval_loader.batch_size,
                        collate_fn=rouge_eval_loader.collate_fn,
                    )
                    client.eval_loader = rouge_eval_subset_loader
                else:
                    client.eval_loader = rouge_eval_loader
                acc_results.append(client.eval(cur_round=r))
                client.eval_loader = prev_loader
                torch.cuda.empty_cache()
            args.eval_print_io = prev_print
            avg_rouge = np.mean(acc_results) if acc_results else 0.0
            eval_rouge_every5.append({"round": r, "rouge": avg_rouge})
            print(f"--- Round {r} Average ROUGE: {avg_rouge} ---")
            args.eval_metric = prev_metric

        if args.log:
            with open(os.path.join(log_dir, "memory.json"), "w") as writer:
                json.dump(memory_record_dic, writer)
            with open(os.path.join(log_dir, "results.json"), "w") as writer:
                payload = {"eval_avg_acc": eval_avg_acc}
                if eval_acc_every5:
                    payload["eval_acc_every5"] = eval_acc_every5
                if eval_rouge_every5:
                    payload["eval_rouge_every5"] = eval_rouge_every5
                json.dump(payload, writer)

    if args.save:
        os.makedirs(log_dir, exist_ok=True)
        for client in client_list:
            state_dict_cpu = {k: v.cpu() for k, v in client.model.state_dict().items()}
            torch.save(
                state_dict_cpu,
                os.path.join(
                    log_dir,
                    f"model_state_dict_client{client.idx}_final_round{args.rounds}.bin",
                ),
            )

    if args.dataset in ["dolly", "sst2"]:
        # --- Final Evaluation on Each Client ---
        print("\n--- Final Evaluation on Each Client's Model ---")
        args.eval_metric = previous_metric
        setup_seed(args.seed)
        if args.dataset == "sst2":
            from utils_data.llm_dataset import LLMDataset, LLMDataCollator
            generation = args.eval_metric != "loss"
            eval_dataset = LLMDataset(
                args.dataset, tokenizer=tokenizer, generation=generation, split="validation"
            )
            data_collator = LLMDataCollator(tokenizer=tokenizer)
            eval_loader_final = DataLoader(
            eval_dataset, batch_size=args.eval_batch_size, collate_fn=data_collator
            )
        else:
            _, eval_loader_final, _ = get_loaders(args, only_eval=True)

        final_eval_results = {}

        # Update all clients with the final eval loader
        for client in client_list:
            client.eval_loader = eval_loader_final

        prev_print = args.eval_print_io
        args.eval_print_io = True
        for client in tqdm(client_list, desc="Final Evaluation for all clients"):
            # Eval directly on persistent model
            eval_result = client.eval(cur_round=args.rounds)
            torch.cuda.empty_cache()

            final_eval_results[f"client_{client.idx}"] = eval_result
            metric_name = "accuracy" if args.dataset == "sst2" else args.eval_metric
            print(f"Client {client.idx} final {metric_name}: {eval_result}")
        args.eval_print_io = prev_print

        if args.log:
            with open(os.path.join(log_dir, "final_eval_all_clients.json"), "w") as writer:
                json.dump(final_eval_results, writer)

        avg_final_eval = (
            np.mean(list(final_eval_results.values())) if final_eval_results else 0.0
        )
        metric_name = "accuracy" if args.dataset == "sst2" else args.eval_metric
        print(f"\nAverage final {metric_name} across all clients: {avg_final_eval}")
