import argparse
import os
import time
import random
import numpy as np
import torch
from server import Server
from client import Client
from utils_data.load_data import get_loaders
from topologies import create_ring_topology, create_full_topology, create_star_topology, create_grid_topology

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # Federation
    parser.add_argument('--num_clients', type=int, default=200, help='N in our paper')
    parser.add_argument('-m', type=float, default=0.05, help='ratio of activate clients in each round (only for centralized FL)')
    parser.add_argument('--rounds', type=int, default=40, help='the total number of rounds')
    parser.add_argument('--local_step', type=int, default=200, help=r'$ tau in our paper')
    parser.add_argument('--batch_or_epoch', type=str, default='batch', choices=['epoch', 'batch'])
    parser.add_argument('--equal_weight', default=False, action='store_true', help='if `true`, the weights among clients for aggregation are the same')
    
    # Gossip-related arguments
    parser.add_argument('--topology', type=str, default='random', choices=['ring', 'full', 'star', 'grid', 'random'], help='Network topology for gossip communication')
    parser.add_argument('--num_neighbors', type=int, default=2, help='Number of neighbors for random gossip aggregation')


    # Data
    ## Arguments related to data on both datasets
    parser.add_argument('--dataset', type=str, default='instruct', choices=['instruct', 'dolly'])
    parser.add_argument('--batch_size', type=int, default=1, help='batch size > 1 may cause error during running')
    parser.add_argument('--max_length', type=int, default=1024, help='the max number of tokens of a data instance')
    parser.add_argument('--use_prompts', default=True, help='if `true`, the prompt template from alpaca is adopted')
    
    ## Arguments related to data only for Dolly-15K
    parser.add_argument('--iid', type=str, default='dir0.5', help=r'`dir{alpha}` means that \alpha in Dirichlet distribution, `0` means IID split')
    parser.add_argument('--zerotask', default=7, type=int, help='the index of the task for evaluation in dolly-15K')
    parser.add_argument('--dataset_subsample', type=float, default=1.0, help='used for sampling a subset from the original dataset, only effective for dolly-15K')

    # Model
    parser.add_argument('--model', type=str, default='datajuicer/LLaMA-1B-dj-refine-150B')

    # Training
    parser.add_argument('--lr', type=float, default=0.001, help=r'learning rate \eta')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay in MeZO')
    parser.add_argument('--grad_clip', type=float, default=-100.0, help='clip the over large loss value, if < 0, disable this feature')

    # Training args only for `FedKSeed`
    parser.add_argument('-K', type=int, default=4096, help='Number of candidate seeds for MeZO')
    parser.add_argument('--zo_eps', type=float, default=0.0005, help=r'eps in MeZO')

    # MeZO Optimizer Arguments
    parser.add_argument('--mezo_optimizer', type=str, default='sgd', choices=['sgd', 'adam', 'muon'], help='Which MeZO optimizer to use.')
    parser.add_argument('--adam_beta1', type=float, default=0.9, help='beta1 for MeZO-Adam')
    parser.add_argument('--adam_beta2', type=float, default=0.999, help='beta2 for MeZO-Adam')
    parser.add_argument('--adam_eps', type=float, default=1e-8, help='epsilon for MeZO-Adam')
    parser.add_argument('--mu', type=float, default=0.9, help='mu for MeZO-Muon')
    parser.add_argument('--ns_steps', type=int, default=5, help='Number of Newton-Schulz steps for Muon')

    # Training args only for `FedKSeed-Pro`
    parser.add_argument('--bias_sampling', default=False, action='store_true', help='if `true`, the probabilities of candidate seeds to be sampled are not identical, i.e., FedKSeed-Pro')
    parser.add_argument('--bias_loss_clip', default=1000.0, type=float, help='scalar gradient whose abstract values exceeds this value will be cliped')
    parser.add_argument('--grad_initial', default=0.0, type=float, help='initial value of scalar gradient history corresponding to each candidate seed')

    # Environment
    parser.add_argument('--device', type=int, default=0, help='index of the targeted cuda device')
    parser.add_argument('--log', default=False, action='store_true', help='if `true`, running logs will be recorded in files')
    parser.add_argument('--log_root', default='logs', help='root path of log files')
    parser.add_argument('--seed', default=42, type=int, help='global seed, for reproducibility')
    
    # Evaluation
    parser.add_argument('--eval_metric', default='rouge', type=str, choices=['rouge', 'loss'], help='metric to evaluate global model in the last round')
    
    # Checkpoints
    parser.add_argument('--save', default=False, action='store_true', help='if `true`, the checkpoint of tuned models will be stored')

    time_stamp = str(time.time())
    args = parser.parse_args()

    eval_avg_acc = []
    memory_record_dic = {}
    
    previous_metric = args.eval_metric
    args.eval_metric = 'loss'
    # set CUDA visibility to targeted cuda device, to avoid the several hundred MB memory consumption of device 0
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    setup_seed(args.seed)
    list_train_loader, eval_loader, _ = get_loaders(args)
    
    if args.dataset == 'instruct':
        args.iid = 'meta'
    log_dir = time_stamp

    if args.log_root != '':
        log_dir = os.path.join(args.log_root, log_dir)
    if args.log:
        os.makedirs(log_dir)
    config = yaml.dump(args, None)
    config = '\n'.join(config.split('\n')[1:])
    print('Configs: ')
    print(config)
    print('=====================')
    if args.log:
        with open(os.path.join(log_dir, 'config.yaml'), 'w') as writer:
            writer.write(config)

    # since only CUDA device is available, load all models on device 0
    args.device = 0
    device = torch.device(f'cuda:{args.device}')

    client_list = []
    
    # sample `K` candidate seeds
    candidate_seeds = np.random.randint(1, 100000000000, args.K)

    # Server is now mainly for evaluation
    server = Server(args, eval_loader=eval_loader, candidate_seeds=candidate_seeds, log_dir=log_dir)
    for idx in range(args.num_clients):
        client_list.append(Client(idx, args, candidate_seeds, list_train_loader[idx]))
    
    # --- Create network topology ---
    print(f"Creating '{args.topology}' topology...")
    if args.topology == 'ring':
        neighborhoods = create_ring_topology(client_list)
    elif args.topology == 'full':
        neighborhoods = create_full_topology(client_list)
    elif args.topology == 'star':
        neighborhoods = create_star_topology(client_list)
    elif args.topology == 'grid':
        neighborhoods = create_grid_topology(client_list)
    else: # 'random'
        neighborhoods = None # Will be determined dynamically in each round
    print("Topology created.")

    # Initialize all client models with the server's initial model (on CPU)
    for client in client_list:
        client.model = deepcopy(server.model)

    # Initial evaluation
    server.model.to(device)
    eval_result = server.eval(cur_round=0, eval_avg_acc=eval_avg_acc)
    server.model.to('cpu')
    eval_avg_acc.append(eval_result)

    if args.log:
        with open(os.path.join(log_dir, 'memory.json'), 'w') as writer:
            json.dump(memory_record_dic, writer)
        with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
            json.dump({
                'eval_avg_acc': eval_avg_acc
            }, writer)

    # --- Gossip Training Loop ---
    for r in range(1, args.rounds + 1):
        print(f"--- Round {r}/{args.rounds} ---")

        # --- Local Training Phase ---
        # In this phase, each client's model is moved to the GPU for training and then left there.
        print("--- Kicking off local training for all clients ---")
        for client in client_list:
            client.local_train(
                pulled_model=client.model, # Starts on CPU, moved to GPU inside local_train
                cur_round=r
            )
        print("--- Local training finished for all clients ---")
        
        # --- Prepare for Aggregation: Move model parameters from GPU to CPU ---
        models_state_dicts_cpu = {
            client.idx: {key: value.cpu() for key, value in client.model.state_dict().items()}
            for client in client_list
        }

        # --- Offload all models from GPU to free up memory ---
        print("--- Offloading all models from GPU to CPU ---")
        for client in client_list:
            client.model.to('cpu')
        torch.cuda.empty_cache()

        # --- Aggregation Phase (Optimized) ---
        print("--- Kicking off aggregation for all clients (Optimized) ---")
        for client in client_list:
            # a. Move the current client's model to GPU for aggregation
            client.model.to(device)

            if neighborhoods: # Static topology
                neighbors = neighborhoods[client.idx]
            else: # Dynamic random topology
                other_clients = [c for c in client_list if c.idx != client.idx]
                num_neighbors = min(args.num_neighbors, len(other_clients))
                neighbors = random.sample(other_clients, num_neighbors)
            
            neighbor_state_dicts = [models_state_dicts_cpu[n.idx] for n in neighbors]

            # b. Perform aggregation (only this client's model is on GPU)
            client.local_gossip_aggregate(
                neighbor_state_dicts=neighbor_state_dicts
            )

            # c. Immediately move the updated model back to CPU
            client.model.to('cpu')
        print("--- Aggregation finished for all clients ---")

        # --- Round Evaluation ---
        # For evaluation, we pick a client's model, move it to the GPU, evaluate, then move it back.
        eval_model = client_list[0].model
        eval_model.to(device)
        server.model = eval_model
        eval_result = server.eval(cur_round=r, eval_avg_acc=eval_avg_acc)
        eval_model.to('cpu')
        eval_avg_acc.append(eval_result)
        
        if args.log:
            with open(os.path.join(log_dir, 'memory.json'), 'w') as writer:
                json.dump(memory_record_dic, writer)
            with open(os.path.join(log_dir, 'results.json'), 'w') as writer:
                json.dump({
                    'eval_avg_acc': eval_avg_acc
                }, writer)


    # --- Final Evaluation for Each Client ---
    print("\n--- Final Evaluation on Each Client's Model ---")
    args.eval_metric = previous_metric
    setup_seed(args.seed)
    _, eval_loader_final, _ = get_loaders(args, only_eval=True)
    server.eval_loader = eval_loader_final
    
    final_eval_results = {}
    for client in client_list:
        print(f"\nEvaluating Client {client.idx}...")
        # Move the client's final model to GPU for evaluation
        client.model.to(device)
        server.model = client.model
        eval_result = server.eval(cur_round=args.rounds, eval_avg_acc=eval_avg_acc)
        # Move it back to CPU after evaluation
        client.model.to('cpu')
        
        final_eval_results[f'client_{client.idx}'] = eval_result
        print(f'Client {client.idx} final {args.eval_metric}: {eval_result}')

    if args.log:
        with open(os.path.join(log_dir, 'final_eval_all_clients.json'), 'w') as writer:
            json.dump(final_eval_results, writer)
    
    avg_final_eval = np.mean(list(final_eval_results.values()))
    print(f'\nAverage final {args.eval_metric} across all clients: {avg_final_eval}')