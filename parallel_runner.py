import torch
import torch.multiprocessing as mp
from copy import deepcopy
from transformers import AutoModelForCausalLM
import os
import traceback

def gpu_worker(gpu_id, task_queue, result_queue, args, candidate_seeds):
    """
    Worker process that runs on a specific GPU.
    It loads the base model once, then continuously processes Clients from the task_queue.
    """
    try:
        # 1. Setup Environment
        device = torch.device(f"cuda:{gpu_id}")
        
        print(f"[Worker {gpu_id}] Initializing on {device}...")

        # 2. Load Base Model (Frozen w0)
        # We load it from scratch to avoid pickling the massive model object from main process
        model_w0 = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map=None, # Manual placement
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        model_w0.to(device)
        model_w0.eval()
        print(f"[Worker {gpu_id}] Model loaded.")

        # 3. Task Loop
        while True:
            task_data = task_queue.get()
            if task_data is None:
                # Sentinel received, exit
                break
            
            # Unpack task: expects (task_type, client)
            if isinstance(task_data, tuple):
                task_type, client = task_data
            else:
                # Legacy support or error? Assume TRAIN if just client
                task_type = 'TRAIN'
                client = task_data

            try:
                # A. Update Client's Model Context (Common for TRAIN and EVAL)
                # IMPORTANT: Set the device for the client to this worker's GPU
                client.device = device
                client.args.device = gpu_id # Update args device too
                
                # Reconstruct model from seed pool using local model_w0
                client.model = deepcopy(model_w0)
                client.model.to(device)
                
                # Apply seed pool updates
                # We pass the model to update_model_by_seed_pool which sets client.model
                client.update_model_by_seed_pool(client.model)
                
                # B. Execute Task
                if task_type == 'TRAIN':
                    # Local Train
                    client.local_train(cur_round=client.current_round_index)
                    
                    # Result: Seed Pool
                    client.model = None
                    client.optimizer_state = {} 
                    result_queue.put((client.idx, client.local_seed_pool))
                    
                elif task_type == 'EVAL':
                    # Local Eval
                    eval_metric = client.eval(cur_round=client.current_round_index)
                    
                    # Result: Metric
                    client.model = None
                    client.optimizer_state = {}
                    result_queue.put((client.idx, eval_metric))
                
            except Exception as e:
                print(f"[Worker {gpu_id}] Error processing client {client.idx} ({task_type}): {e}")
                traceback.print_exc()
                result_queue.put((client.idx, None)) # Error signal

        print(f"[Worker {gpu_id}] Shutting down.")
        
    except Exception as e:
        print(f"[Worker {gpu_id}] Critical Failure: {e}")
        traceback.print_exc()
