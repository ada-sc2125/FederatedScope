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
        # Only visible device for this process is the assigned one to prevent OOM or context conflicts
        # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id) 
        # Note: setting env inside process might not work if cuda already init. 
        # Better to rely on .to(device) with specific index.
        
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
            task = task_queue.get()
            if task is None:
                # Sentinel received, exit
                break
            
            client = task
            # print(f"[Worker {gpu_id}] Processing Client {client.idx}")

            try:
                # A. Update Client's Model Context
                # We essentially perform: client.update_model_by_seed_pool(deepcopy(server.model_w0))
                # But we use the local worker's copy of model_w0
                
                # IMPORTANT: Set the device for the client to this worker's GPU
                client.device = device
                
                # Reconstruct model from seed pool
                # This logic mimics client.update_model_by_seed_pool but uses local model_w0
                # We manually inject the model to avoid re-initializing logic that might create new optimizers prematurely
                client.model = deepcopy(model_w0)
                client.model.to(device)
                
                # Re-initialize the optimizer/framework on this new model & device
                # We call the internal helper or just replicate the logic from client.update_model_by_seed_pool
                # but without passing the model as argument (since we set it above)
                
                # Trigger the seed pool replay
                # Note: We need to make sure update_model_by_seed_pool doesn't try to move things to 'args.device' 
                # if 'args.device' is different from our 'gpu_id'. 
                # We hack args.device locally for this client
                client.args.device = gpu_id 
                
                # We use a slightly modified call logic here to ensure it uses the WORKER'S existing model
                # instead of passing one in, or we pass the one we just created.
                client.update_model_by_seed_pool(client.model)
                
                # B. Local Train
                client.local_train(cur_round=client.current_round_index)
                
                # C. Return Result
                # We only need to return the updated seed pool and the client index
                # Returning the whole client might be heavy but ensures all state is preserved
                # To save bandwidth, we strip the model before sending back
                client.model = None
                client.optimizer_state = {} # Clear optimizer state to save pickling
                
                result_queue.put((client.idx, client.local_seed_pool))
                
            except Exception as e:
                print(f"[Worker {gpu_id}] Error processing client {client.idx}: {e}")
                traceback.print_exc()
                result_queue.put((client.idx, None)) # Error signal

        print(f"[Worker {gpu_id}] Shutting down.")
        
    except Exception as e:
        print(f"[Worker {gpu_id}] Critical Failure: {e}")
        traceback.print_exc()
