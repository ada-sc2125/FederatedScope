import argparse
import math
import os
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from evaluations import rouge_score
from utils_data.default_tokens import DefaultToken
from utils_data.llm_dataset import PROMPT_DICT, load_jsonl


def load_dolly_data(data_dir: str) -> List[Dict[str, str]]:
    candidates = [
        os.path.join(data_dir, "databricks-dolly-15k.jsonl"),
        os.path.join(data_dir, "dolly", "databricks-dolly-15k.jsonl"),
        os.path.join(data_dir, "dolly-15k", "databricks-dolly-15k.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return load_jsonl(
                path,
                instruction="instruction",
                input="context",
                output="response",
                category="category",
            )
    raise FileNotFoundError(
        f"dolly jsonl not found, tried: {', '.join(candidates)}"
    )


def build_prompt(example: Dict[str, str]) -> str:
    if example.get("input"):
        return PROMPT_DICT["prompt_input"].format_map(example)
    return PROMPT_DICT["prompt_no_input"].format_map(example)


def prepare_tokenizer(model_name: str, max_length: int, padding_side: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer.model_max_length = max_length
    tokenizer.padding_side = padding_side
    special_tokens = {}
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
    if special_tokens:
        tokenizer.add_special_tokens(special_tokens)
    return tokenizer, special_tokens


def load_model(model_name: str, checkpoint: str, tokenizer, special_tokens, device):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=True)
    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()
    return model


def batch_iter(items, batch_size):
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def evaluate_loss(
    model, tokenizer, data, batch_size, max_length, device, eval_limit
):
    tokenizer.padding_side = "right"
    loss_total = 0.0
    num_eval = 0
    samples = data[:eval_limit] if eval_limit > 0 else data

    total_batches = math.ceil(len(samples) / batch_size) if samples else 0
    for batch in tqdm(
        batch_iter(samples, batch_size),
        total=total_batches,
        desc="Evaluating loss",
    ):
        prompts = [build_prompt(item) for item in batch]
        targets = [
            f"{(item.get('output') or '')}{tokenizer.eos_token}" for item in batch
        ]
        combined = [p + t for p, t in zip(prompts, targets)]

        enc_combined = tokenizer(
            combined,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc_prompt = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc_combined.input_ids.to(device)
        attention_mask = enc_combined.attention_mask.to(device)
        labels = input_ids.clone()
        prompt_lens = enc_prompt.attention_mask.sum(dim=1).tolist()
        for i, prompt_len in enumerate(prompt_lens):
            labels[i, : int(prompt_len)] = DefaultToken.IGNORE_INDEX.value

        with torch.inference_mode():
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss
        if torch.isnan(loss):
            continue
        loss_total += loss.item() * input_ids.shape[0]
        num_eval += input_ids.shape[0]

    return loss_total / max(num_eval, 1)


def evaluate_rouge(
    model,
    tokenizer,
    data,
    batch_size,
    max_length,
    max_new_tokens,
    device,
    eval_limit,
    eval_print_io,
    eval_print_n,
):
    tokenizer.padding_side = "left"
    rouge_total = 0.0
    num_eval = 0
    printed = 0
    samples = data[:eval_limit] if eval_limit > 0 else data

    total_batches = math.ceil(len(samples) / batch_size) if samples else 0
    for batch in tqdm(
        batch_iter(samples, batch_size),
        total=total_batches,
        desc="Evaluating ROUGE",
    ):
        prompts = [build_prompt(item) for item in batch]
        targets = [
            f"{(item.get('output') or '')}{tokenizer.eos_token}" for item in batch
        ]

        enc_prompt = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc_target = tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc_prompt.input_ids.to(device)
        attention_mask = enc_prompt.attention_mask.to(device)
        label_ids = enc_target.input_ids.to(device)

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=max_new_tokens,
                num_beams=1,
            )

        prompt_len = input_ids.shape[1]
        batch_size_actual = input_ids.shape[0]
        for i in range(batch_size_actual):
            if eval_print_io and printed < max(eval_print_n, 0):
                prompt_text = tokenizer.decode(
                    input_ids[i], skip_special_tokens=True
                )
                output_text = tokenizer.decode(
                    output_ids[i][prompt_len:], skip_special_tokens=True
                )
                print("INPUT:")
                print(prompt_text)
                print("OUTPUT:")
                print(output_text)
                printed += 1
            rouge_total += rouge_score(
                output_ids[i][prompt_len:], label_ids[i], tokenizer
            )
            num_eval += 1

    return rouge_total / max(num_eval, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--eval_metric", type=str, choices=["loss", "rouge"], default="rouge")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--eval_limit", type=int, default=0)
    parser.add_argument("--eval_print_io", action="store_true")
    parser.add_argument("--eval_print_n", type=int, default=2)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    tokenizer, special_tokens = prepare_tokenizer(
        args.model, args.max_length, padding_side="left"
    )
    model = load_model(args.model, args.checkpoint, tokenizer, special_tokens, device)

    data = load_dolly_data(args.data_dir)
    if args.checkpoint:
        log_dir = os.path.dirname(os.path.abspath(args.checkpoint))
    else:
        log_dir = os.getcwd()
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"dolly_eval_{args.eval_metric}.txt")
    if args.eval_metric == "loss":
        metric = evaluate_loss(
            model,
            tokenizer,
            data,
            args.batch_size,
            args.max_length,
            device,
            args.eval_limit,
        )
        result_line = f"Loss: {metric:.4f}"
    else:
        metric = evaluate_rouge(
            model,
            tokenizer,
            data,
            args.batch_size,
            args.max_length,
            args.max_new_tokens,
            device,
            args.eval_limit,
            args.eval_print_io,
            args.eval_print_n,
        )
        result_line = f"ROUGE-L: {metric:.4f}"
    print(result_line)
    with open(log_path, "a", encoding="utf-8") as writer:
        writer.write(result_line + "\n")


if __name__ == "__main__":
    main()
