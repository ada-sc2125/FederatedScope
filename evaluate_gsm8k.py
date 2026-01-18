import argparse
import os
import re
from typing import List, Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_data.default_tokens import DefaultToken
from utils_data.llm_dataset import PROMPT_DICT


def load_gsm8k_split(split: str, data_dir: str) -> pd.DataFrame:
    candidates = [
        os.path.join(data_dir, f"{split}-00000-of-00001.parquet"),
        os.path.join(data_dir, "gsm8k", f"{split}-00000-of-00001.parquet"),
        os.path.join(data_dir, "gsm8k", f"{split}.jsonl"),
        os.path.join(data_dir, f"gsm8k_{split}.jsonl"),
        os.path.join(data_dir, f"gsm8k.{split}.jsonl"),
    ]
    for path in candidates:
        if os.path.exists(path):
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_json(path, lines=True)
    raise FileNotFoundError(
        f"gsm8k {split} split not found, tried: {', '.join(candidates)}"
    )


def build_prompt(question: str) -> str:
    return PROMPT_DICT["prompt_no_input"].format_map(
        {"instruction": question, "input": ""}
    )


def _normalize_number(text: str) -> Optional[str]:
    if text is None:
        return None
    text = text.replace(",", "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def extract_gold_answer(answer_text: str) -> Optional[str]:
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer_text)
    if not match:
        return None
    return _normalize_number(match.group(1))


def extract_pred_answer(output_text: str) -> Optional[str]:
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", output_text.replace(",", ""))
    if not numbers:
        return None
    return _normalize_number(numbers[-1])


def compute_pass_at_k(correct_flags: List[bool], ks: List[int]) -> dict:
    results = {}
    for k in ks:
        results[k] = any(correct_flags[:k])
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--pass_k", type=str, default="5,10")
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    ks = [int(x) for x in args.pass_k.split(",") if x.strip()]
    if not ks:
        raise ValueError("pass_k must contain at least one integer, e.g. 5,10")
    max_k = max(ks)

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.padding_side = "left"
    tokenizer.model_max_length = args.max_length
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cpu",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    state = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(state, strict=True)
    if special_tokens:
        model.resize_token_embeddings(len(tokenizer))
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    df = load_gsm8k_split(args.split, args.data_dir)
    questions = df["question"].tolist()
    answers = df["answer"].tolist()

    em_correct = 0
    pass_at_k_counts = {k: 0 for k in ks}

    total = len(questions)
    for start in tqdm(range(0, total, args.batch_size), desc="Evaluating"):
        batch_questions = questions[start : start + args.batch_size]
        batch_answers = answers[start : start + args.batch_size]
        prompts = [build_prompt(q) for q in batch_questions]
        enc = tokenizer(prompts, return_tensors="pt", padding=True)
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)
        gold_answers = [extract_gold_answer(a) for a in batch_answers]

        input_ids_exp = input_ids.repeat_interleave(max_k, dim=0)
        attention_mask_exp = attention_mask.repeat_interleave(max_k, dim=0)

        with torch.inference_mode():
            outputs = model.generate(
                input_ids=input_ids_exp,
                attention_mask=attention_mask_exp,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        prompt_len = input_ids.shape[1]
        decoded = tokenizer.batch_decode(
            outputs[:, prompt_len:], skip_special_tokens=True
        )

        for i, gold in enumerate(gold_answers):
            offset = i * max_k
            preds = decoded[offset : offset + max_k]
            print("INPUT:")
            print(prompts[i])
            for j, pred_text in enumerate(preds, start=1):
                print(f"OUTPUT[{j}]:")
                print(pred_text)
            correct_flags = []
            for pred_text in preds:
                pred = extract_pred_answer(pred_text)
                correct_flags.append(
                    pred is not None and gold is not None and pred == gold
                )
            if correct_flags[0]:
                em_correct += 1
            pass_flags = compute_pass_at_k(correct_flags, ks)
            for k in ks:
                if pass_flags[k]:
                    pass_at_k_counts[k] += 1

    em = em_correct / total if total else 0.0
    pass_at_k = {k: (pass_at_k_counts[k] / total if total else 0.0) for k in ks}

    print(f"Exact Match Accuracy: {em:.4f}")
    for k in ks:
        print(f"Pass@{k}: {pass_at_k[k]:.4f}")


if __name__ == "__main__":
    main()
