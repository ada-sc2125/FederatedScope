import argparse
import os
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils_data.default_tokens import DefaultToken
from utils_data.llm_dataset import PROMPT_DICT


def load_sst2_split(split: str, data_dir: str) -> pd.DataFrame:
    candidates = [
        os.path.join(data_dir, "sst2", f"{split}-00000-of-00001.parquet"),
        os.path.join(data_dir, f"sst2_{split}.parquet"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return pd.read_parquet(path)
    raise FileNotFoundError(
        f"sst2 {split} split not found, tried: {', '.join(candidates)}"
    )


def build_prompt(sentence: str) -> str:
    instruction = f"Classify the sentiment of the sentence:\n{sentence}"
    return PROMPT_DICT["prompt_no_input"].format_map(
        {"instruction": instruction, "input": ""}
    )


def normalize_label(value: Optional[object]) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip().lower()
    if text.isdigit():
        return int(text)
    if "positive" in text:
        return 1
    if "negative" in text:
        return 0
    return None


def score_label_tokens(model, prompt_ids, label_ids):
    concat_ids = torch.cat([prompt_ids, label_ids], dim=1)
    outputs = model(input_ids=concat_ids)
    logits = outputs.logits
    prompt_len = prompt_ids.shape[1]
    label_len = label_ids.shape[1]
    start = max(prompt_len - 1, 0)
    end = start + label_len
    label_logits = logits[0, start:end, :]
    log_probs = torch.log_softmax(label_logits, dim=-1)
    token_idx = label_ids[0]
    token_positions = torch.arange(label_len, device=label_ids.device)
    return log_probs[token_positions, token_idx].sum().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "validation"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = args.max_length
    tokenizer.padding_side = "left"
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

    df = load_sst2_split(args.split, args.data_dir)
    sentences = df["sentence"].tolist() if "sentence" in df.columns else df["text"].tolist()
    labels = df["label"].tolist() if "label" in df.columns else [None] * len(sentences)

    label_token_ids = {
        0: torch.tensor(
            [tokenizer.encode("negative", add_special_tokens=False)],
            device=device,
        ),
        1: torch.tensor(
            [tokenizer.encode("positive", add_special_tokens=False)],
            device=device,
        ),
    }

    total = len(sentences)
    correct = 0
    for start in tqdm(range(0, total, args.batch_size), desc="Evaluating"):
        batch_sentences = sentences[start : start + args.batch_size]
        batch_labels = labels[start : start + args.batch_size]
        prompts = [build_prompt(s) for s in batch_sentences]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc.input_ids.to(device)
        attention_mask = enc.attention_mask.to(device)

        with torch.inference_mode():
            for i in range(len(prompts)):
                prompt_len = int(attention_mask[i].sum().item())
                prompt_ids = input_ids[i][:prompt_len].unsqueeze(0)
                score_neg = score_label_tokens(model, prompt_ids, label_token_ids[0])
                score_pos = score_label_tokens(model, prompt_ids, label_token_ids[1])
                pred_label = 1 if score_pos >= score_neg else 0
                gold_label = normalize_label(batch_labels[i])

                print("INPUT:")
                print(prompts[i])
                print("OUTPUT:")
                print("positive" if pred_label == 1 else "negative")

                if gold_label is not None:
                    correct += int(pred_label == gold_label)

    if total > 0 and any(label is not None for label in labels):
        acc = correct / total
        print(f"Accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
