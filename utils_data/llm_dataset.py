"""
The implementation of loading dataset `Dolly-15K` is adapted from [FederatedScope](https://github.com/alibaba/FederatedScope/tree/llm)
"""
import copy

from enum import Enum
from torch.utils.data import Dataset
import json
import os
import math
import re
import glob
from dataclasses import dataclass
import torch
import transformers
import pandas as pd


def load_jsonl(file_path,
               instruction='instruction',
               input='input',
               output='output',
               category='category'):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    list_data_dict = []
    with open(file_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            new_item = dict(
                instruction=item[instruction] if instruction in item else None,
                input=item[input] if input in item else None,
                output=item[output] if output in item else None,
                category=item[category] if category in item else None)
            item = new_item
            list_data_dict.append(item)
    return list_data_dict


def _extract_gsm8k_answer_value(answer_text):
    match = re.search(r"####\s*([-+]?\d+(?:\.\d+)?)", answer_text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _gsm8k_answer_category(answer_text):
    value = _extract_gsm8k_answer_value(answer_text)
    if value is None:
        return 0
    abs_value = abs(value)
    if abs_value < 1:
        digits = 1
    else:
        digits = int(math.log10(abs_value)) + 1
    return min(digits, 6)


_GSM8K_QUESTION_CATEGORIES = [
    ("geometry", ["triangle", "rectangle", "square", "circle", "area", "perimeter", "volume", "radius", "diameter"]),
    ("time", ["hour", "minute", "second", "day", "week", "month", "year", "time"]),
    ("money", ["dollar", "$", "cents", "cost", "price", "pay", "paid", "spent", "buy", "sell"]),
    ("ratio_percent", ["percent", "%", "ratio", "proportion", "rate", "per", "of"]),
    ("counting", ["each", "total", "how many", "count", "left", "remaining", "remainder"]),
]


def _gsm8k_question_category(question_text):
    text = question_text.lower()
    for idx, (_, keywords) in enumerate(_GSM8K_QUESTION_CATEGORIES):
        for kw in keywords:
            if kw in text:
                return idx
    return len(_GSM8K_QUESTION_CATEGORIES)


def load_gsm8k_jsonl(file_path, use_question_category=True):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            item = json.loads(line)
            question = item.get("question", "")
            answer = item.get("answer", "")
            list_data_dict.append(
                {
                    "instruction": question,
                    "input": "",
                    "output": answer,
                    "category": _gsm8k_question_category(question)
                    if use_question_category
                    else _gsm8k_answer_category(answer),
                }
            )
    return list_data_dict


def load_gsm8k_parquet(file_path, use_question_category=True):
    df = pd.read_parquet(file_path)
    list_data_dict = []
    for _, row in df.iterrows():
        question = row.get("question", "")
        answer = row.get("answer", "")
        list_data_dict.append(
            {
                "instruction": question,
                "input": "",
                "output": answer,
                "category": _gsm8k_question_category(question)
                if use_question_category
                else _gsm8k_answer_category(answer),
            }
        )
    return list_data_dict


def _normalize_code_solution(solution):
    if solution is None:
        return ""
    if isinstance(solution, str):
        return solution
    if isinstance(solution, list) and solution:
        if isinstance(solution[0], str):
            return solution[0]
        if isinstance(solution[0], dict):
            return solution[0].get("solution", "") or solution[0].get("code", "")
    if isinstance(solution, dict):
        return solution.get("solution", "") or solution.get("code", "")
    return ""


def _code_contests_category(row):
    if "difficulty" in row and row["difficulty"] is not None:
        return str(row["difficulty"])
    if "source" in row and row["source"] is not None:
        return str(row["source"])
    desc = row.get("description", "")
    length = len(desc) if isinstance(desc, str) else 0
    if length < 400:
        return "short"
    if length < 1200:
        return "medium"
    return "long"


def load_code_contests_parquet(file_paths):
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    dfs = [pd.read_parquet(path) for path in file_paths]
    df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]
    list_data_dict = []
    for _, row in df.iterrows():
        description = row.get("description", "")
        solution = _normalize_code_solution(row.get("solutions", row.get("solution", "")))
        list_data_dict.append(
            {
                "instruction": description,
                "input": "",
                "output": solution,
                "category": _code_contests_category(row),
            }
        )
    return list_data_dict


def load_sst2_parquet(file_path):
    df = pd.read_parquet(file_path)
    label_to_text = {0: "negative", 1: "positive"}
    list_data_dict = []
    for _, row in df.iterrows():
        sentence = row.get("sentence", row.get("text", "")) or ""
        label = row.get("label")
        output = label_to_text.get(label, str(label) if label is not None else "")
        list_data_dict.append(
            {
                "instruction": f"Classify the sentiment of the sentence:\n{sentence}",
                "input": "",
                "output": output,
                "category": label if label is not None else output,
            }
        )
    return list_data_dict


class DefaultToken(Enum):
    PAD_TOKEN = "[PAD]"
    EOS_TOKEN = "</s>"
    BOS_TOKEN = "<s>"
    UNK_TOKEN = "<unk>"
    IGNORE_INDEX = -100


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, "
        "paired with an input that provides further context. "
        "Write a response for the task request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:"
        "\n{input}\n\n### Response:"),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response for the task request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"),
}


class LLMDataset(Dataset):
    def __init__(self,
                 dataset,
                 tokenizer,
                 prompt_input=PROMPT_DICT["prompt_input"],
                 prompt_no_input=PROMPT_DICT["prompt_no_input"],
                 generation=False,
                 split=None):
        super(LLMDataset, self).__init__()
        if dataset == 'dolly':
            json_name = 'databricks-dolly-15k.jsonl'
            list_data_dict =  load_jsonl(os.path.join('data', json_name), 
                                        instruction='instruction',
                                        input='context',
                                        output='response',
                                        category='category')
        elif dataset == "gsm8k":
            split_name = split or "train"
            candidates = [
                os.path.join("data", f"{split_name}-00000-of-00001.parquet"),
                os.path.join("data", "gsm8k", f"{split_name}-00000-of-00001.parquet"),
                os.path.join("data", "gsm8k", f"{split_name}.jsonl"),
                os.path.join("data", f"gsm8k_{split_name}.jsonl"),
                os.path.join("data", f"gsm8k.{split_name}.jsonl"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    gsm8k_path = path
                    break
            else:
                raise FileNotFoundError(
                    f"gsm8k {split_name} split not found, tried: {', '.join(candidates)}"
                )
            if gsm8k_path.endswith(".parquet"):
                list_data_dict = load_gsm8k_parquet(gsm8k_path, use_question_category=True)
            else:
                list_data_dict = load_gsm8k_jsonl(gsm8k_path, use_question_category=True)
        elif dataset == "code_contests":
            split_name = split or "train"
            if split_name == "train":
                train_glob = os.path.join("data", "code_contests", "train*.parquet")
                train_paths = sorted(glob.glob(train_glob))
                if train_paths:
                    list_data_dict = load_code_contests_parquet(train_paths)
                else:
                    raise FileNotFoundError(
                        f"code_contests train split not found, tried: {train_glob}"
                    )
            else:
                candidates = [
                    os.path.join("data", "code_contests", f"{split_name}.parquet"),
                    os.path.join(
                        "data", "code_contests", f"{split_name}-00000-of-00001.parquet"
                    ),
                    os.path.join("data", f"code_contests_{split_name}.parquet"),
                ]
                for path in candidates:
                    if os.path.exists(path):
                        cc_path = path
                        break
                else:
                    raise FileNotFoundError(
                        f"code_contests {split_name} split not found, tried: {', '.join(candidates)}"
                    )
                list_data_dict = load_code_contests_parquet(cc_path)
        elif dataset == "sst2":
            split_name = split or "train"
            candidates = [
                os.path.join("data", "sst2", f"{split_name}-00000-of-00001.parquet"),
                os.path.join("data", f"sst2_{split_name}.parquet"),
                os.path.join(os.sep, "data", "sst2", f"{split_name}-00000-of-00001.parquet"),
            ]
            for path in candidates:
                if os.path.exists(path):
                    sst2_path = path
                    break
            else:
                raise FileNotFoundError(
                    f"sst2 {split_name} split not found, tried: {', '.join(candidates)}"
                )
            list_data_dict = load_sst2_parquet(sst2_path)
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != ""
            else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [
            f"{example['output']}{tokenizer.eos_token}"
            for example in list_data_dict
        ]

        data_dict = self.preprocess(sources, targets, tokenizer, generation=generation)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

        categories = [
            example['category'] if 'category' in example else None
            for example in list_data_dict
        ]
        df = pd.DataFrame(categories, columns=["category"])
        self.categories = list(pd.Categorical(df["category"]).codes)

    def _tokenize_fn(self, strings, tokenizer):
        tokenized_list = [
            tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                max_length=tokenizer.model_max_length,
                truncation=True,
            ) for text in strings
        ]
        input_ids = labels = [
            tokenized.input_ids[0] for tokenized in tokenized_list
        ]
        input_ids_lens = labels_lens = [
            tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
            for tokenized in tokenized_list
        ]
        return dict(
            input_ids=input_ids,
            labels=labels,
            input_ids_lens=input_ids_lens,
            labels_lens=labels_lens,
        )

    def preprocess(self, sources, targets, tokenizer, generation):
        if generation:
            sources_tokenized, labels_tokenized = [
                self._tokenize_fn(strings, tokenizer)
                for strings in (sources, targets)
            ]
            input_ids = self._tokenize_fn(sources, tokenizer)["input_ids"]
            labels = self._tokenize_fn(targets, tokenizer)["input_ids"]
        else:
            examples = [s + t for s, t in zip(sources, targets)]
            examples_tokenized, sources_tokenized = [
                self._tokenize_fn(strings, tokenizer)
                for strings in (examples, sources)
            ]
            input_ids = examples_tokenized["input_ids"]
            labels = copy.deepcopy(input_ids)
            for label, source_len in zip(labels,
                                        sources_tokenized["input_ids_lens"]):
                label[:source_len] = DefaultToken.IGNORE_INDEX.value
        return dict(input_ids=input_ids, labels=labels)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i],
                    categories=self.categories[i])


@dataclass
class LLMDataCollator(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=DefaultToken.IGNORE_INDEX.value)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
