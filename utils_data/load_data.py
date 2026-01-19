import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer
from utils_data.default_tokens import DefaultToken
from utils_data.partition_data import partition_idx_labeldir
from collections import Counter


def get_loaders(args, only_eval=False):
    """
    Return: list of train_loaders, eval_loader
    """
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = args.max_length
    special_tokens = dict()
    if tokenizer.pad_token is None:
        special_tokens["pad_token"] = DefaultToken.PAD_TOKEN.value
    if tokenizer.eos_token is None:
        special_tokens["eos_token"] = DefaultToken.EOS_TOKEN.value
    if tokenizer.bos_token is None:
        special_tokens["bos_token"] = DefaultToken.BOS_TOKEN.value
    if tokenizer.unk_token is None:
        special_tokens["unk_token"] = DefaultToken.UNK_TOKEN.value
    tokenizer.add_special_tokens(special_tokens)

    # Generation task
    if args.dataset == 'dolly':
        from utils_data.llm_dataset import LLMDataset, LLMDataCollator
        if args.eval_metric == 'loss':
            raw_datasets = LLMDataset(args.dataset, tokenizer=tokenizer, generation=False)
        else:
            raw_datasets = LLMDataset(args.dataset, tokenizer=tokenizer, generation=True)

        data_collator = LLMDataCollator(tokenizer=tokenizer)

        # only use a subset of raw dataset
        raw_datasets, _ = torch.utils.data.dataset.random_split(raw_datasets, [int(len(raw_datasets) * args.dataset_subsample), len(raw_datasets) - int(len(raw_datasets) * args.dataset_subsample)])
        y_all = np.array([item['categories'] for item in raw_datasets])
        index_eval = np.where(y_all == args.zerotask)[0]
        # delete the indices of eval samples from the all set
        index_train = np.delete(np.arange(len(y_all)), index_eval)
        raw_datasets = np.array(raw_datasets)
        train_set = raw_datasets[index_train]
        eval_set = raw_datasets[index_eval]
        y_train = np.array([item['categories'] for item in train_set])
        counter = Counter(y_train)
        noniid = args.iid
        if 'dir' in noniid:
            split_dic = partition_idx_labeldir(y_train, n_parties=args.num_clients, alpha=float(noniid[3:]), num_classes=len(counter))
            split_trainsets = []
            for _, sample_indices in split_dic.items():
                split_trainsets.append(Subset(train_set, indices=sample_indices))
        else:
            n_parts = [int(len(train_set) / args.num_clients) for _ in range(args.num_clients - 1)]
            n_parts.append(len(train_set) - sum(n_parts))
            split_trainsets = torch.utils.data.dataset.random_split(train_set, n_parts)

        list_train_loader = [
            DataLoader(
                subset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
            ) for subset in split_trainsets
        ]
        eval_loader = DataLoader(
            eval_set, batch_size=args.batch_size, collate_fn=data_collator
        )

    elif args.dataset == "gsm8k":
        from utils_data.llm_dataset import LLMDataset, LLMDataCollator
        generation = args.eval_metric != "loss"
        train_dataset = LLMDataset(
            args.dataset, tokenizer=tokenizer, generation=generation, split="train"
        )
        eval_dataset = LLMDataset(
            args.dataset, tokenizer=tokenizer, generation=generation, split="test"
        )

        data_collator = LLMDataCollator(tokenizer=tokenizer)

        if args.dataset_subsample < 1.0:
            train_dataset, _ = torch.utils.data.dataset.random_split(
                train_dataset,
                [
                    int(len(train_dataset) * args.dataset_subsample),
                    len(train_dataset) - int(len(train_dataset) * args.dataset_subsample),
                ],
            )

        y_train = np.array([item["categories"] for item in train_dataset])
        counter = Counter(y_train)
        noniid = args.iid
        if "dir" in noniid:
            split_dic = partition_idx_labeldir(
                y_train,
                n_parties=args.num_clients,
                alpha=float(noniid[3:]),
                num_classes=len(counter),
            )
            split_trainsets = []
            for _, sample_indices in split_dic.items():
                split_trainsets.append(Subset(train_dataset, indices=sample_indices))
        else:
            n_parts = [int(len(train_dataset) / args.num_clients) for _ in range(args.num_clients - 1)]
            n_parts.append(len(train_dataset) - sum(n_parts))
            split_trainsets = torch.utils.data.dataset.random_split(train_dataset, n_parts)

        list_train_loader = [
            DataLoader(
                subset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
            )
            for subset in split_trainsets
        ]
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, collate_fn=data_collator
        )

    elif args.dataset == "code_contests":
        from utils_data.llm_dataset import LLMDataset, LLMDataCollator
        generation = args.eval_metric != "loss"
        train_dataset = LLMDataset(
            args.dataset, tokenizer=tokenizer, generation=generation, split="train"
        )
        try:
            eval_dataset = LLMDataset(
                args.dataset, tokenizer=tokenizer, generation=generation, split="test"
            )
        except FileNotFoundError:
            eval_dataset = LLMDataset(
                args.dataset, tokenizer=tokenizer, generation=generation, split="valid"
            )

        data_collator = LLMDataCollator(tokenizer=tokenizer)

        if args.dataset_subsample < 1.0:
            train_dataset, _ = torch.utils.data.dataset.random_split(
                train_dataset,
                [
                    int(len(train_dataset) * args.dataset_subsample),
                    len(train_dataset) - int(len(train_dataset) * args.dataset_subsample),
                ],
            )

        y_train = np.array([item["categories"] for item in train_dataset])
        counter = Counter(y_train)
        noniid = args.iid
        if "dir" in noniid:
            split_dic = partition_idx_labeldir(
                y_train,
                n_parties=args.num_clients,
                alpha=float(noniid[3:]),
                num_classes=len(counter),
            )
            split_trainsets = []
            for _, sample_indices in split_dic.items():
                split_trainsets.append(Subset(train_dataset, indices=sample_indices))
        else:
            n_parts = [int(len(train_dataset) / args.num_clients) for _ in range(args.num_clients - 1)]
            n_parts.append(len(train_dataset) - sum(n_parts))
            split_trainsets = torch.utils.data.dataset.random_split(train_dataset, n_parts)

        list_train_loader = [
            DataLoader(
                subset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
            )
            for subset in split_trainsets
        ]
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, collate_fn=data_collator
        )

    elif args.dataset == "sst2":
        from utils_data.llm_dataset import LLMDataset, LLMDataCollator
        generation = args.eval_metric != "loss"
        train_dataset = LLMDataset(
            args.dataset, tokenizer=tokenizer, generation=generation, split="train"
        )
        try:
            eval_dataset = LLMDataset(
                args.dataset, tokenizer=tokenizer, generation=generation, split="validation"
            )
        except FileNotFoundError:
            eval_dataset = LLMDataset(
                args.dataset, tokenizer=tokenizer, generation=generation, split="test"
            )

        data_collator = LLMDataCollator(tokenizer=tokenizer)

        if args.dataset_subsample < 1.0:
            train_dataset, _ = torch.utils.data.dataset.random_split(
                train_dataset,
                [
                    int(len(train_dataset) * args.dataset_subsample),
                    len(train_dataset) - int(len(train_dataset) * args.dataset_subsample),
                ],
            )

        y_train = np.array([item["categories"] for item in train_dataset])
        counter = Counter(y_train)
        noniid = args.iid
        if "dir" in noniid:
            split_dic = partition_idx_labeldir(
                y_train,
                n_parties=args.num_clients,
                alpha=float(noniid[3:]),
                num_classes=len(counter),
            )
            split_trainsets = []
            for _, sample_indices in split_dic.items():
                split_trainsets.append(Subset(train_dataset, indices=sample_indices))
        else:
            n_parts = [int(len(train_dataset) / args.num_clients) for _ in range(args.num_clients - 1)]
            n_parts.append(len(train_dataset) - sum(n_parts))
            split_trainsets = torch.utils.data.dataset.random_split(train_dataset, n_parts)

        list_train_loader = [
            DataLoader(
                subset, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
            )
            for subset in split_trainsets
        ]
        eval_loader = DataLoader(
            eval_dataset, batch_size=args.batch_size, collate_fn=data_collator
        )

    elif args.dataset in ['instruct']:
        from utils_data.natural_instruction_loader import get_instruction_dataset
        list_train_loader, eval_loader = get_instruction_dataset(args, tokenizer, only_eval=only_eval)
    else:
        raise AttributeError(f'dataset {args.dataset} not implemented')
    return list_train_loader, eval_loader, tokenizer
