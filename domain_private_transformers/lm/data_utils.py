import itertools
from typing import Callable, List, Optional, Tuple, Union

import datasets
import torch
from torch.utils.data.dataset import Dataset
from transformers import PreTrainedTokenizer, default_data_collator

from .compiled_args import DataTrainingArguments


# Data preprocessing functions
def get_raw_dataset(
    data_args: DataTrainingArguments,
    file_path: Union[str, List[str]],
    max_examples: int,
) -> Dataset:
    """
    Returns the raw text dataset loaded from `file_path`.
    """
    if isinstance(file_path, list):
        assert all(fp.split(".")[-1] == "txt" for fp in file_path), (
            "All `file_path`s " "must be .txt"
        )
        dss: List[datasets.Dataset] = [
            datasets.load_dataset(
                "text",
                data_files={"split_name": fp},
                keep_linebreaks=not data_args.no_keep_linebreaks,
            )["split_name"]
            for fp in file_path
        ]
        # Select the first max_examples for each dataset
        dss = [ds.select(range(min(max_examples, len(ds)))) for ds in dss]
        return datasets.concatenate_datasets(dss)
    else:
        assert file_path.split(".")[-1] == "txt", "`file_path` must be .txt"
        ds = datasets.load_dataset(
            "text",
            data_files={"split_name": file_path},
            keep_linebreaks=not data_args.no_keep_linebreaks,
        )["split_name"]
        ds = ds.select(range(min(max_examples, len(ds))))
        return ds


def get_tokenized_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizer,
    data_args: DataTrainingArguments,
) -> Dataset:
    column_names = dataset.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_texts(examples):
        return tokenizer(examples[text_column_name])

    return dataset.map(
        tokenize_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )


def get_grouped_dataset(dataset: Dataset, data_args: DataTrainingArguments) -> Dataset:
    block_size = data_args.block_size

    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {
            k: list(itertools.chain(*examples[k])) for k in examples.keys()
        }
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model
        # supported it instead of this drop, you can
        # customize this part to your needs.
        total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Create LM dataset from tokenized by concatenating all texts from
    # our dataset and generating chunks of block_size.
    return dataset.map(
        group_texts,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=False,
        desc=f"Grouping texts in chunks of {block_size}",
    )


def get_dataset_with_path(
    data_args: DataTrainingArguments,
    tokenizer: PreTrainedTokenizer,
    file_path: Union[str, List[str]],
    max_examples: int,
    **_,
):
    if data_args.task_mode == "sgd":
        # Create args for load_dataset function, specifically the extension and
        # data_file
        dataset = get_raw_dataset(data_args, file_path, max_examples)
        dataset = get_tokenized_dataset(dataset, tokenizer, data_args)
        dataset = get_grouped_dataset(dataset, data_args)
    elif data_args.task_mode == "dogo":
        dataset = get_raw_dataset(data_args, file_path, max_examples)
        dataset = get_tokenized_dataset(dataset, tokenizer, data_args)
        dataset = get_grouped_dataset(dataset, data_args)
    else:
        raise ValueError(f"{data_args.task_mode} task data not supported")
    return dataset


def get_all_datasets(
    tokenizer: PreTrainedTokenizer,
    data_args: DataTrainingArguments,
    **_,
) -> Tuple[Dataset, Dataset, Optional[Dataset], Callable]:
    """
    Returns (train, val, (optional) test, data_collator_function)
    If `training_args.do_eval` is `False`, then the test data is `None`.
    """
    kwargs = dict(data_args=data_args, tokenizer=tokenizer)
    train_dataset = get_dataset_with_path(
        **kwargs,
        file_path=data_args.train_data_file,
        max_examples=data_args.max_train_examples,
    )
    val_dataset = get_dataset_with_path(
        **kwargs,
        file_path=data_args.val_data_file,
        max_examples=data_args.max_valid_examples,
    )
    if hasattr(data_args, "eval_data_file") and data_args.eval_data_file is not None:
        eval_dataset = get_dataset_with_path(
            **kwargs,
            file_path=data_args.eval_data_file,
            max_examples=data_args.max_eval_examples,
        )
    else:
        eval_dataset = None

    if data_args.task_mode == "sgd":
        # Use default_data_collator for SGD dataset
        data_collator = default_data_collator
    elif data_args.task_mode == "dogo":
        data_collator = default_data_collator
    else:
        raise ValueError(f"{data_args.task_mode} task data is not supported")

    return train_dataset, val_dataset, eval_dataset, data_collator


def get_prompt_dataset(
    file_path: Union[str, List[str]], tokenizer: PreTrainedTokenizer
) -> List[torch.Tensor]:
    """
    Returns the prompt dataset (for generations) in encoded form. The list's
    ith entry is a tensor of token_ids, shape (1, L) where L is the number of
    tokens in the line i of the file.
    """
    if isinstance(file_path, list):
        lines: List[str] = []
        for fp in file_path:
            with open(fp, "r") as f:
                lines.extend(f.readlines())
    else:
        with open(file_path, "r") as f:
            lines: List[str] = f.readlines()
    encoded_lines = [
        tokenizer.encode(line.strip(), add_special_tokens=False, return_tensors="pt")
        for line in lines
    ]
    return encoded_lines


def get_train_redacted_dataset(
    tokenizer: PreTrainedTokenizer,
    data_args: DataTrainingArguments,
    **_,
) -> Dataset:
    """
    Returns train redacted dataset.
    """
    train_redacted_dataset = get_dataset_with_path(
        data_args=data_args,
        tokenizer=tokenizer,
        file_path=data_args.train_redacted_data_file,
        max_examples=data_args.max_train_examples,
    )
    return train_redacted_dataset
