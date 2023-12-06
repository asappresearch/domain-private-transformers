import logging
import os
import random
from typing import Iterable, Iterator, Optional

import fire
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    Trainer,
    default_data_collator,
)
from transformers.training_args import TrainingArguments

from domain_private_transformers.constants import TASK2DOMAINS2TOKENS
from domain_private_transformers.utils import seed_everything

logger = logging.getLogger(__name__)


def get_domain_dataset(train_ds):
    # train_ds = train_ds.map(partial(company_dataset_preproc), num_proc=num_proc)
    # shuffle and stratified split
    x_trainval, x_test, y_trainval, y_test = train_test_split(
        train_ds["text"],
        train_ds["domain"],
        shuffle=True,
        stratify=train_ds["domain"],
        test_size=0.1,
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_trainval,
        y_trainval,
        shuffle=True,
        stratify=y_trainval,
        test_size=1. / 9,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def random_cutoff_utterances(
    utterance_list: Iterable[str], maximum_start_char=20, minimum_end_char=20
) -> Iterator[str]:
    for utt in utterance_list:
        low = np.random.randint(low=0, high=maximum_start_char + 1)
        high_min = max(low + 1, minimum_end_char)
        high = np.random.randint(low=high_min, high=max(len(utt), high_min + 1))
        yield utt[low:high]


def hf_train_classifier(
    model_name,
    raw_dataset,
    train_epochs=3,
    learning_rate=5e-5,
    pad_to_max_length=True,
    max_seq_length=512,
    do_train=True,
    do_eval=True,
    do_predict=True,
    max_train_samples=None,
    max_eval_samples=None,
    max_predict_samples=None,
    output_dir=None,
):
    # from https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
    label_list = raw_dataset["train"].unique("label")
    label_list.sort()
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        cache_dir=None,
        revision="main",
        use_auth_token=None,
    )
    sentence1_key, sentence2_key = "text", None

    if pad_to_max_length:
        padding = "max_length"
    else:
        padding = False
    label_to_id = {v: i for i, v in enumerate(label_list)}
    model.config.label2id = label_to_id
    model.config.id2label = {id: label for label, id in model.config.label2id.items()}
    max_seq_length = min(max_seq_length, tokenizer.model_max_length)

    def preprocess_function(examples):
        # Tokenize the texts
        args = (
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        )
        result = tokenizer(*args, padding=padding, max_length=max_seq_length, truncation=True)

        # Map labels to IDs (not necessary for GLUE tasks)
        if label_to_id is not None and "label" in examples:
            result["label"] = [(label_to_id[l] if l != -1 else -1) for l in examples["label"]]

        return result

    raw_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        load_from_cache_file=False,
        desc="Running tokenizer on dataset",
    )
    if do_train:
        train_dataset = raw_dataset["train"]
        if max_train_samples is not None:
            max_train_samples = min(len(train_dataset), max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if do_predict:
        if "test" not in raw_dataset and "test_matched" not in raw_dataset:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_dataset["test"]
        if max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))

    if do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    if do_eval:
        if "validation" not in raw_dataset:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_dataset["validation"]
        if max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    data_collator = default_data_collator if pad_to_max_length else None

    t_args = TrainingArguments(
        output_dir="tmp_trainer",
        num_train_epochs=train_epochs,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset if do_train else None,
        eval_dataset=eval_dataset if do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        args=t_args,
    )
    if do_train:
        # default training args loaded from somewhere???
        train_result = trainer.train(resume_from_checkpoint=None)
        training_metrics = train_result.metrics
        # max_train_samples = (
        #     max_train_samples if max_train_samples is not None else len(train_dataset)
        # )
        # metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        training_metrics["train_samples"] = max_train_samples

        logger.info("*** Evaluate ***")

        train_final_metrics = trainer.evaluate(eval_dataset=train_dataset, metric_key_prefix="train_final")
        eval_final_metrics = trainer.evaluate(eval_dataset=eval_dataset)
        train_final_metrics["train_final_num_samples"] = max_train_samples
        eval_final_metrics["eval_num_samples"] = max_eval_samples
        metrics = {**training_metrics, **train_final_metrics, **eval_final_metrics}

    if do_predict:
        logger.info("*** Predict ***")

        # Removing the `label` columns because it contains -1 and Trainer won't like that.
        y_test = predict_dataset["label"]
        predict_dataset = predict_dataset.remove_columns("label")
        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)
        predictions = [model.config.id2label[p] for p in predictions]

        if output_dir is not None:
            output_predict_file = os.path.join(output_dir, f"predict_results.txt")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(predictions):
                        item = label_list[item]
                        writer.write(f"{index}\t{item}\n")

    return trainer, tokenizer, metrics, predictions


# classifiers to predict the domain using validation examples from the dataset
def train_domain_classifier(
    num_per_domain: int = 4000,
    maximum_start_char: int = 10,
    minimum_end_char: int = 10,
    task: str = "dogo",
    domain_tag: str = "dogo",
    split: str = "valid",
    model_type: str = "ptl",
    train_epochs: int = 3,
    learning_rate: float = 5e-5,
    save_path: Optional[str] = None,
    random_seed: int = 42,
):

    assert task in ["dogo", "dogo60"]
    seed_everything(random_seed)
    all_domains = TASK2DOMAINS2TOKENS[domain_tag].keys()

    if split in ["train", "valid"]:
        domains_list = all_domains
        datasets_list = [os.path.join("data", task, split, f"{domain.lower()}.txt") for domain in all_domains]
        downsample_list = [1] * len(all_domains)
        print(f"class labels: {np.unique(all_domains)}")

        tmp_df = pd.concat(
            [
                pd.read_csv(dataset, delimiter="\r", header=None, names=["text"]).assign(domain=domain)
                for domain, dataset in zip(domains_list, datasets_list)
            ]
        )

        replace_strs = [f"{v} " for k, v in TASK2DOMAINS2TOKENS[task].items()]
        tmp_df["text"] = tmp_df["text"].str.replace("|".join(replace_strs), "", regex=True)

        # random spans of text column
        tmp_df["text"] = list(
            random_cutoff_utterances(
                tmp_df["text"], maximum_start_char=maximum_start_char, minimum_end_char=minimum_end_char
            )
        )

        tmp_data = Dataset.from_pandas(tmp_df)
        train_ds = tmp_data.shuffle(seed=42)

        x_train, x_val, x_test, y_train, y_val, y_test = get_domain_dataset(train_ds)

    elif split == "trainval":
        # just shuffle the existing train and val splits
        raise NotImplementedError

    if model_type.startswith("hf"):
        raw_dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(pd.DataFrame({"text": x_train, "label": y_train})),
                "validation": Dataset.from_pandas(pd.DataFrame({"text": x_val, "label": y_val})),
                "test": Dataset.from_pandas(pd.DataFrame({"text": x_test, "label": y_test})),
            },
        )

        model_name = "roberta-base"
        trainer, tokenizer, metrics, y_pred = hf_train_classifier(model_name, raw_dataset, train_epochs=train_epochs, learning_rate=learning_rate)
        training_acc = metrics["train_final_accuracy"]
        val_acc = metrics["eval_accuracy"]
        clf = trainer.model
        labels = list(trainer.model.config.label2id.keys())
        model_state = trainer.model.state_dict()
        model_kwargs = {}  # ?
        y_pred, y_test = np.array(y_pred), np.array(y_test)

    else:
        raise ValueError(f"unsupported model type: {model_type}")

    test_acc = (y_pred == y_test).mean()
    m = confusion_matrix(y_test, y_pred, normalize="true")
    acc_results_str = f"training acc: {training_acc:.4f}, val acc: {val_acc:.4f}, test acc: {test_acc:.4f}"
    print(acc_results_str)
    print(m)
    p = ConfusionMatrixDisplay(m, display_labels=labels).plot()
    plt.title(acc_results_str)

    if save_path is not None:
        output_dict = {
            "model": clf,
            "model_state": model_state,
            "model_kwargs": model_kwargs,
            "tokenizer": tokenizer,
            "labels": labels,
            "training_acc": training_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "test_confusion": m,
        }
        torch.save(output_dict, save_path)
        plt.savefig(save_path.replace(".pt", ".png"), bbox_inches="tight", dpi=300)
        print(f"saved to {save_path}")


if __name__ == "__main__":
    fire.Fire(train_domain_classifier)


# python domain_private_transformers/domain_classifier.py --model_type hf --save_path model/redaction/hf_test.pt
