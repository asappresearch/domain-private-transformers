import os

import fire
from typing import Optional
from domain_private_transformers.redaction import HFClassifierRedaction, KeywordRedaction, NBClassifierRedaction, HFClassifierMergeRedaction, KeywordRedaction


def run_redaction(
    clf_path: str = "model/redaction/as7_hf_epochs10.pt",
    clf_model: str = "hf",
    clf_threshold: Optional[float] = None,
    clf_ngram_range: Optional[int] = None,
    clf_merge_distance: Optional[str] = None,
    output_folder_name: str = "redacted-clf0.95-sgd3dom-hf",
) -> None:
    # ClassifierRedaction
    if clf_model == "nb":
        if clf_threshold is None:
            clf_threshold = 0.9
        redaction_class = NBClassifierRedaction
        redaction_kwargs = {"num_workers": 4, "batch_size": 32}

    elif clf_model in "key":
        redaction_class = KeywordRedaction
        redaction_kwargs = {"ignore_case": True, "num_workers": 4, "batch_size": 32}

    elif clf_model == "hf":
        if clf_threshold is None:
            clf_threshold = 0.95
        if clf_ngram_range is None:
            clf_ngram_range = (1, 2)
        redaction_class = HFClassifierRedaction
        redaction_kwargs = {"ngram_range": clf_ngram_range, "num_workers": 1, "batch_size": 4}

    elif clf_model == "hf-merged":
        if clf_threshold is None:
            clf_threshold = 0.95
        if clf_merge_distance is None:
            clf_merge_distance = "JS"
        redaction_class = HFClassifierMergeRedaction
        redaction_kwargs = {"merge_distance": clf_merge_distance, "num_workers": 1, "batch_size": 4}

    redactor = redaction_class.from_pretrained_path(
        clf_path, threshold=clf_threshold, **redaction_kwargs
    )
    for filename in [
        "../multi-domain-goal-oriented-dialogues-dataset/data/unannotated/airline.tsv",
        "../multi-domain-goal-oriented-dialogues-dataset/data/unannotated/media.tsv",
        "../multi-domain-goal-oriented-dialogues-dataset/data/unannotated/insurance.tsv",
    ]:
        print(
            f"classifer {clf_path}, threshold {clf_threshold} redaction: {os.path.basename(filename)}\n"
        )
        output_filename = os.path.join(
            f"../multi-domain-goal-oriented-dialogues-dataset/data/{output_folder_name}/",
            os.path.basename(filename),
        )
        redactor.redact_dataset(filename, output_filename)


if __name__ == "__main__":
    fire.Fire(run_redaction)
