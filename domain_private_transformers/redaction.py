import json
import os
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from nltk import ngrams
from scipy.spatial.distance import jensenshannon
from torch.nn.functional import softmax

from domain_private_transformers.constants import REDACTED, PUNCTUATION_CHARS


class Redaction(ABC):
    def __init__(self, num_workers=8, batch_size=None, ngram_range=None):
        self.redaction_token = REDACTED
        self.num_workers = num_workers
        self.batch_size = batch_size
        if batch_size is not None and batch_size > 1:
            self.batched = True
        else:
            self.batched = False
            self.batch_size = None
        if isinstance(ngram_range, str):
            tmp_ngram = ngram_range.strip("()")
            ngram_range = tuple(int(n) for n in tmp_ngram.split(","))
        self.ngram_range = ngram_range

    @abstractmethod
    def _is_private(self, x: str) -> bool:
        pass

    @abstractmethod
    def redact_example(self, example) -> str:
        # calls self._is_private
        pass

    def is_redacted_example(self, example: str) -> bool:
        out = self.redact_example({"utterance": [example]})[0]
        return out.find(self.redaction_token) != -1

    def is_redacted_examples(self, examples: Sequence[str]) -> Sequence[bool]:
        out_list = self.redact_example({"utterance": examples})
        return [out.find(self.redaction_token) != -1 for out in out_list]

    def _example_to_utterances(self, example):
        out = example["utterance"]
        if isinstance(out, str):
            return [out]
        else:
            return out

    def _redact_example_map_fn(self, example, output_col: str = "utterance"):
        return {output_col: self.redact_example(example)}

    def redact_dataset(self, input_path, output_path):
        data = pd.read_csv(input_path, sep=",")
        # (empty utterances) replace NaN data with empty string
        data = data.fillna("")
        output_parent_dir = os.path.dirname(output_path)
        os.makedirs(output_parent_dir, exist_ok=True)
        data["domain"] = os.path.basename(input_path).split(".")[0]
        dataset = Dataset.from_pandas(data)
        data = dataset.map(
            self._redact_example_map_fn,
            batched=self.batched,
            batch_size=self.batch_size,
            num_proc=self.num_workers,
            load_from_cache_file=False,
            desc=f"Running redaction on {input_path}",
        ).to_pandas()
        data.drop(columns=["domain"]).to_csv(output_path, index=False, sep=",")


class KeywordRedaction(Redaction):
    """
    token level keyword redaction
    """

    def __init__(self, keyword_list: List[str], ignore_case: bool = True, threshold: Optional[float] = None, **kwargs):
        super(KeywordRedaction, self).__init__(**kwargs)
        self.keywords = [k.lower() for k in keyword_list]
        self.ignore_case = ignore_case

    @classmethod
    def from_pretrained_path(cls, keyword_path, **kwargs):
        loaded_keywords = json.load(open(keyword_path, "r"))
        loaded_keyword_list = [word for v in loaded_keywords.values() for word in v]
        return cls(
            keyword_list=loaded_keyword_list,
            **kwargs,
        )

    def _is_private(self, x: List[str]) -> List[bool]:
        return [example.lower().strip(PUNCTUATION_CHARS) in self.keywords for example in x]

    def _flatten_example_ngrams(self, example_ngrams: List[List[str]]) -> List[str]:
        ngrams_flattened = [x if x else "" for ex in example_ngrams for x in ex]
        return ngrams_flattened if ngrams_flattened else [""]

    def redact_example(self, example):
        # sort by decreasing length before applying substitutions
        utterances = self._example_to_utterances(example)
        example_ngrams = [
            [
                " ".join(x)
                for n in range(3, 0, -1)
                for x in ngrams(utt.split(), min(n, len(utt.split())))
            ]
            for utt in utterances
        ]

        # OOM when example_ngrams is large?
        ngrams_flattened = self._flatten_example_ngrams(example_ngrams)
        is_ngram_private = iter(self._is_private(ngrams_flattened))
        out_list = []
        for out, ws in zip(utterances, example_ngrams):
            for w in ws:
                is_w_private = next(is_ngram_private)
                if is_w_private:
                    out = out.replace(w, " ".join([self.redaction_token] * len(w.split())))
            out_list.append(out)
        return out_list


class PerDomainKeywordRedaction(KeywordRedaction):
    def __init__(self, keyword_dict: Dict[str, List[str]], **kwargs):
        super(PerDomainKeywordRedaction, self).__init__(keyword_list=[], **kwargs)
        self.keywords = {k: [word.lower() for word in v] for k, v in keyword_dict.items()}

    @classmethod
    def from_pretrained_path(cls, keyword_dict, **kwargs):
        loaded_keywords = json.load(open(keyword_dict, "r"))
        return cls(
            keyword_dict=loaded_keywords,
            **kwargs,
        )

    def _is_private(self, x: List[str], domain: str) -> List[bool]:
        return [example.lower().strip(PUNCTUATION_CHARS) in self.keywords[domain] for example in x]

    def redact_example(self, example):
        # sort by decreasing length before applying substitutions
        utterances = self._example_to_utterances(example)
        example_ngrams = [
            [
                " ".join(x)
                for n in range(3, 0, -1)
                for x in ngrams(utt.split(), min(n, len(utt.split())))
            ]
            for utt in utterances
        ]

        # checking all n grams for the same domain
        is_ngram_private = iter(self._is_private(
            [x for ex in example_ngrams for x in ex], example["domain"]
        ))
        out_list = []
        for out, ws in zip(utterances, example_ngrams):
            for w in ws:
                is_w_private = next(is_ngram_private)
                if is_w_private:
                    out = out.replace(w, " ".join([self.redaction_token] * len(w.split())))
            out_list.append(out)
        return out_list

    def is_redacted_example(self, example: str, domain: str) -> bool:
        out = self.redact_example({"utterance": [example], "domain": domain})[0]
        return out.find(self.redaction_token) != -1

    def is_redacted_examples(self, examples: Sequence[str], target_domains: Sequence[str]) -> Sequence[bool]:
        out_list = [self.redact_example({"utterance": examples, "domain": domain}) for domain in target_domains]
        out_iterator = map(list, zip(*out_list))
        return [any([out_domain.find(self.redaction_token) != -1 for out_domain in out]) for out in out_iterator]


class ClassifierRedactionBase(Redaction):
    """
    token level classifier redaction
    """

    def __init__(
        self,
        clf,
        tokenizer,
        threshold,
        training_acc,
        test_acc,
        test_confusion,
        labels,
        vocab_len,
        **kwargs,
    ):
        super(ClassifierRedactionBase, self).__init__(**kwargs)
        self.clf = clf
        self.tokenizer = tokenizer
        self.threshold = threshold
        self.training_acc = training_acc
        self.test_acc = test_acc
        self.test_confusion = test_confusion
        self.labels = labels
        self.vocab_len = vocab_len
        self.device = self.clf.device if hasattr(self.clf, "device") else "cpu"
        if hasattr(self.tokenizer, "ngram_range") and self.ngram_range is None:
            self.ngram_range = self.tokenizer.ngram_range
        if self.device != "cpu":
            assert self.num_workers == 1

    # subclasses still must implement batch _is_private
    def _flatten_example_ngrams(self, example_ngrams: List[List[str]]) -> List[str]:
        ngrams_flattened = [x if x else "" for ex in example_ngrams for x in ex]
        return ngrams_flattened if ngrams_flattened else [""]

    def redact_example_unigram(self, example):
        # return re.sub("|".join(replace_strs), "", example["text"])
        example_ngrams = example["utterance"].split()
        return " ".join(
            [self.redaction_token if self._is_private(w) else w for w in example_ngrams]
        )

    def redact_example_nobatch(self, example):
        # sort by decreasing length before applying substitutions
        example_ngrams = (
            " ".join(x)
            for n in range(self.ngram_range[1], self.ngram_range[0] - 1, -1)
            for x in ngrams(example["utterance"].split(), n)
        )
        out = example["utterance"]
        for w in example_ngrams:
            if self._is_private(w):
                out = out.replace(w, " ".join([self.redaction_token] * len(w.split())))
        return out

    def redact_example(self, example):
        # sort by decreasing length before applying substitutions
        utterances = self._example_to_utterances(example)
        example_ngrams = [
            [
                " ".join(x)
                for n in range(self.ngram_range[1], self.ngram_range[0] - 1, -1)
                for x in ngrams(utt.split(), min(n, len(utt.split())))
            ]
            for utt in utterances
        ]

        # OOM when example_ngrams is large?
        ngrams_flattened = self._flatten_example_ngrams(example_ngrams)
        is_ngram_private = iter(self._is_private(ngrams_flattened))
        out_list = []
        for out, ws in zip(utterances, example_ngrams):
            for w in ws:
                is_w_private = next(is_ngram_private)
                if is_w_private:
                    out = out.replace(w, " ".join([self.redaction_token] * len(w.split())))
            out_list.append(out)
        return out_list


class NBClassifierRedaction(ClassifierRedactionBase):
    @classmethod
    def from_pretrained_path(cls, path, threshold, **kwargs):
        pretrained = torch.load(path)
        assert (
            len(pretrained["tokenizer"].get_feature_names_out())
            == pretrained["model"].feature_log_prob_.shape[1]
        )
        return cls(
            clf=pretrained["model"],
            tokenizer=pretrained["tokenizer"],
            threshold=threshold,
            training_acc=pretrained["training_acc"],
            test_acc=pretrained["test_acc"],
            test_confusion=pretrained["test_confusion"],
            labels=pretrained["labels"],
            vocab_len=len(pretrained["tokenizer"].get_feature_names_out()),
            **kwargs,
        )

    def _is_private(self, x: str) -> bool:
        return (self.clf.predict_proba(self.tokenizer.transform(x)) > self.threshold).any(axis=1)

    def get_keywords(
        self,
        label: Optional[Union[str, List]] = None,
        threshold: Optional[float] = None
    ) -> Dict[str, List[str]]:
        feature_names = self.tokenizer.get_feature_names_out()
        all_feature_probs = self.clf.predict_proba(self.tokenizer.transform(feature_names))
        out_dict = {}
        if isinstance(label, str):
            label = [label]
        elif label is None:
            label = self.labels
        threshold = self.threshold if threshold is None else threshold
        for label_name in label:
            out_idx = np.argsort(all_feature_probs[:, self.labels.index(label_name)])[::-1]
            out_dict[label_name] = [feature_names[o] for o in out_idx if all_feature_probs[o, self.labels.index(label_name)] > threshold]
        return out_dict


class PerDomainNBClassifierRedaction(NBClassifierRedaction):
    def _is_private(self, x: List[str], domain: str) -> np.ndarray:
        return (
            self.clf.predict_proba(self.tokenizer.transform(x))[:, self.labels.index(domain)]
            > self.threshold
        )

    def redact_example(self, example):
        # sort by decreasing length before applying substitutions
        utterances = self._example_to_utterances(example)
        example_ngrams = [
            [
                " ".join(x)
                for n in range(self.ngram_range[1], self.ngram_range[0] - 1, -1)
                for x in ngrams(utt.split(), min(n, len(utt.split())))
            ]
            for utt in utterances
        ]

        # OOM when example_ngrams is large?
        # checking all n grams for the same domain
        is_ngram_private = iter(self._is_private(
            [x for ex in example_ngrams for x in ex], example["domain"]
        ))
        out_list = []
        for out, ws in zip(utterances, example_ngrams):
            for w in ws:
                is_w_private = next(is_ngram_private)
                if is_w_private:
                    out = out.replace(w, " ".join([self.redaction_token] * len(w.split())))
            out_list.append(out)
        return out_list

    def is_redacted_example(self, example: str, domain: str) -> bool:
        out = self.redact_example({"utterance": [example], "domain": domain})[0]
        return out.find(self.redaction_token) != -1

    def is_redacted_examples(self, examples: Sequence[str], target_domains: Sequence[str]) -> Sequence[bool]:
        out_list = [self.redact_example({"utterance": examples, "domain": domain}) for domain in target_domains]
        out_iterator = map(list, zip(*out_list))
        return [any([out_domain.find(self.redaction_token) != -1 for out_domain in out]) for out in out_iterator]


class HFClassifierRedaction(ClassifierRedactionBase):
    @classmethod
    def from_pretrained_path(cls, path, threshold, **kwargs):
        pretrained = torch.load(path)
        # pretrained["model"].evaluate()
        model = pretrained["model"]
        model.eval()
        return cls(
            clf=model,
            tokenizer=pretrained["tokenizer"],
            threshold=threshold,
            training_acc=pretrained["training_acc"],
            test_acc=pretrained["test_acc"],
            test_confusion=pretrained["test_confusion"],
            labels=pretrained["labels"],
            vocab_len=None,
            **kwargs,
        )

    def _is_private(self, x: str, batch_size=128) -> bool:
        out = []
        for b in range(0, len(x), batch_size):
            tmp = self.tokenizer(x[b:b + batch_size], return_tensors="pt", padding=True).to(self.device)
            logits = self.clf(**tmp)["logits"]
            out.append(softmax(logits, dim=1).ge(self.threshold).any(dim=1).detach().to("cpu").numpy())
        return np.concatenate(out)


class PerDomainHFClassifierRedaction(HFClassifierRedaction):
    def _is_private(self, x: List[str], domain: str, batch_size=128) -> np.ndarray:
        out = []
        for b in range(0, len(x), batch_size):
            tmp = self.tokenizer(x[b:b + batch_size], return_tensors="pt", padding=True).to(self.device)
            # tmp = self.tokenizer([x], return_tensors="pt").to(self.device)
            logits = self.clf(**tmp)["logits"]
            out.append(softmax(logits, dim=1).ge(self.threshold)[:, self.labels.index(domain)].detach().to("cpu").numpy())
        return np.concatenate(out)

    def redact_example(self, example):
        # sort by decreasing length before applying substitutions
        utterances = self._example_to_utterances(example)
        example_ngrams = [
            [
                " ".join(x)
                for n in range(self.ngram_range[1], self.ngram_range[0] - 1, -1)
                for x in ngrams(utt.split(), min(n, len(utt.split())))
            ]
            for utt in utterances
        ]

        # OOM when example_ngrams is large?
        is_ngram_private = iter(self._is_private(
            [x for ex in example_ngrams for x in ex], example["domain"]
        ))
        out_list = []
        for out, ws in zip(utterances, example_ngrams):
            for w in ws:
                is_w_private = next(is_ngram_private)
                if is_w_private:
                    out = out.replace(w, " ".join([self.redaction_token] * len(w.split())))
            out_list.append(out)
        return out_list

    def is_redacted_example(self, example: str, domain: str) -> bool:
        out = self.redact_example({"utterance": [example], "domain": domain})[0]
        return out.find(self.redaction_token) != -1

    def is_redacted_examples(self, examples: Sequence[str], target_domains: Sequence[str]) -> Sequence[bool]:
        out_list = [self.redact_example({"utterance": examples, "domain": domain}) for domain in target_domains]
        out_iterator = map(list, zip(*out_list))
        return [any([out_domain.find(self.redaction_token) != -1 for out_domain in out]) for out in out_iterator]


class HFClassifierMergeRedaction(HFClassifierRedaction):
    # inspired by Algorithm 1 in for span extraction https://arxiv.org/abs/2205.04515
    def __init__(self, merge_distance="JS", **kwargs):
        super(HFClassifierMergeRedaction, self).__init__(**kwargs)
        # merge_threshold, merge_tree?
        if merge_distance == "JS":
            def merge_function(x, y):
                return jensenshannon(np.maximum(x, 0), np.maximum(y, 0), axis=1)
        else:
            raise ValueError
        self.merge_dist = merge_function

    def _get_private_prob(self, x: List[str], batch_size=128) -> np.ndarray:
        out = []
        for b in range(0, len(x), batch_size):
            tmp = self.tokenizer(x[b:b + batch_size], return_tensors="pt", padding=True).to(self.device)
            probs = softmax(self.clf(**tmp)["logits"], dim=1)
            out.append(probs.detach().to("cpu").numpy())
        return np.concatenate(out)

    def _merge_private_spans(self, private_probs: np.ndarray, orig_utterances: List[str]):
        if len(orig_utterances) == 1:
            return orig_utterances
        assert private_probs.shape[0] == len(orig_utterances)
        # merge along dim 0 based on distribution distance across dim 1
        prob_distances = self.merge_dist(private_probs[1:, :], private_probs[:-1, :])
        median_distance = np.median(prob_distances)
        get_logical_idx = np.arange(len(orig_utterances), dtype=int)
        for d_idx in np.argsort(prob_distances):
            if prob_distances[d_idx] < median_distance:
                current_idx = get_logical_idx[d_idx]
                get_logical_idx[d_idx+1:] -= 1
                next_idx = get_logical_idx[d_idx + 1]
                tmp = orig_utterances.pop(current_idx)  # index d_idx+1 now in location d_idx
                orig_utterances[next_idx] = " ".join([tmp, orig_utterances[next_idx]])
        return orig_utterances

    def redact_example(self, example):
        # use merge_distance to merge bottom-up spans
        utterances = self._example_to_utterances(example)
        example_ngrams = []
        for utterance in utterances:
            utterance_split = self._flatten_example_ngrams([utterance.split()])
            private_probs = self._get_private_prob(utterance_split)
            merged_utterance = self._merge_private_spans(private_probs, utterance_split)
            example_ngrams.append(merged_utterance)

        ngrams_flattened = self._flatten_example_ngrams(example_ngrams)  # necessary?
        is_ngram_private = iter(self._is_private(ngrams_flattened))
        out_list = []
        for out, ws in zip(utterances, example_ngrams):
            for w in ws:
                is_w_private = next(is_ngram_private)
                if is_w_private:
                    out = out.replace(w, " ".join([self.redaction_token] * len(w.split())))
            out_list.append(out)
        return out_list


class PerDomainHFClassifierMergeRedaction(HFClassifierMergeRedaction):
    def _get_private_prob(self, x: List[str], domain: str, batch_size=128) -> np.ndarray:
        out = []
        for b in range(0, len(x), batch_size):
            tmp = self.tokenizer(x[b:b + batch_size], return_tensors="pt", padding=True).to(self.device)
            probs = softmax(self.clf(**tmp)["logits"], dim=1)
            probs = probs[:, self.labels.index(domain)].unsqueeze(1)
            out.append(probs.detach().to("cpu").numpy())
        return np.concatenate(out)

    def _is_private(self, x: List[str], domain: str, batch_size=128) -> np.ndarray:
        out = []
        for b in range(0, len(x), batch_size):
            tmp = self.tokenizer(x[b:b + batch_size], return_tensors="pt", padding=True).to(self.device)
            logits = self.clf(**tmp)["logits"]
            out.append(softmax(logits, dim=1).ge(self.threshold)[:, self.labels.index(domain)].detach().to("cpu").numpy())
        return np.concatenate(out)

    def redact_example(self, example):
        utterances = self._example_to_utterances(example)
        example_ngrams = []
        for utterance in utterances:
            utterance_split = self._flatten_example_ngrams([utterance.split()])
            private_probs = self._get_private_prob(utterance_split, example["domain"])
            merged_utterance = self._merge_private_spans(private_probs, utterance_split)
            example_ngrams.append(merged_utterance)

        ngrams_flattened = self._flatten_example_ngrams(example_ngrams)  # necessary?
        is_ngram_private = iter(self._is_private(ngrams_flattened, example["domain"]))
        out_list = []
        for out, ws in zip(utterances, example_ngrams):
            for w in ws:
                is_w_private = next(is_ngram_private)
                if is_w_private:
                    out = out.replace(w, " ".join([self.redaction_token] * len(w.split())))
            out_list.append(out)
        return out_list

    def is_redacted_example(self, example: str, domain: str) -> bool:
        out = self.redact_example({"utterance": [example], "domain": domain})[0]
        return out.find(self.redaction_token) != -1

    def is_redacted_examples(self, examples: Sequence[str], target_domains: Sequence[str]) -> Sequence[bool]:
        out_list = [self.redact_example({"utterance": examples, "domain": domain}) for domain in target_domains]
        out_iterator = map(list, zip(*out_list))
        return [any([out_domain.find(self.redaction_token) != -1 for out_domain in out]) for out in out_iterator]
