"""
Generate samples with GPT-2 and filter out those that are likely to be
memorized samples from the training set.
"""
import os
import logging
import string
from typing import Optional, List, Tuple, Union

import fire
import numpy as np
import pandas as pd
from pprint import pprint
import torch
import zlib
from transformers import GPT2Tokenizer, GPT2LMHeadModel, \
    StoppingCriteriaList, set_seed
from tqdm import tqdm
from domain_private_transformers import constants
from domain_private_transformers.redaction import KeywordRedaction, \
    PerDomainNBClassifierRedaction, PerDomainHFClassifierRedaction, \
        PerDomainHFClassifierMergeRedaction, PerDomainKeywordRedaction
from domain_private_transformers.lm.decoding_utils import StopTokenStoppingCriteria


KEYWORDS_AIRLINES = [
    "check",
    "boarding",
    "pass",
    "confirmation",
    "booking",
    "help",
    "number",
    "seat",
]
KEYWORDS_MEDIA = [
    "zip",
    "purchase",
    "internet",
    "data",
    "services",
    "plan",
    "service",
    "help",
    "cable",
]

logging.basicConfig(level='ERROR')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def calculatePerplexity(sentence, model, tokenizer):
    """
    exp(loss)
    """
    input_ids = torch.tensor(tokenizer.encode(sentence)).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    return torch.exp(loss)


def print_best(metric, prompts_of_samples, samples, success, name1, scores1, name2=None, scores2=None, n=10):
    """
    print the `n` best samples according to the given `metric`
    """
    # these values then don't show up in the largest vals
    metric[np.isnan(metric)] = -np.inf
    idxs = np.argsort(metric)[::-1][:n]

    for i, idx in enumerate(idxs):
        if scores2 is not None:
            print(f"{i+1}: success={success[idx]}, {name1}={scores1[idx]:.3f}, {name2}={scores2[idx]:.3f}, score={metric[idx]:.3f}")
        else:
            print(f"{i+1}: {name1}={scores1[idx]:.3f}, , score={metric[idx]:.3f}")

        print()
        pprint(f"PROMPT: {prompts_of_samples[idx]} --- TEXT: {samples[idx]}")
        print()
        print()
        

def parse_prompts(prompts_files: List[str]) -> List[str]:
    lines: List[str] = []
    for fp in prompts_files:
        with open(fp, "r") as f:
            lines.extend(f.readlines())
    lines = [line.strip() for line in lines]
    return lines


def main(
    target_model: str = "gpt2-medium",
    ref_model: str = "gpt2",
    N: int = 1000,
    batch_size: int = 10,
    use_prompts: bool = False,
    task_mode: str = "dogo",
    ctx_domains: Union[str, Tuple[str]] = "ALL",
    data_folder: Optional[str] = None,
    data_split: str = "test",
    redaction_model: Optional[str] = None,
    redaction_model_path: Optional[str] = None,
    redaction_target_domain: Optional[Union[str, Tuple[str]]] = None,
    redaction_model_threshold: Optional[float] = None,
    redaction_model_ngram_range: Optional[Tuple[int, int]] = None,
    redaction_model_merge_distance: Optional[str] = None,
    save_results: bool = True,
    save_results_basename: Optional[str] = None,
    seed: int = 0,
    **kwargs,
):
    set_seed(seed)
    print(f"using device: {device}")

    # sample from the top_k tokens output by the model
    top_k = 40

    print("Loading GPT2...")
    tokenizer1 = GPT2Tokenizer.from_pretrained(target_model)
    tokenizer2 = GPT2Tokenizer.from_pretrained(ref_model)
    tokenizer1.padding_side = "left"
    tokenizer1.pad_token = tokenizer1.eos_token

    model1 = GPT2LMHeadModel.from_pretrained(target_model, return_dict=True).to(device)
    model1.config.pad_token_id = model1.config.eos_token_id
    model2 = GPT2LMHeadModel.from_pretrained(ref_model, return_dict=True).to(device)
    model1.eval()
    model2.eval()
    if redaction_model == "keyword":
        if redaction_target_domain == "AIRLINE":
            keywords = KEYWORDS_AIRLINES
        elif redaction_target_domain == "MEDIA":
            keywords = KEYWORDS_MEDIA
        elif ("MEDIA" in redaction_target_domain) and ("AIRLINE" in redaction_target_domain):
            keywords = KEYWORDS_AIRLINES + KEYWORDS_MEDIA
        else:
            raise ValueError(f"{redaction_target_domain} not supported")
        Redaction = KeywordRedaction(keyword_list=keywords, **kwargs)
    elif redaction_model == "key":
        redaction_model_threshold = None
        clf_path, redaction_class, redaction_kwargs, clf_threshold = (
            redaction_model_path, PerDomainKeywordRedaction,
            {"ignore_case": True, "num_workers": 4, "batch_size": 32}, redaction_model_threshold
        )
        Redaction = redaction_class.from_pretrained_path(
            clf_path, threshold=clf_threshold, **redaction_kwargs
        )
    elif redaction_model in ("nb", "bert", "bert-merged"):
        if redaction_model == "nb":
            if redaction_model_threshold is None:
                redaction_model_threshold = 0.9
            clf_path, redaction_class, redaction_kwargs, clf_threshold = (
                redaction_model_path, PerDomainNBClassifierRedaction,
                {"num_workers": 4, "batch_size": 32}, redaction_model_threshold,
            )
        elif redaction_model == "bert-merged":
            if redaction_model_threshold is None:
                redaction_model_threshold = 0.95
            if redaction_model_merge_distance is None:
                redaction_model_merge_distance = "JS"
            print(redaction_model_threshold, redaction_model_merge_distance)
            clf_path, redaction_class, redaction_kwargs, clf_threshold = (
                redaction_model_path, PerDomainHFClassifierMergeRedaction,
                {"merge_distance": redaction_model_merge_distance, "num_workers": 1, "batch_size": 4}, redaction_model_threshold
            )
        else:
            # TODO check this runs
            if redaction_model_threshold is None:
                redaction_model_threshold = 0.95
            if redaction_model_ngram_range is None:
                redaction_model_ngram_range = (1, 2)
            print(redaction_model_threshold, redaction_model_ngram_range)
            clf_path, redaction_class, redaction_kwargs, clf_threshold = (
                redaction_model_path, PerDomainHFClassifierRedaction,
                {"ngram_range": redaction_model_ngram_range, "num_workers": 1, "batch_size": 4}, redaction_model_threshold
            )
        Redaction = redaction_class.from_pretrained_path(
            clf_path, threshold=clf_threshold, **redaction_kwargs
        )
    else:
        Redaction = None

    # Convert the param to a string
    if isinstance(redaction_target_domain, str):
        redaction_target_domain = [redaction_target_domain]
    elif isinstance(redaction_target_domain, tuple):
        redaction_target_domain = list(redaction_target_domain)
    else:
        raise ValueError(f"{redaction_target_domain} not supported")
    print(f"target domain(s): {redaction_target_domain}")

    if use_prompts:
        print(f"Loading prompts for {task_mode}...")
        task_domains = list(constants.TASK2DOMAINS2TOKENS[task_mode].keys())
        if ctx_domains == "ALL":
            ctx_domains = ",".join(task_domains)
        elif isinstance(ctx_domains, tuple):
            ctx_domains: List[str] = list(ctx_domains)
        elif isinstance(ctx_domains, str) and (ctx_domains in task_domains):
            ctx_domains = [ctx_domains]
        else:
            raise ValueError(f"{ctx_domains} not supported")

        prompts_files = [
            os.path.join(data_folder, f"{data_split}/{domain.lower()}_prompts.txt")
            for domain in ctx_domains
        ]
        prompts = parse_prompts(prompts_files)
        encoded_prompts = [
            tokenizer1.encode(line, add_special_tokens=False, return_tensors="pt")
            for line in prompts
        ]

        bad_words_ids = tokenizer1(
            [constants.REDACTED], add_prefix_space=True, add_special_tokens=False
        ).input_ids

        stop_tokens: set[str] = set(constants.get_stop_tokens(task_mode))
        stop_tokens.add(tokenizer1.eos_token)
        stopping_criteria = StopTokenStoppingCriteria(
            stop_tokens=set(
                tokenizer1.encode(token)[0]
                for token in stop_tokens
                if token in tokenizer1.get_vocab()
            )
        )

        n_prompts = len(prompts)
        total_N = N * n_prompts
    else:
        total_N = N

    seq_len = 256  # number of tokens to generate
    n_empty_gens = 0
    samples = []
    prompts_of_samples = []
    scores = {"TARGET": [], "REF": [], "Lower": [], "zlib": [], "success": []}

    with tqdm(total=total_N, desc="generating outputs") as pbar:
        prompts_of_texts = []
        texts = []
        if use_prompts:
            # Generate N for each prompt
            for i in range(N):
                for prompt, encoded_prompt in zip(prompts, encoded_prompts):
                    input_ids = encoded_prompt  # shape (1, input_len)
                    input_len = len(input_ids[0])

                    output_ids = model1.generate(
                        input_ids=input_ids.to(device),
                        max_length=input_len + seq_len,
                        min_length=0,
                        do_sample=True,
                        top_k=top_k,
                        top_p=1.0,
                        num_return_sequences=1,
                        bad_words_ids=bad_words_ids,
                        stopping_criteria=StoppingCriteriaList([stopping_criteria]),
                    )

                    # Throw away batch dimension
                    output_ids = output_ids.squeeze(dim=0)
                    input_ids = input_ids.squeeze(dim=0)

                    # Slice the output string
                    whole_str: str = tokenizer1.decode(output_ids, skip_special_tokens=False)
                    prompt_str: str = tokenizer1.decode(input_ids, skip_special_tokens=False)
                    output_str: str = whole_str[len(prompt_str) :]
                    del whole_str, prompt_str

                    # Remove potential stop_tokens at the end.
                    if len(stop_tokens) > 0:
                        eos_pos = [output_str.find(tok) for tok in stop_tokens]
                        eos_pos = [e for e in eos_pos if e != -1]
                        if len(eos_pos) == 0:
                            eos_pos = None
                        else:
                            eos_pos = min(eos_pos)
                    else:
                        eos_pos = None
                    output_str = output_str[:eos_pos]
                    output_str = output_str.strip()
                    texts.append(output_str)
                    prompts_of_texts.append(prompt)

                pbar.update(n_prompts)
        else:
            num_batches = int(np.ceil(total_N / batch_size))
            for i in range(num_batches):
                # encode the prompts
                prompts = ["<|endoftext|>"] * batch_size
                input_len = 1
                inputs = tokenizer1(prompts, return_tensors="pt", padding=True)

                # batch generation
                output_sequences = model1.generate(
                    input_ids=inputs['input_ids'].to(device),
                    attention_mask=inputs['attention_mask'].to(device),
                    max_length=input_len + seq_len,
                    do_sample=True,
                    top_k=top_k,
                    top_p=1.0
                )

                batch_texts = tokenizer1.batch_decode(output_sequences, skip_special_tokens=False)
                texts.extend(batch_texts)
                prompts_of_texts.extend(["" for _ in batch_texts])

                pbar.update(batch_size)

        for text, prompt in zip(texts, prompts_of_texts):
            if len(text) == 0:
                n_empty_gens += 1
                # placeholders
                p1 = np.nan
                p2 = np.nan
                p_lower = np.nan
                zlib_entropy = np.nan
                is_success = np.nan
            else:
                # perplexity of TARGET and REF
                # p1 = calculatePerplexity(text, model1, tokenizer).item()
                # p2 = calculatePerplexity(text, model2, tokenizer).item()
                p1_prompt = calculatePerplexity(prompt, model1, tokenizer1).item()
                p1_prompt_gen = calculatePerplexity(f"{prompt} {text}", model1, tokenizer1).item()
                p1 = p1_prompt_gen / p1_prompt

                p2_prompt = calculatePerplexity(prompt, model2, tokenizer2).item()
                p2_prompt_gen = calculatePerplexity(f"{prompt} {text}", model2, tokenizer2).item()
                p2 = p2_prompt_gen / p2_prompt

                # perplexity on lower-case sample
                # p_lower = calculatePerplexity(text.lower(), model1, tokenizer).item()
                p_lower_prompt = calculatePerplexity(prompt.lower(), model1, tokenizer1).item()
                p_lower_prompt_gen = calculatePerplexity(
                    f"{prompt.lower()} {text.lower()}", model1, tokenizer1
                ).item()
                p_lower = p_lower_prompt_gen / p_lower_prompt

                # Zlib "entropy" of sample
                zlib_entropy = len(zlib.compress(bytes(text, 'utf-8')))

                # redact text to check attack success
                if Redaction is not None:
                    if redaction_model == "keyword":
                        formatted_text = text.lower().translate(
                            str.maketrans('', '', string.punctuation)
                        )
                        is_success = Redaction.is_redacted_example(formatted_text)
                    elif redaction_model in ("nb", "bert", "bert-merged", "key"):
                        # Check whether clf predicts any of the target domains
                        redaction_target_domain_example = redaction_target_domain
                        is_success = any(
                            Redaction.is_redacted_example(text, domain)
                            for domain in redaction_target_domain_example
                        )
                    else:
                        raise ValueError(redaction_model)
                else:
                    is_success = None

            samples.append(text)
            prompts_of_samples.append(prompt)
            scores["TARGET"].append(p1)
            scores["REF"].append(p2)
            scores["Lower"].append(p_lower)
            scores["zlib"].append(zlib_entropy)
            scores["success"].append(is_success)

    # Save the results
    os.makedirs(f"logs/mia_{redaction_model}", exist_ok=True)
    if save_results_basename is None:
        results_save_path = f"logs/mia_{redaction_model}/{task_mode}/results__ctx_{''.join(ctx_domains)}__redact_target_{''.join(redaction_target_domain)}__{target_model.replace('/', '-')}__{ref_model.replace('/', '-')}.csv"
    else:
        results_save_path = f"logs/mia_{redaction_model}/{task_mode}/results__{save_results_basename}.csv"
    df = pd.DataFrame({
        "prompt": prompts_of_samples,
        "gen": samples,
        "target_ppl": scores["TARGET"],
        "ref_ppl": scores["REF"],
        "lower_ppl": scores["Lower"],
        "zlib": scores["zlib"],
        "success": scores["success"],
    })
    if save_results:
        df.to_csv(results_save_path, index=False, header=True, sep=",")

    scores["TARGET"] = np.asarray(scores["TARGET"])
    scores["REF"] = np.asarray(scores["REF"])
    scores["Lower"] = np.asarray(scores["Lower"])
    scores["zlib"] = np.asarray(scores["zlib"])
    scores["success"] = np.asarray(scores["success"])

    print(f"======== num empty gens: {n_empty_gens}/{total_N} ========")
    print(f"======== num empty gens ratio: {n_empty_gens/total_N} ========")
    print()
    print()

    # Sort by ratio of log perplexities of REF and TARGET models
    metric = np.log(scores["REF"]) / np.log(scores["TARGET"])
    print(f"======== top sample by ratio of {ref_model} and {target_model} perplexities: ========")
    print_best(metric, prompts_of_samples, samples, scores["success"],
               f"PPL-{target_model}", scores["TARGET"],
               f"PPL-{ref_model}", scores["REF"])
    print()
    print()

    print(f"======== Overall Attack Success: {np.nansum(scores['success'])}/"
          f"{len(scores['success'])-n_empty_gens} ========")
    print(f"======== Overall Attack Success Rate: {np.nanmean(scores['success'])} ========")
    return scores


if __name__ == '__main__':
    tmp = fire.Fire(main)

# example usage:

# PYTHONPATH="."
# python LM_Memorization/extraction.py \
#   --use_prompts \
#   --target_model outputs/dogo_pub__airline_media__lr5e-5 \
#   --ref_model outputs/dogo_pub__airline__lr5e-5 \
#   --redaction_model keyword \
#   --task_mode "dogo" \
#   --task_domains "AIRLINE,MEDIA" \
#   --data_folder data/dogo
