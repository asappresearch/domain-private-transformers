import logging
from typing import Iterable, Optional, Union, List

import torch
from tqdm import tqdm
import transformers
from transformers import add_start_docstrings
from transformers.generation_stopping_criteria import (
    STOPPING_CRITERIA_INPUTS_DOCSTRING,
    StoppingCriteria,
    StoppingCriteriaList,
)

from domain_private_transformers import constants

logger = logging.getLogger(__name__)

DECODING_TYPES = ("greedy", "sampling", "top-k", "top-p")


class StopTokenStoppingCriteria(StoppingCriteria):
    def __init__(self, stop_tokens: Iterable[int]):
        self.stop_tokens = stop_tokens

    @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        return input_ids[-1][-1].item() in self.stop_tokens


def generate(
    model: transformers.PreTrainedModel,
    tokenizer: transformers.PreTrainedTokenizer,
    prompt_dataset: Iterable[torch.Tensor],
    stop_tokens: Optional[Iterable[str]] = None,
    bad_words: Optional[List[str]] = None,
    decoding_type: str = "greedy",
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    device: Union[str, torch.device] = "cpu",
    disable_tqdm: bool = False,
    return_prompts: bool = False,
):
    """
    Returns a list of generations for each prompt in `prompt_dataset`.
    Optionally also returns prompts if flag is set.
    """
    assert not model.training, "Generation must be when `model` is in eval mode."

    # Decoding kwargs
    num_beams = 1
    if decoding_type == "greedy":
        do_sample = False
        top_k = None
        top_p = None
    elif decoding_type == "sampling":
        do_sample = True
        top_k = None
        top_p = None
    elif decoding_type == "top-k":
        do_sample = True
        top_k = top_k
        top_p = None
    elif decoding_type == "top-p":
        do_sample = True
        top_k = 50  # only consider top 50 words even when >50 needed to add to p
        top_p = top_p
    else:
        raise ValueError(f"Decoding type {decoding_type} not supported")

    pad_token = (
        constants.PAD_TOKEN
        if constants.PAD_TOKEN in tokenizer.get_vocab()
        else tokenizer.eos_token
    )
    pad_token_id = tokenizer.encode(pad_token)[0]
    if (bad_words is not None) and (len(bad_words) > 0):
        bad_words_ids = tokenizer(
            bad_words, add_prefix_space=True, add_special_tokens=False
        ).input_ids
    else:
        bad_words_ids = None

    # Stop generation at a stop token
    stop_tokens: set[str] = set(stop_tokens) if stop_tokens is not None else set()
    stop_tokens.add(tokenizer.eos_token)
    stopping_criteria = StopTokenStoppingCriteria(
        stop_tokens=set(
            tokenizer.encode(token)[0]
            for token in stop_tokens
            if token in tokenizer.get_vocab()
        )
    )

    generations = []
    prompts = []
    prompt_dataset = (
        prompt_dataset if disable_tqdm else tqdm(prompt_dataset, desc="generation")
    )
    for input_ids in prompt_dataset:
        # input_ids has shape (1, prompt_len)
        prompt_len = len(input_ids[0])
        if return_prompts:
            # Prompt might contain special tokens, don't skip them
            prompt = tokenizer.decode(input_ids[0], skip_special_tokens=False)
            prompts.append(prompt)
        input_ids = input_ids.to(device)

        # Generate output ids
        output_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length + prompt_len,
            min_length=0,
            temperature=temperature,
            do_sample=do_sample,
            num_beams=num_beams,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=1.0,
            num_return_sequences=1,
            pad_token_id=pad_token_id,
            bad_words_ids=bad_words_ids,
            stopping_criteria=StoppingCriteriaList([stopping_criteria]),
        )

        # Throw away batch dimension
        output_ids = output_ids.squeeze(dim=0)
        input_ids = input_ids.squeeze(dim=0)

        # Slice the output string
        whole_str: str = tokenizer.decode(output_ids, skip_special_tokens=False)
        prompt_str: str = tokenizer.decode(input_ids, skip_special_tokens=False)
        output_str: str = whole_str[len(prompt_str) :]
        del whole_str, prompt_str

        # Remove potential stop_tokens at the end.
        if len(stop_tokens) > 0:
            eos_pos = [output_str.find(tok) for tok in stop_tokens]
            eos_pos = [e for e in eos_pos if e != -1]
            if len(eos_pos) == 0:
                # Didn't generate eos_token; that's okay -- just skip!
                eos_pos = None
            else:
                eos_pos = min(eos_pos)
        else:
            eos_pos = None
        output_str = output_str[:eos_pos]
        output_str = output_str.strip()

        generations.append(output_str)

    if return_prompts:
        return generations, prompts
    else:
        return generations
