import json
import os
from typing import List, Iterable, Optional

import fire
import numpy as np
import pandas as pd
from domain_private_transformers.constants import (
    USR_START,
    SYS_START,
    BOT_START,
    CONVO_START,
    CONVO_END,
    TASK2DOMAINS2TOKENS,
)

DOGO_DEFAULT_DATA_DIR = "../multi-domain-goal-oriented-dialogues-dataset/data/"
DOGO_DEFAULT_INPUT_DIR = os.path.join(DOGO_DEFAULT_DATA_DIR, "unannotated")


def get_author_token(author: str) -> str:
    if author == "customer":
        author_token = USR_START
    elif author == "agent":
        author_token = SYS_START
    elif author == "bot":
        author_token = BOT_START
    else:
        raise ValueError(f"Author {author} not supported")
    return author_token


def convo_grp2str(df: pd.DataFrame, domain_token: Optional[str]) -> List[str]:
    """
    Returns the list of (formatted) utterances from the conversation.
    """
    convo: List[str] = []
    for i, row in df.iterrows():
        author_token = get_author_token(row["authorRole"])
        utterance = "" if domain_token is None else f"{domain_token} "
        utterance += f"{author_token} {row['utterance'].strip()}"
        convo.append(utterance)
    return convo


def convo_grp2prompts(
    df: pd.DataFrame, domain_token: Optional[str], max_prompts_per_convo: int
) -> List[List[str]]:
    """
    Returns the list of (formatted) prompts from the conversation.

    max_prompts_per_convo: used when non-negative.
    """
    n_utts = len(df)
    convo: List[List[str]] = []
    prompt_so_far: List[str] = []
    for i, (row_idx, row) in enumerate(df.iterrows()):
        # Don't create more prompts if over limit
        if i >= max_prompts_per_convo >= 0:
            break
        # Last utterance cannot be used for prompting, so use just the token
        if i == n_utts - 1:
            break
        author_token = get_author_token(row["authorRole"])
        prompt_utt = "" if domain_token is None else f"{domain_token} "
        prompt_utt += f"{author_token} {row['utterance'].strip()}"
        prompt_so_far.append(prompt_utt)
        prompt_token = "" if domain_token is None else f"{domain_token} "
        prompt_token += f"{get_author_token(df.iloc[i + 1]['authorRole'])}"
        # copy() necessary as prompt_so_far mutates
        convo.append(prompt_so_far.copy() + [prompt_token])
    return convo


def load_convos_from_tsv(domain_tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(domain_tsv_path, sep=",")
    # NaNs were likely created due to space strings, replace with " "
    df["utterance"].fillna(" ", inplace=True)
    df = df.sort_values(by=["conversationId", "turnNumber"]).reset_index(drop=True)
    # remove conversations that appear as duplicates except conversationId ending in "-1" or "-2"
    df["conversationId"] = df["conversationId"].str.rstrip("-1").str.rstrip("-2")
    df = df.drop_duplicates()
    return df


def create_splits(domains: Iterable[str], input_dir: str, split_convos_info_path: str, valid_frac, test_frac):
    split_data = {}
    for domain in domains:
        print(f"** Splitting {domain}...")
        # Load convos from tsv file
        domain_tsv_path = os.path.join(input_dir, f"{domain.lower()}.tsv")
        df = load_convos_from_tsv(domain_tsv_path)

        # Split convos into train, valid, test
        convoIds = df["conversationId"].unique()
        np.random.shuffle(convoIds)
        n_convos = len(convoIds)
        n_test_convos = int(test_frac * n_convos)
        n_valid_convos = int(valid_frac * n_convos)

        split_data[domain] = {
            "test": convoIds[:n_test_convos].tolist(),
            "valid": convoIds[n_test_convos : n_test_convos + n_valid_convos].tolist(),
            "train": convoIds[n_test_convos + n_valid_convos :].tolist(),
        }

    with open(split_convos_info_path, "w") as f:
        json.dump(split_data, f, indent=2)


def main(
    split_convos_info_path: str = f"data/dogo__split_convos_info.json",
    input_dir: str = DOGO_DEFAULT_INPUT_DIR,
    output_dir: str = "data/dogo/",
    task_str: str = "dogo",
    use_domain_tokens: bool = True,
    prompts_max_convos_per_split: int = 10,
    prompts_max_prompts_per_convo: int = 10,
    valid_frac: Optional[float] = 0.1,
    test_frac: Optional[float] = 0.1,
    seed: int = 0,
):
    np.random.seed(seed)

    # Create splits from unannotated and save if it does not exist
    if not os.path.exists(split_convos_info_path):
        print(f"* Splits don't exist, creating for all domains using unannotated...")
        create_splits(TASK2DOMAINS2TOKENS[task_str].keys(), input_dir, split_convos_info_path, valid_frac, test_frac)

    with open(split_convos_info_path, "r") as f:
        # domain |-> (split_name |-> [convoID1, convoID2, ...])
        split_data = json.load(f)

    for domain, domain_token in TASK2DOMAINS2TOKENS[task_str].items():
        print(f"* Processing domain: {domain}, domain tokens: {use_domain_tokens}...")

        if not use_domain_tokens:
            domain_token = None
        # Load convos from tsv file
        domain_tsv_path = os.path.join(input_dir, f"{domain.lower()}.tsv")
        if not os.path.isfile(domain_tsv_path):
            print(f"* {domain_tsv_path} does not exist, skipping...")
            continue
        df = load_convos_from_tsv(domain_tsv_path)
        data = split_data[domain]
        for split_name, split_convosIds in data.items():
            split_df = df[df["conversationId"].isin(split_convosIds)]

            # Full convo (each convo is a string)
            split_convos: List[str] = split_df.groupby(by="conversationId").apply(
                lambda grp: f" ".join(convo_grp2str(grp, domain_token))
            )
            split_convos = [
                f"{CONVO_START} {convo} {CONVO_END}" for convo in split_convos
            ]
            split_path = os.path.join(output_dir, split_name, f"{domain.lower()}.txt")
            os.makedirs(os.path.dirname(split_path), exist_ok=True)
            pd.DataFrame(split_convos).to_csv(
                split_path, index=False, header=False, sep="\n"
            )

            # Convo prompts (each convo is a list of prompts)
            split_df_prompts = split_df[
                split_df["conversationId"].isin(
                    split_convosIds[:prompts_max_convos_per_split]
                )
            ]
            split_prompts: List[List[str]] = split_df_prompts.groupby(
                by="conversationId"
            ).apply(
                lambda grp: [
                    f" ".join(prompt)
                    for prompt in convo_grp2prompts(
                        grp, domain_token, prompts_max_prompts_per_convo
                    )
                ]
            )
            split_prompts = [
                [f"{CONVO_START} {prompt}" for prompt in convo]
                for convo in split_prompts
            ]
            split_path = os.path.join(
                output_dir, split_name, f"{domain.lower()}_prompts.txt"
            )
            os.makedirs(os.path.dirname(split_path), exist_ok=True)
            with open(split_path, "w") as f:
                for convo in split_prompts:
                    for prompt in convo:
                        print(prompt, file=f)

            # Convo prompts full (each convo is a list of prompts)
            split_df_prompts = split_df[
                split_df["conversationId"].isin(
                    split_convosIds[:prompts_max_convos_per_split]
                )
            ]
            split_prompts: List[List[str]] = split_df_prompts.groupby(
                by="conversationId"
            ).apply(
                lambda grp: [
                    f" ".join(prompt)
                    for prompt in convo_grp2prompts(grp, domain_token, -1)
                ]
            )
            split_prompts = [
                [f"{CONVO_START} {prompt}" for prompt in convo]
                for convo in split_prompts
            ]
            split_path = os.path.join(
                output_dir, split_name, f"{domain.lower()}_prompts_full.txt"
            )
            os.makedirs(os.path.dirname(split_path), exist_ok=True)
            with open(split_path, "w") as f:
                for convo in split_prompts:
                    for prompt in convo:
                        print(prompt, file=f)


if __name__ == "__main__":
    fire.Fire(main)
