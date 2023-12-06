from typing import List

CONVO_START = "<_soc_>"
CONVO_END = "<_eoc_>"
USR_START = "USR:"
USR_END = "<_cus_end_>"
SYS_START = "SYS:"
SYS_END = "<_rep_end_>"
BOT_START = "BOT:"
BOT_END = "<_bot_end_>"
PAD_TOKEN = "<pad>"
END_TOKEN = "<|endoftext|>"
REDACTED = "<REDACTED>"

CONVO_STOP_TOKENS = [
    USR_START,
    SYS_START,
    PAD_TOKEN,
    CONVO_START,
    CONVO_END,
    END_TOKEN,
]

PUNCTUATION_CHARS = ".,?!"

TASK2DOMAINS2TOKENS = {
    "dogo": {
        "AIRLINE": "<AIRLINE>",
        "FASTFOOD": "<FASTFOOD>",
        "FINANCE": "<FINANCE>",
        "INSURANCE": "<INSURANCE>",
        "MEDIA": "<MEDIA>",
        "SOFTWARE": "<SOFTWARE>",
    },
    "dogo60": {
        "AIRLINE": "<AIRLINE>",
        "FASTFOOD": "<FASTFOOD>",
        "FINANCE": "<FINANCE>",
        "INSURANCE": "<INSURANCE>",
        "MEDIA": "<MEDIA>",
        "SOFTWARE": "<SOFTWARE>",
    },
    "airline_media_ins": {
        "AIRLINE": "<AIRLINE>",
        "MEDIA": "<MEDIA>",
        "INSURANCE": "<INSURANCE>",
    },
    "airline_media_ins60": {
        "AIRLINE": "<AIRLINE>",
        "MEDIA": "<MEDIA>",
        "INSURANCE": "<INSURANCE>",
    },
}


def get_stop_tokens(task_mode: str) -> List[str]:
    return CONVO_STOP_TOKENS + list(TASK2DOMAINS2TOKENS[task_mode].values())
