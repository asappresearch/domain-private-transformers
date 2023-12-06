"""Compilation of all the arguments."""
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

import transformers

from .. import constants
from . import decoding_utils

MODEL_CONFIG_CLASSES = list(transformers.MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

TRUE_TAGS = ('y', 'yes', 't', 'true')


# See all possible arguments in src/transformers/training_args.py
# or by passing the --help flag to this script.
# We now keep distinct sets of args, for a cleaner separation of concerns.
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from "
                    "scratch."
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )

    static_lm_head: str = field(default='no')
    static_embedding: str = field(default='no')
    attention_only: str = field(default="no")

    def __post_init__(self):
        self.static_lm_head = self.static_lm_head.lower() in TRUE_TAGS
        self.static_embedding = self.static_embedding.lower() in TRUE_TAGS
        self.attention_only = self.attention_only.lower() in TRUE_TAGS


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    DATA_FILE_EXTS = ['txt']
    data_folder: Optional[str] = field(
        default=None, metadata={"help": "Path to folder with all the data."}
    )
    redacted_data_folder: Optional[str] = field(
        default=None, metadata={"help": "Path to folder with all redacted data."}
    )
    train_data_file: Optional[str] = field(
        default=None,
        metadata={"help": f"A {DATA_FILE_EXTS} file containing the training "
                          f"data."}
    )
    val_data_file: Optional[str] = field(
        default=None,
        metadata={"help": f"A {DATA_FILE_EXTS} file containing the validation "
                          f"data."}
    )
    eval_data_file: Optional[str] = field(
        default=None,
        metadata={"help": f"A {DATA_FILE_EXTS} file containing the testing "
                          f"data."}
    )

    # Useful for truncating the dataset.
    max_train_examples: int = field(default=sys.maxsize)
    max_valid_examples: int = field(default=sys.maxsize)
    max_eval_examples: int = field(default=sys.maxsize)

    line_by_line: bool = field(
        default=True,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    task_mode: Optional[str] = field(
        default=None, metadata={"help": "The name of the task."}
    )
    task_domains: str = field(
        default="ALL", metadata={"help": "Comma-sep list of task domains to load"}
    )
    format_mode: Optional[str] = field(
        default='cat', metadata={"help": "The mode of data2text format (cat, peek, nopeek)"}
    )
    max_source_length: Optional[int] = field(
        default=512, metadata={"help": "the max source length of summarization data. "}
    )
    train_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for training data. "}
    )
    val_max_target_length: Optional[int] = field(
        default=100, metadata={"help": "the max target length for dev data. "}
    )
    block_size: int = field(
        default=-1,
        metadata={
            "help": "Optional input sequence length after tokenization."
                    "The training dataset will be truncated in block of this size for training."
                    "Default to the model max input length for single sentence inputs (take into account special "
                    "tokens)."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the "
                          "preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    no_keep_linebreaks: bool = field(
        default=False, metadata={"help": "Do not keep line breaks when using TXT files."}
    )
    max_seq_len: int = field(default=sys.maxsize)

    def __post_init__(self):
        if self.data_folder is not None:
            logging.warning(f'Overriding dataset paths using '
                            f'`data_folder`:{self.data_folder}')

            if self.task_mode == "sgd":
                self.train_data_file = os.path.join(self.data_folder, "train/dial.txt")
                self.val_data_file = os.path.join(self.data_folder, "valid/dial.txt")
                self.eval_data_file = os.path.join(self.data_folder, "test/dial.txt")

                self.train_prompt_file = os.path.join(self.data_folder, "train/dial_eval.txt")
                self.val_prompt_file = os.path.join(self.data_folder, "valid/dial_eval.txt")
                self.eval_prompt_file = os.path.join(self.data_folder, "test/dial_eval.txt")
            elif self.task_mode == "dogo":
                all_domains = list(constants.TASK2DOMAINS2TOKENS[self.task_mode].keys())
                if self.task_domains == "ALL":
                    self.task_domains = ",".join(all_domains)
                self.task_domains: List[str] = self.task_domains.split(',')
                assert all(
                    domain in all_domains for domain in self.task_domains
                ), f"At least domain {self.task_domains} not supported"

                self.train_data_file: List[str] = [
                    os.path.join(self.data_folder, f"train/{domain.lower()}.txt")
                    for domain in self.task_domains
                ]
                if self.redacted_data_folder is not None:
                    self.train_redacted_data_file: List[str] = [
                        os.path.join(self.redacted_data_folder,
                                     f"train/{domain.lower()}.txt")
                        for domain in self.task_domains
                    ]
                self.val_data_file: List[str] = [
                    os.path.join(self.data_folder, f"valid/{domain.lower()}.txt")
                    for domain in self.task_domains
                ]
                self.eval_data_file: List[str] = [
                    os.path.join(self.data_folder, f"test/{domain.lower()}.txt")
                    for domain in self.task_domains
                ]

                self.train_prompt_file: List[str] = [
                    os.path.join(self.data_folder,
                                 f"train/{domain.lower()}_prompts.txt")
                    for domain in self.task_domains
                ]
                self.val_prompt_file: List[str] = [
                    os.path.join(self.data_folder,
                                 f"valid/{domain.lower()}_prompts.txt")
                    for domain in self.task_domains
                ]
                self.eval_prompt_file: List[str] = [
                    os.path.join(self.data_folder,
                                 f"test/{domain.lower()}_prompts.txt")
                    for domain in self.task_domains
                ]
            else:
                raise ValueError(f'{self.task_mode} task data not supported')
        else:
            # Sanity check for data file format
            for data_file in [
                self.train_data_file,
                self.val_data_file,
                self.eval_data_file,
                self.train_prompt_file,
                self.val_prompt_file,
                self.eval_prompt_file,
            ]:
                if data_file is not None:
                    ext = data_file.split('.')[-1]
                    assert ext in self.DATA_FILE_EXTS, \
                        f"{data_file} should be one of {self.DATA_FILE_EXTS}"
                    assert os.path.isfile(data_file), f"{data_file} does not exist"


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    max_eval_batches: int = field(default=-1, metadata={"help": "Maximum number of evaluation steps to run."})
    skip_generation: str = field(default="no")
    decoding_type: str = field(default="greedy")
    decoding_temperature: float = field(default=1.0)
    decoding_top_k: Optional[int] = field(default=None)
    decoding_top_p: Optional[float] = field(default=None)
    decoding_max_generations: Optional[int] = field(
        default=None,
        metadata={"help": "max #generations decoded per file, must be >=0"}
    )
    decoding_allow_redaction_token: bool = field(
        default=False,
        metadata={"help": f"Flag to allow {constants.REDACTED} while decoding"}
    )

    tokenizer_add_redaction_token: bool = field(
        default=False,
        metadata={"help": f"Flag to force add {constants.REDACTED} to vocab"}
    )
    tokenizer_init_lmhead_w_input_emb: bool = field(
        default=False,
        metadata={"help": f"Flag to init LM head with input embeddings"}
    )

    train_redaction_schedule: Optional[str] = field(
        default=None,
        metadata={"help": "Use a redaction sampler with dec step schedule"}
    )

    train_redaction_exp_scale: float = field(
        default=1.0,
        metadata={"help": "Scale factor for redaction sampler exp schedules"}
    )

    ema_model_averaging: str = field(default="no")
    ema_model_gamma: float = field(default=0.99)
    ema_model_start_from: int = field(default=1000)
    lr_decay: str = field(default="yes")
    eval_epochs: int = field(default=10)

    do_test: bool = field(default=False, metadata={"help": "Do test"})
    evaluate_base_model: bool = field(
        default=False, metadata={"help": "Evaluate base model for sanity."}
    )
    evaluate_before_training: str = field(
        default="yes",
        metadata={"help": "Run evaluation before training."},
    )
    evaluate_after_training: str = field(
        default="yes",
        metadata={"help": "Run evaluation during training at each logging step."},
    )
    save_at_last: str = field(default="no", metadata={"help": "Save at the end of training."})

    def __post_init__(self):
        super(TrainingArguments, self).__post_init__()
        self.skip_generation = self.skip_generation.lower() in ('y', 'yes')
        self.ema_model_averaging = (self.ema_model_averaging.lower() in ('y', 'yes'))
        self.lr_decay = (self.lr_decay.lower() in ('y', 'yes'))
        self.evaluate_after_training = (self.evaluate_after_training in ('y', 'yes'))
        self.evaluate_before_training = (self.evaluate_before_training in ('y', 'yes'))
        self.save_at_last = (self.save_at_last in ('y', 'yes'))

        if self.train_redaction_schedule:
            assert (
                self.tokenizer_add_redaction_token,
                "Must specify --tokenizer_add_redaction_token flag with" \
                " --train_redaction_schedule"
            )
            assert self.train_redaction_schedule in (
                "step", "linear", "exp_concave", "exp_convex"
            )

        assert (
            self.decoding_type in decoding_utils.DECODING_TYPES
        ), f"Decoding type {self.decoding_type} not supported"
        assert (
            (self.decoding_max_generations is None) or
            (self.decoding_max_generations >= 0)
        ), f"`decoding_max_generations` must be non-negative"


@dataclass
class PrivacyArguments:
    """Arguments for differentially private training."""
    per_example_max_grad_norm: float = field(
        default=.1, metadata={
            "help": "Clipping 2-norm of per-sample gradients."
        }
    )
    noise_multiplier: float = field(
        default=None, metadata={
            "help": "Standard deviation of noise added for privacy; if `target_epsilon` is specified, "
                    "use the one searched based budget"
        }
    )
    target_epsilon: float = field(
        default=None, metadata={
            "help": "Privacy budget; if `None` use the noise multiplier specified."
        }
    )
    target_delta: float = field(
        default=None, metadata={
            "help": "Lax probability in approximate differential privacy; if `None` use 1 / len(train_data)."
        }
    )
    accounting_mode: str = field(
        default="rdp", metadata={"help": "One of `rdp`, `glw`, `all`."}
    )
    non_private: str = field(default="no")
    clipping_mode: str = field(default="default")

    def __post_init__(self):
        self.non_private = self.non_private.lower() in ('y', 'yes')


@dataclass
class AuxiliaryArguments:
    eval_spectrum: str = field(default="no")
    max_spectrum_batches: int = field(default=100)
    max_lanczos_iter: int = field(default=100)

    store_grads: str = field(default="no")
    orthogonal_projection_path: Optional[str] = field(default=None)
    orthogonal_projection_rank: int = field(default=100)
    verbose: bool = field(default=False)

    def __post_init__(self):
        self.eval_spectrum = self.eval_spectrum.lower() in TRUE_TAGS  # noqa
        self.store_grads = self.store_grads.lower() in TRUE_TAGS  # noqa
