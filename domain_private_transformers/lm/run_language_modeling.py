# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc.
# team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import logging

import torch
from ml_swissknife import utils
from transformers import HfArgumentParser, MODEL_WITH_LM_HEAD_MAPPING, set_seed
from transformers.models.gpt2 import GPT2Tokenizer
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.optimization import get_linear_schedule_with_warmup

from private_transformers import PrivacyEngine
from .compiled_args import (
    AuxiliaryArguments,
    DataTrainingArguments,
    ModelArguments,
    PrivacyArguments,
    TrainingArguments,
)
from .. import constants
from .data_utils import get_all_datasets, get_prompt_dataset, get_train_redacted_dataset
from .trainer import Trainer

logger = logging.getLogger(__name__)
logger.propagate = False

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


def evaluate_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: TrainingArguments,
    privacy_args: PrivacyArguments,
    auxiliary_args: AuxiliaryArguments,
):
    """
    Evaluates the model with tokenized data.
    """
    # Get dataset
    train_dataset, val_dataset, eval_dataset, data_collator = get_all_datasets(
        tokenizer=tokenizer,
        data_args=data_args,
    )

    # Materialize the prompts.
    generation_stuff = dict(
        train_prompts=get_prompt_dataset(data_args.train_prompt_file, tokenizer),
        val_prompts=get_prompt_dataset(data_args.val_prompt_file, tokenizer),
    )

    # Save the original training args
    orig_output_dir = training_args.output_dir
    training_args.output_dir = os.path.join(orig_output_dir, "base_model_eval_log")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        data_args=data_args,
        privacy_args=privacy_args,
        auxiliary_args=auxiliary_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        generation_stuff=generation_stuff,
    )

    trainer.evaluate(
        eval_dataset=None,
        log_results=True,
        epoch="base",
    )

    # Restore the original training args
    training_args.output_dir = orig_output_dir

    # Delete everything
    del train_dataset, val_dataset, eval_dataset, data_collator
    del generation_stuff
    del trainer


def main():
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            TrainingArguments,
            PrivacyArguments,
            AuxiliaryArguments,
        )
    )
    args = parser.parse_args_into_dataclasses()
    model_args, data_args, training_args, privacy_args, auxiliary_args = args

    model_args: ModelArguments
    data_args: DataTrainingArguments
    training_args: TrainingArguments
    privacy_args: PrivacyArguments
    auxiliary_args: AuxiliaryArguments

    if data_args.eval_data_file is None and training_args.do_test:
        raise ValueError(
            "Cannot do evaluation without an evaluation data file. Either "
            "supply a file to --eval_data_file or remove the --do_test "
            "argument."
        )

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists "
            f"and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    if auxiliary_args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = (
            logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN
        )
    logger.setLevel(log_level)
    # Create file handler for logging
    os.makedirs(training_args.output_dir, exist_ok=True)
    fh = logging.FileHandler(os.path.join(training_args.output_dir, "lm.log"), mode="w")
    fh.setLevel(log_level)
    # Create console handler for logging
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    # Create formatter and add to handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.warning(
        f"Process rank: {training_args.local_rank}, "
        f"device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {bool(training_args.local_rank != -1)}, "
        f"16-bits training: {training_args.fp16}",
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed
    set_seed(training_args.seed)

    # Debug mode
    if training_args.debug:
        import warnings
        warnings.filterwarnings("error")

    from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel

    # Config.
    config = GPT2Config.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )
    config.return_dict = True
    config.tie_word_embeddings = False

    # Tokenizer; `bos_token` and `eos_token` is the same for GPT2; both are
    # 50256.
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_args.model_name_or_path, cache_dir=model_args.cache_dir
    )

    if data_args.block_size <= 0:
        data_args.block_size = tokenizer.model_max_length
    else:
        data_args.block_size = min(data_args.block_size, tokenizer.model_max_length)

    # Model.
    gpt2 = GPT2LMHeadModel.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
    )
    logger.debug(f"base gpt2 model: {model_args.model_name_or_path}")
    logger.debug(gpt2)

    # Evaluate the base model
    if training_args.evaluate_base_model:
        logger.info("*** Evaluate the base model ***")
        evaluate_model(
            model=gpt2,
            tokenizer=tokenizer,
            model_args=model_args,
            data_args=data_args,
            training_args=training_args,
            privacy_args=privacy_args,
            auxiliary_args=auxiliary_args,
        )

    if training_args.tokenizer_init_lmhead_w_input_emb:
        # Clone the embedding into the lm_head for better initialization.
        lm_head = gpt2.get_output_embeddings()
        embedding = gpt2.get_input_embeddings()
        lm_head.weight.data.copy_(embedding.weight.data)
        logger.debug(
            f"Cloning initial embedding into lm_head, "
            f"checking norms... \n"
            f"\tlm_head: {lm_head.weight.norm()}, embedding: "
            f"{embedding.weight.norm()}"
        )
        torch.testing.assert_allclose(lm_head.weight, embedding.weight)
        del lm_head, embedding

    # Adjust tokenizer and model embeddings.
    logger.debug("adapt tokenizer to include special tokens")
    logger.debug(f"before len(tokenizer) = {len(tokenizer)}")
    orig_tokenizer_vocab = tokenizer.get_vocab()
    additional_tokens = set(constants.get_stop_tokens(data_args.task_mode))
    additional_tokens.add(constants.PAD_TOKEN)
    if training_args.tokenizer_add_redaction_token:
        additional_tokens.add(constants.REDACTED)
    tokens_to_add = [token for token in additional_tokens
                     if token not in orig_tokenizer_vocab]
    num_tokens_to_add = len(tokens_to_add)
    logger.debug(f"Adding {num_tokens_to_add} special tokens to tokenizer")
    if num_tokens_to_add > 0:
        tokenizer.add_special_tokens({
            "additional_special_tokens": tokens_to_add
        })
        logger.debug(f"after len(tokenizer) = {len(tokenizer)}")
        logger.debug(
            f"tokenizer.eos_token: {tokenizer.eos_token}, " f"{tokenizer.eos_token_id}"
        )
        logger.debug(
            f"tokenizer.bos_token: {tokenizer.bos_token}, " f"{tokenizer.bos_token_id}"
        )

        logger.debug("adapt size of lm_head and input_embeddings to include "
                     "additional tokens")
        logger.debug("use avg-based initialization")

        input_embeddings_before = gpt2.get_input_embeddings().weight
        lm_head_before = gpt2.get_output_embeddings().weight
        gpt2.resize_token_embeddings(len(tokenizer))

        input_embeddings_after = gpt2.get_input_embeddings().weight
        lm_head_after = gpt2.get_output_embeddings().weight
        logger.debug(
            f"before lm_head.weight.size() = {lm_head_before.size()}, "
            f"input_embeddings_before.size() = {input_embeddings_before.size()}"
        )
        logger.debug(
            f"after lm_head.weight.size() = {lm_head_after.size()}, "
            f"after input_embeddings_after.size() = {input_embeddings_after.size()}"
        )
        lm_head_after.data[-num_tokens_to_add:] = lm_head_before.mean(dim=0)
        input_embeddings_after.data[-num_tokens_to_add:] = input_embeddings_before.mean(dim=0)

        torch.testing.assert_allclose(lm_head_before, lm_head_after[:-num_tokens_to_add])
        logger.debug("pre-chunk equal for lm_head")
        torch.testing.assert_allclose(input_embeddings_before,
                                      input_embeddings_after[:-num_tokens_to_add])
        logger.debug("pre-chunk equal for input_embeddings")

    logger.debug("double check: ")
    logger.debug(f"embedding size {gpt2.get_input_embeddings().weight.size()}")
    logger.debug(f"lm_head size {gpt2.get_output_embeddings().weight.size()}")
    model = gpt2

    train_dataset, val_dataset, eval_dataset, data_collator = get_all_datasets(
        tokenizer=tokenizer,
        data_args=data_args,
    )

    if training_args.train_redaction_schedule:
        train_redacted_dataset = get_train_redacted_dataset(
            tokenizer=tokenizer,
            data_args=data_args,
        )
    else:
        train_redacted_dataset = None

    # Materialize the prompts.
    generation_stuff = dict(
        train_prompts=get_prompt_dataset(data_args.train_prompt_file, tokenizer),
        val_prompts=get_prompt_dataset(data_args.val_prompt_file, tokenizer),
    )
    if training_args.do_test:
        generation_stuff["eval_prompts"] = get_prompt_dataset(data_args.eval_prompt_file, tokenizer)

    # Do not evaluate on eval_dataset (this could be test!)
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        model_args=model_args,
        data_args=data_args,
        privacy_args=privacy_args,
        auxiliary_args=auxiliary_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        train_redacted_dataset=train_redacted_dataset,
        eval_dataset=None,
        data_collator=data_collator,
        generation_stuff=generation_stuff,
    )

    # Massage the parameters.
    if model_args.attention_only:
        model.requires_grad_(False)
        for name, param in model.named_parameters():
            if "c_attn.weight" in name:
                param.requires_grad_(True)
    else:
        model.requires_grad_(True)
        if model_args.static_lm_head:
            model.get_output_embeddings().requires_grad_(False)
        if model_args.static_embedding:
            model.get_input_embeddings().requires_grad_(False)
            model.transformer.wpe.requires_grad_(False)
    params = tuple(param for param in model.parameters() if param.requires_grad)
    names = tuple(
        name for name, param in model.named_parameters() if param.requires_grad
    )
    num_trainable_params = sum(param.numel() for param in params)
    logger.debug(f"Num trainable params: {num_trainable_params / 1e6:.4f} M")
    logger.debug(json.dumps(names, indent=4))

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [{
        "params": [p for n, p in model.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": training_args.weight_decay,
    }, {
        "params": [p for n, p in model.named_parameters()
                   if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
    }]
    optimizer = torch.optim.AdamW(
        params=optimizer_grouped_parameters,
        lr=training_args.learning_rate,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
    )
    trainer.optimizer = optimizer

    # Create the lr_scheduler.
    num_update_steps_per_epoch = (
        len(trainer.get_train_dataloader()) // trainer.args.gradient_accumulation_steps
    )
    num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
    t_total = int(num_update_steps_per_epoch * trainer.args.num_train_epochs)
    if training_args.lr_decay:
        trainer.lr_scheduler = get_linear_schedule_with_warmup(
            trainer.optimizer,
            num_warmup_steps=training_args.warmup_steps,
            num_training_steps=t_total,
        )
    else:
        trainer.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            trainer.optimizer, lambda _: 1.0
        )

    # Hacky way to set noise_multiplier.
    if privacy_args.non_private:
        privacy_args.noise_multiplier = 0.0
        privacy_args.per_example_max_grad_norm = None
    else:
        actual_batch_size = (
            training_args.per_device_train_batch_size
            * training_args.gradient_accumulation_steps
        )
        privacy_engine = PrivacyEngine(
            module=model,
            batch_size=actual_batch_size,
            sample_size=len(train_dataset),
            epochs=training_args.num_train_epochs,
            max_grad_norm=privacy_args.per_example_max_grad_norm,
            noise_multiplier=privacy_args.noise_multiplier,
            target_epsilon=privacy_args.target_epsilon,
            target_delta=privacy_args.target_delta,
            accounting_mode=privacy_args.accounting_mode,
            clipping_mode=privacy_args.clipping_mode,
        )
        # Originally, these could have been null.
        privacy_args.noise_multiplier = privacy_engine.noise_multiplier
        privacy_args.target_delta = privacy_engine.target_delta

        logger.debug("privacy_args: ")
        logger.debug(json.dumps(privacy_args.__dict__, indent=4))
        privacy_engine.attach(optimizer)

    # Training.
    if training_args.do_train:
        logger.info("*** Train ***")
        B = training_args.per_device_train_batch_size
        grad_acc_steps = training_args.gradient_accumulation_steps
        logger.info(
            f"Training set size: {len(train_dataset)}, "
            f"per_device_train_batch_size: {B}, "
            f"gradient_accumulation_steps: {grad_acc_steps}"
        )
        trainer.train(model_path=None)
        if training_args.save_at_last:
            trainer.save_model()

    # Evaluation
    if training_args.do_test:
        logger.info("*** Evaluate on test dataset ***")

        output = trainer.evaluate(eval_dataset=eval_dataset, log_results=False)
        utils.jdump(
            output,
            os.path.join(training_args.output_dir, "final_results.json"),
        )

        logger.info("***** Test results *****")
        logger.info(output)


if __name__ == "__main__":
    main()
