#!/bin/bash

output_dir=${1}
data_dir=${2}
task_mode=${3}
task_domains=${4:-"ALL"}
model_name_or_path=${5:-"gpt2"} # One of distilgpt2, gpt2, gpt2-medium, gpt2-large
target_epsilon=${6:-"8"}
cache_dir=${7:-"$HOME/.cache/huggingface/transformers/"}
clipping_mode=${8:-"ghost"} # Fill 'default' to turn this off.
non_private=${9:-"no"}

log_steps=500
max_int=9223372036854775807
max_examples=${max_int}

case ${task_mode} in
  "sgd")
    num_train_epochs=50
#    learning_rate=5e-5
    ;;
  "dogo")
    # num_train_epochs=5
    num_train_epochs=2.5
#    learning_rate=5e-5
    ;;
esac

for lr in "5e-5" "5e-4"; do
  output_dir_new="${output_dir}__lr${lr}"
  python -u -m domain_private_transformers.lm.run_language_modeling \
    --task_mode "${task_mode}" \
    --task_domains "${task_domains}" \
    --output_dir "${output_dir_new}" --overwrite_output_dir \
    --model_name_or_path "${model_name_or_path}" \
    --tokenizer_name "${model_name_or_path}" \
    --do_train \
    --save_steps ${log_steps} --save_total_limit 1 --save_at_last yes \
    --logging_dir "${output_dir_new}" --logging_steps -1 \
    --max_train_examples ${max_examples} --max_valid_examples ${max_examples} --max_eval_examples ${max_examples} \
    --data_folder "${data_dir}" \
    --per_example_max_grad_norm 0.1 --target_epsilon "${target_epsilon}" \
    --learning_rate ${lr} --lr_decay "no" --num_train_epochs ${num_train_epochs} --per_device_train_batch_size 4 \
    --eval_steps ${log_steps} --evaluation_strategy steps --evaluate_before_training "yes" --evaluate_after_training "yes" --per_device_eval_batch_size 10 \
    --decoding_type "top-k" --decoding_temperature 5.0 --decoding_top_k 5 --decoding_max_generations ${max_int} \
    --tokenizer_add_redaction_token \
    --non_private "${non_private}" \
    --clipping_mode "${clipping_mode}" \
    --cache_dir "${cache_dir}" \
    --seed 0 \
    --evaluate_base_model \
    --verbose
done
