#!/usr/bin/env bash

target_models=(
  "dogo_pub__airline_media__lr5e-4"
)

for target_model in "${target_models[@]}"; do
  bash training_scripts/eval_dogo.sh \
    "outputs/eval__${target_model}" \
    "data/dogo" \
    "dogo" \
    "AIRLINE,MEDIA" \
    "outputs/${target_model}" 2>&1 | tee "logs/eval__${target_model}.log"
done
