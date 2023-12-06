#!/bin/sh

export PYTHONPATH="/persist/git/domain-private-transformers" 

export domain_list="AIRLINE,MEDIA,INSURANCE"
export prompt_domain="AIRLINE"
export domain_tag=airline_media_ins60
export task_mode="dogo"
export domain_classifier_task="dogo60"
# export domain_tokens="False"
# export task_tag="dogo-nodomaintok"
# export dataset_path="data/dogo60-nodomaintok/"
export domain_tokens="True"
export task_tag="dogo"
export dataset_path="data/dogo60/"


### preprocess data
python training_scripts/preprocess_dogo.py \
  --split_convos_info_path "data/dogo60__split_convos_info.json" \
  --input_dir "../multi-domain-goal-oriented-dialogues-dataset/data/unannotated" \
  --output_dir ${dataset_path} \
  --use_domain_tokens ${domain_tokens} \
  --valid_frac 0.2 \
  --test_frac 0.2

### TRAINING ###
#################################
###### PUB on NON-REDACTED ######
### public fine-tuning baseline #
#################################
bash training_scripts/train_pub.sh \
    /persist/outputs/${task_tag}_pub__${domain_tag}__eps8 \
    ${dataset_path} \
    ${task_mode} \
    ${domain_list}


####################################
###### DP-SGD on NON-REDACTED ######
### private fine-tuning baseline ###
####################################
bash training_scripts/train_pvt__eps8__gradacc16.sh \
    /persist/outputs/${task_tag}_pvt__${domain_tag}__50epochs__eps8__gradacc16 \
    ${dataset_path} \
    ${task_mode} \
    ${domain_list} \
    50


### domain classifier
python domain_private_transformers/domain_classifier.py --split train --model_type hf ---train_epochs 5 \
--learning_rate 1e-5 --task ${domain_classifier_task} --domain_tag ${domain_tag} --maximum_start_char 0 \
--save_path model/redaction/dogo3dom_train60_hf_5epochs_1e-5_best.pt

### reference lm for attacks and for threshold of domain classifier redaction
bash training_scripts/train_pub.sh \
    /persist/outputs/${task_tag}_pub__${prompt_domain,,}60__20epochs__eps8 \
    ${dataset_path} \
    ${task_mode} \
    ${prompt_domain} \
    20

### Sweep to find the best threshold on eval set, add threshold to model names


### REDACT ###
### run keyword-based redaction
redaction_model="model/redaction/dogo60_keywords.json"
redaction_model_type="key"
redaction_threshold=""
redacted_data_tag="redacted-key${redaction_threshold}-dogo60-3dom"
redacted_model_tag="redacted_key${redaction_threshold}"

# redact data
cmd_redaction="python scripts/run_redaction.py \
        --clf_model ${redaction_model_type} --clf_path ${redaction_model} \
        --output_folder_name ${redacted_data_tag}"
echo ${cmd_redaction}
eval ${cmd_redaction}

### run classifier-based redaction
redaction_model="model/redaction/dogo3dom_train60_hf_5epochs_1e-5_best.pt"
redaction_model_type="hf"
redaction_threshold=0.96
redaction_ngram=16
redacted_data_tag="redacted-clf${redaction_threshold}-dogo60-3dom-rob"
redacted_model_tag="redacted_clf-rob${redaction_threshold}"

cmd_redaction="python scripts/run_redaction.py \
        --clf_model ${redaction_model_type} --clf_path ${redaction_model} \
        --clf_threshold ${redaction_threshold} --clf_ngram_range ${redaction_ngram},${redaction_ngram} \
        --output_folder_name ${redacted_data_tag}"
echo ${cmd_redaction}
eval ${cmd_redaction}
### REDACT ###


### preprocess redacted data (2 steps)
cmd_prered="python training_scripts/preprocess_dogo.py \
  --split_convos_info_path data/dogo60__split_convos_info.json \
  --task_str ${task_mode} \
  --input_dir ../multi-domain-goal-oriented-dialogues-dataset/data/${redacted_data_tag} \
  --output_dir data/${task_tag}-${redacted_data_tag}/ \
  --use_domain_tokens ${domain_tokens} \
  --valid_frac 0.2 \
  --test_frac 0.2"
echo ${cmd_prered}
eval ${cmd_prered}


### TRAINING ###
#############################
###### PUB on REDACTED ######
###      (2 steps)        ###
#############################
cmd_pft="bash training_scripts/train_pub_redacted.sh \
    /persist/outputs/${task_tag}_pub_${redacted_model_tag}__${domain_tag} \
    data/${task_tag}-${redacted_data_tag} \
    ${task_mode} \
    ${domain_list}"
echo ${cmd_pft}
eval ${cmd_pft}


############################################################################
###### (Redaction Schedule) DP-SGD on NON-REDACTED || PUB on REDACTED ######
###              1 step private training and eval                        ###
############################################################################
cmd="bash training_scripts/train_pvt_redacted_schedule.sh \
  /persist/outputs/${task_tag}_pvt_sch_${redacted_model_tag}__exp_concave_sch15__${domain_tag}__eps8 \
  ${dataset_path} data/${task_tag}-${redacted_data_tag} ${task_mode} ${domain_list} exp_concave 15"
echo ${cmd}
eval ${cmd}

####################################################################
###### (JFT) DP-SGD on NON-REDACTED with init=PUB on REDACTED ######
####################################################################
cmd_jft="bash training_scripts/train_pvt__on_pub_redacted.sh \
   /persist/outputs/${task_tag}_pvt__on_pub_${redacted_model_tag}__${domain_tag}__eps8 \
   ${dataset_path} \
   ${task_mode} \
   ${domain_list} \
   /persist/outputs/${task_tag}_pub_${redacted_model_tag}__${domain_tag}__lr5e-4"
echo ${cmd_jft}
eval ${cmd_jft}


### EVALUATION ###

export domain_list="AIRLINE,MEDIA,INSURANCE"
export domain_tag=airline_media_ins60
export task_mode="dogo"
export task_tag="dogo"
export domain_classifier_task="dogo60"
export dataset_path="data/dogo60/"
export prompt_domain="AIRLINE"
export target_domain_list="MEDIA,INSURANCE"
export target_domain_str="MEDIAINSURANCE"

# example models to evaluate
redaction_model="model/redaction/dogo3dom_train60_hf_5epochs_1e-5_best.pt"
redaction_type="bert"
redaction_threshold=0.96
redaction_ngram=16
reference_model="${task_tag}_pub__${prompt_domain,,}60__20epochs__eps8__lr1e-5"

results_prefix="ctx_${prompt_domain}__redact_target_${target_domain_str}__-persist-outputs"

target_models=(
    # example models
    dogo_pvt__on_pub_redacted_clf-rob0.96__airline_media_ins60__eps8__lr5e-5
    dogo_pvt__on_pub_redacted_clf-rob0.96__airline_media_ins60__eps8__lr5e-4
)

for target_model in "${target_models[@]}"; do
    # PPL
    cmd="bash /persist/git/domain-private-transformers/training_scripts/eval_dogo.sh \
    /persist/outputs/eval/eval__${prompt_domain}__${target_model} \
    /persist/git/domain-private-transformers/${dataset_path} \
    ${task_mode} \
    ${prompt_domain} \
    /persist/outputs/${target_model} 2>&1 | tee /persist/outputs/eval/logs/eval__${prompt_domain}__${target_model}.log"
    echo ${cmd}
    eval ${cmd}

    uuid_gen=$(uuidgen)
    mia_results_prefix="${results_prefix}-${target_model}__-persist-outputs-${reference_model}__${redaction_type}_thresh${redaction_threshold}__tdc3dom__testset"
    echo "${mia_results_prefix},${uuid_gen}" >> logs/mia_${redaction_type}/${domain_classifier_task}/sorted/uuid_lookup.csv

    # MIA
    cmd2="python LM_Memorization/extraction.py \
        --use_prompts \
        --task_mode ${domain_classifier_task} \
        --ctx_domains ${prompt_domain} \
        --data_folder ${dataset_path} \
        --redaction_model ${redaction_type} \
        --redaction_model_path ${redaction_model} \
        --redaction_target_domain ${target_domain_list} \
        --redaction_model_threshold ${redaction_threshold} \
        --redaction_model_ngram_range ${redaction_ngram},${redaction_ngram} \
        --N 10 \
        --target_model /persist/outputs/${target_model} \
        --ref_model /persist/outputs/${reference_model} \
        --save_results_basename ${uuid_gen}"
    echo ${cmd2}
    eval ${cmd2}

    cmd3="python scripts/process_mia_results.py \
    --mia-results-file logs/mia_${redaction_type}/${domain_classifier_task}/results__${uuid_gen}.csv \
    --plot-file plots/mia_${redaction_type}/${domain_classifier_task}/${uuid_gen}.png \
    --ratio-file logs/mia_${redaction_type}/${domain_classifier_task}/sorted/${uuid_gen}.csv \
    --target-lm target_ppl \
    --ref-lm ref_ppl"
    echo ${cmd3}
    eval ${cmd3}
done
