# domain-private-transformers
Repository for the paper [Domain Private Transformers for Multi-Domain Dialog Systems](https://arxiv.org/abs/2305.14208), in Findings of EMNLP 2023

This merges several repos for langauge model training:
- [lm_privacy](https://github.com/wyshi/lm_privacy)
- huggingface langauge-modeling ([example](https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_clm_no_trainer.py))
- [private-transformers](https://github.com/lxuechen/private-transformers)

## Installation

This requires python 3.9.
```bash
conda create -n <name_env> python=3.9

# Get submodules
git submodule update --init --recursive

# Install other requirements (or requirements-macos.txt)
pip install -r requirements-linux-gpu.txt

# Install submodules and/or their requirements
pip uninstall private-transformers
cd private-transformers
pip install -e .
cd ..
```

## Training a transformer on MultiDoGO dataset

```bash
# example script to train models
bash scripts/train_evaluate_multidogo_models.sh
```

Download MultiDomain Goal Oriented Dialogues (MultiDoGO) dataset [link](https://github.com/awslabs/multi-domain-goal-oriented-dialogues-dataset/)
at `../` location (if at another path, use that path).

0. Use a redaction function to redact data.
```bash
PYTHONPATH="." python scripts/run_redaction.py \
        --clf_model hf --clf_path model/redaction/dogo3dom_train60_hf_5epochs_1e-5_best.pt \
        --clf_threshold 0.96 --clf_ngram_range 16,16 \
        --output_folder_name redacted-clf0.96-dogo60-3dom-rob
```

1. Preprocess data and create splits for DOGO dataset (other default values of 
params are reasonable).

```bash
# This cmd loads all <domain>.tsv files in the input dir and creates split 
# folders in output_dir

# Create splits for unannotated data
PYTHONPATH="." python training_scripts/preprocess_dogo.py \
  --input_dir "../multi-domain-goal-oriented-dialogues-dataset/data/unannotated" \
  --output_dir "data/dogo/"
  --use_domain_tokens True \
  --valid_frac 0.2 \
  --test_frac 0.2

# Create splits for redacted data
PYTHONPATH="." python training_scripts/preprocess_dogo.py \
  --split_convos_info_path data/dogo60__split_convos_info.json \
  --task_str dogo \
  --input_dir ../multi-domain-goal-oriented-dialogues-dataset/data/redacted-clf0.96-dogo60-3dom-rob \
  --output_dir data/dogo-redacted-clf0.96-dogo60-3dom-rob/ \
  --use_domain_tokens True \
  --valid_frac 0.2 \
  --test_frac 0.2

```

2. Run the "exp_concave" redaction schedule with this redacted data. Other 
   options are "step", "linear", "exp_convex". The script loops over a few 
   learning rates, you can edit the script to only do one learning rate

```bash
# This cmd loads the AIRLINE and MEDIA data (from the non-redacted in data/dogo 
# and redacted in data/dogo-redacted-clf0.9)

bash training_scripts/train_pvt_redacted_schedule.sh \ 
  outputs/dogo_pvt_sch_redacted_clf-rob0.96_exp_concave__airline_media__eps8 \ 
  data/dogo \
  data/dogo-redacted-clf0.96-dogo60-3dom-rob/ \
  "dogo" \
  "AIRLINE,MEDIA" \
  "exp_concave"
```

3. Get test ppl on data/dogo for the trained model
```bash
bash training_scripts/eval_dogo.sh \
  outputs/eval__dogo_pvt_sch_redacted_clf-rob0.96_exp_concave__airline_media__eps8__lr5e-5 \
  data/dogo \
  "dogo" \
  "AIRLINE,MEDIA" \
  outputs/dogo_pvt_sch_redacted_clf-rob0.96_exp_concave__airline_media__eps8__lr5e-5
```

4. Train public model on only airline data (the name says 
   w_redacted_token because the redacted token was added to the tokenizer. but 
   don't worry, only non-redacted data was used. I needed to add the redacted 
   token to all models to keep the tokenizer vocab same)

```bash
bash training_scripts/train_pub.sh \ 
  outputs/dogo_pub_w_redacted_tok__airline \
  data/dogo \
  "dogo" \
  "AIRLINE"
```

5. Run Membership Inference Attack (MIA) with `keyword` classifier 
   on the target model using (4.) as ref model. Use `AIRLINE` contexts 
   (prompts) and `MEDIA` domain for checking the generations. So we prompt 
   with `AIRLINE` data and check if model leaks other domain info.

```bash
PYTHONPATH="." python LM_Memorization/extraction.py \
  --use_prompts \
  --task_mode "dogo" \
  --ctx_domains "AIRLINE" \
  --data_folder data/dogo \
  --redaction_model "keyword" \
  --redaction_target_domain "MEDIA" \
  --N 10 \
  --target_model outputs/dogo_pvt_sch_redacted_clf-rob0.96_exp_concave__airline_media__eps8__lr5e-5 \
  --ref_model outputs/dogo_pub_w_redacted_tok__airline__lr5e-4

PYTHONPATH="." python LM_Memorization/extraction.py \
  --use_prompts \
  --task_mode "dogo" \
  --ctx_domains "AIRLINE" \
  --data_folder data/dogo \
  --redaction_model "bert" \
  --redaction_model_path model/redaction/dogo3dom_train60_hf_5epochs_1e-5_best.pt \
  --redaction_model_threshold 0.96 \
  --redaction_model_ngram_range 16,16 \      
  --redaction_target_domain "MEDIA" \
  --N 10 \
  --target_model outputs/dogo_pvt_sch_redacted_clf-rob0.96_exp_concave__airline_media__eps8__lr5e-5 \
  --ref_model outputs/dogo_pub_w_redacted_tok__airline__lr5e-4
  
# Saves logs in logs/mia_bert/
```

## Citation
```
@inproceedings{kabra2023domain,
    title={Domain Private Transformers for Multi-Domain Dialog Systems}, 
    author={Anmol Kabra and Ethan R. Elenberg},
    booktitle={Findings of the Association for Computational Linguistics: EMNLP 2023},
    year={2023},
}
```
