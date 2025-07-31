#!/bin/bash -l

#SBATCH --job-name=sd3_train_lora
#SBATCH --output=/ptmp/wangyd/out/sd3_train_lora.out
#SBATCH --error=/ptmp/wangyd/err/sd3_train_lora.err
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ywang@mpib-berlin.mpg.de

# Activate env

source ~/mpib/chm-artistic-social-determinism/Art_social_determinism/art_soc/bin/activate

# # Load args from sweep_params.txt
# PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" sweep_params.txt)
# echo "Running training with: $PARAMS"

# Set paths
export WANDB_PROJECT=artistic_social_determinism
export WANDB_ENTITY=yangyud01-university-of-chicago
export MODEL_PATH=/u/wangyd/sd3-custom
export HF_DATASETS_CACHE="/ptmp/$USER/hf_cache_$SLURM_JOB_ID"

FOLDER=/u/wangyd/mpib/chm-artistic-social-determinism/Data/dreambooth_dataset
#shuf -n 1000 --random-source=<(yes 42) $CAPTIONS > /ptmp/wangyd/captions_1000.jsonl

OUTDIR=/ptmp/wangyd/sd3_lora_runs/run_lora_$SLURM_JOB_ID
mkdir -p "$OUTDIR"

# Run training
accelerate launch --config_file /u/wangyd/.cache/huggingface/accelerate/default_config.yaml \
  modified_train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path "$MODEL_PATH" \
  --dataset_name "$FOLDER" \
  --image_column image \
  --caption_column text \
  --output_dir "$OUTDIR" \
  --dataloader_num_workers 6 \
  --instance_prompt "painting" \
  --optimizer "prodigy" \
  --mixed_precision fp16 \
  --resolution 512 \
  --train_batch_size 8 \
  --gradient_accumulation_steps 2 \
  --seed 42 \
  --max_train_steps 40000 \
  --checkpointing_steps 500 \
  --validation_prompt "A painting of a swirling night sky, bright stars, crescent moon over a quiet village landscape.<loc_Saint-rémy-de-provence> <loc_France> <year_1889> <gender_male> <artist_vincent_van_gogh>" \
  --validation_epochs 25 \
  --report_to wandb \
  --train_text_encoder \
  --repeats 1 \
  --learning_rate 1. \
  --lr_scheduler constant
