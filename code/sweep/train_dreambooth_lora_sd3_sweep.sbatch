#!/bin/bash -l

#SBATCH --job-name=sd3_sweep
#SBATCH --output=/ptmp/wangyd/out/job_%A_%a.out
#SBATCH --error=/ptmp/wangyd/err/job_%A_%a.err
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ywang@mpib-berlin.mpg.de

# Activate env

source ~/mpib/chm-artistic-social-determinism/Art_social_determinism/art_soc/bin/activate

# Load args from sweep_params.txt
PARAMS=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" sweep_params.txt)
echo "Running training with: $PARAMS"

# Set paths
export WANDB_PROJECT=artistic_social_determinism
export WANDB_ENTITY=yangyud01-university-of-chicago
export MODEL_PATH=/u/wangyd/sd3-custom

FOLDER=/u/wangyd/mpib/chm-artistic-social-determinism/Data/dreambooth_dataset
#shuf -n 1000 --random-source=<(yes 42) $CAPTIONS > /ptmp/wangyd/captions_1000.jsonl

OUTDIR=/ptmp/wangyd/sd3_lora_runs/array_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$OUTDIR"

# Run training
accelerate launch --config_file /u/wangyd/.cache/huggingface/accelerate/default_config.yaml \
  ~/diffusers/examples/dreambooth/modified_train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path "$MODEL_PATH" \
  --dataset_name "$FOLDER" \
  --image_column image \
  --caption_column text \
  --output_dir "$OUTDIR" \
  --dataloader_num_workers 6 \
  --instance_prompt "painting <token>" \
  --mixed_precision fp16 \
  --resolution 512 \
  --train_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --lr_scheduler cosine \
  --lr_warmup_steps 800 \
  --max_train_steps 19000 \
  --validation_prompt "A painting of a swirling night sky, bright stars, crescent moon over a quiet village landscape <loc_Saint-rémy-de-provence> <loc_France> <year_1889> <gender_male> <artist_vincent_van_gogh>" \
  --validation_epochs 5 \
  --report_to wandb \
  --train_text_encoder \
  --repeats 1 \
  $PARAMS
