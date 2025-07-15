#!/bin/bash -l

#SBATCH -o ./job.out
#SBATCH -e ./job.err

#SBATCH -D ./
#SBATCH -J db_lora_sd3
#SBATCH --nodes=1
#SBATCH --constraint="gpu"   #   providing GPUs.
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18   #   using 18 cores each.
#SBATCH --mem=125000
#SBATCH --mail-type=END
#SBATCH --mail-user=ywang@mpib-berlin.mpg.de
#SBATCH --time=04:00:00

source ~/mpib/chm-artistic-social-determinism/Art_social_determinism/art_soc/bin/activate

#export HF_HUB_OFFLINE=1
#export WANDB_MODE=offline
export WANDB_PROJECT=artistic_social_determinism
export WANDB_ENTITY=yangyud01-university-of-chicago
export MODEL_PATH=/u/wangyd/sd3-custom

CAPTIONS=/ptmp/wangyd/captions_100.jsonl
OUTDIR=/ptmp/wangyd/sd3_lora_runs/$SLURM_JOB_ID

mkdir -p "$OUTDIR"

accelerate launch --config_file /u/wangyd/.cache/huggingface/accelerate/default_config.yaml \
  ~/diffusers/examples/dreambooth/train_dreambooth_lora_sd3.py \
  --pretrained_model_name_or_path "$MODEL_PATH" \
  --dataset_name "$CAPTIONS" \
  --image_column image \
  --caption_column prompt \
  --instance_prompt "painting style <token>" \
  --output_dir "$OUTDIR" \
  --mixed_precision fp16 \
  --resolution 512 \
  --train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 4e-4 \
  --lr_scheduler constant \
  --lr_warmup_steps 0 \
  --max_train_steps 100 \
  --validation_prompt "A painting of 13 dogs running on the grass <gender_female> <year_1982> <loc_france>" \
  --validation_epochs 25 \
  --seed 42 \
  --report_to wandb \
  --train_text_encoder \
  --text_encoder_lr 5e-5
