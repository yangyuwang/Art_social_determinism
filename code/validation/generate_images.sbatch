#!/bin/bash -l

#SBATCH --job-name=sd3_sweep
#SBATCH --output=/ptmp/wangyd/out/job_%A.out
#SBATCH --error=/ptmp/wangyd/err/job_%A.err
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120000
#SBATCH --time=1:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=ywang@mpib-berlin.mpg.de

# Activate env

source ~/mpib/chm-artistic-social-determinism/Art_social_determinism/art_soc/bin/activate

# Set paths
export WANDB_PROJECT=artistic_social_determinism
export WANDB_ENTITY=yangyud01-university-of-chicago
export MODEL_PATH=/u/wangyd/sd3-custom

PROMPT_FILE=/u/wangyd/mpib/chm-artistic-social-determinism/Art_social_determinism/examples/prompts.txt
INFER_DIR=/u/wangyd/mpib/chm-artistic-social-determinism/Data/inference
mkdir -p "$INFER_DIR"

LORADIR=/ptmp/wangyd/sd3_lora_runs/array_20468420_6

# Generate images
python generate_images.py \
  --base_model "$MODEL_PATH" \
  --lora_dir   "$LORADIR" \
  --prompts    "$PROMPT_FILE" \
  --out_dir    "$INFER_DIR"