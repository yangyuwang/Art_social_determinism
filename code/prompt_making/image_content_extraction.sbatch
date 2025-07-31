#!/bin/bash -l

#SBATCH --job-name=caption_lava
#SBATCH --output=/ptmp/wangyd/out/caption_%a.out
#SBATCH --error=/ptmp/wangyd/err/caption_%a.err
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --constraint="gpu"
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ywang@mpib-berlin.mpg.de

source ~/mpib/chm-artistic-social-determinism/Art_social_determinism/art_soc/bin/activate

python image_content_extraction.py \
  /u/wangyd/mpib/chm-artistic-social-determinism/Data/dreambooth_dataset/train \
  llava-hf/llava-v1.6-mistral-7b-hf \
  --out /u/wangyd/mpib/chm-artistic-social-determinism/Data/painting_content_${SLURM_ARRAY_TASK_ID}.jsonl \
  --lengths 40 32 24 16 8
