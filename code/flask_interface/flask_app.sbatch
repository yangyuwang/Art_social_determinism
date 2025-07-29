#!/bin/bash -l

#SBATCH --job-name=flask
#SBATCH --output=/ptmp/wangyd/out/flask.out
#SBATCH --error=/ptmp/wangyd/err/flask.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=125000
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=ywang@mpib-berlin.mpg.de

# Activate environment
source ~/mpib/chm-artistic-social-determinism/Art_social_determinism/art_soc/bin/activate

# Run flask
python /u/wangyd/mpib/chm-artistic-social-determinism/Art_social_determinism/code/flask_interface/app.py \
  --token_dict /u/wangyd/mpib/chm-artistic-social-determinism/Data/natural_prompt_dataset/special_token_dict.json \
  --lora_dir /u/wangyd/mpib/chm-artistic-social-determinism/Art_social_determinism/model/checkpoint-22500 \
  --output_dir /u/wangyd/mpib/chm-artistic-social-determinism/Data/website_generate
