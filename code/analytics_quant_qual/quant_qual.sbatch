#!/bin/bash -l

#SBATCH --job-name=sd3_eval_ckpts
#SBATCH --output=/ptmp/wangyd/out/sd3_eval_%A_%a.out
#SBATCH --error=/ptmp/wangyd/err/sd3_eval_%A_%a.err
#SBATCH --array=0-15
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

# Paths
BASE=/u/wangyd/sd3-custom
LORA_ROOT=/u/wangyd/models/prodigy_running
JSONL=/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/test_data.jsonl

# Checkpoints every 2500 steps from 2500 to 40000
CHECKPOINT_STEPS=(2500 5000 7500 10000 12500 15000 17500 20000 22500 25000 27500 30000 32500 35000 37500 40000)

# Select checkpoint based on SLURM task ID
STEP=${CHECKPOINT_STEPS[$SLURM_ARRAY_TASK_ID]}
LORA_DIR=${LORA_ROOT}/checkpoint-${STEP}
OUT_DIR="/raven/u/wangyd/mpib/chm-artistic-social-determinism/Data/test/step${STEP}"

echo "Evaluating checkpoint at step ${STEP}"

# Run evaluation
python quant_qual.py \
  --base_model "$BASE" \
  --lora_dir "$LORA_DIR" \
  --jsonl "$JSONL" \
  --out_root "$OUT_DIR" \
  --steps 25 \
  --seed 42
