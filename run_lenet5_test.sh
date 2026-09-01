#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

DELTA=$1
BATCH_SIZE=$2
T1=$3
T2=$4

if [ -z "$DELTA" ] || [ -z "$BATCH_SIZE" ]; then
  echo "Usage: sbatch run_lenet_test.sh <delta> <batch_size>"
  exit 1
fi

LOG_DIR=$WORK/acardia0/LeonardoTests
mkdir -p "$LOG_DIR"

LAST=$(find "$LOG_DIR" -maxdepth 1 -type f -name 'Leonardo_test_*.log' -printf '%f\n' | grep -oE '^Leonardo_test_[0-9]+\.log$' | grep -oE '[0-9]+' | sort -n | tail -1)

if [ -z "$LAST" ]; then
  NEXT=1
else
  NEXT=$((LAST+1))
fi

LOG_FILE=$LOG_DIR/Leonardo_test_${NEXT}.log

module load profile/deeplrn
module load cineca-ai/4.3.0

export OMP_NUM_THREADS=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONWARNINGS="ignore::UserWarning"

cd "$WORK/acardia0/METaQ"

torchrun --standalone --nproc_per_node=2 train_on_gpus.py \
  --model_name "LeNet-5" \
  --delta "$DELTA" \
  --data_root ./data \
  --batch_size "$BATCH_SIZE" \
  --T1 "$T1" \
  --T2 "$T2" \
  > "$LOG_FILE" 2>&1
