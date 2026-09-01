#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

DELTA=$1
BATCH_SIZE=$2
T1=$3
T2=$4
EPOCH_FRACTION=$5	

if [ -z "$DELTA" ] || [ -z "$BATCH_SIZE" ] || [ -z "$T1" ] || [ -z "$T2" ] || [ -z "$EPOCH_FRACTION" ]; then
  echo "Usage: sbatch run_alexnet_test.sh <delta> <batch_size> <T1> <T2> <epoch_fraction>"
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

export PYTHONPATH="$HOME/.pydeps/cineca-ai43:${PYTHONPATH}"
export OMP_NUM_THREADS=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONWARNINGS="ignore::UserWarning"

export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

cd "$WORK/acardia0/METaQ"

srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 bash -lc '
if [ "$SLURM_NODEID" -eq 0 ]; then
  torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    train_on_gpus.py \
    --model_name AlexNet \
    --delta '"$DELTA"' \
    --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
    --train_workers 4 \
    --val_workers 2 \
    --batch_size '"$BATCH_SIZE"' \
    --T1 '"$T1"' \
    --T2 '"$T2"' \
    --epoch_fraction '"$EPOCH_FRACTION"' \
    --n_epochs 10 \
    --lr 1e-4 \
    --max_iterations 3 \
    --metrics_interval 1 \
    --entropy_warmup_epochs 1 \
    --entropy_every 4 \
    --check_ddp_sync \
    > '"$LOG_FILE"' 2>&1
else
  torchrun \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    train_on_gpus.py \
    --model_name AlexNet \
    --delta '"$DELTA"' \
    --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
    --train_workers 4 \
    --val_workers 2 \
    --batch_size '"$BATCH_SIZE"' \
    --T1 '"$T1"' \
    --T2 '"$T2"' \
    --epoch_fraction '"$EPOCH_FRACTION"' \
    --n_epochs 10 \
    --lr 1e-4 \
    --max_iterations 3 \
    --metrics_interval 1 \
    --entropy_warmup_epochs 1 \
    --entropy_every 4 \
    --check_ddp_sync \
    > /dev/null 2>&1
fi
'
