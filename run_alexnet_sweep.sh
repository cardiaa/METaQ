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

# First untuned METaQ sweep point for pretrained ImageNet AlexNet.
# It deliberately reuses the cross-architecture regularizer recipe (C=16,
# T1/T2/T3 = 1e-5/1e-7/3e-8) but keeps AlexNet's pretrained fine-tuning LR.
# Pass the number of epochs as the first sbatch argument: 3 for calibration,
# 20 for the full run. The second argument overrides entropy_coeff and the
# third overrides the per-GPU batch size (default 128, global 2048 on 16 GPUs).

N_EPOCHS=${1:-20}
ENTROPY_COEFF=${2:-5e-8}
BATCH_SIZE=${3:-128}
DISTILLATION=${4:-N}
NO_METAQ=${5:-N}
C_LEVELS=${6:-16}
LSQ_PER_CHANNEL=${7:-N}
if [ "$NO_METAQ" = "Y" ]; then
    PERSPECTIVE_COEFF=0
    ENTROPY_COEFF=0
    SPARSITY_COEFF=0
else
    PERSPECTIVE_COEFF=1e-5
    SPARSITY_COEFF=1e-7
fi
if [ "$DISTILLATION" = "Y" ]; then
    DISTILL_ALPHA=0.5
else
    DISTILL_ALPHA=0.0
fi

LOG_DIR=$WORK/acardia0/LeonardoTests
mkdir -p "$LOG_DIR"
LAST=$(find "$LOG_DIR" -maxdepth 1 -type f -name 'Leonardo_test_*.log' -printf '%f\n' | grep -oE '^Leonardo_test_[0-9]+\.log$' | grep -oE '[0-9]+' | sort -n | tail -1)
NEXT=$(( ${LAST:-0} + 1 ))
LOG_FILE=$LOG_DIR/Leonardo_test_${NEXT}.log
export LOG_FILE

module load profile/deeplrn
module load cineca-ai/4.3.0

METAQ_PYTHON=$WORK/acardia0/venvs/metaq/bin/python
if [ ! -x "$METAQ_PYTHON" ]; then
    echo "metaq environment not found: $METAQ_PYTHON" > "$LOG_FILE"
    exit 1
fi
export METAQ_PYTHON
export OMP_NUM_THREADS=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONWARNINGS="ignore::UserWarning"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

cd "$WORK/acardia0/METaQ"
if [ "$NO_METAQ" = "Y" ]; then
    mkdir -p "$WORK/acardia0/METaQCheckpoints/AlexNet"
    export METAQ_CHECKPOINT_PATH="$WORK/acardia0/METaQCheckpoints/AlexNet/AlexNetCheckpoint1.pth"
else
    unset METAQ_CHECKPOINT_PATH
fi

srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 bash -lc '
    if [ "$SLURM_NODEID" -eq 0 ]; then OUTPUT_TARGET="$LOG_FILE"; else OUTPUT_TARGET=/dev/null; fi
    "$METAQ_PYTHON" -m torch.distributed.run \
        --nnodes=$SLURM_JOB_NUM_NODES \
        --nproc_per_node=4 \
        --node_rank=$SLURM_NODEID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        --rdzv_id=$SLURM_JOB_ID \
        train_on_gpus_pretrained.py \
        --model_name AlexNet \
        --delta -10 \
        --gamma 2 \
        --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
        --train_workers 4 \
        --val_workers 2 \
        --batch_size '"$BATCH_SIZE"' \
        --n_epochs '"$N_EPOCHS"' \
        --lr 1e-4 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff '"$PERSPECTIVE_COEFF"' \
        --entropy_coeff '"$ENTROPY_COEFF"' \
        --sparsity_coeff '"$SPARSITY_COEFF"' \
        --perspective Y \
        --flat_schedule N \
        --mag_prune_ratio 0 \
        --quantization Y \
        --quantizer lsq \
        --lsq_scale_lr 1e-5 \
        --lsq_init mse \
        --lsq_grad_scaling N \
        --lsq_per_channel '"$LSQ_PER_CHANNEL"' \
        --joint_lsq_metaq Y \
        --distillation '"$DISTILLATION"' \
        --distill_alpha '"$DISTILL_ALPHA"' \
        --distill_tau 1.0 \
        --bn_recalibration_batches 50 \
        --C '"$C_LEVELS"' \
        --max_iterations 3 \
        --metrics_interval 1 \
        --entropy_warmup_epochs 1 \
        --entropy_every 4 \
        --dual_step 3e-9 \
        --check_ddp_sync \
        --pretrained Y \
        > "$OUTPUT_TARGET" 2>&1
'
