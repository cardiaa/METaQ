#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# First untuned METaQ sweep point for pretrained ImageNet AlexNet.
# It deliberately reuses the cross-architecture regularizer recipe (C=16,
# T1/T2/T3 = 1e-5/1e-7/3e-8) but keeps AlexNet's pretrained fine-tuning LR.
# Arguments 17--19 split the run into diagnostic, common-ramp, and common-flat
# phases and must sum to argument 1. Argument 20 optionally supplies one target
# z-sparsity per quantized tensor; T2_l is calibrated from the corresponding
# |w_l| quantile after the diagnostic phase. Argument 21 couples the LSQ-scale
# learning-rate schedule to the weight schedule.
# Test_219 configuration: A=30, B=1, C=24, D=5, with targets at 90% of the
# test_218 values.
# Argument 22 explicitly controls optimizer weight decay; use 0 to isolate the
# METaQ T1 ridge from the optimizer's independent ridge term.

N_EPOCHS=${1:-20}
ENTROPY_COEFF=${2:-5e-8}
BATCH_SIZE=${3:-128}
DISTILLATION=${4:-N}
NO_METAQ=${5:-N}
C_LEVELS=${6:-16}
LSQ_PER_CHANNEL=${7:-N}
SAVE_CHECKPOINT=${8:-N}
PRETRAINED_CHECKPOINT=${9:-/leonardo_work/IscrC_ObCTDoNN/acardia0/alexnet_checkpoints/alexnet-owt-7be5be79.pth}
CHECKPOINT_OUTPUT_NAME=${10:-AlexNetCheckpoint1.pth}
LR=${11:-1e-4}
FLAT_SCHEDULE=${12:-N}
ENTROPY_WARMUP=${13:-1}
Z_PRUNING=${14:-N}
SPARSITY_COEFF=${15:-1e-7}
DISTILL_ALPHA_OVERRIDE=${16:-0.5}
DIAGNOSTIC_EPOCHS=${17:-0}
METAQ_RAMP_EPOCHS=${18:-0}
METAQ_FLAT_EPOCHS=${19:-$N_EPOCHS}
LAYERWISE_T2_TARGETS=${20:-}
LSQ_SCALE_LR_SCHEDULE=${21:-N}
OPTIMIZER_WEIGHT_DECAY=${22:-1e-4}

if [ "$N_EPOCHS" -ne $((DIAGNOSTIC_EPOCHS + METAQ_RAMP_EPOCHS + METAQ_FLAT_EPOCHS)) ]; then
    echo "Invalid epoch schedule: total=$N_EPOCHS but diagnostic=$DIAGNOSTIC_EPOCHS + metaq_ramp=$METAQ_RAMP_EPOCHS + metaq_flat=$METAQ_FLAT_EPOCHS" >&2
    exit 2
fi
if [ "$NO_METAQ" = "Y" ]; then
    PERSPECTIVE_COEFF=0
    ENTROPY_COEFF=0
    SPARSITY_COEFF=0
else
    PERSPECTIVE_COEFF=1e-5
fi
if [ "$DISTILLATION" = "Y" ]; then
    DISTILL_ALPHA=$DISTILL_ALPHA_OVERRIDE
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
if [ "$NO_METAQ" = "Y" ] || [ "$SAVE_CHECKPOINT" = "Y" ]; then
    mkdir -p "$WORK/acardia0/METaQCheckpoints/AlexNet"
    export METAQ_CHECKPOINT_PATH="$WORK/acardia0/METaQCheckpoints/AlexNet/$CHECKPOINT_OUTPUT_NAME"
else
    unset METAQ_CHECKPOINT_PATH
fi

srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 bash -lc '
    if [ "$SLURM_NODEID" -eq 0 ]; then OUTPUT_TARGET="$LOG_FILE"; else OUTPUT_TARGET=/dev/null; fi
    export METAQ_CHECKPOINT_PATH="'"$METAQ_CHECKPOINT_PATH"'"
    if [ "$SLURM_NODEID" -eq 0 ]; then
        echo "[CHECKPOINT TARGET] ${METAQ_CHECKPOINT_PATH:-disabled}" >> "$OUTPUT_TARGET"
    fi
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
        --diagnostic_epochs '"$DIAGNOSTIC_EPOCHS"' \
        --metaq_ramp_epochs '"$METAQ_RAMP_EPOCHS"' \
        --metaq_flat_epochs '"$METAQ_FLAT_EPOCHS"' \
        --layerwise_t2_targets '"$LAYERWISE_T2_TARGETS"' \
        --lr '"$LR"' \
        --optimizer_weight_decay '"$OPTIMIZER_WEIGHT_DECAY"' \
        --perspective_coeff '"$PERSPECTIVE_COEFF"' \
        --entropy_coeff '"$ENTROPY_COEFF"' \
        --sparsity_coeff '"$SPARSITY_COEFF"' \
        --perspective Y \
        --flat_schedule '"$FLAT_SCHEDULE"' \
        --mag_prune_ratio 0 \
        --z_pruning '"$Z_PRUNING"' \
        --quantization Y \
        --quantizer lsq \
        --lsq_scale_lr 1e-5 \
        --lsq_init mse \
        --lsq_grad_scaling N \
        --lsq_per_channel '"$LSQ_PER_CHANNEL"' \
        --lsq_scale_lr_schedule '"$LSQ_SCALE_LR_SCHEDULE"' \
        --joint_lsq_metaq Y \
        --distillation '"$DISTILLATION"' \
        --distill_alpha '"$DISTILL_ALPHA"' \
        --distill_tau 1.0 \
        --bn_recalibration_batches 50 \
        --C '"$C_LEVELS"' \
        --max_iterations 3 \
        --metrics_interval 1 \
        --entropy_warmup_epochs '"$ENTROPY_WARMUP"' \
        --entropy_every 4 \
        --dual_step 3e-9 \
        --check_ddp_sync \
        --pretrained Y \
        --pretrained_checkpoint '"$PRETRAINED_CHECKPOINT"' \
        >> "$OUTPUT_TARGET" 2>&1
'
