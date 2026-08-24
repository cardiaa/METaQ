#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Test 203 ablation: twice the dual iterations used in test 199.

# test_251: ViT-B/16, METaQ. Same script as the control with the three
# coefficients on, to be launched only after test_250 has confirmed the recipe.
#
# COEFFICIENTS FROM DeiT-Small, which is the closest network we have calibrated:
# T1 1e-5, T2 5e-8, T3 1.5e-8, the knee of that frontier. They are a transfer
# guess and absolute coefficients have not transferred between distant
# architectures before, so this is a scouting run and its dose will move.
#
# --dual_step 1.3e-2 relative, and this part does transfer exactly: in relative
# mode the convergence rate is dual_step * (1/C) * N_dual * calls_per_epoch, and
# at batch 64 this network has 1251 steps and entropy_every 4, hence the same 313
# calls as ResNet-18 and ResNet-50.
#
# COST: the dual solver is dominated by per-tensor overhead rather than by weight
# count, measured at 0.93s per call on ResNet-18 with 21 tensors and 2.69s with
# ResNet-50's 54. ViT-B/16 has about 50 quantized tensors, so expect a per-call
# cost near ResNet-50's and roughly 1400-1600s per epoch in total. Ten epochs is
# then four to five hours; the eight-hour wall is slack on an estimate that has
# never been measured on a transformer of this size.
#
# WHAT TO COMPARE AGAINST. NNCodec reports ViT-B/16 at 32.87MB of 346.27MB, that
# is 9.49 per cent or 10.5x, with the full-precision accuracy given as 81.07. The
# accuracy at the compressed size has to be taken from their paper before it goes
# in our table; do not quote a compression figure of theirs without it.

LOG_DIR=$WORK/acardia0/LeonardoTests
mkdir -p "$LOG_DIR"

LAST=$(find "$LOG_DIR" -maxdepth 1 -type f -name 'Leonardo_test_*.log' -printf '%f\n' | grep -oE '^Leonardo_test_[0-9]+\.log$' | grep -oE '[0-9]+' | sort -n | tail -1)

if [ -z "$LAST" ]; then
    NEXT=1
else
    NEXT=$((LAST+1))
fi

LOG_FILE=$LOG_DIR/Leonardo_test_${NEXT}.log
export LOG_FILE

module load profile/deeplrn
module load cineca-ai/4.3.0

METAQ_PYTHON=$WORK/acardia0/venvs/metaq/bin/python
if [ ! -x "$METAQ_PYTHON" ]; then
    echo "DeiT-Small environment not found: $METAQ_PYTHON" > "$LOG_FILE"
    exit 1
fi
export METAQ_PYTHON

export OMP_NUM_THREADS=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONWARNINGS="ignore::UserWarning"
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

cd "$WORK/acardia0/METaQ"

srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 bash -lc '
    if [ "$SLURM_NODEID" -eq 0 ]; then
        OUTPUT_TARGET="$LOG_FILE"
    else
        OUTPUT_TARGET=/dev/null
    fi

    "$METAQ_PYTHON" -m torch.distributed.run \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$SLURM_JOB_ID \
    train_on_gpus_pretrained.py \
    --model_name ViT-B-16 \
    --delta -10 \
    --gamma 2 \
    --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
    --train_workers 4 \
    --val_workers 2 \
    --batch_size 64 \
    --n_epochs 10 \
    --lr 5e-5 \
    --optimizer_weight_decay 0 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 1.5e-8 \
    --sparsity_coeff 5e-8 \
    --perspective Y \
    --flat_schedule N \
    --mag_prune_ratio 0 \
    --quantization Y \
    --quantizer lsq \
    --lsq_scale_lr 1e-5 \
    --lsq_init mse \
    --lsq_grad_scaling N \
    --lsq_scale_lr_schedule Y \
    --min_lr 2.5e-6 \
    --joint_lsq_metaq Y \
    --distillation Y \
    --distill_alpha 0.9 \
    --distill_tau 1.0 \
    --bn_recalibration_batches 0 \
    --C 16 \
    --max_iterations 6 \
    --metrics_interval 1 \
    --entropy_warmup_epochs 1 \
    --entropy_every 4 \
    --dual_step 1.3e-2 \
    --dual_step_mode relative \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
