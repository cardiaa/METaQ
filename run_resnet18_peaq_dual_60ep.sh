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

# test_238: ResNet-18 at T3=1e-7 with sixty epochs. ONE flag differs from
# test_237: --n_epochs 40 -> 60. This is the run the whole ResNet-18 sequence has
# been pointing at.
#
#   test_174   LSQ only,       20 ep   69.888  +0.154   9.34%   10.71x
#   test_236   T3=3e-8,        40 ep   70.080  +0.346   5.73%   17.45x
#   test_237   T3=1e-7,        40 ep   69.406  -0.328   4.21%   23.75x
#
# WHY SIXTY. Every run in this sequence ended with both axes still moving, and
# test_237 more than any of them: over its last five epochs it gained 0.069
# accuracy points per epoch, against 0.047 for test_236, while its packet was
# still falling. It needs 0.33 points to reach the checkpoint and it was earning
# them at a rate that closes the gap in five epochs. A sixty-epoch cosine also
# holds the learning rate higher for longer than a forty-epoch one, so the extra
# epochs buy healing and compression at the same time rather than trading them.
#
# WHAT IT WOULD MEAN. Lossless at roughly 4.1% of FP32 is about 24x, which is
# 1.9MB against the 2.5MB of HEMP+LOBSTER \citep{tartaglione2021hemp} at a
# comparable accuracy deficit. That is a strict win on both axes with one method
# where they stack two, and it would replace test_236 as the headline row.
#
# WHAT IT WOULD MEAN IF IT FAILS. The accuracy plateaus below the checkpoint and
# the run becomes a better frontier point than test_237, at maybe 4.0% and -0.2.
# The lossless headline then stays test_236 at 17.45x and the frontier gains a
# fourth measured point. Either outcome is reportable; only the headline changes.
#
# NOTHING ELSE MOVES. T3 stays at 1e-7 and the dual step at 1.3e-2 relative.
# test_237 showed the relative step handles the higher coefficient exactly as
# designed: its xi_pinned_frac peaked at 0.0034, LOWER than the 0.148 of
# test_236 at a third of the coefficient, because the step scales with the
# interval it has to cross.
#
# COST: 432s/epoch measured, so about seven hours and a quarter for sixty
# epochs. The twelve-hour wall is slack; there is no resume in the trainer.

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
    --model_name ResNet-18 \
    --delta -10 \
    --gamma 2 \
    --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
    --train_workers 4 \
    --val_workers 2 \
    --batch_size 64 \
    --n_epochs 60 \
    --lr 1e-2 \
    --optimizer_weight_decay 1e-4 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 1e-7 \
    --sparsity_coeff 1e-7 \
    --perspective Y \
    --flat_schedule N \
    --mag_prune_ratio 0 \
    --quantization Y \
    --quantizer lsq \
    --lsq_scale_lr 1e-5 \
    --lsq_scale_lr_schedule Y \
    --min_lr 1e-4 \
    --lsq_init mse \
    --lsq_grad_scaling N \
    --joint_lsq_metaq Y \
    --bn_recalibration_batches 50 \
    --C 16 \
    --max_iterations 3 \
    --metrics_interval 1 \
    --entropy_warmup_epochs 1 \
    --entropy_every 4 \
    --dual_step 1.3e-2 \
    --dual_step_mode relative \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
