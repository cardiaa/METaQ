#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=1:30:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_189: one more doubling of the weight rate.
#
# The learning rate turned out to be the second thing we had been getting wrong,
# after the missing anchor. It had been pinned at 2e-5 since tests 169 to 171
# for one reason, to limit drift damage, and once distillation removed the drift
# that constraint became a brake we carried for a dozen experiments. Raising it
# to 5e-5 moved the control from 79.592 to 79.719 over the last three epochs
# (test_188 against test_185), which is 0.023 from the 79.742 FP32 baseline and
# within one standard error of it. We now have a quantization-aware training
# procedure that reproduces the pretrained accuracy, which is the base the
# contribution has to sit on.
#
# Two points are not a response curve, so this run doubles once more. The size
# of the step is chosen by measurement resolution rather than by caution: the
# standard error of a three-epoch mean is 0.033, so distinguishing two settings
# needs about 0.07. On the observed slope, 2.5x of rate bought 0.127, a step to
# 7e-5 would buy roughly 0.05 and be unreadable, while a doubling should buy
# about 0.10 and be measurable. If it degrades instead, the optimum is bracketed
# between 5e-5 and 1e-4, which costs the same hour and is equally useful. The
# value is also the one that WAS harmful without the anchor, since test_170 ran
# at 1e-4 and ended at 77.85, so it is the sharpest available check that the
# regime has genuinely changed.
#
# Everything else is test_188: distillation at alpha 0.9, the step-size schedule
# fix, no per-channel and no eight-bit ends, both of which test_187 showed cost
# 0.65 of sparse ratio and returned nothing once the network is anchored.
# Twenty-five epochs keep the cosine identical in shape.
#
# One caution on reading it. In test_188 the block means run 79.31, 79.35,
# 79.46, 79.63, 79.71, which looks like a trajectory still climbing at epoch
# twenty-five, but between epochs 16 and 25 the rate falls thirty-eightfold and
# that rise is the cosine consolidating. The same artefact has misled us in the
# late climb of test_179, the apparent ceiling of test_178 and the compression
# plateau of test_186. Read the last-three mean, not the slope.
#
# WHAT TO READ, against the 79.719 and 11.28% of test_188: the sparse accuracy
# as a mean over the last three epochs, and the sparse ratio, which crept from
# 11.09% to 11.28% when the rate went up and will probably creep further, since
# more movement means a less concentrated weight distribution.
#
# About an hour.

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
    --model_name DeiT-Small \
    --delta -10 \
    --gamma 2 \
    --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
    --train_workers 4 \
    --val_workers 2 \
    --batch_size 128 \
    --n_epochs 25 \
    --lr 1e-4 \
    --optimizer_weight_decay 0 \
    --perspective_coeff 0 \
    --entropy_coeff 0 \
    --sparsity_coeff 0 \
    --perspective Y \
    --flat_schedule N \
    --mag_prune_ratio 0 \
    --quantization Y \
    --quantizer lsq \
    --lsq_scale_lr 1e-5 \
    --lsq_init mse \
    --lsq_grad_scaling N \
    --lsq_scale_lr_schedule Y \
    --joint_lsq_metaq Y \
    --distillation Y \
    --distill_alpha 0.9 \
    --distill_tau 1.0 \
    --bn_recalibration_batches 0 \
    --C 16 \
    --max_iterations 3 \
    --metrics_interval 1 \
    --entropy_warmup_epochs 1 \
    --entropy_every 4 \
    --dual_step 3e-9 \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
