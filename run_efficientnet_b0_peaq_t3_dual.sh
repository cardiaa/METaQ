#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_234: EfficientNet-B0, does the entropy channel open at all once the DUAL
# is allowed to converge. A mechanism test, not a frontier trace, and it is the
# run that test_232 and test_233 should have been.
#
# WHAT THOSE TWO ESTABLISHED. They differ only in entropy_coeff, 3e-7 against
# 3e-6, and:
#   - their beta* norms at epoch one agree to four significant figures,
#     2.756366e-04 against 2.756720e-04;
#   - the values stream compressed by -0.0887 and -0.0735 over four epochs,
#     against -0.0992 for the SAME recipe with the entropy term switched off;
#   - xi covered 25-42% of its useful range in test_232 and 4% in test_233.
# Ten times the coefficient, no change in the applied gradient, and less
# compression than not running the term at all.
#
# WHY. entropy_coeff reaches the weights through exactly one channel: the dual
# xi. The per-weight subproblem is solved against the RAW bucket costs xi_b --
# knapsack_perspective_leonardo states it, "entropy_coeff does NOT scale the
# bucket costs of the x-subproblem" -- so the entropy term is exactly as strong
# as the dual is converged and no stronger. And the dual ascends by
# xi += dual_step * g with g normalized to [-1,1] (the test_134 fix), so an
# ABSOLUTE dual_step moves xi a fixed distance per iteration, while the range it
# must cover is xi_hi - xi_lo = entropy_coeff * log2(upper_c/lower_c), which is
# PROPORTIONAL to entropy_coeff. Raising the coefficient pushes the target away
# exactly as fast as it raises it. Traversals of the dual range per epoch:
#
#   test_168  ResNet-18, lossless   0.195
#   test_220  AlexNet, T3 alone     0.0135
#   test_232  EffNet                0.0050
#   test_233  EffNet                0.0005    <- 386x slower than test_168
#
# Two compounding causes, and only one of them is the coefficient: the other is
# that global batch 4096 buys speed with steps, so this network gets 78 dual
# calls per epoch against test_168's 313. The batch decision that made T3
# affordable here is the same one that starved its dual.
#
# THE FIX IS IN THE CODE, not in this script: --dual_step_mode relative makes
# --dual_step a fraction of the dual range per iteration, so it follows
# entropy_coeff, the layer size and the call rate on its own. Absolute remains
# the default and every previous run is bit-identical.
# --dual_step 1.3e-2 relative reproduces test_168's 0.195 traversals per epoch
# at 78 calls; scratchpad/verify_dual_step_scaling.py derives it and checks that
# the resulting rate no longer depends on entropy_coeff.
#
# WHY A FIXED T3 AND NOT A RAMP. A ramp moves the target the dual is chasing,
# and worst exactly where it is smallest: from epoch one to two a linear ramp
# DOUBLES the dual range. Ramps are usable again now that the step follows the
# coefficient, but not before the mechanism is confirmed once. So: T3 fixed at
# 3e-8, the value that was lossless on ResNet-18, with the trainer's own
# exponential settle over the first four epochs.
#
# TEN EPOCHS, and the schedule is byte-for-byte test_230's: same cosine, same
# warm-up, no --min_lr. Every epoch therefore has a paired control from a run
# whose only difference is that the entropy term is off, which is the strongest
# form this measurement can take and also the cheapest.
#
# WHAT COUNTS AS THE ANSWER. values_zstd_ratio against test_230's 9.0936,
# 9.0579, 9.0217, 8.9944, 8.9779, 8.9637, 8.9523, 8.9482, 8.9472, 8.9412, which
# is a decay of only 0.15 over the whole run and flat from epoch seven on -- so
# anything that keeps falling after epoch seven is the entropy term. Separation
# by epoch four or five means the channel opens and the dose can then be traced
# with a ramp. No separation, with xi now covering its range, means the entropy
# term does not pay on this network at any dose the dual can reach, and THAT is
# a result worth having: the EfficientNet row becomes T2-only at 23-27%
# sparsity, --sparsity_coeff 2.28e-7 to 3.16e-7 fixed over twenty epochs, and
# the paper says plainly that the entropy channel is network-dependent.
#
# WATCH xi FIRST, before the compression. xi_pinned_frac at zero with the spread
# growing toward entropy_coeff*29 means converging. xi_pinned_frac heading to
# one means the step is now too large and the dual is chattering against its
# bounds, which is the test_133 failure in the other direction: kill it and
# halve --dual_step. That check is available on epoch one.
#
# COST: 365s/epoch measured, ten epochs, about an hour.

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
        --model_name EfficientNet-B0 \
        --delta -10 \
        --gamma 2 \
        --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
        --train_workers 4 \
        --val_workers 2 \
        --batch_size 256 \
        --n_epochs 10 \
        --lr 2e-2 \
        --lr_warmup_epochs 2 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff 1e-5 \
        --entropy_coeff 3e-8 \
        --sparsity_coeff 0 \
        --perspective Y \
        --flat_schedule N \
        --mag_prune_ratio 0 \
        --quantization Y \
        --quantizer lsq \
        --lsq_scale_lr 1e-3 \
        --lsq_scale_lr_mode relative \
        --lsq_scale_lr_schedule Y \
        --lsq_init mse \
        --lsq_grad_scaling N \
        --lsq_per_channel Y \
        --joint_lsq_metaq Y \
        --distillation Y \
        --distill_alpha 0.5 \
        --distill_tau 1.0 \
        --bn_recalibration_batches 50 \
        --C 16 \
        --max_iterations 3 \
        --metrics_interval 1 \
        --entropy_warmup_epochs 0 \
        --entropy_every 4 \
        --dual_step 1.3e-2 \
        --dual_step_mode relative \
        --check_ddp_sync \
        --pretrained Y \
        > "$OUTPUT_TARGET" 2>&1
'
