#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=20:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_229bis: AlexNet, PRESTO complete (T1 + T2 + T3) on the frozen test_225
# recipe. NOT LAUNCHED. Parked here as the next AlexNet run, correctly dosed, so
# that the choice to spend fourteen hours on it stays a choice and not a
# rebuild. Andrea suspended AlexNet after test_229 to look at EfficientNet-B0
# first (test_230).
#
# WHY 60 EPOCHS AND NOT 20. A 20-epoch budget cannot be lossless on this recipe
# no matter what the coefficients do, measured on the recipe itself against
# 56.524 FP32:
#
#   test_223   20 epochs   55.956   -0.568   (uniform C=16, no 8-bit input convs)
#   test_224   40 epochs   56.452   -0.072
#   test_225   60 epochs   56.552   +0.028   <- frozen baseline
#
# and test_225's own epoch 20 sits at 56.108, -0.416. So a short run starts
# between two and five tenths in the hole before T2 and T3 add anything. The
# other half of the same fact: test_229 does NOT already contain the lossless
# row. Its 42.4% sparsity point is at epoch 10, where the baseline itself is
# only at 55.910; the -0.144 quoted there is against the baseline at that epoch,
# not against FP32. Only a full-length run at the right dose can put a lossless
# point on the frontier.
#
# T2 DOSE: --t2_scale 0.050, and the number means something different from the
# 0.31 of test_229. What transfers between runs is the CUMULATIVE EXPOSURE
# T2 * S_linear, not the scale: the schedule sums absorb the epoch count, the
# learning-rate shape and the duty cycle, so equal exposure means equal
# displacement. Read off test_229's own trajectory, in units of its T2:
#
#   exposure   sparsity   packet
#     31.4       42.4%     6.34%   <- epoch 10, target of this run
#     62.6       51.0%     5.41%
#     96.1       57.3%     4.80%
#    191.7       70.6%     3.66%   <- epoch 60, the whole of test_229
#
# This configuration (60 epochs, ordinary cosine, T3 on at entropy_every 4)
# has schedule sums (192.59, 192.59), so 0.050 lands at exposure 31.3-31.9 on
# the layers that carry the weight. Two traps that cost test_230-as-drafted a
# rebuild, both now handled by the code:
#   - the 20-epoch legacy schedule sums to 265.0 against 191.7 for the 60-epoch
#     ramped one. A short run delivers MORE integrated force at equal
#     coefficient, not less: the ramp throws away the early exposure and the
#     flat tail runs at lr/20.
#   - with T3 on, the whole PRESTO gradient moves into the FISTA branch and is
#     applied one step in entropy_every. _l1_schedule_sums now weights by that
#     duty cycle, so --t2_scale means the same thing with T3 on and off. Change
#     --entropy_every and the calibration follows on its own; change it against
#     an OLD script and the dose moves by the same factor.
#
# T3 DOSE: --entropy_coeff 3e-8, the value that was lossless on ResNet-18 in
# test_168 at the same T1 and the same entropy_every. Careful reading logs older
# than test_169: the names T2/T3 are printed in the opposite positions there, so
# the 168's "T2=3e-08" IS the entropy and its "T3=1e-07" is the sparsity
# (l1_push_near_zero=2e-6 = 2*sqrt(T1*T2) confirms it).
#
# ENTROPY WARMUP 0, deliberately. test_168 used one warm-up epoch to get a free
# LSQ-only control; here the control already exists as test_225, and a warm-up
# epoch would run T2 at full force and full duty for one epoch and at quarter
# duty afterwards, which is exactly the non-uniformity the schedule sums assume
# away.
#
# EXPECTED: 42% sparsity, 6.3% packet before T3 touches it, 5.4-6.0% after if
# T3 takes off the values what it took on DeiT, i.e. 16-18x against 9.30x for
# the LSQ baseline, at an accuracy deficit of order 0.15. Borderline lossless,
# which is the honest description: it is a coin flip and it costs fourteen
# hours.
#
# COST: ~2.1s per dual call measured on AlexNet (test_220: 315s/epoch with 78
# calls against ~155s of base), 313 calls per epoch here, so ~810-860s/epoch and
# 13.5-14.5h over sixty epochs. There is no resume in the trainer, only a final
# save, so it must fit in one job: hence --time 20:00:00. If that wall is not
# available, --entropy_every 8 halves the cost and the calibration rescales T2
# on its own; T3 would then run at half its anchored exposure, which breaks the
# test_168 anchor and is the reason not to do it lightly.

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
        --model_name AlexNet \
        --delta -10 \
        --gamma 2 \
        --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
        --train_workers 4 \
        --val_workers 2 \
        --batch_size 64 \
        --n_epochs 60 \
        --lr 2e-3 \
        --lr_warmup_epochs 1 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff 1e-5 \
        --entropy_coeff 3e-8 \
        --sparsity_coeff 0 \
        --layerwise_t2_targets 0.112,0.434,0.455,0.441,0.441,0.637,0.637,0.525 \
        --t2_calibration displacement \
        --t2_scale 0.050 \
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
        --bn_recalibration_batches 0 \
        --C 16 \
        --layer_C 256,256,16,16,16,16,16,16 \
        --max_iterations 3 \
        --metrics_interval 1 \
        --entropy_warmup_epochs 0 \
        --entropy_every 4 \
        --dual_step 3e-9 \
        --check_ddp_sync \
        --pretrained Y \
        > "$OUTPUT_TARGET" 2>&1
'
