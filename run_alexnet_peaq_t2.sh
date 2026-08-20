#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_227: AlexNet, second rung of the ladder, with the T2 calibration fixed.
# PEAQ with the sparsity term T2 only; the entropy term T3 stays off.
#
# test_226 ran this exact configuration and was INERT. At epoch 39 of 60, with
# the ramp at 96.7% of full dose, it had 17.12% sparsity against the 16.11% the
# same recipe reaches with T2 switched off entirely, and zstd 10.87% against
# 10.75%: one point of sparsity bought, and the packet slightly WORSE. Accuracy
# was untouched at 56.44, because nothing was happening.
#
# The cause is not a bug in the solver, it is the calibration rule. The dual was
# doing its job perfectly (mean_z_star 0.5405, z>0.5 on 60.6% of the weights);
# what never happened was the transmission of that opinion to the deployed model.
# T2 does not prune: it applies a constant-magnitude L1 push of 2*sqrt(T1*T2) and
# the LSQ zero bin absorbs whatever reaches it. The historical rule
# T2 = 4*T1*q^2 sets the DUAL threshold at q and says nothing about whether the
# push can carry a weight that far. On test_226 it produced coefficients of
# 4.7e-9 to 2.4e-8, a total displacement of about 2e-4, against the 0.010 to
# 0.022 needed to reach the requested quantiles. Three to four orders of
# magnitude short.
#
# The arithmetic was checked against the run itself: modelling the deployed
# sparsity as the fraction of weights with |w| < D + a/2 predicts 16.2% global,
# and the run measured 17.12%. The model holds to within one point, so the dose,
# not the machinery, is what failed.
#
# --t2_calibration displacement (now the default) inverts the relation instead:
# it solves 2*sqrt(T1*T2)*S = q_target - a/2, where S is the exact schedule sum
# the trainer computes from its own learning-rate curve, ramp shape and step
# count. On this schedule S = 230.1, verified offline against a hand calculation.
# The resulting coefficients are 4e-5 to 2e-4, i.e. six to eleven thousand times
# the old ones, and the implied L1 force is 15% to 30% of the typical task
# gradient: strong, but an ordinary L1 strength rather than an absurd one.
#
# Note the model deliberately IGNORES the task gradient, which resists on the
# weights that matter. The realized sparsity will therefore land BELOW the 85.3%
# target, which is the safe direction to be wrong in.
#
# Everything else is byte-for-byte the test_226 configuration, which is itself
# the frozen test_225 recipe plus the phase schedule and --min_lr.
#
# Baseline to beat, test_225 (run_alexnet_lsq_lossless.sh, LSQ only):
#   A_Q 56.552 and sparse 56.556 against 56.524 FP32, both ABOVE the checkpoint,
#   at zstd 10.75% dense / 10.76% sparse, i.e. 9.30x, with 16.11% of the weights
#   landing in the LSQ zero bin on their own.
#
# THE RECIPE IS FROZEN. Everything below is identical to test_225 except the
# three things this experiment is about, listed at the end. Do not tune anything
# else here: the whole point of the ladder is that the LSQ row and the PEAQ rows
# differ only by the coefficients.
#
# 1. --layerwise_t2_targets 0.15,0.55,0.60,0.60,0.60,0.88,0.88,0.65
#    One target sparsity per quantized tensor, in order conv1..conv5, fc6, fc7,
#    fc8. After the diagnostic phase the trainer calibrates each layer's
#    coefficient as T2_l = 4*T1*q_target(|w_l|)^2, so the dose is set from the
#    weight distribution the network actually has rather than by guessing a
#    scalar. The profile is Deep Compression's own lossless AlexNet pruning
#    profile (conv1 16%, conv2 62%, conv3-5 63-65%, fc6/fc7 91%, fc8 75%),
#    trimmed slightly because we are pruning AND quantizing to 4 bits, whereas
#    they pruned first. Global target 85.3%; the estimated packet is about 3.5%,
#    i.e. roughly 29x, against the 9.30x of the baseline.
#    Note that T2 does not prune anything directly: it puts an L1-like push of
#    about 2*sqrt(T1*T2) on the small weights, and the LSQ zero bin then absorbs
#    them. mag_prune_ratio and target_sparsity stay at zero, as they must under
#    --lsq_per_channel Y.
#
# 2. Phase schedule 10 + 30 + 20 = 60 epochs.
#    Ten diagnostic epochs at T2=0, which reproduce the first ten epochs of
#    test_225 exactly (during them layerwise_t2_current is all zero, so the
#    trainer takes the T1-only fast path and the step sizes stay untouched);
#    thirty epochs ramping T2 linearly to its calibrated value; twenty flat
#    epochs at full dose. A long ramp is what an 85% target needs. The ramp also
#    means that even if the accuracy breaks we read WHERE it breaks, which is the
#    frontier point this run is meant to produce.
#
# 3. --min_lr 1e-4.
#    Structurally necessary, and the one deviation from the frozen recipe. Under
#    the phase schedule the rate falls to the floor by the end of the ramp and is
#    then HELD for the whole flat phase; at the historical 1% floor that is 2e-5,
#    where test_225 measured level_change at 0.115% per epoch, i.e. a frozen
#    network. The compression would arrive on a model that can no longer heal.
#    At 1e-4 (5% of the base rate) test_225 measured 0.5% to 0.8% of weights
#    changing level per epoch, so the flat phase is a genuine healing tail.
#
# WHAT TO CHECK ON EPOCH 1: training_time. The general perspective path replaces
# the T1-only fast path once T2 goes live, and it runs on every step over all 61M
# weights. The estimate is about six seconds per epoch of extra traffic against
# the 157s of test_225, so anything up to ~170s is fine. If epoch 11, the first
# with T2 alive, jumps well past 200s the run will not fit the slot and should be
# killed rather than left to time out.
#
# WHAT TO CHECK ON EPOCH 11: the [LAYERWISE T2 CALIBRATION] line, and that
# metaq_scale_grad_last stops being all zeros. Both confirm the gate handing over
# from the ridge regime to the full envelope exactly when T2 becomes live.

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
        --diagnostic_epochs 10 \
        --metaq_ramp_epochs 30 \
        --metaq_flat_epochs 20 \
        --lr 2e-3 \
        --lr_warmup_epochs 1 \
        --min_lr 1e-4 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff 1e-5 \
        --entropy_coeff 0 \
        --sparsity_coeff 0 \
        --layerwise_t2_targets 0.15,0.55,0.60,0.60,0.60,0.88,0.88,0.65 \
        --t2_calibration displacement \
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
