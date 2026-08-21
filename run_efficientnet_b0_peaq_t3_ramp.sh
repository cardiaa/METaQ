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

# test_232: EfficientNet-B0, T3 dose-response curve. Same ramp trick as test_231,
# other coefficient. T2 stays at zero so the two frontiers stay separable.
#
# WHY T3 AND NOT THE COMBINATION YET. test_231 measured T2's frontier and it is
# shallow on this network: the knee sits between 23% and 27% sparsity, against
# 42-51% on AlexNet, and it buys 8.67x -> 9.75x for free and 10.46x for -0.63.
# But it also measured where the redundancy actually is. Values stream, in bits
# per SURVIVING weight:
#
#   test_230 baseline    3.39 bits   (8.94% of the packet, 84.3% survivors)
#   test_231 endpoint    2.41 bits   (4.53% of the packet, 60.2% survivors)
#   AlexNet test_229     1.12 bits   <- T2 alone had already done T3's work
#
# On AlexNet T2 flattened the surviving levels onto almost nothing before T3 was
# ever switched on, which is why T3 was expected to add little there. Here the
# survivors are still carrying 2.4 bits out of the 4 the codebook allows. THIS is
# the network where the entropy term has something to work on, and measuring it
# alone is worth an hour and a half before anything is combined.
#
# TOP OF THE RAMP: 3e-7, ten times test_168's 3e-8. The ramp is linear over
# twenty epochs, so epoch 1 runs at half the ResNet-18 value, epoch 2 lands
# exactly on it, and the sweep continues to ten times it. Careful reading logs
# older than test_169: the names T2/T3 are printed in the opposite positions
# there, so the 168's "T2=3e-08" IS the entropy (l1_push_near_zero=2e-6 =
# 2*sqrt(T1*T2) confirms which is which).
#
# There is no dimensional argument carrying T3 between networks the way step
# sizes carried T2 in test_231: the entropy gradient goes as T3*log2(c_b) over
# relaxed bucket counts, which barely depends on the network, while the task
# gradient it competes with does. So the anchor is a run that worked and the
# ramp does the rest. The dimensionless number to watch is entropy_fraction in
# perspective_entropy_diag, which test_168 held at 0.0026-0.0040 in its early
# epochs; that is the scale at which the term was productive and harmless.
#
# WHEN TO KILL IT. If the values stream has not started falling by epoch six
# (three times the ResNet-18 dose) the top was too low: multiply by ten. If A_Q
# falls off a cliff, or xi_pinned_frac in perspective_entropy_diag goes to one --
# the dual chattering against its bounds, the test_133 failure -- the useful rows
# are already in the log and the rest is noise.
#
# READ IT AGAINST test_230 AT THE SAME EPOCH for the first ten rows, which is the
# only same-recipe reference there is, and note that the baseline is only ten
# epochs long: rows 11 to 20 have no baseline and can only be read against its
# plateau of 76.590. The schedules also drift apart after epoch ten, since
# test_230's cosine is dead by then while this one is at 45% of base rate.
#
# THE DOSE THIS PRODUCES IS NOT THE DOSE TO RE-RUN. A ramp row reports the
# CUMULATIVE exposure up to that epoch, not the coefficient held for a whole run.
# On test_231 the difference is a factor of 4.4: the epoch-5 row reads T2=1e-6,
# but reproducing that row with a fixed coefficient over a full twenty-epoch
# cosine takes 2.28e-7. Compute the same integral before re-running any row from
# this log; scratchpad/verify_t2_exposure.py has the arithmetic.
#
# COST. First time the dual runs on this network. 147s/epoch in test_231, plus 78
# FISTA calls per epoch over 82 tensors, so somewhere between 230 and 310s per
# epoch and between an hour and twenty and an hour and forty-five. The four-hour
# wall is slack for a cost that has never been measured here, not an estimate.

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
        --n_epochs 20 \
        --diagnostic_epochs 0 \
        --metaq_ramp_epochs 20 \
        --metaq_flat_epochs 0 \
        --lr 2e-2 \
        --lr_warmup_epochs 2 \
        --min_lr 1e-3 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff 1e-5 \
        --entropy_coeff 3e-7 \
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
        --dual_step 3e-9 \
        --check_ddp_sync \
        --pretrained Y \
        > "$OUTPUT_TARGET" 2>&1
'
