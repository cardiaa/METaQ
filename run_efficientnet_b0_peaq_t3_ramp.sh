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

# test_233: EfficientNet-B0, T3 dose-response curve, second attempt. Same ramp
# trick as test_231, other coefficient. T2 stays at zero so the two frontiers
# stay separable.
#
# WHAT test_232 MEASURED, killed at epoch four. Top of the ramp 3e-7, so epoch 4
# ran T3 at 6e-8, twice the value that was productive on ResNet-18. Change over
# those four epochs, against the two runs that bracket it:
#
#                     values    mask   sparsity   H_Q bits/weight
#   test_230 LSQ      -0.099   +0.024   +0.34       -0.0259
#   test_231 T2 ramp  -1.335   +0.343   +4.99       -0.3272
#   test_232 T3 ramp  -0.089   +0.025   +0.36       -0.0217
#
# i.e. indistinguishable from the run with every PRESTO coefficient at zero, and
# marginally BELOW it. Not a malfunction: xi_pinned_frac went 0.0128 -> 0, xi grew
# smoothly, beta_commonmode sat at 0.077-0.107 inside test_168's 0.035-0.138 band.
# The dual is healthy and the dose is inert.
#
# WHY, AND WHY THE DIAGNOSTIC LIED. test_232's entropy_fraction read 0.0054-0.0091
# against test_168's 0.0026-0.0040, which looks like an OVER-dose. It is not:
# applied_entropy_norm measures bstar, the complete perspective gradient with the
# T1 ridge in it (see the FISTA block: "beta* = dphi/dw (ridge + entropy)"), and
# at these T3 values the ridge dominates. The evidence is in the run itself --
# T3 rose fourfold, 1.5e-8 to 6e-8, and the norm rose 1.42x. Solving the two
# components gives a constant ridge near 2.37e-4 with the entropy part going from
# 14% to 39% of the total. entropy_fraction is not a clean T3 signal at low T3
# and must not be used as one.
#
# The honest comparison is absolute: the push here is 2.8e-4 against test_168's
# 1.3e-3, and there are 312 steps per epoch against 1251, so the entropy exposure
# per epoch is about an eighth of the run this dose was anchored on. That puts
# the productive coefficient near 2.4e-7, which the old ramp only reached at
# epoch 16, with the cosine already at its floor and the network too cold to
# respond.
#
# SO: TOP OF THE RAMP 3e-6, ten times test_232. Epoch 2 now runs at 3e-7, just
# above the 2.4e-7 the exposure argument predicts, so the curve should start
# moving around epochs two to three and the sweep continues to a hundred times
# the ResNet-18 value while the learning rate is still alive. The per-epoch push
# peaks at epochs eight and nine and decays afterwards -- the cosine falls faster
# than the ramp climbs -- so anything not crossed by epoch twelve will not be
# crossed at all.
#
# The premise that motivated this measurement is unchanged: test_231 left the
# surviving weights carrying 2.41 bits of the 4 the codebook allows, against 1.12
# on AlexNet. The room for the entropy term is real; only the dose was wrong.
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
# The ResNet-18 anchor this was all scaled from is test_168's 3e-8. Careful
# reading logs older than test_169: the names T2/T3 are printed in the opposite
# positions there, so the 168's "T2=3e-08" IS the entropy and its "T3=1e-07" is
# the sparsity (l1_push_near_zero=2e-6 = 2*sqrt(T1*T2) confirms which is which).
#
# There is no dimensional argument carrying T3 between networks the way step
# sizes carried T2 in test_231: the entropy gradient goes as T3*log2(c_b) over
# relaxed bucket counts, which barely depends on the network, while the task
# gradient it competes with does. Hence a ramp rather than a derivation, twice.
#
# WHAT TO WATCH, and it is not entropy_fraction. The signal is values_zstd_ratio
# in sparse_zstd_components, read against test_230 at the same epoch: that run is
# the same recipe with every coefficient at zero, so anything it also does is not
# T3. Its first four epochs are 9.094, 9.058, 9.022, 8.994.
#
# WHEN TO KILL IT. If values_zstd_ratio has not separated from that baseline by
# epoch four -- which is now thirty times the ResNet-18 dose -- the entropy
# channel does not open on this network at any affordable dose, and that is the
# result: stop, and put the remaining budget on T2. If A_Q falls off a cliff, or
# xi_pinned_frac goes to one (the dual chattering against its bounds, the
# test_133 failure), the useful rows are already in the log and the rest is
# noise.
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
# COST, now measured rather than guessed: test_232 ran 365s/epoch, so about two
# hours for the twenty epochs. The four-hour wall stays as slack.

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
        --entropy_coeff 3e-6 \
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
