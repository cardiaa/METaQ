#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=16:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_263: ResNet-18 at T3=2.6e-7 with one hundred epochs. ONE flag differs
# from test_257: --entropy_coeff 1e-7 -> 2.6e-7.
#
# WHY THIS RUN EXISTS. Our ResNet-18 frontier stops at 23.6x and the literature
# does not. Metz, Bichler and Dupret, "Efficient Neural Compression with
# Inference-time Decoding", ISCAS 2024, Table II: ResNet-18 at 69.8 per cent and
# 46.8MB coded to 1.67MB at 69.0 per cent, that is 3.57 per cent of the model,
# 28.0x, at -0.8. Mixed-precision quantization with a learned zero point driven
# by a Fracbits-style precision parameter against an entropy target, coded with
# tabled ANS. Beyond our deepest measured point we have nothing, so everything we
# might say about that region is extrapolation.
#
# THE TARGET IS NEITHER THEIR SIZE NOR THEIR ACCURACY. Both single-axis choices
# were considered and both are worse.
#
# Aiming at their SIZE, 3.57 per cent, needs T3=1.7e-7 and lands near -0.22. Safe,
# because 1.7x the last measured dose is a short extrapolation, but it produces
# one claim only: more accurate at the same size. It does not dominate them,
# since 3.52 against 3.57 per cent is a wash.
#
# Aiming at their ACCURACY, -0.8, needs T3 near 3.6e-7. The claim would be
# stronger if it lands, but the outcome depends entirely on how hard the frontier
# bends, and the spread is wide. Starting from test_257 at (4.15 per cent,
# +0.126) and spending down to -0.8 at a marginal cost of 0.485 accuracy points
# per point of packet gives 2.33 per cent, at 0.65 gives 2.81, at 0.85 gives 3.15.
# The last of those is only twelve per cent smaller than they are, which is a
# weak sentence bought with thirteen hours.
#
# AIMING BETWEEN THEM DOMINATES IN EVERY SCENARIO. Landing near -0.5:
#   marginal cost 0.485 (no bend)         -> 2.95 per cent of model, 33.9x
#   marginal cost 0.65  (ResNet-50 bend)  -> 3.28 per cent, 30.5x
#   marginal cost 0.85  (sharp bend)      -> 3.50 per cent, 28.6x
# All three are SMALLER than their 3.57 per cent and MORE ACCURATE than their
# -0.8. One measured point that beats them on both axes at once, with no
# interpolation on our side and no extrapolation on theirs, and the conclusion
# does not depend on which scenario is true.
#
# THE DOSE. Regressing our two sixty-epoch points, T3=6e-8 at 4.82 per cent and
# 69.936 against T3=1e-7 at 4.14 and 69.606, the packet moves -1.331 points and
# the accuracy -0.646 per unit of ln(T3). Adding the +0.254 that sixty to a
# hundred epochs bought at T3=1e-7, T3=2.6e-7 predicts 2.87 per cent of packet at
# -0.49. Substituting the flatter ResNet-50 slopes measured at a deeper operating
# point, -1.163 and -0.726, gives 3.03 per cent at -0.57. The run should land in
# 2.9 to 3.1 per cent of the model at -0.5 to -0.6.
#
# WHY A HUNDRED EPOCHS. At T3=1e-7 the packet SATURATED between sixty and a
# hundred epochs, 4.14 to 4.15 per cent, while the accuracy gained 0.254 points.
# On this branch the extra budget buys accuracy at a fixed size, which is the
# axis this run needs. The packet prediction is read off the sixty-epoch slope
# precisely because the extra epochs do not move it.
#
# THE ASSUMPTION MOST LIKELY TO BE WRONG is that the +0.254 healing bonus
# transfers to a dose two and a half times larger, where the compression pressure
# is higher throughout. If it halves we land near -0.62, if it vanishes near
# -0.75. Both still beat their -0.8, and the packet is unaffected either way, so
# the domination claim survives even the worst case; only its margin shrinks.
#
# NOTHING ELSE MOVES. T2 stays at 1e-7, T1 at 1e-5, the dual step at 1.3e-2
# relative, min_lr at 1e-4, one hundred epochs.
#
# COST: 474s per epoch measured on test_257, so about thirteen hours and ten
# minutes. The sixteen-hour wall is the one test_257 cleared; there is no resume
# in the trainer, so it has to fit in one allocation.

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
    --n_epochs 100 \
    --lr 1e-2 \
    --optimizer_weight_decay 1e-4 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 2.6e-7 \
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
