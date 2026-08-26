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

# test_259: ViT-B/16, METaQ at one fifth of the test_256 dose. This run exists
# to put a second point on the ViT-B/16 curve, and specifically the one at the
# near-lossless end, which is the half of the frontier we do not have.
#
# WHERE WE ARE. Against 81.056 full precision, on the corrected tensor set:
#   test_255  LSQ only                20 ep   80.816  -0.240   10.19%
#   test_256  T2=5e-8, T3=1.5e-8      20 ep   80.372  -0.684    4.11%
# Six points of packet separate them and nothing sits in between, while every
# other network in the paper carries two to four points. The text calls this a
# frontier and at the moment it is two endpoints.
#
# WHY ONE FIFTH AND NOT ONE HALF. The response saturates, so halving the
# coefficients would barely move the result. The transport rule is the one we
# already use for T2: what carries between runs is the CUMULATIVE EXPOSURE,
# coefficient times the schedule sum, never the coefficient. With ramp_end 1 and
# a five per cent floor this schedule sums to 11.45 over twenty epochs, so a
# fixed dose alpha reproduces the exposure test_256 had accumulated by epoch k
# when alpha = S(k)/S(20). Reading test_256's own packet trajectory against that
# map:
#
#   alpha    exposure of      packet there    sparsity
#   0.507    epoch 6            4.79%          66.07%
#   0.347    epoch 4            5.46%          61.18%
#   0.261    epoch 3            6.11%          55.72%
#   0.175    epoch 2            7.50%          46.87%
#
# Halving the dose lands at 4.8 per cent, which is not a new point. One fifth,
# T2 1e-8 and T3 3e-9, falls between the epoch-2 and epoch-3 rows and should
# land near 6.8 to 7.0 per cent, squarely in the gap.
#
# THE RULE PREDICTS THE PACKET, NOT THE ACCURACY. Those trajectory rows are
# mid-flight at a high learning rate; this run reaches the same exposure with a
# full cosine behind it, so the accuracy should be well above test_256's -0.684.
# A landing between -0.30 and -0.45 is the expectation. If it comes out at 7 per
# cent and -0.65 the frontier is much flatter than we think and the interesting
# object becomes the accuracy, not the size.
#
# WATCH THE MASK. At the sparsity this dose implies, near fifty per cent, the
# support mask is at its most expensive: its cost is H(p)/32 and that is maximal
# at p = 1/2. test_237 and test_238 showed the mask getting CHEAPER as sparsity
# went past fifty per cent, so this point pays the worst mask of the three ViT
# runs and its packet is a pessimistic reading of what the dose buys.
#
# NOTHING ELSE MOVES from test_256: same twenty epochs, same rate, same
# distillation, same relative dual step at 1.3e-2, same T1.
#
# COST: 2229s per epoch measured on test_256, whose first epoch is cheap because
# the entropy warm-up leaves the dual off. That is 11h50 for twenty epochs, and
# test_256 cleared its twelve-hour wall with ten minutes to spare. Not worth
# repeating: the wall here is sixteen hours.

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
    --n_epochs 20 \
    --lr 5e-5 \
    --optimizer_weight_decay 0 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 3e-9 \
    --sparsity_coeff 1e-8 \
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
