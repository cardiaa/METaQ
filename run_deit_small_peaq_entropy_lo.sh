#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=4:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_194: map the frontier with a lower entropy dose.
#
# Budget is exhausted as a lever. Three runs at the standard dose land on the
# same point regardless of schedule: 79.24 at 7.78% (test_191, 25 epochs), 79.17
# at 7.41% (test_192, stretched cosine, more compression not more accuracy), and
# 79.25 at 7.71% (test_193, fifteen floor-rate healing epochs that added +0.01
# accuracy, refuting the compress-then-heal idea). The frontier sits at about
# 79.2 and 7.7%, which is -0.5 from the 79.742 FP32 line, and the ~0.5 gap is the
# amount of compression, not a lack of healing time.
#
# The lever for lossless is therefore to compress LESS, moving up the frontier,
# and the coefficient that controls compression is entropy, since test_191
# showed the sparsity coefficient is a weak, largely emergent lever here. This
# run halves entropy_coeff from 3e-8 to 1.5e-8, changing nothing else with
# respect to test_191, to place ONE intermediate point on the accuracy-ratio
# frontier and reveal its shape.
#
# The shape is the decision. With the baseline at 79.79 / 11.43% and the full
# dose at 79.2 / 7.7%, a CONVEX frontier, flat near the baseline, means lossless
# costs little ratio and is worth taking. A LINEAR one means lossless lands back
# near the uncompressed baseline, in which case the honest headline is the
# near-lossless point, which at -0.5 already beats Q-ViT 3-bit and sits inside
# the range L2 reports as near-lossless (-0.27 to -0.86) on ResNet-18.
#
# WHAT TO READ. Sparse accuracy against 79.24 and 79.742, and sparse ratio
# against 7.78%: where this midpoint falls between the baseline and the full-dose
# point tells us whether the curve bends toward us or not. Also watch sparsity,
# which should be lower than test_191's 44% since less entropy pressure means
# fewer weights driven into the zero bin.
#
# About three hours with the entropy solver, 25 epochs.

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
    --min_lr 5e-6 \
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
