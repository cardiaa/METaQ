#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=05:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Test 203 ablation: twice the dual iterations used in test 199.

# test_250: ViT-B/16, LSQ-only control at ten epochs. First run on this network,
# and it goes alone: the METaQ run follows once this one has said whether the
# recipe trains here at all.
#
# WHY ViT-B/16. It is the only architecture besides ResNet-50 on which a method
# that measures size after lossless coding publishes numbers: NNCodec, the
# reference implementation of the ISO/IEC 15938-17 standard, reports it at
# 346.27MB full precision and 81.07 top-1, compressed down to 32.87MB in its
# best configuration. That is 9.49 per cent, or 10.5x. Their accuracy AT that
# size is not in the material we could retrieve and must be read off the paper
# before it goes into any table of ours.
#
# THE RECIPE IS THE DeiT-Small ONE, halved. Transformers do not take the ResNet
# recipe: they use ADAM, a much smaller rate and distillation from the
# full-precision teacher. This script is test_194 with the model changed, the
# batch halved to 64 per GPU because ViT-B/16 carries four times DeiT-Small's
# parameters and the teacher has to fit beside the student, and the rate halved
# with it, from 1e-4 to 5e-5, by linear scaling.
#
# THE RATE IS THE GUESS, and this control is what tests it. Watch
# weight_percentiles: on the networks where the recipe worked the first
# percentile of the latent weights moved a few per cent over the run (-7.76 on
# ResNet-18, -3.55 on ResNet-50), and on the one where it did not it moved
# +0.01. Near zero means the network is frozen and the rate is too low, whatever
# the accuracy says.
#
# WHAT WOULD MAKE THE CELL WORTH OPENING. On both ResNets four-bit LSQ finishes
# ABOVE the full-precision checkpoint, which is the headroom METaQ then spends.
# On EfficientNet-B0 it finished 0.98 below and nothing could be spent. If this
# control lands above 81.07 there is a cell here; if it lands a point below,
# ViT-B/16 is EfficientNet again and we should say so and stop.
#
# THE CHECKPOINT MUST BE STAGED FIRST. ResNet-50 failed on its first launch for
# this reason. From a login node:
#   wget -P /leonardo_work/IscrC_ObCTDoNN/acardia0/imagenet_checkpoints \
#        https://download.pytorch.org/models/vit_b_16-c867db91.pth
#
# COST: never measured. ViT-B/16 has about 3.6 times DeiT-Small's compute per
# image, so expect roughly 450-550s per epoch without the dual, under two hours
# for ten. The four-hour wall is slack on an estimate.

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
    --lr 3e-5 \
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
