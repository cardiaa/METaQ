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

# DeiT-Small / ImageNet transfer test for the complete joint LSQ-METaQ method.
# Four 4-GPU nodes give a global batch size of 2048.
#
# test_171 changes the FINE-TUNING RECIPE only; every PEAQ coefficient is left
# exactly as in the successful ResNet-18 test-168. Measurements from test_170
# motivate this: against an FP32 baseline of 79.742, one epoch of plain LSQ QAT
# with PEAQ still disabled by the warm-up already cost 1.58 points (78.164),
# while PEAQ added only 0.31 over the following nine epochs (77.850). Test_169,
# same learning rate but four times the optimizer steps per epoch, lost 3.23
# points in that same PEAQ-free first epoch. The damage therefore scales with
# the number of optimizer steps, not with PEAQ.
#
# Accordingly: the learning rate drops from 1e-4 to 2e-5, weight decay goes to
# zero (the optimizer is Adam with coupled L2, which shrinks weights toward the
# zero bin and duplicates the perspective ridge), and the run is extended from
# 10 to 20 epochs because PEAQ enters through the gradient and is throttled by
# the same factor as the learning rate.
#
# Epoch 1 runs with PEAQ disabled (entropy_warmup_epochs=1) and is therefore a
# free LSQ-only control for the new recipe: at or above 79.3 the learning rate
# was the whole story, whereas a value near 78.2 would indicate a
# representational floor of the per-tensor 16-level codebook instead.

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
    --n_epochs 20 \
    --lr 2e-5 \
    --optimizer_weight_decay 0 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 3e-8 \
    --sparsity_coeff 1e-7 \
    --perspective Y \
    --flat_schedule N \
    --mag_prune_ratio 0 \
    --quantization Y \
    --quantizer lsq \
    --lsq_scale_lr 1e-5 \
    --lsq_init mse \
    --lsq_grad_scaling N \
    --joint_lsq_metaq Y \
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
