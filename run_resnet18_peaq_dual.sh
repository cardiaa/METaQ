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

# test_235: ResNet-18, the run behind the paper's headline, repeated with the
# entropy dual allowed to converge. ONE flag differs from test_196, which is why
# this is worth two hours: --dual_step_mode relative.
#
# WHAT test_196 DELIVERED, and its matched control:
#
#   test_174   LSQ alone, every METaQ coefficient zero   69.888   9.34%   10.7x
#   test_196   METaQ, T1 1e-5, T2 1e-7, T3 3e-8          69.840   6.08%   16.4x
#   test_197   the same at forty epochs                  69.578   5.21%   19.2x
#
# against 69.734 FP32, so the twenty-epoch pair is lossless on BOTH rows and the
# regularizer removes 35% of the packet for 0.05 points. That table is the paper.
#
# WHY REPEAT IT. test_233 and test_234 on EfficientNet-B0 established that the
# entropy term is exactly as strong as its dual is converged, and that the dual's
# convergence rate had never been a controlled quantity: dual_step was absolute
# while the range xi must cover, entropy_coeff*log2(upper_c/lower_c), is
# proportional to entropy_coeff. On EfficientNet, fixing only that -- same
# coefficient, same everything -- took the term from indistinguishable-from-off
# to four times the control on all three compression measures at 0.07 points of
# accuracy.
#
# ResNet-18 was never starved the way EfficientNet was. Traversals of the dual
# range per epoch: test_196 0.195, test_232 0.005. That is why the method has
# always worked here. But 0.195 means the dual crosses its own range once every
# five epochs, which over twenty epochs is four crossings on a target that is
# itself moving, and there is no evidence that this is converged rather than
# merely adequate. --dual_step 1.3e-2 relative is the same step that worked on
# EfficientNet and reproduces 0.76 traversals per epoch here, i.e. 3.9 times
# test_196's rate, with every coefficient left exactly where it was.
#
# WHAT WOULD MAKE THIS A RESULT. The packet below 6.08% at an accuracy still at
# or above 69.734. If it lands there, the headline improves without touching a
# single coefficient and the ablation table gains a row that explains WHY the
# entropy term works, which is currently missing from the paper.
#
# WHAT WOULD MAKE IT A NEGATIVE, and that is fine too: same packet, same
# accuracy, meaning test_196's dual was already converged and 0.195 traversals
# per epoch is enough. Then the EfficientNet fix stays a fix for the networks
# with few dual calls, and the ResNet-18 numbers stand as they are.
#
# CHECK ON EPOCH ONE. xi_pinned_frac must stay near zero. If it climbs toward
# one the dual is now chattering against its clamps, which is the test_133
# failure: kill it and halve --dual_step. Everything else in this run is
# test_196 byte for byte, including --entropy_warmup_epochs 1, which leaves
# epoch one as a free LSQ-only control.
#
# COST: 378s/epoch measured on test_196, twenty epochs, about two hours.

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

    torchrun \
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
    --n_epochs 20 \
    --lr 1e-2 \
    --optimizer_weight_decay 1e-4 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 3e-8 \
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
