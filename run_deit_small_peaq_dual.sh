#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=06:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Test 203 ablation: twice the dual iterations used in test 199.

# test_241: DeiT-Small, the knee of the frontier repeated with the entropy dual
# allowed to converge. ONE flag differs from test_194: --dual_step_mode relative,
# with --dual_step rescaled to the same convergence rate that worked on
# ResNet-18. Everything else, including all three coefficients, is untouched.
#
# WHY THIS RUN AND NOT A RECALIBRATION. The DeiT hyperparameters are already
# calibrated: test_194 is the measured knee of the accuracy-size curve on this
# network, 79.490 at 9.04% of FP32 against a 79.742 checkpoint, and its marginal
# cost per point of packet is the lowest of the four points on that curve. The
# open question is not the dose, it is whether the dual was converged when the
# dose was chosen. It was not: test_194 crossed its dual interval 0.189 times per
# epoch. On ResNet-18 the same figure was 0.195, and raising it fourfold there
# took the packet from 6.08% to 5.74% at an accuracy cost inside the run-to-run
# noise, then to 4.82% at sixty epochs while staying above the checkpoint.
#
# THE STEP. In relative mode the convergence rate is dual_step * (1/C) * N_dual *
# calls_per_epoch and no longer depends on the coefficient or on the tensor size.
# ResNet-18 ran at 1.3e-2 with 313 calls per epoch, that is 0.763 traversals.
# DeiT-Small has 625 steps and entropy_every 4, so 156 calls, and 2.6e-2
# reproduces the same 0.763. That is four times test_194's rate, the same factor
# that worked on ResNet-18.
#
# WHAT WE EXPECT. If the transfer holds, the packet moves from 9.04% toward
# roughly 8.5% at an accuracy cost inside the noise. That widens the margin over
# Q-ViT at three bits, which reports 79.0 at a nominal 9.86%, on both axes at
# once, and it is the second cell of the paper.
#
# WHAT WOULD MAKE IT A NULL. Same packet, same accuracy: then 0.189 traversals
# per epoch was already enough on this network and test_194 stands as it is.
#
# CHECKS. xi_pinned_frac was 0.0065 on the first entropy epoch of test_194 and
# zero afterwards; on ResNet-18 the faster step made it peak at 0.148 and then
# fall to zero, which is a transient and not the test_133 chattering. If it stays
# high past the second entropy epoch, halve --dual_step to 1.3e-2.
#
# COST: 441s/epoch measured on test_194, so about three hours for twenty-five
# epochs. DeiT-Small carries twice the parameters of ResNet-18 but runs at global
# batch 2048 against 1024, so it takes half the steps per epoch and the two
# networks cost almost exactly the same per epoch, 441s against 432s.

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
    --max_iterations 6 \
    --metrics_interval 1 \
    --entropy_warmup_epochs 1 \
    --entropy_every 4 \
    --dual_step 2.6e-2 \
    --dual_step_mode relative \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
