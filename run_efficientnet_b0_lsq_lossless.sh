#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_230: EfficientNet-B0, LSQ-only baseline. The test_223 correction, applied
# to the second network the lr diagnosis condemned.
#
# WHAT THIS RUN IS FOR. test_222 put EfficientNet-B0 at 75.682 best against
# 77.566 FP32, i.e. -1.88, and the diagnosis says why: the QAT displacement
# budget was 0.19 quantization steps over the whole run, so the network could
# not move a single weight to a different level and A_Q was PTQ plus noise. The
# identical failure on AlexNet (0.21 steps, -1.64) was fixed by one change of
# regime -- global batch 1024 instead of 4096, a learning rate two orders of
# magnitude higher, per-channel scales on a relative schedule, distillation --
# and that took AlexNet from -1.64 to +0.028 across test_223/224/225. This run
# asks whether the same correction transfers, on the cheapest budget that can
# answer the question.
#
# WHAT CHANGES FROM test_222, and nothing else:
#   lr            2e-4 -> 2e-2     a hundredfold; see below
#   lr_warmup     none -> 2 epochs  Goyal's large-batch rule, not decoration
#   lsq_scale_lr  1e-5 absolute -> 1e-3 relative, with schedule
#   distillation  off -> on, alpha 0.5, tau 1.0
#   weight_decay  0 -> 1e-4        the value both lossless runs use
#   n_epochs      20 -> 10
# The batch stays at 256 per GPU, global 4096. The AlexNet correction also moved
# the batch to 1024, but that was never the operative half of it: what broke
# test_201/205 was moving to 4096 and NOT rescaling the rate. The budget that
# diagnosed the failure is the product lr * n_step, so global 4096 at 2e-2 and
# global 1024 at 5e-3 deliver the same displacement, and the cheaper one leaves
# the extra hour for the T2 run that follows.
#
# THE LEARNING RATE IS THE ONE GUESS IN HERE. Both lossless runs set it at about
# a fortieth of the network's own from-scratch rate: ResNet-18 1e-2 at global
# 1024, AlexNet 2e-3. EfficientNet-B0 has no comparable anchor -- the checkpoint
# is rwightman's, trained with RMSProp, and an RMSProp rate does not transfer to
# SGD -- so the bracket is all we have, and 2e-2 at global 4096 is its middle
# (4.9e-6 per sample, half of ResNet-18's). On the displacement scale that
# diagnosed the failure it buys about 9.5 quantization steps against the 0.19 of
# test_222: fifty times more than a network that could not move, and
# deliberately below ResNet-18's 176, because ten epochs cannot repay a large
# first-epoch dip and EfficientNet is the more delicate network.
#
# THE COST OF THE LARGE BATCH, stated up front so it is not rediscovered later:
# 3120 optimizer steps in total, which is HALF the step count of the run that
# failed. The displacement budget does not care -- it is a product -- but level
# reassignment happens per step, and the two networks that came out lossless
# both healed with more steps rather than fewer (AlexNet 55.96 -> 56.45 -> 56.55
# from 20 to 40 to 60 epochs; ResNet-18 still climbing at epoch 20). DeiT is the
# counterexample, damaged by steps rather than healed. Which regime EfficientNet
# is in is exactly what nobody knows yet. So: if this run lands short while
# level_change is still healthy and the accuracy curve is still rising at epoch
# 10, the diagnosis is the step budget and the follow-up is --batch_size 64 at
# --lr 5e-3, same per-sample rate and four times the steps, NOT more lr.
#
# HOW TO READ THE FIRST EPOCHS. test_222's epoch 1 was 73.78 with a frozen
# network, which is roughly the PTQ level. The warm-up puts epochs 1 and 2 at a
# third and two thirds of the rate, so epoch 3 is the first at full 2e-2 and the
# first that can go wrong. Above 73 and rising, still climbing at epoch 10: the
# rate is working and the answer is more epochs, exactly as it was for AlexNet
# at test_223. Falling off a cliff at epoch 3 and not recovering by epoch 5: the
# rate is too hot, drop to 8e-3 and rerun, still under an hour of nodes.
#
# T1 IS OFF, on purpose. This is a pure LSQ baseline: T1=T2=T3=0, so the METaQ
# gradient block never runs and what comes out is the quantizer alone. Two
# reasons rather than one. It is the control the paper is missing, and on THIS
# network T1 is not free: with T2=T3=0 the boundary multiplier contributes
# +T1*s*q_edge^2 to every clipped weight, one-signed, and test_222 measured the
# resulting drift directly -- mean clipped fraction 2.58% -> 3.93%, and
# features.7.0.block.1.0 from 9.3% to 34.7% clipped, a third of a depthwise
# layer pinned at the codebook edge with the straight-through mask zeroing its
# gradient. Whatever this run reports is therefore the quantizer's own number,
# not a number with a ridge in it.
#
# NOTE for the T2 run that follows: it will have to switch T1 back on, because
# the displacement calibration needs T1 > 0. So the first thing to check in its
# log is the clipped fraction per layer against this run's, and if the depthwise
# layers start climbing, that is the test_222 feedback loop returning and not
# T2 doing its job.
#
# --layer_C IS NOT SET. Uniform C=16, four bits everywhere. On AlexNet the 8-bit
# input convolutions were worth +0.28 for 0.068% of the packet and they are the
# first thing to try if this lands a couple of tenths short -- but EfficientNet-B0
# has 82 quantized tensors, so that flag needs an 82-entry list, and it is not a
# change to make blind on the first pass.
#
# COST. test_222 ran 160s/epoch at this batch; the FP teacher's forward pass
# under no_grad adds about a third, so expect 210-230s/epoch and roughly fifty
# minutes for ten epochs and evaluation. The teacher also has to fit alongside
# the student at batch 256 -- it holds no activations for backward, so it
# should, but a CUDA OOM on epoch 1 is the one failure mode that would be the
# batch's fault and not the recipe's.

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
        --n_epochs 10 \
        --lr 2e-2 \
        --lr_warmup_epochs 2 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff 0 \
        --entropy_coeff 0 \
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
