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

# test_266: DeiT-Small, METaQ con duale relativo, punto 1 (un quinto della dose).
#
# COSA CAMBIA DAI RUN 189/191/192/194 CHE SONO NEL PAPER. Solo il duale, ma e'
# la parte che conta: quei quattro run portano --dual_step 3e-9 ASSOLUTO. Il
# test_233 ha dimostrato che con passo assoluto il termine entropico non e'
# governabile, perche' l'intervallo duale da attraversare vale
# entropy_coeff*log2(upper_c/lower_c), cioe' e' proporzionale a T3, mentre il
# passo non lo e': alzando T3 di dieci volte la norma del gradiente applicato
# resta identica a quattro cifre significative. La frontiera DeiT del paper e'
# quindi misurata con un duale che sappiamo non convergere. Qui si passa a
# --dual_step 1.3e-2 --dual_step_mode relative, che e' quello che usano
# ResNet-18, ResNet-50, EfficientNet-B0 e ViT-B/16.
#
# E --max_iterations 3, non 6. Lo script da cui questo deriva ne aveva 6, che e'
# la stessa anomalia dei run ViT-B/16: con il passo relativo la velocita' di
# convergenza duale e' dual_step * N_dual * chiamate/epoca, quindi N_dual non e'
# una tolleranza di solutore ma raddoppia la velocita' del duale. Tre ovunque.
#
# IL CONTROLLO NON VA RIFATTO. test_189, LSQ + distillazione, ha i tre
# coefficienti a zero, quindi il duale non viene mai invocato e il suo 79.832 a
# 11.98 per cento del modello resta valido. Servono solo i due punti METaQ.
#
# LA DOSE SCENDE, E DI MOLTO. Con il duale relativo il termine entropico diventa
# molto piu' forte a parita' di coefficiente: su ResNet-18 il solo passaggio da
# lento a veloce a T3 invariato porto' il pacchetto da 6.08 a 5.74 per cento. La
# coppia (5e-8, 1.5e-8) del test_194 quasi certamente sfonda. Questi due run
# prendono un quinto e la meta' di quella coppia, che e' esattamente la manovra
# che ha centrato il ginocchio su ViT-B/16 al primo colpo (test_259: previsto
# 6.8-7.0 per cento a -0.30/-0.45, misurato 6.84 a -0.374).
#
# PERCHE' DUE E NON TRE. Controllo piu' due punti METaQ e' lo standard della
# casa su ResNet-50, ViT-B/16 ed EfficientNet-B0. Vanno lanciati insieme, non in
# sequenza: bracchettano invece di inseguire.
#
# COSTO: 435s per epoca misurati sul test_194, cioe' circa tre ore per
# venticinque epoche. Il wall di cinque ore e' abbondante.

LOG_DIR=$WORK/acardia0/LeonardoTests
mkdir -p "$LOG_DIR"

# NUMERAZIONE FISSA. Gli script di questo blocco partono insieme e
# l'auto-incremento li farebbe collidere sullo stesso file di log.
NEXT=266
LOG_FILE=$LOG_DIR/Leonardo_test_${NEXT}.log
if [ -e "$LOG_FILE" ]; then
    echo "Leonardo_test_266.log esiste gia': rifiuto di sovrascriverlo." >&2
    exit 1
fi
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
    --dual_step 1.3e-2 \
    --dual_step_mode relative \
    --check_ddp_sync \
    --pretrained Y \
    > "$OUTPUT_TARGET" 2>&1
'
