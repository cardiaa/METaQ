#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_265: DeiT-Small, CONTROLLO LSQ + distillazione rifatto con --min_lr 5e-6.
# Sostituisce il test_189 come riga di controllo del capitolo DeiT.
#
# PERCHE' VA RIFATTO, ed e' un difetto di controllo non di dose. Il test_189
# porta min_lr=None, cioe' il pavimento di default all'1 per cento del rate,
# mentre i tre run METaQ che gli sono confrontati (191, 192, 194) portano
# min_lr=5e-6, cioe' il 5 per cento. Il controllo ricuoce quindi a un learning
# rate cinque volte piu' basso dei run che dovrebbe controllare, e la coda di un
# coseno e' esattamente dove si guadagnano gli ultimi decimi di accuratezza. Il
# capitolo DeiT del paper e' costruito su differenze contro quella riga, e una
# riga di controllo che non condivide lo scheduler non e' un controllo.
#
# IL VERSO DEL BIAS E' A NOSTRO SFAVORE, il che e' l'unica buona notizia: un
# pavimento piu' basso di solito regala accuratezza finale, quindi il test_189 e'
# lusingato e il costo di METaQ che il paper riporta, 0.34 punti, e' semmai una
# sovrastima. Ma resta un confronto non appaiato e va sistemato, non spiegato.
#
# LO SI RIFA' ADESSO PERCHE' COSTA NIENTE. Con i tre coefficienti a zero il duale
# non viene mai invocato e l'epoca costa 133s misurati sul test_189: venticinque
# epoche sono cinquantasei minuti. E' il run piu' economico di tutto il
# programma, e senza di esso i due punti METaQ nuovi non hanno un controllo
# appaiato contro cui essere letti.
#
# ALLINEATO ALLA RICETTA COMUNE come i due punti METaQ: max_iterations 3, duale
# 1.3e-2 relativo (inerte qui ma identico), entropy_warmup 1 che fissa la forma
# del coseno, per-tensore, mse, C=16, distillazione 0.9 dal teacher a piena
# precisione, batch globale 2048, lr 1e-4, min_lr 5e-6.

LOG_DIR=$WORK/acardia0/LeonardoTests
mkdir -p "$LOG_DIR"

# NUMERAZIONE FISSA. Gli script di questo blocco partono insieme e
# l'auto-incremento li farebbe collidere sullo stesso file di log.
NEXT=265
LOG_FILE=$LOG_DIR/Leonardo_test_${NEXT}.log
if [ -e "$LOG_FILE" ]; then
    echo "Leonardo_test_265.log esiste gia': rifiuto di sovrascriverlo." >&2
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
