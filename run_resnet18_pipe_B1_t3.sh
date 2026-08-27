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

# test_274: ResNet-18, pipeline contro congiunta -- direzione B stadio 1, solo entropia.
# Coefficienti (T1,T2,T3) = (1e-5, 0, 3e-8) su venti epoche.
# Direzione B = l'ipotesi di Andrea, che T2 a valle distrugga cio' che T3 ha
# fatto a monte. Questo stadio salva il checkpoint che il test_275 ricarica.
#
# IL DISEGNO, e perche' venti epoche e non trenta. Il riferimento congiunto deve
# esistere gia' e deve avere ESATTAMENTE questi coefficienti: e' il test_235,
# 20 epoche, (1e-5, 1e-7, 3e-8), duale relativo 1.3e-2, max_iterations 3,
# per-tensore, 69.668 al 5.83 per cento del modello con sparsita' 49.18. E'
# identico al test_239 in ogni flag tranne n_epochs ed entropy_coeff, quindi e'
# un riferimento a tutti gli effetti e non costa nulla perche' e' gia' girato.
#
# PERCHE' 20+20 CONTRO 20 E NON 30+30 CONTRO 60. Con 30+30 contro un run da 60 il
# congiunto avrebbe UN coseno e la pipeline DUE, con un riavvio del learning rate
# a meta': un riavvio e' di per se' una tecnica che cambia l'accuratezza, quindi
# qualunque differenza misurata sarebbe indistinguibile da esso. Con 20+20 contro
# 20 ogni stadio ha lo stesso coseno da venti epoche del riferimento e le rampe
# sono confrontabili. Resta un confondimento, il budget, sessanta epoche-
# equivalenti contro venti: ma e' a NOSTRO SFAVORE, perche' la pipeline riceve il
# doppio delle epoche. Se perde comunque, la conclusione non ha repliche.
#
# NON ESISTE UNA SCORCIATOIA CON UN FLAG. Un run unico da 40 epoche con
# --entropy_warmup_epochs 20 accenderebbe T3 a meta' lasciando un coseno solo, ma
# nel trainer quel flag e' accoppiato allo scheduler,
# _ramp_end = entropy_warmup_epochs + max(0, sparsity_warmup_epochs), quindi
# terrebbe il learning rate piatto per venti epoche e poi farebbe un coseno da
# venti: lo stesso confondimento spostato di posto. E per T2 il gate simmetrico
# non esiste, quindi la direzione inversa e' comunque inesprimibile.
#
# TUTTI E QUATTRO GLI STADI PORTANO --entropy_warmup_epochs 1 anche quando T3 e'
# zero. Non e' una svista: quel flag fissa _ramp_end e quindi la forma del
# coseno, e lasciarlo a 1 ovunque e' cio' che rende i quattro stadi e il
# riferimento identici nello scheduler.
#
# LO STADIO 2 RICARICA I PESI LATENTI MA NON GLI STEP SIZE, che vengono
# re-inizializzati con MSE sui latenti caricati. Per una pipeline e' la
# definizione giusta -- Deep Compression pota e poi quantizza da capo sui pesi
# potati -- ed e' anche parte del meccanismo con cui il secondo stadio puo'
# distruggere il lavoro del primo, che e' esattamente cio' che l'esperimento
# vuole misurare. Va dichiarato nel paper, non nascosto.
#
# COSTO: 139s per epoca senza il duale e 480 con, misurati sui test_245 e
# test_239. Ogni direzione e' quindi 20x139 + 20x480 = 3 ore e 26 minuti in
# totale sui due stadi.

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
mkdir -p "$WORK/acardia0/METaQCheckpoints/ResNet-18"
export METAQ_CHECKPOINT_PATH="$WORK/acardia0/METaQCheckpoints/ResNet-18/pipeB_stage1_t3.pt"

srun --ntasks=$SLURM_NTASKS --ntasks-per-node=1 bash -lc '
    export METAQ_CHECKPOINT_PATH="'"$METAQ_CHECKPOINT_PATH"'"
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
    --n_epochs 20 \
    --lr 1e-2 \
    --optimizer_weight_decay 1e-4 \
    --perspective_coeff 1e-5 \
    --entropy_coeff 3e-8 \
    --sparsity_coeff 0 \
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
