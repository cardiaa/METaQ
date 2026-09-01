#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --account=IscrC_ObCTDoNN
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=14:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# test_282: AlexNet, ablazione dei termini -- PRESTO completo.
# Coefficienti (T1,T2,T3) = (1e-5,3e-7,6e-8). Ogni altro flag e' identico negli
# altri quattro bracci: la sola differenza fra i cinque run e' questa terna.
#
# PAVIMENTO DEL LEARNING RATE ALL'1 PER CENTO, non al 5, e la ragione e'
# diagnostica. Con --min_lr 2e-5 questo braccio differisce dal test_225 in UN
# SOLO flag, --lsq_per_channel, quindi il confronto fra i due isola esattamente
# il costo del passaggio a per-tensore su una rete di otto tensori enormi. Con
# 1e-4 le differenze sarebbero state due e non avremmo saputo a chi attribuire
# un eventuale calo. La cautela storica sul pavimento all'1 per cento (test_225
# misurava level_change 0.115 per cento per epoca, rete congelata) riguardava la
# FASE PIATTA del phase schedule con T2 a piena forza; qui metaq_ramp_epochs e'
# zero, il coseno copre tutte le sessanta epoche e il pavimento viene sfiorato
# solo alla fine. ResNet-18 gira l'intera sua frontiera con T3 acceso a questo
# stesso pavimento dell'1 per cento.
#
# IL BRACCIO (0,0,0) NON VIENE ESEGUITO. Su ResNet-18 l'ablazione mostra che il
# solo ridge non fa nulla, 9.79 contro 9.88 per cento del controllo, e quel
# risultato non ha bisogno di essere ripetuto. Il controllo di questa ablazione
# e' quindi il braccio solo-T1, che e' anche l'unico direttamente confrontabile
# con il test_225.
#
# T3 RIDOSATO DA 6e-8 A 2.3e-8 (28 agosto). La dose trasferita da ResNet-18 e'
# risultata pesantemente sovradosata su AlexNet, cosa che l'aspettativa non
# prevedeva: pensavamo che dopo T2 non restasse spazio per l'entropia, e invece
# T3 da SOLO comprime piu' di T2 da solo. Il test_279 all'epoca 25 stava gia' a
# 2.55 per cento del modello, 39.3x, ma a -1.96 di accuratezza, contro il 3.04
# per cento e -0.754 che il braccio T2 raggiunge in sessanta epoche complete.
# Comprime di piu' e costa il doppio.
#
# LA NUOVA DOSE VIENE DALLA TRAIETTORIA DEL test_279, non da un modello. Quel run
# e' di fatto la curva dose-risposta di T3 su AlexNet letta un'epoca alla volta,
# che e' il metodo che questo progetto usa da quando tre modelli analitici di
# fila hanno sbagliato. La somma di schedule entropica su sessanta epoche vale
# 30.29 e all'epoca 13, dove il pacchetto tocca 2.96 per cento, ne sono state
# accumulate 11.65: frazione 0.385, quindi la dose fissa di pari esposizione e'
# 6e-8 x 0.385 = 2.3e-8. A quella dose il pacchetto atteso e' vicino ai 3 per
# cento del braccio T2, il che mette i due termini a confronto ALLA STESSA
# DIMENSIONE, che e' l'asse su cui l'ablazione deve pronunciarsi.
#
# WALL DA 10 A 14 ORE. La stima di 400s per epoca veniva da run AlexNet storici
# di un'altra fase del progetto. Il costo misurato e' 682s per il braccio
# entropico e 670 per il completo, cioe' 533s di duale sopra i 149 di base:
# sessanta epoche sono 11.2 ore e i due run precedenti sarebbero morti intorno
# all'epoca 53. Quattordici ore lasciano margine anche se il cluster rallenta.
#
# COSA NON CAMBIA: T2 resta a 3e-7 nel braccio completo, quindi i test_277 e
# test_278 restano validi come bracci dell'ablazione, perche' in entrambi T3 e'
# zero e il ridosaggio non li tocca.
#
# PERCHE' L'ABLAZIONE ALEXNET VA RIFATTA DA ZERO. Frangioni ha chiesto quanto
# contribuisce il termine di potatura. Su ResNet-18 l'ablazione a cinque bracci
# esiste e risponde. Su AlexNet quello che avevamo NON e' confrontabile, per
# quattro motivi tutti verificati sui log:
#   1. il braccio T2 (test_229) e' una RAMPA, --metaq_ramp_epochs 40, mentre
#      ogni altro risultato del paper e' a dose fissa;
#   2. T2 non era un coefficiente piatto ma la regola analitica per-layer,
#      --layerwise_t2_targets con --t2_calibration displacement e --t2_scale;
#   3. --lsq_per_channel Y contro N di tutte le altre cinque architetture;
#   4. --entropy_warmup_epochs 0 contro 1, e nessun --min_lr.
# E il passo duale relativo su AlexNet non e' mai stato usato, non per svista ma
# perche' nessun run della ricetta congelata ha mai acceso T3: il duale non e'
# mai stato invocato e il --dual_step 3e-9 assoluto che quegli script portano e'
# rimasto innocuo. Qui il duale serve davvero, in due bracci su cinque.
#
# COSA E' STATO ALLINEATO alla ricetta comune: per-tensore, entropy_warmup 1,
# min_lr 1e-4, nessuna rampa, --sparsity_coeff piatto, duale 1.3e-2 relativo,
# max_iterations 3, entropy_every 4, C=16 con --layer_C agli estremi a 8 bit.
# COSA RESTA SPECIFICO DI ALEXNET e va dichiarato in appendice: lr 2e-3 con
# warm-up di una epoca (rete senza normalizzazione), distillazione a 0.5,
# --layer_C sulle due convoluzioni di ingresso, e --lsq_scale_lr 1e-3 in
# modalita' relativa invece di 1e-5 assoluta. Su quest'ultimo l'equivalenza e'
# stretta proprio qui: il passo medio iniziale di AlexNet e' 0.00742, quindi
# 1e-3 relativo vale 7.4e-6 assoluti contro i 1e-5 di ResNet, entro il 25 per
# cento. Su EfficientNet-B0 lo scarto e' invece di otto volte, e li' e' la
# modalita' relativa a essere quella giusta.
#
# LA DOSE T2 E' L'UNICO NUMERO INCERTO. Non abbiamo mai fatto girare AlexNet con
# un --sparsity_coeff piatto sulla ricetta congelata: l'unico tentativo, il
# test_226, era inerte perche' usava una regola duale tre o quattro ordini di
# grandezza troppo debole. Il valore qui, 3e-7, e' la media pesata per parametri
# del vettore per-layer calibrato nel test_229, dominata da fc6 (2.61e-7) e fc7
# (3.83e-7) che da soli sono il 95 per cento della rete. Il braccio T1+T2 costa
# meno di tre ore e non tocca il duale: va letto per primo, e se la sparsita' che
# riporta e' lontana dal ginocchio misurato (42 per cento) i due bracci
# entropici vanno ridosati prima di spendere le tredici ore che costano.
#
# T3 = 6e-8 e' la dose produttiva di ResNet-18, e trasferisce per costruzione:
# batch 64 su sedici GPU danno 1251 passi per epoca ed entropy_every 4, cioe' le
# stesse 313 chiamate duali per epoca.

LOG_DIR=$WORK/acardia0/LeonardoTests
mkdir -p "$LOG_DIR"
# NUMERAZIONE FISSA. I bracci partono a coppie e l'auto-incremento
# li farebbe collidere sullo stesso file di log.
NEXT=282
LOG_FILE=$LOG_DIR/Leonardo_test_${NEXT}.log
if [ -e "$LOG_FILE" ]; then
    echo "Leonardo_test_282.log esiste gia': rifiuto di sovrascriverlo." >&2
    exit 1
fi
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
        --model_name AlexNet \
        --delta -10 \
        --gamma 2 \
        --data_root /leonardo_work/IscrC_ObCTDoNN/acardia0/datasets \
        --train_workers 4 \
        --val_workers 2 \
        --batch_size 64 \
        --n_epochs 60 \
        --lr 2e-3 \
    --min_lr 2e-5 \
        --lr_warmup_epochs 1 \
        --optimizer_weight_decay 1e-4 \
        --perspective_coeff 1e-5 \
        --entropy_coeff 2.3e-8 \
        --sparsity_coeff 3e-7 \
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
        --lsq_per_channel N \
        --joint_lsq_metaq Y \
        --distillation Y \
        --distill_alpha 0.5 \
        --distill_tau 1.0 \
        --bn_recalibration_batches 0 \
        --C 16 \
        --layer_C 256,256,16,16,16,16,16,16 \
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
