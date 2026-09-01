# PRESTO — every completed run, in the paper's own metric

Compiled 26 August 2026 from the archived logs in `LeonardoTests/`, complete
through test_282.

**How to read every number.** `A_Q` is top-1 accuracy of the deployed quantized
model. `Δ FP32` is `A_Q` minus the full-precision checkpoint measured by the same
harness. **Packet** is the serialized sparse package (support mask + nonzero
values + metadata) divided by the FP32 bytes of the *quantized weight tensors*;
lower is better. **Model** is the same packet divided by the bytes of the *whole
checkpoint*, which is the accounting every published method uses; `CR` is its
reciprocal and is therefore the number to quote against the literature. Sparsity
is the fraction of zero symbols emerging from the LSQ zero bin — no external
pruning rule is ever applied. All runs use a 16-symbol codebook unless noted,
weights only, activations in FP32.

**Converting between the two.** Writing `f` for the quantized fraction of the
model, `Model = Packet × f + (1 − f)`. Every log prints `f` on its
`[QUANTIZED TENSORS]` line.

| network | f | 1 − f | FP32 size |
|---|---|---|---|
| ResNet-18 | 99.909% | +0.091 | 46.76 MB |
| ResNet-50 | 99.788% | +0.212 | 102.2 MB |
| DeiT-Small | 99.374% | +0.626 | 88.2 MB |
| ViT-B/16 | 99.682% | +0.318 | 346.3 MB |
| AlexNet | 99.983% | +0.017 | 233.1 MB |
| EfficientNet-B0 | 99.010% | +0.990 | 21.4 MB |
| RegNetX-600MF | 99.447% | +0.553 | 24.5 MB |

**Coefficient convention.** `T1` = `perspective_coeff` (ridge), `T2` =
`sparsity_coeff`, `T3` = `entropy_coeff`. Logs *before* test_169 print `T2` and
`T3` in the opposite positions; check the log date before reading old values.

---

## 1. ResNet-18 / ImageNet — the headline cell

FP32 checkpoint: **69.734**

| run | configuration | ep | A_Q | Δ FP32 | packet | model | CR | sparsity |
|---|---|---|---|---|---|---|---|---|
| test_174 | LSQ only | 20 | 69.888 | +0.154 | 9.34% | 9.42% | 10.61× | 20.09% |
| test_240 | LSQ only | 60 | 70.490 | **+0.756** | 9.76% | 9.84% | 10.16× | 18.16% |
| test_196 | PRESTO, T3 = 3e-8, slow dual | 20 | 69.840 | +0.106 | 6.08% | 6.17% | 16.21× | 45.80% |
| test_235 | PRESTO, T3 = 3e-8, fast dual | 20 | 69.668 | −0.066 | 5.74% | 5.83% | 17.15× | 49.18% |
| test_236 | PRESTO, T3 = 3e-8 | 40 | 70.080 | **+0.346** | 5.73% | 5.82% | 17.18× | 51.53% |
| test_197 | PRESTO, T3 = 6e-8, slow dual$^{\dagger}$ | 40 | 69.578 | −0.156 | 5.21% | 5.30% | 18.87× | 54.40% |
| **test_239** | **PRESTO, T3 = 6e-8** | **60** | **69.936** | **+0.202** | **4.82%** | **4.91%** | **20.37×** | 61.83% |
| test_247 | seed repeat of test_239 | 60 | 70.022 | +0.288 | 4.80% | 4.89% | 20.45× | 61.85% |
| test_248 | seed repeat of test_239 | 60 | 70.102 | +0.368 | 4.79% | 4.88% | 20.51× | 61.84% |
| test_237 | PRESTO, T3 = 1e-7 | 40 | 69.406 | −0.328 | 4.21% | 4.30% | 23.26× | 67.37% |
| test_238 | PRESTO, T3 = 1e-7 | 60 | 69.606 | −0.128 | 4.14% | 4.23% | 23.64× | 68.55% |
| **test_257** | **PRESTO, T3 = 1e-7** | **100** | **69.860** | **+0.126** | **4.15%** | **4.24%** | **23.58×** | 69.66% |

$^{\dagger}$ **test_197 non è appaiato agli altri**: è l'unico run del progetto
con `lsq_scale_lr_schedule=False`, cioè il learning rate degli step size tenuto
costante invece che agganciato al coseno. Non entra nel paper e non va usato per
calcolare costi marginali.

Common recipe: batch 64 per GPU (global 1024, 16 A100), SGD lr 1e-2, `min_lr`
1e-4, wd 1e-4, C = 16, no distillation, `bn_recalibration_batches` 50, T1 = 1e-5,
T2 = 1e-7, `entropy_every` 4, `max_iterations` 3. "Fast dual" is
`--dual_step 1.3e-2 --dual_step_mode relative`; the slow-dual rows predate it.

**Seed study** (test_239 / 247 / 248, identical configuration):
accuracy 69.936 / 70.022 / 70.102 → **mean 70.020, s.d. 0.083**;
packet 4.82 / 4.80 / 4.79 → **mean 4.803%, s.d. 0.015** (model 4.894%, 20.43×).
The margin above FP32 is **+0.286 ± 0.083, i.e. 3.4 σ**.

**Term ablation**, 60 epochs, everything else identical:

| run | (T1, T2, T3) | A_Q | Δ FP32 | packet | model | CR |
|---|---|---|---|---|---|---|
| test_240 | (0, 0, 0) | 70.490 | +0.756 | 9.76% | 9.84% | 10.16× |
| test_244 | (1e-5, 0, 0) — ridge only | 70.392 | +0.658 | 9.79% | 9.87% | 10.13× |
| test_245 | (1e-5, 1e-7, 0) — no entropy | 70.312 | +0.578 | 6.88% | 6.96% | 14.36× |
| test_246 | (1e-5, 0, 6e-8) — no sparsity | 70.090 | +0.356 | 4.94% | 5.03% | 19.88× |
| test_239/247/248 | full | 70.020 | +0.286 | 4.80% | 4.89% | 20.45× |

Three conclusions. **T1 alone does nothing**: 9.79% against the control's 9.76%,
which kills the objection that the gain is a ridge in disguise. **T3 is the
workhorse on this network**, 19.88× alone against T2's 14.33× alone — the
opposite of AlexNet and EfficientNet-B0. **The two do not add**: the full method
reaches 4.80% against T3-alone's 4.94%, 2.8% more packet for 0.070 accuracy
points. T2 is a refinement, not a multiplier.

### Pipeline contro ottimizzazione congiunta (test_273-276, 27 agosto 2026)

Due pipeline da 20+20 epoche contro il congiunto, stessi coefficienti
(T1,T2,T3) = (1e-5, 1e-7, 3e-8), stessa ricetta. Ogni stadio è un coseno da 20
epoche identico a quello del test_235, quindi le rampe sono confrontabili; il
confondimento residuo è il budget e **favorisce la pipeline**, che ne riceve il
doppio. Lo stadio 2 riparte dai pesi latenti dello stadio 1 e ri-inizializza gli
step size con MSE, che è la definizione giusta di pipeline — Deep Compression pota
e poi quantizza da capo — ed è anche parte del meccanismo.

| run | contenuto | ep | A_Q | Δ FP32 | packet | model | CR | sparsity |
|---|---|---|---|---|---|---|---|---|
| test_235 | congiunto | 20 | 69.668 | −0.066 | 5.74% | 5.83% | 17.17× | 49.18% |
| **test_236** | **congiunto** | **40** | **70.080** | **+0.346** | **5.73%** | **5.82%** | **17.19×** | 51.53% |
| test_273 | A stadio 1, solo T2 | 20 | 70.098 | +0.364 | 7.38% | 7.46% | 13.40× | 38.58% |
| test_275 | **A = T2 poi T3** | **40** | 70.114 | +0.380 | 5.99% | **6.08%** | 16.46× | 48.81% |
| test_274 | B stadio 1, solo T3 | 20 | 69.782 | +0.048 | 6.00% | 6.09% | 16.43× | 46.29% |
| test_276 | **B = T3 poi T2** | **40** | 70.108 | +0.374 | 7.13% | **7.21%** | 13.86× | 41.02% |

> I Δ stampati nei log di test_275 e test_276 sono **spuri** (+68.6 e +59.9): la
> riga `PRETRAINED BASELINE` valuta in FP32 i pesi latenti caricati dallo stadio 1,
> che senza quantizzazione non significano nulla. Il riferimento resta 69.734.

**A budget appaiato le tre accuratezze coincidono** — 70.080, 70.114, 70.108,
entro 0.034 punti contro una sigma di 0.083 — quindi il confronto è puramente
sulla dimensione, senza compromessi da discutere. **Il congiunto è il 4.3% più
piccolo della pipeline migliore e il 19.4% della peggiore.**

**Cosa comprano le venti epoche in più**, partendo dal congiunto a 20:

| | accuratezza | dimensione |
|---|---|---|
| congiunto → 40 ep | +0.412 | **−0.010 punti** |
| pipeline A | +0.446 | **+0.250 punti** |
| pipeline B | +0.440 | **+1.389 punti** |

Tutte e tre convertono il budget extra nella stessa accuratezza. **Solo il
congiunto conserva la dimensione**; le pipeline la peggiorano.

**IL MECCANISMO, misurato sull'entropia dei simboli non nulli.** La
ri-quantizzazione fra i due stadi costa **0.343 bit** in direzione A e **0.445**
in direzione B: la griglia si re-inizializza sui latenti caricati e rimappa i
simboli. Lo stadio entropico se li riguadagna (−0.742 bit in A/2); **lo stadio di
sparsità no**, ne aggiunge altri +0.040 invece di recuperarli. Bilancio della
direzione B, che è l'ipotesi di Andrea: lo stadio T3 guadagna 0.759 bit, la
re-inizializzazione ne restituisce 0.445 e T2 ne aggiunge 0.040, così la pipeline
finisce **0.274 bit sopra il punto da cui lo stadio 1 era partito**. Tutto il
lavoro entropico distrutto, e un po' oltre. **L'ipotesi è confermata e
quantificata**, ed è anche il motivo per cui l'ordine conta: A batte B di 1.13
punti di pacchetto.

---

**Budget buys both axes.** test_237 → test_238 (40 → 60 epochs) gained +0.200
accuracy *and* −0.07 packet. test_238 → test_257 (60 → 100) gained a further
+0.254 accuracy at the same packet. test_257 was still gaining 0.03 points per
epoch when the schedule ended. Every ResNet-18 run in the sequence has been
budget-limited, never dose-limited.

---

## 2. ResNet-50 / ImageNet — the transfer cell

FP32 checkpoint: **76.122** (torchvision `IMAGENET1K_V1`, the checkpoint on which
DeepCABAC and HEMP report 76.13).

| run | configuration | ep | A_Q | Δ FP32 | packet | model | CR | sparsity |
|---|---|---|---|---|---|---|---|---|
| test_242 | LSQ only | 20 | 76.554 | +0.432 | 9.78% | 9.97% | 10.03× | 22.08% |
| **test_243** | **PRESTO, T3 = 3e-8** | **20** | **76.318** | **+0.196** | **4.92%** | **5.12%** | **19.53×** | 60.03% |
| test_249 | PRESTO, T3 = 1e-7 | 20 | 75.444 | −0.678 | 3.52% | 3.72% | 26.85× | 75.26% |

Recipe identical to ResNet-18 with the model name changed. At a matched budget
PRESTO **halves** the control's packet (−49.7%) for 0.236 accuracy points and
stays above the checkpoint — quantitatively the same trade as ResNet-18 (−50.6%
for 0.554 points, +0.202 above). The method transfers inside the family.

Marginal cost 243 → 249: 0.624 accuracy points per point of packet.
test_243 was still rising at +0.197 per epoch when the schedule ended.

---

## 3. DeiT-Small / ImageNet — the transformer frontier

FP32 checkpoint: **79.742**

| run | configuration | ep | A_Q | Δ FP32 | packet | model | CR | sparsity |
|---|---|---|---|---|---|---|---|---|
| test_189 | LSQ + distillation | 25 | 79.832 | +0.090 | 11.43% | 11.98% | 8.34× | 13.74% |
| **test_194** | **PRESTO, T3 = 1.5e-8** | **25** | **79.490** | −0.252 | **9.04%** | **9.61%** | **10.41×** | 37.73% |
| test_191 | PRESTO, T3 = 3e-8 | 25 | 79.294 | −0.448 | 7.76% | 8.34% | 11.99× | 44.15% |
| test_192 | PRESTO, T3 = 3e-8 | 40 | 79.200 | −0.542 | 7.36% | 7.94% | 12.60× | 49.90% |
| test_241 | PRESTO, T3 = 1.5e-8, fast dual | 9/25 | 78.648 | −1.094 | 9.06% | 9.63% | — | — |

**BLOCCO RIFATTO CON DUALE RELATIVO E CONTROLLO APPAIATO (27 agosto 2026).**

| run | (T2, T3) | ep | A_Q | Δ FP32 | packet | model | CR | sparsity |
|---|---|---|---|---|---|---|---|---|
| **test_265** | **controllo (0,0,0)** | 25 | **79.706** | **−0.036** | 11.44% | 11.99% | 8.34× | 13.66% |
| test_266 | (1e-8, 3e-9) — un quinto | 25 | 79.694 | −0.048 | 11.43% | 11.98% | 8.35× | 21.54% |
| **test_267** | **(2.5e-8, 7.5e-9) — metà** | 25 | **79.704** | −0.038 | **10.07%** | **10.63%** | **9.40×** | 30.89% |
| **test_270** | **(5e-8, 1.5e-8) — dose piena** | 25 | **79.378** | −0.364 | **8.49%** | **9.06%** | **11.03×** | 39.67% |
| test_271 | (1e-7, 3e-8) — doppia | 25 | 78.948 | −0.794 | 7.16% | 7.74% | 12.92× | 48.83% |

Il controllo appaiato atterra **0.126 sotto il test_189**, come previsto: il
pavimento al 5% ricuoce meno. DeiT-Small esce quindi dalla classe "controllo
lossless" ed entra in quella di ViT-B/16, frontiera contro controllo appaiato.

**La frontiera è ancora tutta nel tratto gratis.** test_267 toglie 1.36 punti di
pacchetto per **due millesimi** di accuratezza; test_266 non comprime affatto pur
portando la sparsità dal 13.66 al 21.54%, perché sotto il 50% ogni zero in più
rende la maschera più cara di quanto i valori risparmino. **Con la dose piena e la doppia la frontiera diventa reale.** Costo marginale in
punti di accuratezza per punto di pacchetto: **0.001** dal controllo al 267
(0.002 punti di accuratezza su 1.37 di pacchetto; il valore 0.007 riportato in
precedenza era un errore di calcolo),
**0.208** dal 267 al 270, **0.325** dal 270 al 271. Monotono crescente, ginocchio
al test_270. E il confronto diretto col vecchio duale assoluto: il test_194, con
gli **stessi** coefficienti del 270, arrivava a 9.61% del modello a 79.490; il 270
col duale relativo arriva a **9.06% a 79.378**. Costo marginale implicito 0.205,
cioè i due punti stanno sulla stessa curva — **il passo relativo muove lungo la
frontiera, non la sposta**.

Recipe: batch 128 per GPU (global 2048), ADAM lr 1e-4, wd 0, `min_lr` 5e-6,
distillation from the FP teacher at α = 0.9, T1 = 1e-5, T2 = 5e-8.

Four points on one curve from a control that is itself lossless. The marginal
cost rises monotonically — 0.143, 0.167, 0.190 accuracy points per point of
packet — which locates the knee at test_194.

test_241 was **killed by the wall-time limit at epoch 9 of 25** and is not a
comparison point.

**IL CONTROLLO NON È APPAIATO (trovato il 26 agosto 2026).** test_189 ha
`min_lr=None`, cioè il pavimento di default all'1% del rate, mentre test_191,
test_192 e test_194 hanno `min_lr=5e-6`, cioè il 5%. Il controllo ricuoce a un
learning rate cinque volte più basso dei run che dovrebbe controllare, e la coda
del coseno è dove si guadagnano gli ultimi decimi. Il verso del bias favorisce il
controllo, quindi il costo di PRESTO qui riportato è semmai una sovrastima — ma
non è un confronto appaiato. Si somma al passo duale assoluto: **tutto il blocco
DeiT va rifatto**, controllo compreso, con
`run_deit_small_lsq_control_matched.sh` che costa 56 minuti.

---

## 4. ViT-B/16 / ImageNet — the widest margin

FP32 checkpoint: **81.056** (torchvision, the checkpoint NNCodec measures at
81.07).

### 4.1 The five recipe-search controls (superseded accounting)

Tests 250–254 ran before the packed-QKV fix and quantized **65,058,816 of
86,567,656 parameters, 75.15% of the model**: torchvision keeps the packed
query/key/value projection of `nn.MultiheadAttention` in a bare parameter named
`in_proj_weight`, which the `.weight`-suffix predicate skipped. **Their packet
numbers are not comparable to anything** and are omitted here. Their accuracies
remain valid and are what selected the recipe:

| run | ep | lr | A_Q final | Δ FP32 | best |
|---|---|---|---|---|---|
| test_250 | 10 | 5e-5 | 80.814 | −0.242 | 80.819 |
| **test_251** | **20** | **5e-5** | **80.914** | **−0.142** | 80.975 (ep 18) |
| test_252 | 20 | 1e-4 | 80.830 | −0.226 | 80.879 |
| test_253 | 20 | 3e-5 | 80.874 | −0.182 | 80.966 (ep 19) |
| test_254 | 30 | 5e-5 | 80.834 | −0.222 | 80.921 (ep 28) |

Three learning rates within a factor of three land within 0.09 of each other,
thirty epochs are worse than twenty, and `level_change` runs 10.5% → 1.1% so the
network is not frozen. **Four-bit LSQ simply sits about a tenth of a point under
this checkpoint and stays there.** There is no headroom to spend, so ViT-B/16 is
reported as a frontier against a budget-matched control, not as a lossless claim.

### 4.2 The runs that count

| run | configuration | ep | A_Q | Δ FP32 | packet | model | CR | sparsity |
|---|---|---|---|---|---|---|---|---|
| test_255 | LSQ + distillation | 20 | 80.816 | −0.240 | 10.19% | 10.48% | 9.55× | 21.81% |
| **test_256** | **PRESTO (1e-5, 5e-8, 1.5e-8)** | **20** | **80.372** | −0.684 | **4.11%** | **4.42%** | **22.65×** | **72.93%** |
| test_269 | stesso punto ma **N_dual 3** | 20 | 80.260 | −0.796 | 4.14% | 4.44% | 22.50× | 72.62% |

Recipe: batch 64 per GPU (global 1024), ADAM lr 5e-5, wd 0, `min_lr` 2.5e-6,
distillation α = 0.9, C = 16, fast dual.

PRESTO removes **60% of the control's packet** for 0.444 accuracy points. The
sparsity, 72.93%, is the highest of any run in the project. The run is clearly
budget-limited: accuracy dips to 79.195 at epoch 8 and then climbs monotonically
to 80.372, still gaining 0.05 points per epoch when the schedule ends.

---

## 5. AlexNet / ImageNet — the sparsity-mechanism cell

FP32 checkpoint: **56.524**

| run | configuration | ep | A_Q | Δ FP32 | packet | model | CR | sparsity |
|---|---|---|---|---|---|---|---|---|
| test_223 | LSQ only, uniform C = 16 | 20 | 55.956 | −0.568 | 10.72% | 10.74% | 9.32× | 15.96% |
| test_224 | LSQ only, 8-bit input convs | 40 | 56.452 | −0.072 | 10.81% | 10.83% | 9.24× | 15.98% |
| test_225 | **LSQ only, frozen recipe** | 60 | 56.552 | **+0.028** | 10.76% | 10.78% | 9.28× | 16.11% |
| test_229 | PRESTO, T2 ramp to full dose | 60 | 54.938 | −1.586 | 3.66% | 3.68% | 27.20× | 70.60% |

### Ablazione sulla ricetta comune (test_277-278, 28 agosto 2026)

Quattro bracci a sessanta epoche, per-tensore, dose fissa con `--sparsity_coeff`
piatto, pavimento all'1%, duale relativo. Il braccio ridge differisce dal
test_225 in **un solo flag**, `--lsq_per_channel`, quindi il confronto fra i due
prezza esattamente il passaggio a per-tensore.

| run | (T1, T2, T3) | ep | A_Q | Δ FP32 | packet | model | CR | sparsity |
|---|---|---|---|---|---|---|---|---|
| test_225 | (1e-5, 0, 0), **per-canale** | 60 | 56.552 | +0.028 | 10.76% | 10.78% | 9.28× | 16.11% |
| **test_277** | **(1e-5, 0, 0), per-tensore** | 60 | **56.500** | −0.024 | 10.56% | **10.58%** | **9.46×** | 17.23% |
| **test_278** | **(1e-5, 3e-7, 0)** | 60 | **55.770** | −0.754 | **3.02%** | **3.04%** | **32.93×** | **78.20%** |

**Il gate è passato.** Il per-tensore costa **0.052** punti e restituisce **0.20**
punti di pacchetto, cioè 3.8 punti per punto di accuratezza. Scomposizione:
metadata −0.017, maschera +0.102, **valori −0.287**. Su AlexNet il metadata è solo
l'**8%** del guadagno, perché otto tensori non generano metadata: il resto è
lunghezza di codice. **Seconda conferma indipendente che il per-canale si paga due
volte**, e qui in forma quasi pura.

**Il braccio T2 è il miglior punto AlexNet che abbiamo mai avuto.** Contro il
test_229, che è la vecchia ricetta (rampa, regola per-layer, per-canale):
**+0.832 di accuratezza a sparsità più alta e dimensione più piccola**, 32.93×
contro 27.20×. Il divario con Deep Compression (2.86%, 35×, perdita nulla) si è
quasi dimezzato: eravamo al 3.68% e −1.586, siamo al 3.04% e −0.754.

**La dose non è sovradosata**, malgrado il 78.20% di sparsità. Il tasso di cambio
fra 277 e 278 è **10.3 punti di pacchetto per punto di accuratezza**, in linea con
i ginocchi misurati altrove (10.75 su EfficientNet-B0, ~10.6 su ResNet-18). Il
ginocchio di AlexNet si è spostato a destra con la nuova ricetta: il test_229 al
70.60% di sparsità pagava già −1.61 dal suo controllo, qui al 78.20% paghiamo
−0.73.

**E resta più spazio a T3 di quanto credessimo.** Attenzione a non confondere due
quantità: 1.688 è l'**entropia zero-order** dei simboli non nulli, non il tasso
serializzato. Sul test_282 i valori pesano 0.854% dei byte FP32 al 79.66% di
sparsità, cioè **1.34 bit per peso sopravvissuto misurati**, sotto l'entropia
perché zstd sfrutta struttura che l'istogramma non descrive. I pesi sopravvissuti
portano **1.688 bit** di entropia ciascuno dopo T2, contro gli 1.12 misurati sulla vecchia ricetta. I
bracci 279 e 280 partono da un margine reale, non da zero.

---

**L'ABLAZIONE È COMPLETA (test_281-282, 28 agosto 2026).** Quattro bracci a
sessanta epoche, una terna di differenza, tutto il resto identico.

| braccio | (T1, T2, T3) | A_Q | Δ FP32 | packet | model | CR | sparsity | H(nz) |
|---|---|---|---|---|---|---|---|---|
| test_277 | (1e-5, 0, 0) — ridge | 56.500 | −0.024 | 10.56% | 10.58% | 9.46× | 17.23% | — |
| test_278 | (1e-5, 3e-7, 0) | 55.770 | −0.754 | 3.02% | 3.04% | 32.93× | 78.20% | 1.688 |
| **test_281** | **(1e-5, 0, 2.3e-8)** | **55.882** | −0.642 | 3.46% | **3.48%** | **28.77×** | 73.72% | 1.721 |
| **test_282** | **completo** | **55.644** | −0.880 | **2.83%** | **2.85%** | **35.13×** | 79.66% | 1.686 |

**IL RIDOSAGGIO HA CAMBIATO LA CONCLUSIONE.** Con T3 = 6e-8 (il test_279 ucciso)
l'entropia sembrava molto meno efficiente della sparsità, 4.15 contro 10.33
punti di pacchetto per punto di accuratezza. A dose corretta i due sono **alla
pari**: T2 rende 10.33, T3 rende **11.49**. La lezione è che il tasso di cambio
si misura al punto operativo giusto, non a una dose trasferita.

**CORREZIONE A UN'AFFERMAZIONE DEL PAPER.** Dicevamo "su ResNet-18 T3 è il
cavallo da tiro, 19.88× contro 14.33×". È vero, ma riguarda **quanto lontano
arriva un termine**, non quanto è efficiente. I tassi di cambio dal controllo:

| | T2 solo | T3 solo |
|---|---|---|
| ResNet-18 | **16.17** pkt/acc, arriva a 14.36× | 12.04 pkt/acc, arriva a **19.89×** |
| AlexNet | 10.33 pkt/acc, arriva a **32.93×** | **11.49** pkt/acc, arriva a 28.77× |

Su entrambe le reti il termine **meno** efficiente per punto è quello che arriva
**più lontano**, e quale sia si inverte fra le due. La formulazione corretta è
sulla portata, non sull'efficienza.

**CORREZIONE (30 agosto 2026) — I TASSI DI CAMBIO NON VANNO USATI PER
CLASSIFICARE I TERMINI.** Ogni *rate* qui sopra è la pendenza di una secante dal
controllo all'estremo di un braccio, su una frontiera concava, e i due bracci di
un blocco non finiscono alla stessa dimensione. Il braccio dosato più lontano
mostra per costruzione la secante più piatta. Su ResNet-18 il braccio più lontano
è T3 (4.94 contro 6.88 di pacchetto) e ha il rate peggiore; su AlexNet è T2 (3.02
contro 3.46) e ha il rate peggiore. **L'inversione fra architetture è esattamente
ciò che una frontiera concava prevede senza nessuna differenza fra i termini.**
In più i denominatori sono dentro il rumore su ResNet-18 (0.178 punti contro
σ_seed 0.083) e la lettura a epoca appaiata fa oscillare il rate di T2 fra −154 e
+16.5 invertendo l'ordine all'epoca 30. Su AlexNet l'ordine è invece stabile a
ogni epoca appaiata, ma l'ultima epoca del test_281 sta 0.082 sotto il proprio
plateau (unico braccio dei nove in questa condizione): sulla media delle ultime
otto epoche il confronto diventa 15.28 contro 10.10, non "alla pari".
**Nel paper la classifica per efficienza è stata rimossa**; resta la portata, che
è un fatto, e la quota di guadagno catturata da ciascun termine.

**LA QUOTA DI GUADAGNO È LA STATISTICA GIUSTA** (scrivendo come 100% la riduzione
di dimensione dell'obiettivo completo, sulla contabilità *model*; identica sul
pacchetto):

| rete | controllo | solo T2 | solo T3 | completo | quota T2 | quota T3 |
|---|---|---|---|---|---|---|
| ResNet-18 | 9.84% | 6.96% | 5.03% | 4.89% | 58.1% | **97.2%** |
| AlexNet | 10.58% | 3.04% | 3.48% | 2.85% | **97.5%** | 91.8% |

**COSTO, mai riportato prima nell'ablazione.** Con T3=0 il solver duale non viene
mai chiamato (`entropy_steps = 0` per tutto il run in 240/244/245/277/278): il
braccio di sparsità gira a 108 s/epoca su entrambe le reti, identico al
controllo. T3 costa **4.0× su ResNet-18 e 6.3× su AlexNet**.

**I DUE TERMINI NON SI SOMMANO — replicato su una seconda architettura.**
Il completo contro il miglior braccio singolo: **1.5** punti di pacchetto per
punto di accuratezza su AlexNet, **2.0** su ResNet-18, contro i 10-16 che i
termini rendono da soli.

**E ORA SAPPIAMO PERCHÉ.** L'entropia dei simboli non nulli:

| | inizio → fine |
|---|---|
| T2 solo | 2.892 → **1.688** |
| T3 solo | 3.326 → **1.721** |
| completo | 2.892 → **1.686** |

**I due termini portano l'entropia dei simboli allo stesso pavimento, ~1.69-1.72
bit, per strade diverse, e combinarli non scende sotto.** Il completo finisce a
1.686 contro gli 1.688 di T2 da solo: identici. È il meccanismo della
non-additività, misurato.

**CONTRO DEEP COMPRESSION.** Loro 2.86% del modello (35.0×) a perdita nulla; il
test_282 è a **2.85% (35.13×) a −0.880**. **Stessa identica dimensione**, 0.88
punti sotto. La vecchia ricetta (test_229) stava a 3.68% e −1.586: il divario si
è dimezzato ed è ora puramente sull'accuratezza, a parità di dimensione.

---

**T2 dose–response curve** (test_229, one row per epoch, deficit against test_225
at the *same* epoch):

| sparsity | packet | CR | deficit |
|---|---|---|---|
| 29.4% | 8.07% | 12.4× | −0.136 |
| 35.5% | 7.18% | 13.9× | −0.202 |
| **42.4%** | **6.34%** | **15.8×** | **−0.144** |
| 51.0% | 5.41% | 18.5× | −0.466 |
| 57.3% | 4.80% | 20.8× | −0.818 |
| 64.8% | 4.14% | 24.2× | −0.842 |
| 70.6% | 3.66% | 27.3× | −1.614 |

The knee sits between 42% and 51%: the deficit triples across that gap and never
recovers.

**STESSO CONFONDIMENTO DI EFFICIENTNET, più altri due.** test_229 ha
`min_lr=1e-4` (pavimento 5%) e test_225 `min_lr=None` (1%); in più test_229 dosa
T2 con la regola per-layer su una rampa mentre ogni altro risultato del paper è a
dose fissa con un coefficiente piatto, e usa `lsq_per_channel Y` contro N di
tutte le altre architetture. **Superato dall'ablazione a cinque bracci**
(test_267–271), internamente appaiata su tutto. These are mid-run rows, so they are not lossless operating points.

**Why T3 has no room here.** After T2, the surviving weights carry **1.12 bits
each** — T2, by inflating the step sizes (fc6 by 5.46×), has already done T3's
job. Contrast EfficientNet-B0 at 2.41 bits and ViT-B/16 at 3.37.

**SUPERATO dal test_282**: il confronto corrente con Deep Compression è 2.85%
contro 2.86% — **stessa dimensione**, e la differenza è tutta sull'accuratezza
(−0.880 contro perdita nulla). Le righe 3.66%/27.2×/−1.59 qui sopra sono la
vecchia ricetta. OPQ resta avanti a entrambi (1.94%, 51.5×, +0.46 nella nostra
contabilità). AlexNet è una vetrina pruning-first: il 95.96% dei parametri sta in
tre layer fully connected. Resta nel paper come metodologia dose–risposta e come
sede delle ablazioni, non come riga competitiva.

---

## 6. EfficientNet-B0 / ImageNet — the depthwise cell

FP32 checkpoint: **77.566**

| run | configuration | ep | A_Q | Δ FP32 | packet | model | CR | sparsity |
|---|---|---|---|---|---|---|---|---|
| test_222 | LSQ, pre-correction recipe | 20 | 75.390 | −2.176 | 11.62% | 12.49% | 8.00× | 15.98% |
| test_230 | **LSQ only, corrected recipe** | 10 | 76.590 | −0.976 | 11.55% | 12.43% | 8.05× | 16.19% |
| test_234 | T3 only, converged dual | 10 | 76.452 | −1.114 | 11.19% | 12.07% | 8.29× | 18.06% |
| test_258 | LSQ only, 30 ep, 5% lr floor | 30 | 76.487 | −1.079 | 11.51% | 12.39% | 8.07× | 16.42% |
| **test_260** | **LSQ only, 8-bit depthwise** | **30** | **76.849** | **−0.717** | 12.18% | 13.05% | 7.66× | 15.48% |
| **test_261** | **LSQ only, 8-bit dw + per-tensor** | **30** | **76.811** | **−0.755** | **11.15%** | **12.03%** | **8.31×** | 16.55% |
| **test_268** | **LSQ only, + 9 hot layer a 8 bit** | **30** | **77.034** | **−0.532** | 11.17% | **12.05%** | **8.30×** | 16.51% |
| **test_272** | **PRESTO (1e-5, 2.1e-7, 6e-8)** | **30** | **76.820** | **−0.746** | **9.25%** | **10.15%** | **9.85×** | 27.99% |

**EFFICIENTNET-B0 È VINTA.** test_272 sta a **10.15% del modello a 76.820**;
NNCodec al suo punto di Tabella 1 sta a **11.25% a 76.78**. Più piccoli del 9.8%
**e** più accurati di 0.04. E a pari dimensione la loro frontiera dà 76.30, quindi
il margine solido è **+0.52**. Cautela da dichiarare: il margine di accuratezza
sul loro punto di punta è 0.04 e sta dentro il rumore (media ultime cinque 76.696);
quello sulla dimensione, 1.10 punti, no.
| test_231 | T2 ramp, endpoint | 20 | 74.746 | −2.820 | 8.15% | 9.06% | 11.04× | 39.83% |

**T2 dose–response curve** (test_231, deficit against test_230 at the same epoch):

| sparsity | packet | model | CR | deficit |
|---|---|---|---|---|
| 20.70% | 10.67% | 11.55% | 8.65× | −0.18 |
| **22.96%** | **10.26%** | **11.15%** | **8.97×** | **−0.12** |
| 25.15% | 9.89% | 10.78% | 9.28× | −0.42 |
| 27.23% | 9.56% | 10.46% | 9.56× | −0.63 |
| 30.95% | 9.21% | 10.11% | 9.89× | −0.97 |
| 32.55% | 8.98% | 9.88% | 10.12× | −1.14 |

**QUESTA CURVA HA DUE CONFONDIMENTI.** test_231 gira a 20 epoche con
`min_lr=1e-3` (pavimento 5%), test_230 a 10 epoche con `min_lr=None` (pavimento
1%): il "deficit alla stessa epoca" confronta run con coseni di lunghezza e
pavimento diversi. La localizzazione del ginocchio regge, perché è un effetto
grande e monotono, ma i deficit numerici non sono differenze a configurazione
appaiata e non vanno citati come tali nel paper.

The knee arrives at 23–27% sparsity, far earlier than AlexNet's 42–51%.
EfficientNet-B0 has 5.3M parameters produced by a search whose objective was
accuracy per parameter: it has little redundancy to give.

**This is the only network where a published method is ahead of us on both
axes.** NNCodec delivers **76.78 (−0.89) at 11.25% of the model, 8.89×**; our
best is 76.590 (−0.976) at 12.43%. Their full Pareto frontier, read from their
Figure 6 (21.45 MB, baseline 77.67):

| model | CR | top-1 | Δ |
|---|---|---|---|
| 13.69% | 7.30× | 77.01 | −0.66 |
| **11.25%** | **8.89×** | **76.78** | **−0.89** |
| 10.08% | 9.92× | 76.29 | −1.38 |
| 9.03% | 11.07× | 75.86 | −1.81 |
| 8.07% | 12.39× | 75.23 | −2.44 |

**THE BUDGET LEVER DOES NOT WORK HERE (test_258, 26 August 2026).** Tripling the
budget bought nothing: 76.487 at thirty epochs against 76.590 at ten. The network
did train — the first percentile of the latents moved **−8.93%** against
test_230's −3.07%, three times as far — and landed in the same place.
`level_change` held near 4.5% for twenty epochs before the cosine put it away.
**This is a representational ceiling, not an optimization one.** Unlike every
other network, LSQ-only here neither fattens nor slims the packet: 11.66% → 11.51%
over thirty epochs.

**WHERE THE CEILING IS.** Clipping fractions at epoch 30, the sixteen depthwise
tensors against the other sixty-six:

| | mean, ep 1 | mean, ep 30 | max, ep 30 |
|---|---|---|---|
| depthwise (16 tensors, 3.48% of weights) | 5.50% | **6.58%** | **28.91%** |
| everything else (66 tensors) | 2.43% | 1.93% | 9.14% |

Three and a half times the clipping, and moving the **wrong way** over training
while the rest of the network improves. Five of the eight worst layers are
depthwise; `features.7.0.block.1.0`, the 5×5 depthwise of the last MBConv stage,
ends with 28.91% of its weights pinned to the grid endpoints. A depthwise kernel
gives each output channel a single input channel, so its quantization error never
averages over a sum. test_260 probes this with `--layer_C` at 8 bits on those
sixteen tensors plus the input convolution, which costs at most 0.435% of FP32.

**THE DEPTHWISE DIAGNOSIS WAS RIGHT (test_260, 26 August 2026).** Putting the
sixteen depthwise convolutions and the input convolution at 8 bits took the
depthwise mean clipping from **6.58% to 1.57%**, indistinguishable from the 1.84%
of the rest of the network, and bought **+0.362** accuracy points on the final
epoch, **+0.446** on the mean of the last five. The parameter-weighted mean
clipping is now 1.01% and the classifier, 24.45% of the weights, sits at 0.33%:
no second bottleneck of that kind is left.

**BUT IT DID NOT MOVE THE FRONTIER, ONLY THE CEILING.** It cost 0.67 packet
points, almost all in the values stream (8.89% → 9.47%), because 8-bit tensors
hand the coder a wider alphabet. Exchange rates on this network, in packet points
bought per accuracy point spent:

| channel | rate |
|---|---|
| T2 up to its knee (test_231, 11.55 → 10.26) | **10.75** |
| T3 (test_234, 10 ep at 3e-8) | 2.61 |
| 8-bit depthwise, read in reverse | 1.85 |
| T2 beyond the knee (test_231 rows 3–6) | 1.17 |

Projected to NNCodec's operating point of 11.25% of the model, the 4-bit branch
lands at −1.19 and the 8-bit-depthwise branch at −1.03, against their −0.89. And
at the size T2's knee reaches on its own, 11.75%, we project −0.84 where their
own frontier interpolates to −0.843: **a tie, not a win.**

**WHAT IS LEFT IS THE METADATA, AND IT IS EXACTLY THE SIZE OF THE GAP.** The
per-channel step sizes cost a hard **0.599%** of FP32, identical across
test_230/231/258/260 because it is a fixed count of fp32 numbers. Per-tensor
scales would replace it with 0.0016%, worth 0.597 packet points — 0.23 accuracy
points at T3's exchange rate, against a deficit of 0.14. Per-channel exists to
give each output channel its own scale where a shared one fits badly, which on
this network meant the depthwise layers; those now carry 8 bits, so we are paying
for two substitutable mechanisms at once. test_261 tests dropping one.

**PER-TENSOR SCALES ARE PAID FOR TWICE (test_261, 26 August 2026).** Dropping
per-channel step sizes cost **0.098** accuracy points on the mean of the last
five epochs and bought **1.03** packet points — a rate of **10.5 packet points
per accuracy point**, as cheap as T2 at its knee and five times cheaper than the
8-bit depthwise trade. Only 0.597 of that was the predicted metadata saving:

| | metadata | values | mask | total |
|---|---|---|---|---|
| test_260 per-channel | 0.599% | 9.473% | 2.110% | 12.18% |
| test_261 per-tensor | 0.002% | 9.128% | 2.016% | 11.15% |
| difference | −0.597 | **−0.345** | −0.094 | −1.03 |

One shared grid per tensor gives the coder **one** symbol distribution instead of
one per output channel, and the pooled stream is more concentrated. Per-channel
quantization therefore costs code length as well as metadata, which is a result
worth stating on its own. It also puts this network back in line with the other
four, all of which run per-tensor.

**Two more facts from this network that belong in the paper.** First, the per-channel
LSQ scales cost a fixed **0.599% of FP32** — 5.2% of the packet at the control
and 7.3% at 8.15%, so metadata becomes a growing share as compression improves,
and NNCodec does not pay it because its local scaling factors are merged into the
folded BatchNorm parameters. Second, the network never reaches lossless: the
corrected recipe took the gap from −2.18 to −0.98 on a ten-epoch budget, and the
curve was still rising.

---

## 7. RegNetX-600MF — ESCLUSA DAL PAPER (30 agosto 2026)

**Questa architettura non compare piu' nel paper.** L'unico run e' stato ucciso dal
wall time a 15 epoche su 20, quindi non e' un esperimento completato e non e'
appaiato a un controllo. La riga resta qui come registro, ma il paper non la cita
piu' in nessun punto: ne' nella tabella delle architetture aggiuntive, ne' nella
nota sulle architetture di trasferimento, ne' in bibliografia.


FP32 checkpoint: **73.676**. test_198 (T1 = 1e-5, T2 = 1e-7, T3 = 3e-8) was
stopped by the wall-time limit after 15 of 20 epochs: **72.888 (−0.788) at 8.79%
packet, 9.30% of the model, 10.76×**, sparsity 31.82%. Not a comparison point; it
documents that the recipe transfers to grouped convolutions.

---

## 8. Results that cut across the networks

### 8.1 Quantization-aware training on its own makes the model *bigger*

Packet at epoch 1 against packet at the end, and the entropy of the nonzero
symbols over the same interval, for every LSQ-only control:

| control | network | packet ep 1 → end | H(nonzero) ep 1 → end |
|---|---|---|---|
| test_174 | ResNet-18, 20 ep | 8.63% → 9.34% (+8.2%) | 2.796 → 3.080 (**+0.284**) |
| test_240 | ResNet-18, 60 ep | 8.59% → 9.76% (**+13.6%**) | 2.788 → 3.215 (**+0.427**) |
| test_242 | ResNet-50, 20 ep | 8.83% → 9.78% (+10.8%) | 2.854 → 3.147 (**+0.293**) |
| test_189 | DeiT-Small, 25 ep | 10.86% → 11.43% (+5.2%) | 3.354 → 3.545 (**+0.191**) |
| test_255 | ViT-B/16, 20 ep | 9.67% → 10.19% (+5.4%) | 3.169 → 3.316 (**+0.147**) |

Four architectures, no exception: QAT alone makes the symbol distribution more
uniform, i.e. **less** compressible, while it gains accuracy. PRESTO does not
improve a trend, it inverts one. The same quantity under the regularizer:
test_236 −0.468, test_238 −0.717, test_239 −0.599, test_243 −0.553, test_249
−0.825, test_257 −0.643, test_194 −0.540, **test_256 −1.415**.

### 8.2 The two terms are two mechanisms, not two dials

**La tabella va fatta su ResNet-18, dove l'ablazione è appaiata.** Quella su
EfficientNet-B0 più sotto confronta test_231 (5% di pavimento, 20 epoche) con
test_230 e test_234 (1%, 10 epoche), quindi il braccio T2 è confuso. I cinque
bracci di ResNet-18 sono identici in ogni flag tranne la terna, e dicono la
stessa cosa più nettamente. Movimento del primo percentile dei latenti,
composizione del pacchetto ed entropia dei simboli non nulli, dall'inizio alla
fine di ciascun run:

| braccio | latenti p1 | maschera | valori | H(non nulli) |
|---|---|---|---|---|
| test_240 controllo (0,0,0) | +7.76% | 2.417 → 2.103 | 6.171 → 7.660 | 2.788 → **3.215** |
| test_244 solo T1 | +11.66% | 2.417 → 2.108 | 6.077 → 7.682 | 2.782 → 3.210 |
| **test_245 T1+T2** | **+14.31%** | 2.632 → **2.914** | 5.213 → 3.971 | 2.502 → **2.526** |
| **test_246 T1+T3** | +11.25% | 2.419 → 2.787 | 6.088 → **2.153** | 2.784 → **1.926** |
| test_239 completo | +11.53% | 2.639 → 2.755 | 5.019 → 2.066 | 2.501 → 1.902 |

**T2 contrae i latenti più di ogni altro braccio** (14.31% contro l'11.66% del
solo ridge). **ERRATA (30 agosto 2026): non è l'unico la cui maschera cresce** —
la maschera del braccio T3 cresce 2.419 → 2.787, cioè *più* di quella di T2
(2.632 → 2.914), e T3 da solo arriva al 60.30% di sparsità contro il 45.56% di
T2. Entrambi i termini producono zeri; il discriminante è l'entropia dei simboli,
non la maschera. **T3 lascia i latenti dove li lascia il solo ridge** (+11.25% contro
+11.66%) e fa tutto il suo lavoro sui **valori** (−65%) e sull'**entropia dei
simboli** (−0.858 bit). T2 l'entropia dei simboli non la tocca (+0.024, piatta);
il controllo la fa **salire** di +0.427. Due meccanismi separabili, su cinque run
che differiscono solo nella terna dei coefficienti.

La tabella qui sotto misura la stessa cosa su EfficientNet-B0 ed è quella
storicamente citata, ma i suoi tre run non sono appaiati sul pavimento del
learning rate né sul numero di epoche:

| | step size (median) | latent p1 | mask | values | H(nonzero) |
|---|---|---|---|---|---|
| control (test_230) | ×0.980 | −3.07% | +0.037 | −0.152 | −0.036 |
| **T2** (test_231) | **×1.359** | **−6.99%** | +1.049 | −4.558 | −1.177 |
| **T3** (test_234) | ×0.988 | −3.23% | +0.172 | −0.649 | **−0.156** |

**T2 works by moving the grid and the weights**: it inflates the step size by 36%
and reshapes the latent distribution twice as much as the control, which widens
the zero bin and produces sparsity. This is the same mechanism CoDeQ
(arXiv 2512.12981, Dec 2025) identifies independently — the dead zone of a scalar
quantizer *is* magnitude pruning.

**T3 works without moving either**: its step sizes and latent percentiles are
indistinguishable from the control, and all of its gain comes from reassigning
weights among symbols.

**Which channel is efficient depends on the network.** On ResNet-18 T3 alone
reaches 19.88× against T2 alone at 14.33×. On AlexNet and EfficientNet-B0 the
ordering reverses and T2 is about four times the cheaper channel. The predictor
is how many bits the survivors still carry after T2: 1.12 on AlexNet (nothing
left for T3), 2.41 on EfficientNet-B0, 3.37 on ViT-B/16.

### 8.3 The mask is most expensive at 50% sparsity

Packet composition, ResNet-18:

| run | sparsity | mask | values | metadata |
|---|---|---|---|---|
| test_174 (LSQ) | 20.09% | 2.211% | 7.132% | 0.000197% |
| test_236 | 51.53% | 2.905% | 2.828% | 0.000197% |
| test_239 | 61.83% | 2.755% | 2.066% | 0.000197% |
| test_238 | 68.55% | 2.563% | 1.573% | 0.000197% |
| test_257 | 69.66% | 2.522% | 1.632% | 0.000197% |

The cost of the support mask is `H(p)/32`, maximal at p = 1/2, so past 50%
sparsity every extra zero is rewarded twice. test_236 sits almost exactly on the
maximum. Metadata is negligible with per-tensor scales (0.0002%) and a hard floor
with per-channel ones (0.599% on EfficientNet-B0, 0.00006% on ViT-B/16 which uses
per-tensor).

### 8.4 The dual-convergence result

The entropy term is exactly as strong as its dual is converged, because the
coefficient reaches the weights only through the multiplier. The dual range is
proportional to T3 while an absolute dual step is not, so the convergence rate
was never a controlled quantity. Traversals of the dual range per epoch:

| run | network | traversals / epoch | outcome |
|---|---|---|---|
| test_196 | ResNet-18 | 0.195 | works |
| test_194 | DeiT-Small | 0.189 | works |
| test_220 | AlexNet | 0.0135 | works, under-powered |
| test_232 | EfficientNet-B0 | 0.0050 | inert |
| test_233 | EfficientNet-B0 | 0.0005 | inert, worse than test_232 |
| test_234 | EfficientNet-B0 | 0.76 | opens the channel |

test_232 and test_233 differ tenfold in T3 and their applied gradient norms at
epoch one agree to four significant figures, 2.756366e-04 against 2.756720e-04.
Both compressed *less* than the same recipe with the entropy term switched off.

`--dual_step_mode relative` fixes it by measuring the step as a fraction of the
dual range. Derivation and check: `scratchpad/verify_dual_step_scaling.py`.

**The lesson, generalized.** When a coefficient looks inert, check first that the
dual variable it acts through is converging, and only then touch the coefficient.
The stop criterion originally written for test_233 would have produced the wrong
conclusion.

### 8.5 The dose is measured, then transported by its integral

Three analytic dose models in a row were wrong (test_226 inert, test_227 runaway,
test_228 overdosed 10×). What transfers between runs is the **cumulative
exposure**, coefficient times the schedule sum, never the coefficient itself. A
ramp row reports the exposure accumulated up to that epoch: la riga di **epoca 6**
del test_231 legge T2 = 1e-6 e si riproduce su un coseno di venti epoche a
**3.16e-7, fattore 3.2**; la riga di epoca 5 vale 2.28e-7, fattore 4.4. (Prima
qui erano accoppiati l'epoca 5 e il fattore 4.4 con il valore 3.16e-7: erano due
righe diverse.) La dose di EfficientNet nel paper, 2.1e-7 su trenta epoche, viene
dalla riga di epoca 6. Arithmetic in `scratchpad/verify_t2_exposure.py`.

### 8.6 Budget is the lever that never costs the other axis

Every frontier in this document ended with both axes still moving:

| pair | change | Δ accuracy | Δ packet |
|---|---|---|---|
| test_237 → test_238 | 40 → 60 ep | +0.200 | −0.07 |
| test_238 → test_257 | 60 → 100 ep | +0.254 | +0.01 |
| test_235 → test_236 | 20 → 40 ep | +0.412 | −0.01 |
| test_191 → test_192 | 25 → 40 ep | −0.094 | −0.40 |

Raising or lowering T3 trades one axis for the other; lengthening the schedule
does not. **When accuracy is the binding constraint, the right knob is the
budget, not the coefficient.**

---

## 9. Where we stand against the literature

All rows on the **model** accounting. Our own rows in bold.

### ResNet-18 (baselines differ; Δ is the comparable axis)

| method | model | CR | Δ baseline |
|---|---|---|---|
| HEMP | 7.69% | 13.0× | −0.96 |
| HEMP + LOBSTER | 5.34% | 18.7× | −0.06 |
| **PRESTO, T3 = 6e-8, 60 ep (3 seeds)** | **4.89%** | **20.45×** | **+0.286 ± 0.083** |
| EPR-DFT (Oktay et al., ICLR 2020) | 4.22% | 23.7× | 0.0 |
| **PRESTO, T3 = 1e-7, 100 ep** | **4.24%** | **23.58×** | **+0.126** |
| Metz et al. (ISCAS 2024) | 3.57% | 28.0× | −0.8 |
| LilNetX (ICLR 2023) | 2.16% | 46× | −1.2 |

**A tie on size with EPR-DFT, and we do not claim more.** Their error is quoted
to one decimal and equals their baseline exactly, so their Δ is unresolvable
below ±0.05. What we add is a deployable four-bit codebook, where EPR
reconstructs a dense FP32 network through a learned decoder.

**The tail is theirs until we measure a run in it.** Beyond 24× we have no
measured point. Metz et al. (ISCAS 2024, arXiv 2406.06237), Table II: ResNet-18
at 69.8/46.8 MB coded to **1.67 MB = 3.57% = 28.0× at −0.8**, with
mixed-precision quantization driven by a Fracbits-style precision parameter
against an entropy target and a tabled-ANS decoder they argue is cheap enough to
run at inference time. LilNetX is further out at 2.16% and −1.2.

test_262 (T3 = 2.6e-7, 100 ep) aims **between** their two coordinates rather than
at either. Landing near −0.5 gives 2.95% of the model if the frontier does not
bend, 3.28% at the ResNet-50 marginal cost of 0.624, and 3.50% under a sharp
bend — **smaller than their 3.57% and more accurate than their −0.8 in all
three**, so one measured point dominates them on both axes without interpolation
on our side or extrapolation on theirs. Aiming at their size instead (T3 = 1.7e-7,
→ 3.52% at −0.22) yields only the accuracy claim; aiming at their accuracy
(T3 ≈ 3.6e-7) spans 2.33% to 3.15% depending on the bend, and its worst case is
only 12% smaller than they are.

### ResNet-50 — no competitor is on our frontier

| method | model | CR | Δ baseline |
|---|---|---|---|
| DeepCABAC | 10.14% | 9.86× | −1.62 |
| **PRESTO LSQ control** | **9.97%** | **10.03×** | **+0.432** |
| HEMP | 8.88% | 11.26× | −1.61 |
| NNCodec | 7.43% | 13.5× | −1.01 |
| Deep Compression (quoted second-hand) | 6.15% | 16.3× | −7.18 |
| EPR-DFT | 5.38% | 18.6× | −1.0 |
| **PRESTO, T3 = 3e-8** | **5.12%** | **19.53×** | **+0.196** |
| Metz et al. (ISCAS 2024) | 4.41% | 22.7× | −0.5 |
| **PRESTO, T3 = 1e-7** | **3.72%** | **26.85×** | **−0.678** |
| LilNetX | 3.53% | 29× | −2.9 |

### DeiT-Small — the entropy-coded cell is empty

| method | model | CR | Δ baseline | note |
|---|---|---|---|---|
| Q-ViT 4-bit | 12.93% | 7.7× | +1.0 | nominal, 300 ep, distilled |
| **PRESTO, T3 = 1.5e-8** | **9.61%** | **10.41×** | **−0.252** | measured |
| Q-ViT 3-bit | 9.86% | 10.1× | −0.9 | nominal |
| **PRESTO, T3 = 3e-8, 40 ep** | **7.94%** | **12.60×** | **−0.542** | measured |
| FTerViT W2A8 (2026) | 6.59% | 15.2× | −2.39 | nominal, bit-packed |

### ViT-B/16 — the widest margin in the paper

NNCodec Pareto frontier, read from their Figure 7 (346.27 MB, baseline 81.07):

| method | model | CR | top-1 | Δ |
|---|---|---|---|---|
| NNCodec | 14.94% | 6.69× | 80.92 | −0.15 |
| NNCodec | 12.10% | 8.26× | 80.81 | −0.26 |
| **PRESTO LSQ control** | **10.48%** | **9.55×** | **80.816** | **−0.240** |
| NNCodec | 10.44% | 9.58× | 80.41 | −0.66 |
| NNCodec | 9.49% | 10.53× | 79.95 | −1.12 |
| NNCodec | 8.24% | 12.14× | 78.26 | −2.81 |
| NNCodec | 5.97% | 16.75× | 76.93 | −4.14 |
| NNCodec | 4.85% | 20.62× | 75.22 | −5.85 |
| **PRESTO (1e-5, 5e-8, 1.5e-8)** | **4.41%** | **22.65×** | **80.372** | **−0.684** |

At the size closest to ours they deliver 75.22 and we deliver 80.372: **5.15
accuracy points at a smaller size**, and less than half the size at half the
accuracy loss. The comparison is unusually clean — both weight-only, both
measuring bytes after lossless coding, both starting from the same public
checkpoint (measured 81.07 by them, 81.056 by us).

### EfficientNet-B0 — the one cell we are losing

NNCodec **76.78 (−0.89) at 11.25%, 8.89×**, against our 76.590 (−0.976) at
12.43%, 8.05×. Behind on both axes. See §6 for the frontier and §11 for the plan.

---

## 10. Wall-clock cost

Sixteen A100-SXM-64GB across four Leonardo nodes, global batch 1024, per-epoch
training time excluding the evaluation and serialization pass.

| run | epochs | s / epoch | first epoch |
|---|---|---|---|
| ResNet-18, LSQ only | 60 | 108 ± 1 | 109 |
| ResNet-18, ridge only | 60 | 108 ± 0 | 108 |
| ResNet-18, no entropy | 60 | 108 ± 1 | 108 |
| ResNet-18, no sparsity | 60 | 427 ± 31 | 110 |
| ResNet-18, full PRESTO | 60 | 447 ± 33 | 108 |
| ResNet-18, full PRESTO | 100 | 474 ± 36 | 139 |
| ResNet-50, LSQ only | 20 | 109 ± 0 | 109 |
| ResNet-50, full PRESTO | 20 | 997 ± 107 | 157 |
| ResNet-50, T3 = 1e-7 | 20 | 920 ± 50 | 157 |
| ViT-B/16, LSQ only | 20 | 218 ± 1 | 218 |
| ViT-B/16, full PRESTO | 20 | 2229 ± 45 | 279 |

**T1 and T2 are free** (closed-form per-step path). All of the cost is the T3
dual.

**CORREZIONE (31 agosto 2026) — il confronto per chiamata va fatto a N_dual
appaiato.** La riga ViT-B/16 da 2229 s/epoca è a **N_dual = 6**, le due ResNet a
N_dual = 3: i 6.43 s/chiamata che citavamo per ViT sono quindi il doppio del
dovuto, e la conclusione "lo sweep per peso domina su ViT" era un artefatto. A
N_dual = 3 (test_269, 1213 s/epoca) il costo per chiamata è **1.08 s su
ResNet-18, 2.84 su ResNet-50, 3.18 su ViT-B/16**; per tensore 51.6, 52.5 e 63.6
ms (quasi piatto), per milione di pesi 93, 111 e 37 ms (un fattore tre). ViT-B/16
ha 3.4× i pesi di ResNet-50 su quattro tensori in meno e costa solo il 12% in
più: **il solver è dominato dall'overhead per tensore su tutto l'intervallo
misurato**. Fit ai minimi quadrati sui tre punti appaiati: **48 ms per tensore +
9 ns per peso**, con il termine per tensore che vale il 76-92% della chiamata.

---

## 11. What is still missing

- **The ViT-B/16 knee.** One PRESTO point only, and it is the deep one. test_259
  (T2 = 1e-8, T3 = 3e-9, one fifth of the test_256 exposure) targets ~7%.
- **EfficientNet-B0 against NNCodec.** test_258 landed and the budget lever does
  not work on this network and test_260 moved the ceiling but not the frontier
  (§6). test_261 (LSQ control, per-tensor scales) attacks the 0.599% metadata
  floor, which is the size of the remaining gap; test_262 (full PRESTO on the
  8-bit-depthwise recipe) is aimed at a tie with NNCodec and runs regardless,
  because it is the only matched control-and-PRESTO pair we will have here.
- **A measured ResNet-18 point in the 28× tail.** test_263 (T3 = 2.6e-7, 100 ep,
  one flag from test_257) targets 2.9–3.1% of the model at −0.5 to −0.6, against
  Metz et al. at 3.57% and −0.8, chosen to dominate them on both axes under every
  plausible bend of the frontier. Dose from regression on ln(T3) over the two
  60-epoch points, **in unità packet e sul run singolo test_239** (sulle righe
  pubblicate, contabilità model e media dei tre semi, le pendenze sono −1.29 e
  −0.81): −1.331 packet points and −0.646 accuracy points per unit, with
  the flatter ResNet-50 slopes (−1.163, −0.726) as the pessimistic bracket.
- **Seeds on a second network.** Only ResNet-18 has three.
- **Published sizes** are converted from reported megabytes rather than
  recomputed by our serializer.
- **Per-tensor symbol histograms**, which would settle *which* symbols T3 empties.
- **AlexNet** needs its closing paragraph as a methodological contribution rather
  than a competitive row.
