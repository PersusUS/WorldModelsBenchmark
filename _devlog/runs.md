# Registro de ejecuciones

---

## R0 — Resultados originales (pre-correcciones) · INVALIDADOS

225 corridas en `results/`, generadas antes de las correcciones F1/F2.
Medidas sobre ruido gaussiano y con el objetivo de NLL equivocado.

Media de WMF sobre 5 semillas, conservada aquí solo como referencia histórica:

```
method               mini_min  mini_med  mini_max  gymn_min  gymn_med  gymn_max  dmco_min  dmco_med  dmco_max
finetuning            +0.0812   +0.0107   +0.0147   +0.0065   +0.0053   +0.0089   +0.0072   +0.0417   +0.0145
replay_infinite       +0.0619   +0.0404   +0.0775   +0.0065   +0.0061   +0.0071   +0.0071   +0.0377   +0.0416
ewc                   +0.0004   +0.0014   +0.0014   -0.0002   -0.0000   +0.0000   -0.0001   -0.0000   -0.0004
progressive_nets      -0.0124   -0.0240   -0.0141   +0.0079   +0.0073   +0.0083   +0.0076   +0.0212   +0.0132
ug_mtm                +0.0000   +0.0000   -0.0000   +0.0000   +0.0000   -0.0000   +0.0000   -0.0000   +0.0000
```

PIS = 0.0 en las 225. FT medio de UG-MTM = −1.207.

**No usar para el paper.**

---

## R1 — Smoke test tras las correcciones F1/F2/F7 + gating en eval

MiniGrid `distance_med`, semilla 999, 150 pasos, 8 rollouts (corrida corta solo
para verificar que el pipeline funciona end-to-end).

```
finetuning         WMF=+0.0014  PF=-0.0062  RD=+0.0095  FT=+0.1600
ewc                WMF=+0.0012  PF=+0.0007  RD=+0.0024  FT=+0.1618
progressive_nets   WMF=+0.0224  PF=+0.0101  RD=+0.0460  FT=+0.0048
ug_mtm             WMF=-0.0000  PF=+0.0000  RD=-0.0000  FT=-2.7496
```

**Lectura.** Las métricas ya responden para los baselines. UG-MTM sigue en
0.0000 exacto → llevó al diagnóstico F3.

Directorios `*_999` eliminados tras la prueba.

---

## R2 — Diagnóstico de la puerta de UG-MTM (F3)

Modelo entrenado 150 pasos sobre MiniGrid-Empty-8x8, medido sobre 100
transiciones reservadas:

```
tau               = 0.446077
u_t  mean/max     = 1.327e-01 / 1.498e-01
u_t > tau         : 0 / 100
mean gate expert0 = 0.998084    expert1 = 1.916e-03
```

Prueba complementaria: perturbando el experto nuevo con `N(0, 1000)`, PF se
mueve 3.8e-6 y RD 4.5e-5 — por debajo de la precisión reportada.

---

---

## R3 — ¿Discrimina alguna señal la tarea A de la B?

AUC = P(señal_B > señal_A) sobre 200 transiciones reservadas por tarea,
tras entrenar solo en la tarea A. 0.5 = ninguna discriminación.

**Antes de corregir F0** (VAE colapsado), 400 y 2000 pasos:

| Señal | med | max |
| --- | --- | --- |
| UncertaintyHead MC-dropout | 0.519 | 0.508 |
| MC-dropout sobre la transición | 0.481 | 0.515 |
| Error de predicción a un paso | 0.769 | 0.533 |
| Distancia latente media L2 | 1.4e-03 | 3.7e-04 |

Ninguna señal servía, y el error de predicción oscilaba erráticamente entre
corridas (0.914 / 0.559 / 0.769 / 0.533) — puro ruido.

**Después de corregir F0**, 2000 pasos:

| Señal | med | max |
| --- | --- | --- |
| UncertaintyHead MC-dropout | 0.294 | **0.864** |
| MC-dropout sobre la transición | 0.283 | 0.780 |
| Error de predicción a un paso | 0.693 | 0.862 |
| Distancia latente media L2 | **2.89** | **5.24** |

Las distancias latentes suben de ~1e-3 a 2.9–5.2: el VAE por fin codifica los
entornos. Ver F3 para la lectura.

---

## R4 — Diagnóstico del VAE (lleva a F0)

2000 pasos sobre MiniGrid-Empty-8x8:

```
step    0  recon=0.077582  kl=0.071340
step  500  recon=0.015919  kl=0.001034
step 1000  recon=0.014296  kl=0.000544
step 1999  recon=0.014381  kl=0.000342

per-dim std across data = 3.4671e-05
active dims (std > 1e-2) = 0 / 32
```

Comparativa controlada de escalados (dos VAE idénticos, mismos datos):

```
current   active dims= 0/32   mean per-dim std=1.217e-04   recon MSE=1.6585e-02
fixed     active dims=32/32   mean per-dim std=2.890e-01   recon MSE=5.4610e-04
```

---

---

## R5 — Smoke run del benchmark con el VAE corregido

MiniGrid, 1000 pasos, 20 rollouts, semilla 999. Mismos ajustes que el runner
real salvo la semilla.

```
########## minigrid / distance_med ##########
  finetuning         WMF=  +99.1917  PF= -0.7696  RD=  +248.7487  FT=+17.0501
  replay_infinite    WMF= +104.2663  PF= -2.1532  RD=  +262.8188  FT=+20.1040
  ewc                WMF=  +59.6248  PF= +0.6768  RD=  +148.3854  FT=+16.0712
  progressive_nets   WMF= +126.7080  PF= +5.0160  RD=  +311.7539  FT=+17.6709
  ug_mtm             WMF=  +40.0676  PF=+12.3122  RD=   +87.8568  FT=+15.8940

########## minigrid / distance_max ##########
  finetuning         WMF=  +42.4791  PF= +3.7973  RD=  +102.4004  FT=+16.2371
  replay_infinite    WMF=  +36.8989  PF= -3.2697  RD=   +95.5170  FT=+15.6742
  ewc                WMF=  +30.3800  PF= +0.6159  RD=   +75.3342  FT=+15.6021
  progressive_nets   WMF= +111.2457  PF=+14.4773  RD=  +263.6369  FT=+16.9329
  ug_mtm             WMF=+5487.6541  PF=+10.3102  RD=+13708.8252  FT=+17.3855
```

**Lecturas.**

1. Las métricas ya no son ruido alrededor de cero: hay variación real entre
   métodos, y el FT pasa a ser positivo (~+16 a +20) en todos.
2. **RD domina el WMF por completo** (F14). El agregado es de facto `0.4·RD`.
3. **RD explota en UG-MTM a distancia máxima** (13708). Rollout a lazo abierto
   de 15 pasos sin cota.
4. El orden entre métodos cambia según la distancia: UG-MTM es el mejor a
   `med` y con diferencia el peor a `max`.
5. Estos números **no son comparables con R0** y tampoco son definitivos:
   quedan por resolver F13 (la KL está mal) y F14 (agregación).

No usar para el paper. Sirve para confirmar que el pipeline discrimina y para
destapar F13/F14.

Directorios `*_999` eliminados tras la prueba.

---

## Pendiente
- [ ] Decidir P5 (peso del término de incertidumbre) y P6 (inversión a med)
- [ ] Ejecución completa de 225 corridas
- [ ] Regenerar figura y tabla desde los nuevos `metrics.json`

---

## R6 — Smoke run con la KL corregida (F13)

Mismos ajustes que R5 (MiniGrid, 1000 pasos, 20 rollouts, semilla 999) para que
sea comparable celda a celda. Usa los propios `run_baseline` / `run_ug_mtm`.

```
########## minigrid / distance_med ##########
  finetuning         WMF=  +8.3846  PF= -1.5010  RD=  +22.4640  FT=+16.85
  replay_infinite    WMF=  +3.8228  PF= -3.8330  RD=  +13.3877  FT=+16.61
  ewc                WMF= +11.5643  PF= +0.8830  RD=  +28.0278  FT=+16.98
  progressive_nets   WMF= +15.0628  PF= +5.7730  RD=  +31.8828  FT=+16.72
  ug_mtm             WMF=  +8.0290  PF= +5.0112  RD=  +15.0612  FT=+16.86

########## minigrid / distance_max ##########
  finetuning         WMF=  +9.2753  PF= +3.7414  RD=  +19.4467  FT=+16.82
  replay_infinite    WMF=  +6.0419  PF= -2.7976  RD=  +17.9023  FT=+16.74
  ewc                WMF=  +5.1065  PF= +1.5643  RD=  +11.2021  FT=+17.53
  progressive_nets   WMF= +16.1367  PF= +7.1086  RD=  +33.2331  FT=+16.34
  ug_mtm             WMF= +81.5012  PF= +6.8324  RD=+196.9205  FT=+17.60
```

Reparto dentro de `WMF = 0.4·PF + 0.4·RD` (PIS=0):

| Celda | % PF | % RD |
| --- | --- | --- |
| med / finetuning | 6.3 | 93.7 |
| med / replay_infinite | 22.3 | 77.7 |
| med / ewc | 3.1 | 96.9 |
| med / progressive_nets | 15.3 | 84.7 |
| med / ug_mtm | 25.0 | 75.0 |
| max / finetuning | 16.1 | 83.9 |
| max / replay_infinite | 13.5 | 86.5 |
| max / ewc | 12.3 | 87.8 |
| max / progressive_nets | 17.6 | 82.4 |
| max / ug_mtm | 3.4 | 96.7 |

**Lecturas.**

1. **La explosión desaparece.** UG-MTM a distancia máxima: `RD = 13708` → `196.9`.
   El rango completo de RD queda en 11–197 en vez de 75–13708.
2. **F14 se reduce pero no se cierra.** PF pasa de aportar ~1% a aportar 3–25%
   (mediana ~15%). La brecha `RD/|PF|` baja de 1–3 órdenes de magnitud a 3–32×.
   RD sigue dominando con 75–97% del agregado.
3. **No comparable con R5 celda a celda.** PF cambió entre R5 y R6
   (med/finetuning: −0.770 → −1.501) aunque F13 no tocó `compute_nll`. Ver R7:
   es no-determinismo, no el arreglo. La columna de ratios R5↔R6 mezcla los dos
   efectos y no sirve para medir la fórmula.

Directorios `*_999` eliminados tras la prueba.

---

## R7 — Aislar la fórmula y comprobar reproducibilidad

MiniGrid `distance_med`, `finetuning`, semilla 999, **la misma celda dos veces**.
En cada corrida se calculan la fórmula antigua y la nueva **sobre el mismo par de
modelos**, así que la única diferencia es la fórmula.

```
Device: cuda    torch.backends.cudnn.deterministic = False

run 1:  PF= 0.2373  RD_nueva=19.9702  RD_antigua=256.1147  ratio=12.82  d_trans=3.7260
run 2:  PF=-0.2101  RD_nueva=17.7509  RD_antigua=212.3382  ratio=11.96  d_trans=5.2808
```

**(1) El pipeline no es reproducible.** Misma semilla, números distintos:

```
PF:  +0.237284  vs  -0.210058   delta=0.447
RD:  19.970211  vs  17.750872   delta=2.219
```

**A PF le cambia el signo.** El delta entre dos corridas de la misma semilla
(0.447) es mayor que |PF| en varias celdas de R6. Confirma I5 empíricamente
— ver F16, promovido a bloqueante.

**(2) El arreglo de la KL vale ~12×, no 5–70×.** Con los modelos fijos, la
inflación es 12.82× y 11.96× — consistente entre corridas, y del mismo orden que
el 9.3× que se midió analíticamente sobre entradas sintéticas. Los ratios de 5–70×
de la comparativa R5↔R6 eran sobre todo ruido entre corridas.

**(3) Primeros valores de `d_trans` (Ec. 9) que produce el proyecto.** 3.73 y
5.28 — pero ojo: es `d_trans(model_i, model_k)`, la distancia entre el modelo
antes y después de la tarea B, **no** `d_trans(E_A, E_B)` como la define la Ec. 9.
Ver F15.

---

## R8 — Verificación de F16: ¿reproduce el benchmark?

Repite R7 por los runners ya corregidos. La misma celda, dos veces, semilla 999,
en tres métodos con rutas de código distintas (`finetuning` simple, `ewc` con
Fisher, `ug_mtm` con MC-dropout y hooks de escalado de gradiente).

```
=== finetuning ===
  OK WMF    8.5205296834  vs    8.5205296834   delta=0.000e+00
  OK PF    -1.7783794403  vs   -1.7783794403   delta=0.000e+00
  OK RD    23.0797036489  vs   23.0797036489   delta=0.000e+00
  OK FT    16.5521640778  vs   16.5521640778   delta=0.000e+00

=== ewc ===
  OK WMF    8.2884732437  vs    8.2884732437   delta=0.000e+00
  OK PF     1.3305854797  vs    1.3305854797   delta=0.000e+00
  OK RD    19.3905976295  vs   19.3905976295   delta=0.000e+00
  OK FT    16.5521640778  vs   16.5521640778   delta=0.000e+00

=== ug_mtm ===
  OK WMF   10.5560239029  vs   10.5560239029   delta=0.000e+00
  OK PF     5.5788383484  vs    5.5788383484   delta=0.000e+00
  OK RD    20.8112214088  vs   20.8112214088   delta=0.000e+00
  OK FT    19.0437374115  vs   19.0437374115   delta=0.000e+00
```

**Los 12 valores idénticos bit a bit.** F16 resuelto.

Comparar con R7, la misma comprobación antes del arreglo: `PF = +0.2373` vs
`PF = −0.2101`, cambiando de signo.

**Ojo al comparar con R6.** Sembrar los entornos cambia qué episodios se recogen,
así que los valores de R6 y los de aquí no son la misma medición. R6/R7 son
pre-F16; todo lo posterior es la línea base nueva.

---

## R9 — F14 sobre el pipeline reproducible (línea base nueva)

MiniGrid, 1000 pasos, 20 rollouts, semilla 999, ya con F13 y F16 corregidos.
**Esta es la línea base con la que comparar de aquí en adelante**; R5 y R6 son
históricas (KL rota y entornos sin sembrar, respectivamente).

```
########## minigrid / distance_med ##########   (Empty-8x8 -> FourRooms)
  finetuning         WMF=  +8.5205  PF= -1.7784  RD= +23.0797  FT=+16.5522
  replay_infinite    WMF=  +4.3507  PF= -3.2520  RD= +14.1288  FT=+16.5522
  ewc                WMF=  +8.2885  PF= +1.3306  RD= +19.3906  FT=+16.5522
  progressive_nets   WMF= +16.0843  PF= +4.0210  RD= +36.1899  FT=+16.6668
  ug_mtm             WMF= +10.5560  PF= +5.5788  RD= +20.8112  FT=+19.0437

########## minigrid / distance_max ##########   (FourRooms -> KeyCorridorS3R1)
  finetuning         WMF= +11.4297  PF= +3.4658  RD= +25.1085  FT=+17.9050
  replay_infinite    WMF=  +5.1162  PF= -1.9539  RD= +14.7443  FT=+17.9050
  ewc                WMF=  +6.4417  PF= +1.8692  RD= +14.2349  FT=+17.9050
  progressive_nets   WMF= +16.5258  PF= +9.2617  RD= +32.0528  FT=+17.9829
  ug_mtm             WMF= +68.7789  PF= +4.8561  RD=+167.0912  FT=+16.3713
```

Reparto dentro de `WMF = 0.4·PF + 0.4·RD` (PIS=0):

| Celda | % PF | % RD | RD/&#124;PF&#124; |
| --- | --- | --- | --- |
| med / finetuning | 7.2 | 92.9 | 13.0 |
| med / replay_infinite | 18.7 | 81.3 | 4.3 |
| med / ewc | 6.4 | 93.6 | 14.6 |
| med / progressive_nets | 10.0 | 90.0 | 9.0 |
| med / ug_mtm | 21.1 | 78.9 | 3.7 |
| max / finetuning | 12.1 | 87.9 | 7.2 |
| max / replay_infinite | 11.7 | 88.3 | 7.5 |
| max / ewc | 11.6 | 88.4 | 7.6 |
| max / progressive_nets | 22.4 | 77.6 | 3.5 |
| max / ug_mtm | 2.8 | 97.2 | 34.4 |

**Lecturas.**

1. **Reproducibilidad entre procesos, no solo dentro de uno.** Las tres celdas
   que R9 comparte con R8 (med/finetuning, med/ewc, med/ug_mtm) dan valores
   idénticos, y fueron **procesos distintos**. Es una comprobación más fuerte
   que la de R8.
2. **F14 confirmado y estable.** PF se lleva el **2.8–22.4%** del agregado
   (mediana 11.7%), RD el **77.6–97.2%**. Prácticamente lo mismo que R6 (3–25%),
   así que la conclusión no dependía de los datos concretos: es estructural.
3. **Sembrar los entornos movió RD entre ×0.72 y ×1.45** respecto a R6. Es decir,
   los órdenes de magnitud de R6 eran correctos; solo cambiaban los datos.
4. **UG-MTM a distancia máxima sigue siendo el caso raro**: `RD = 167` frente a
   14–32 del resto, y `RD/|PF| = 34.4`, el peor de la tabla. Ya no explota
   (era 13708 con la KL rota) pero es un outlier de 5–10× sobre los demás
   métodos. Merece mirarse al decidir P7 y al interpretar UG-MTM.
5. **`replay_infinite` tiene el WMF más bajo en ambos niveles** (4.35 y 5.12) y
   PF negativo. Con la Tabla 3 original pasaba lo contrario (replay peor que
   finetuning en MiniGrid, el Finding 4, el único que se sostenía). Con una sola
   semilla no se puede concluir nada, pero **es lo primero que hay que mirar en
   la ejecución completa**: si el Finding 4 se invierte, cambia el resultado más
   sólido del paper.

Directorios `*_999` eliminados tras la prueba.

---

## R10 — Verificación del refactor + primera instrumentación de F17

MiniGrid `distance_med`, semilla 999, tres métodos con rutas de código distintas.
Mismo protocolo que R9 (20 rollouts, 1000 pasos, batch 8, `seq_len` 5), ejecutado
con el runner reescrito de la sesión 6.

### ¿Cambió el refactor algún resultado? No.

```
=== finetuning ===
  OK  WMF  R9=    8.5205296834  now=    8.5205296834  delta=3.099e-11
  OK  PF   R9=   -1.7783794403  now=   -1.7783794403  delta=7.617e-12
  OK  RD   R9=   23.0797036489  now=   23.0797036489  delta=1.491e-11
  OK  FT   R9=   16.5521640778  now=   16.5521640778  delta=4.121e-11
=== ewc ===
  OK  WMF  R9=    8.2884732437  now=    8.2884732437  delta=1.338e-11
  OK  PF   R9=    1.3305854797  now=    1.3305854797  delta=3.633e-11
  OK  RD   R9=   19.3905976295  now=   19.3905976295  delta=4.712e-11
  OK  FT   R9=   16.5521640778  now=   16.5521640778  delta=4.121e-11
=== ug_mtm ===
  OK  WMF  R9=   10.5560239029  now=   10.5560239029  delta=6.933e-12
  OK  PF   R9=    5.5788383484  now=    5.5788383484  delta=1.133e-11
  OK  RD   R9=   20.8112214088  now=   20.8112214088  delta=4.399e-11
  OK  FT   R9=   19.0437374115  now=   19.0437374115  delta=9.770e-13
```

Las 12 cifras coinciden. Los deltas de ~1e-11 son la precisión con la que estaban
anotadas en R9 (10 decimales), no una diferencia real. Un runner reescrito de
arriba abajo —protocolo desde el config, dos funciones unificadas en una,
instrumentación nueva en seis puntos— y **ningún resultado se movió**. Es lo que
`preserve_rng_state` y el haber conservado el orden exacto de llamadas compran.

### Lo que se ve ahora que antes no se guardaba

| Señal | `finetuning` | `ewc` | `ug_mtm` |
| --- | --- | --- | --- |
| Recon. entrenamiento, 1er paso de A | 931.76 | 931.76 | 931.77 |
| Recon. entrenamiento, último paso de A | 8.17 | 8.17 | 9.89 |
| **Recon. reservada A, tras la tarea A** | **6.49** | **6.49** | **7.66** |
| **Recon. reservada A, tras la tarea B** | **725.27** | **718.01** | **7.66** |
| Recon. entrenamiento, último paso de B | 19.66 | 20.06 | **533.19** |
| NLL sobre `D_A` tras A | 22.28 | 22.28 | 21.29 |
| NLL sobre `D_A` tras B | 20.50 | 23.61 | 26.89 |
| NLL sobre `D_A`, modelo sin entrenar | 38.84 | 38.84 | 40.34 |
| Pasos con NaN (A + B) | 0 | 0 | 0 |

**Cuatro lecturas, y la segunda es la importante.**

1. **La tarea A se aprende.** 931.76 → 8.17 en entrenamiento (×114) y 6.49 sobre
   datos reservados, que sumados sobre 12288 píxeles son 5.3e-04 por píxel
   (RMSE ≈ 0.023 en `[0,1]`). La objeción de F17 tiene respuesta numérica.
2. **PF y RD no ven el olvido donde ocurre → F18.** `finetuning` degrada la
   reconstrucción de la tarea A **×112** (6.49 → 725.27) y PF sale **negativo**
   (−1.78). `compute_nll` opera sobre latentes congelados y nunca llama a
   `encode`: el codificador puede derivar hasta ser inútil sin que ninguna métrica
   del paper lo registre. Ver F18 en `findings.md`.
3. **EWC no protege el codificador** (718.01, prácticamente igual que
   `finetuning`), aunque su PF sí sea mejor que el de `finetuning`. Los dos
   números cuentan historias opuestas sobre el mismo modelo.
4. **UG-MTM: 7.663551597595215 → 7.663551597595215, idéntico bit a bit.** Congela
   el VAE, así que su aislamiento del codificador es total por construcción. El
   precio está en la fila de la tarea B: **533.19** frente a 19.66. No olvida
   porque no aprende. Es la cuantificación de F3.

Directorio de resultados en el scratchpad, fuera de `results/`.

---

## R11 — Prueba de escalado de la tarea A (F17)

`experiments/convergence_A.py`, MiniGrid `distance_med`, `finetuning`, semilla
999. **Una** corrida de 10000 pasos, evaluada en 1×, 2×, 5× y 10× el `n_train`
del config (1000).

```
  steps=   1000 (1x) | held-out recon=     6.486 | train recon=    49.417 | NLL(own)=   22.283 | gap vs random=   16.612
  steps=   2000 (2x) | held-out recon=     3.013 | train recon=     4.316 | NLL(own)=   21.071 | gap vs random=   16.521
  steps=   5000 (5x) | held-out recon=     1.606 | train recon=     1.944 | NLL(own)=   14.338 | gap vs random=   24.267
  steps=  10000 (10x) | held-out recon=     1.367 | train recon=     1.060 | NLL(own)=    8.586 | gap vs random=   30.900
```

### La equivalencia del diseño está verificada

El punto 1× da `held-out recon = 6.486` y `NLL(own) = 22.283`. R10, proceso
distinto y celda real del benchmark, dio **6.485961380004883** y
**22.283300399780273**. Coinciden. Es decir: evaluar por el camino con
`preserve_rng_state` **no perturba la corrida**, y el punto `m` de la sonda es de
verdad lo que habría dado una corrida independiente de `m × n_train` pasos. Por
eso la sonda cuesta un 10× en vez de 1×+2×+5×+10×.

### Qué dice sobre el presupuesto

Leer la columna de reconstrucción reservada, que es la única comparable entre
presupuestos:

| Presupuesto | Recon. reservada | Mejora sobre el punto anterior |
| --- | --- | --- |
| 1× (1000) | 6.486 | — |
| 2× (2000) | 3.013 | −53.5% |
| 5× (5000) | 1.606 | −46.7% |
| 10× (10000) | 1.367 | **−14.9%** |

**El presupuesto actual está en la parte empinada de la curva.** A 1000 pasos la
reconstrucción de la tarea A es **4.7× peor** que a 10000. Pero los rendimientos
decrecientes se ven claramente: entre 5× y 10× solo se gana un 15%.

- **5× (5000 pasos) es el punto dulce**: queda a un 17% del valor de 10× por la
  mitad de cómputo.
- 10× no está del todo convergido tampoco, pero ya es plano a efectos de un
  argumento de "había algo que olvidar".

**Coste, orden de magnitud** (medido en MiniGrid, ~2 min por 1000 pasos en esta
GPU; las otras dos familias son más lentas al recoger rollouts):

| Presupuesto | 225 corridas |
| --- | --- |
| 1× (hoy) | ~19 h |
| 2× | ~31 h |
| 5× | **~3.3 días** |
| 10× | ~6.5 días |

### La NLL latente no se estabiliza, y eso confirma el motivo de no fiarse de ella

`NLL(own)` cae 22.28 → 8.59 sin señal de plateau, y la separación contra el modelo
sin entrenar **crece** de 16.6 a 30.9. Es exactamente el efecto que hacía que la
NLL no fuera comparable entre presupuestos: el codificador sigue afinando el
espacio latente, así que a cada punto se le mide una tarea distinta. La
reconstrucción en píxeles se aplana; la NLL latente no. Si el apéndice del paper
usara la NLL para argumentar convergencia, argumentaría lo contrario de lo que
pretende.

### Salvedades

Una semilla, una celda, una familia, y solo `finetuning`. Sirve para fijar el
presupuesto y para la figura de apéndice; no es una afirmación sobre las otras
ocho celdas. Repetirla en `dmcontrol` (la familia visualmente más difícil) es
barato y conviene antes de cerrar el presupuesto.

---

## R12 — Chequeo temprano del Finding 4 (5 semillas, 20 celdas)

`finetuning` vs `replay_infinite`, MiniGrid, `distance_med` y `distance_max`,
semillas 0–4. Protocolo actual (1000 pasos). Resultados en
`_devlog/archive/results-R12-finding4/`, **fuera de `results/`** para no
comprometer la decisión de presupuesto.

### El Finding 4 se invierte, y no por poco

| Métrica | Celda | `finetuning` | `replay_infinite` | Δ | Gana en | perm p | d_z |
| --- | --- | --- | --- | --- | --- | --- | --- |
| WMF | med | +10.016 ± 3.420 | **+4.936 ± 1.811** | −5.08 | 5/5 | 0.0625 | −1.97 |
| WMF | max | +12.703 ± 4.631 | **+4.689 ± 1.992** | −8.01 | 5/5 | 0.0625 | −2.27 |
| PF | med | −0.472 ± 1.174 | **−2.318 ± 0.686** | −1.85 | 5/5 | 0.0625 | −3.07 |
| PF | max | +2.696 ± 2.310 | **−2.467 ± 0.433** | −5.16 | 5/5 | 0.0625 | −2.27 |
| RD | med | +25.512 ± 8.055 | **+14.659 ± 4.034** | −10.85 | 5/5 | 0.0625 | −1.66 |
| RD | max | +29.063 ± 9.437 | **+14.190 ± 4.629** | −14.87 | 5/5 | 0.0625 | −2.17 |

**Replay olvida menos que finetuning en las 10 comparaciones semilla a semilla, en
las tres métricas, en los dos niveles.** Ni una excepción. `p = 0.0625` es el
**suelo** que permite n=5 en un test exacto emparejado, o sea la máxima evidencia
que 5 semillas pueden dar; los `d_z` de −1.66 a −3.07 son efectos grandes.

El paper afirmaba lo contrario: *"Replay is insufficient in low-capacity
settings"*, replay **peor** que finetuning en MiniGrid con p<0.001, y era el
**único de los cinco Findings que resistía el escrutinio**. Con la medición
corregida (F0, F1, F2, F13, F16) se invierte. Ya no se sostiene ninguno.

Y conviene decirlo claro: la dirección nueva es **la que uno esperaría a priori** —
replay entrena sobre A+B, así que debería olvidar menos. La afirmación llamativa
era la vieja, y era un artefacto.

### El mecanismo se ve en la columna de píxeles

| Recon. tarea A | med: tras A → tras B | max: tras A → tras B |
| --- | --- | --- |
| `finetuning` | 5.84 → **739.81** (×127) | 43.20 → **1005.00** (×23) |
| `replay_infinite` | 5.84 → **5.86** (×1.003) | 43.20 → **45.51** (×1.05) |

El codificador de replay **no se degrada**: sigue entrenando sobre datos de A.
El de finetuning se va por un factor de 127. Esa es la separación real entre los
dos métodos, y es **×126 mayor que la que ve PF** (1.85 nats). F18 en acción: la
métrica del paper detecta el orden correcto, pero mide una sombra del efecto.

Nota: las dos primeras columnas son idénticas entre métodos (5.839 ± 0.532) porque
comparten el entrenamiento de la tarea A semilla a semilla. Es la comprobación de
que el emparejado es real.

### Un hallazgo que salió de aquí: F20

**FT es idéntico entre los dos métodos en las 10 semillas, delta exactamente
0.000.** No es casualidad: `FT = NLL(M_random, D_A) − NLL(M_i, D_A)` solo depende
del modelo post-tarea-A, y los métodos que comparten arquitectura solo se
diferencian **después** del cambio de tarea. FT no puede distinguirlos. Ver F20 en
`findings.md`.

---

## R13 — Qué mueve arreglar el Fisher de EWC (I3), y qué no

MiniGrid `distance_med`, semilla 999, 1000 pasos: **la misma celda que R10**, para
que la comparación sea directa. Dos métodos, `ewc` y `finetuning`, el segundo como
control.

### El control primero

```
finetuning   WMF R10=8.5205296834  ahora=8.5205296834  delta=3.1e-11
             PF  R10=-1.7783794403 ahora=-1.7783794403 delta=7.6e-12
             RD  R10=23.0797036489 ahora=23.0797036489 delta=1.5e-11
```

Y eso **después** de: gradientes por muestra en el Fisher, extraer la recogida de
datos a una función compartida, una cuarta recogida de episodios (los reservados
de la tarea B), tres llamadas de evaluación nuevas y dos métricas nuevas. Los
deltas de 1e-11 son la precisión con la que R10 quedó anotado. La disciplina de
`preserve_rng_state` sigue comprando exactamente lo que se pagó por ella.

### Y ahora EWC

| | R10 | R13 | |
| --- | --- | --- | --- |
| PF | +1.3306 | **+0.0029** | ×459 más pequeño |
| RD | 19.3906 | **17.8688** | −7.8% |
| WMF | 8.2885 | 7.1487 | −13.8% |
| Recon. entrenamiento final en B | 20.06 | **19.59** | sin pérdida de plasticidad |
| Recon. reservada de A tras B | 718.01 | **725.76** | igual de destruida |

Tres lecturas:

1. **EWC pasa a conservar la NLL latente de la tarea A casi exactamente.** PF de
   0.003 significa que `M` sale del cambio de tarea donde entró. Con el Fisher
   infraestimado la penalización era casi inerte: se estaba evaluando un
   `finetuning` con pasos de más y llamándolo EWC.
2. **Y no paga plasticidad por ello.** Su ajuste final a la tarea B (19.59) es el
   mismo que el de `finetuning` (19.66). No es el equilibrio estabilidad-plasticidad
   habitual: a este presupuesto, sale gratis.
3. **El codificador sigue destruido, y ahora se sabe por qué.** 725.76 frente a
   725.27 de `finetuning`: idéntico. El Fisher es **exactamente cero** sobre los 20
   parámetros del VAE, porque no entran en `log P(z'|z,a)`. Ver F21.

Las cifras de EWC coinciden hasta el último decimal entre dos procesos distintos
(la corrida aislada del arreglo y esta), así que el cambio es reproducible.

---

## R14 — Primera medición de FT real y `d_trans` (smoke)

Dos corridas cortas, solo para comprobar que la maquinaria nueva produce números
con sentido. **No son resultados**: 200 y 50 pasos, una semilla.

**MiniGrid `distance_med`, semilla 7, 200 pasos.** `d_trans = 13.03`.

| Método | PF | RD | **FT** | recon B tras B |
| --- | --- | --- | --- | --- |
| `finetuning` | +1.34 | 23.20 | **+54.69** | 47.88 |
| `ug_mtm` | +0.70 | 12.88 | **−516.10** | 618.67 |

Referencia desde cero: recon B = 102.58.

La métrica vieja daba delta 0.000 entre métodos por construcción. La nueva separa
por 570 unidades y cuenta una historia coherente con F18: UG-MTM congela el VAE,
así que no olvida la tarea A **porque no aprende la tarea B**. Las dos caras del
mismo hecho, y ahora las dos están medidas.

**DMControl `distance_min`, semilla 3, 50 pasos** — el nivel que hasta esta sesión
comparaba `cheetah/run` consigo mismo (F8). Con la perturbación de gravedad ya
aplicada: `d_trans = 5.75`, FT = +44.06 para `finetuning`. La casilla produce
números por primera vez.

---

## R15 — Gymnasium de punta a punta, y el coste real de una corrida

`HalfCheetah-v4` `distance_min` (gravedad 9.8 → 7.0), semilla 11, 200 pasos, 3
rollouts. La única familia que no se había ejecutado con el runner de la sesión
7. **Funciona**: referencia, `d_trans`, FT y las dos celdas, 0 pasos con NaN.

```
referencia  | d_trans = 8.913 | recon B desde cero = 250.24
finetuning  | PF=-4.3357 RD=48.8408 FT=170.1600 | recon A: 250.41 -> 75.44
ewc         | PF=-0.2893 RD=27.7059 FT=168.6201 | recon A: 250.41 -> 77.49
```

**La reconstrucción de la tarea A mejora tras entrenar en B** (250 → 75), y PF
sale negativo en los dos métodos. No es olvido negativo: a 200 pasos y 3 rollouts
nada ha convergido, y `distance_min` en esta familia son dos gravedades del mismo
HalfCheetah, visualmente casi idénticos. Entrenar en B sigue enseñando a
reconstruir A. Es lo que un nivel de distancia mínima *debería* hacer, pero a este
presupuesto no distingue eso de un artefacto: no leer estas cifras como resultado.

### El dato que importaba: cuánto cuesta la ejecución completa

Medido en aislamiento, sin nada más en la GPU:

| | Coste |
| --- | --- |
| Entrenamiento | **46 s / 1000 pasos** |
| Recogida, MiniGrid | **0.5 s / episodio** |
| Recogida, Gymnasium | **2.7 s / episodio** (renderiza cada paso) |

A 5000 pasos: 2 entrenamientos por celda (~8 min) y 2 por referencia. La recogida
son 60 episodios por `(familia, distancia, semilla)` — y **se hacía seis veces**,
una por método más la referencia, sobre datos idénticos por construcción. En
Gymnasium eso eran 13.5 min tirados por casilla y semilla. Hoisted en esta misma
sesión; ver el changelog.

**Estimación con el arreglo: ~2 días** para las 225 corridas más las 45 parejas
de referencia.

### Una salvedad operativa

La corrida tardó ~40 min de reloj **consumiendo 4.8 min de CPU**. Corría a la vez
que la suite de integración, que construye entornos MuJoCo y dm_control; los
contextos de render se serializan. No lances el benchmark con `pytest` de
integración en paralelo.

---

## R16 — La ejecución completa · TERMINADA (2 ago 10:08)

Lanzada el **30 jul 2026 a las 18:54:58**, PID **1428**, en segundo plano y
desligada de la sesión.

```
Protocolo:  5000 pasos (D12), 20 rollouts, batch 8, seq_len 5, semillas 0-4
Plan:       225 celdas + 45 parejas de referencia
Salida:     cf_worldmodels/results/
Log:        _devlog/full-run-2026-07-30.log   (stdout: una línea por celda)
            _devlog/full-run-2026-07-30.err   (stderr: barras de tqdm)
PID:        _devlog/full-run.pid
```

Estimación ~2 días. El orden es minigrid → gymnasium → dmcontrol, y dentro de
cada familia min → med → max, semilla a semilla.

**Cómo mirar cómo va:**

```bash
tail -20 _devlog/full-run-2026-07-30.log
```

```bash
ls cf_worldmodels/results/*/ | wc -l
```

**Si hay que pararla:** `Stop-Process -Id 1428`. Es reanudable — relanzar el
mismo comando salta lo que ya tenga `metrics.json` y se niega a mezclar
protocolos. Lo único que se pierde es la celda a medias.

**Mientras corre, no lances `pytest` con marcador `integration`**: construye
entornos MuJoCo y dm_control, y los contextos de render se serializan con los
de la corrida (ver R15).

Cuando termine, las tablas salen de:

```bash
python experiments/summarize_results.py
```

---

### Resultado final

**225 celdas, 45 parejas de referencia, 0 pasos con NaN, un solo protocolo, y
cada resultado con su par de tareas registrado.** Del 30 jul 18:54 al 2 ago
10:08, en tres arranques.

| Familia | Celdas | Notas |
| --- | --- | --- |
| minigrid | 75 | 30 jul 18:55 → 31 jul 07:52, ~13 min/celda |
| gymnasium | 75 | 31 jul → 1 ago 08:37 |
| dmcontrol | 75 | `distance_max` rehecho dos veces (F25, F26), ~43 min/celda |

Tres interrupciones, **cero resultados perdidos**: F24 (contexto OpenGL al
cambiar de familia), F25 (par con anchos de acción distintos) y F26 (par que
resultó ser el mismo entorno; sus 25 celdas están archivadas). La reanudabilidad
y la comprobación de protocolo hicieron su trabajo las tres veces.

### El resultado que importa: F27

El olvido **hace pico en el nivel intermedio en las tres familias**, así que las
etiquetas `min/med/max` no aciertan el orden en ninguna. Lo que sí lo predice, por
correlación de rangos sobre las 9 celdas: la dificultad de la tarea B (+0.62) y
`d_trans` (+0.57), frente a +0.13 del nivel etiquetado. Desarrollado en **F27**,
que es el hallazgo principal de toda la ejecución.

### `replay_infinite` contra `finetuning`, las 9 celdas

Replay olvida menos en RD **en 6 de las 9 celdas por unanimidad** (5/5 semillas):
las tres de minigrid, `med` de gymnasium, y `min` y `med` de dmcontrol. Pierde
por unanimidad en **`gymnasium/distance_max`** (0/5, d_z +1.79) y empata en las
otras dos. El Finding 4 invertido se sostiene como tendencia, **no como ley**.

---

### MiniGrid, las tres distancias, 5 semillas

Copia íntegra en `minigrid-summary.txt`. Lo que hay que leer:

**PF** (NLL a un paso sobre `D_A`, latentes congelados)

| Método | min | med | max |
| --- | --- | --- | --- |
| `finetuning` | −5.70 ± 1.15 | −5.44 ± 1.50 | +0.30 ± 2.09 |
| `replay_infinite` | −6.03 ± 0.68 | −7.15 ± 1.10 | −6.16 ± 1.47 |
| `ewc` | **−0.06 ± 0.06** | **−0.01 ± 0.08** | **+0.08 ± 0.08** |
| `progressive_nets` | +3.23 ± 0.99 | +5.45 ± 2.29 | +14.99 ± 3.36 |
| `ug_mtm` | +3.17 ± 1.95 | +12.58 ± 2.36 | +22.89 ± 13.40 |

**RD** (KL entre rollouts imaginados a 15 pasos)

| Método | min | med | max |
| --- | --- | --- | --- |
| `finetuning` | 37.67 ± 4.77 | 82.30 ± 16.07 | 46.62 ± 7.68 |
| `replay_infinite` | **23.02 ± 5.53** | **41.21 ± 7.63** | **23.59 ± 3.42** |
| `ewc` | 29.19 ± 2.72 | 61.91 ± 6.86 | 22.92 ± 1.91 |
| `progressive_nets` | 39.27 ± 6.04 | 76.29 ± 16.78 | 52.66 ± 3.47 |
| `ug_mtm` | 15.78 ± 7.87 | 32.81 ± 8.11 | **1103.42 ± 1647.03** |

**Reconstrucción de la tarea A en píxeles: tras A → tras B**

| Método | min | med | max |
| --- | --- | --- | --- |
| `finetuning` | 0.51 → **398.78** | 2.12 → **682.56** | 43.94 → **1338.72** |
| `replay_infinite` | 0.51 → **0.75** | 2.12 → **1.72** | 43.94 → **44.16** |
| `ewc` | 0.51 → **403.28** | 2.12 → **693.38** | 43.94 → **1287.10** |
| `progressive_nets` | 0.51 → 394.49 | 2.12 → 684.48 | 43.94 → 1308.81 |
| `ug_mtm` | 0.48 → 0.48 | 1.47 → 1.47 | 44.41 → 44.41 |

**`d_trans`**: 19.33 ± 5.34 · 20.26 ± 6.11 · 24.92 ± 4.10.

**Comparación emparejada `replay_infinite − finetuning`** (5 semillas por celda):

| Nivel | RD delta | gana | p | d_z |
| --- | --- | --- | --- | --- |
| min | −14.65 | 5/5 | 0.0625 | −1.99 |
| med | −41.09 | 5/5 | 0.0625 | −2.64 |
| max | −23.03 | 5/5 | 0.0625 | −2.14 |

En PF solo es claro a distancia máxima (delta −6.46, 5/5, d_z −6.08).

### Cinco lecturas de MiniGrid

1. **F18 a escala completa.** Tres de los cinco métodos pierden la
   reconstrucción de la tarea A por un factor de ~790 y PF sale **negativo**
   (`finetuning`, −5.70) o casi cero (`ewc`, −0.06). La métrica del paper dice
   que mejoraron. Con 3 niveles × 5 semillas esto deja de ser anécdota.
2. **EWC es el caso limpio de F21.** PF de −0.06 / −0.01 / +0.08: conserva `M`
   casi exactamente. Y su codificador acaba igual de destruido que el de
   `finetuning` (403 frente a 399). Protege lo que su Fisher cubre y nada más.
   Consecuencia visible en la columna de reparto: para EWC, **RD aporta el
   99.7–99.9% del WMF**, porque PF es cero. Es el mejor argumento posible para
   D10.
3. **El Finding 4 invertido se sostiene en MiniGrid.** Replay olvida menos en
   RD en las 15 comparaciones emparejadas, con la p en su suelo. Pero ver abajo:
   en Gymnasium no se replica.
4. **`d_trans` ordena los niveles** (19.3 < 20.3 < 24.9) pero con ±5–6 **min y
   med se solapan**. Separa `max` del resto, no `min` de `med`.
5. **La tarea A no se aprende igual en los tres niveles** (0.51 / 2.12 / 43.94:
   son entornos distintos), así que los factores de degradación **no son
   comparables entre niveles** en términos absolutos.

### Gymnasium `distance_min`: una celda sin nada que olvidar

| Método | PF | RD | recon A: tras A → tras B | recon B | FT |
| --- | --- | --- | --- | --- | --- |
| `finetuning` | −8.20 ± 0.69 | 75.28 ± 13.22 | 27.77 → **25.25** | 24.60 | +1.73 |
| `replay_infinite` | −8.16 ± 0.77 | 82.84 ± 30.18 | 27.77 → **24.69** | 25.54 | +0.79 |
| `ewc` | +0.02 ± 0.01 | 41.32 ± 6.38 | 27.77 → 25.24 | 24.62 | +1.71 |
| `progressive_nets` | −0.16 ± 0.51 | 81.50 ± 9.24 | 27.77 → 25.71 | 25.04 | +1.28 |
| `ug_mtm` | −0.06 ± 0.06 | 10.92 ± 4.93 | 27.16 → 27.16 | 31.52 | −5.19 |

`d_trans = 28.81 ± 5.14`.

**Nadie olvida.** La reconstrucción de la tarea A **mejora** para los cinco
métodos tras entrenar en B. Es coherente: las dos tareas son el mismo
HalfCheetah con gravedad 9.8 y 7.0, visualmente casi idénticos, así que
entrenar en B sigue enseñando a reconstruir A. Ver **F22**.

Dos consecuencias inmediatas:

- **El Finding 4 invertido no se replica aquí**: replay iguala a finetuning en
  PF (−8.16 vs −8.20) y sale *peor* en RD (82.8 vs 75.3), con el triple de
  dispersión. Lo de MiniGrid puede ser un resultado de MiniGrid.
- **El handicap de UG-MTM desaparece cuando las tareas se parecen**: su
  reconstrucción de B es 31.52 frente a ~25 del resto, no los 557 frente a 2 de
  MiniGrid. Congelar el codificador cuesta poco si el codificador viejo sirve.

---

## R17 — Diagnóstico de P12: dos celdas reproducidas y abiertas por dentro

`minigrid/distance_max`, `ug_mtm`, semillas **2** (RD = 4364) y **3** (RD = 17.7).
Reproducción completa de la celda con `_devlog/diagnose-p12.py`.

**Lo primero, la validación**: RD 4364.4502 contra 4364.4502 guardado y PF
39.9796 contra 39.9796; en la semilla 3, 17.7188 y 18.5238 clavados. La
reproducción es exacta, así que lo que sigue describe el objeto real.

| | s2 paso 0 | s2 paso 4 | s2 paso 14 | s3 paso 0 | s3 paso 14 |
| --- | --- | --- | --- | --- | --- |
| KL | 80.3 | **7872.0** | 2966.5 | 20.6 | 15.2 |
| término de medias | 58.2 | **7690.8** | 2926.4 | 17.5 | 5.1 |
| término de traza | 41.5 | 202.9 | 57.4 | 15.9 | 8.6 |
| `min log_var_k` | −7.26 | **−9.70** | −8.41 | −4.16 | −0.99 |
| \|mu_i\| / \|mu_k\| | 0.28/0.27 | 0.26/0.43 | 0.29/0.57 | 0.34/0.33 | 0.35/0.60 |

**Colapso de varianza, no divergencia del rollout.** Las medias no se mueven
(`|mu_i|` clavado en 0.27 los quince pasos); lo que explota es `var_k`, que baja
a `e^-9.9 = 5e-05` (σ = 0.007). El término `(mu_i − mu_k)²/var_k` hace el resto.
Detalle en F23, política de reporte en P12.

Nota de método: los checkpoints no se guardan (D3), así que diagnosticar exige
reproducir. Que la reproducción salga exacta es la prueba de que `set_seed` +
`preserve_rng_state` + el protocolo en disco bastan para eso, ocho días y tres
familias después de la corrida original.

---

## R18 — Sondeo a 2× presupuesto: el par anidado de Gymnasium (sesión 9, D20)

`run_full_benchmark.py --families gymnasium --distances distance_med distance_max
--methods finetuning --seeds 0 1 2 3 4 --steps 10000 --results-dir results-2x`

10 celdas + 10 parejas de referencia, 30 entrenamientos a 10000 pasos. 0 NaN.
Log en `probe-2x.log`. **La pregunta y la regla de parada estaban escritas en
D20 antes de mirar.**

### La comparación, mismo método y mismas celdas

| | 5000 pasos | 10000 pasos |
| --- | --- | --- |
| RD `distance_med` | 118.04 | 144.70 |
| RD `distance_max` | **58.79** | **102.18** |
| razón med/max | 2.01 | 1.42 |
| pico | **med** | **med** |
| dificultad B, med | 27.10 | 24.65 |
| dificultad B, max | 17.31 | 15.14 |
| `d_trans` med | 30.86 | **57.30** |
| `d_trans` max | 24.71 | **71.51** |
| pérdida en píxeles de A, med | ~0% (control) | +0.8% (control) |
| pérdida en píxeles de A, max | — | +1465.7% |

### 1. El pico aguanta · F27 REFORZADO

El nivel máximo —superconjunto estricto de perturbación del medio— sigue
olvidando **menos** al doble de entrenamiento. Por la regla de D20, se sigue con
el resto del plan.

**Y la objeción no solo sobrevive, se refuta.** El ataque era «a distancia
máxima tu modelo no llega a ajustar B, así que no sobrescribe». Los datos dicen
lo contrario: **B es más FÁCIL de ajustar en el máximo** (15.14 frente a 24.65),
a los dos presupuestos. El cheetah a masa ×3, fricción ×0.5 y gravedad 4.0 se
mueve poquísimo, así que genera observaciones menos variadas. La perturbación
más fuerte de la familia produce la tarea B **más fácil** y el menor olvido —que
es la tesis funcionando, no una excepción a ella.

Nota de paso, y no es menor: **la premisa del eje falla en su propio nivel
superior.** Se supone que perturbar más hace la tarea más difícil. Aquí la hace
más fácil.

**Salvedad honesta: la brecha se estrecha.** La razón med/max baja de 2.01 a
1.42. La dirección aguanta; la magnitud depende del presupuesto. Se reporta así
en `results.tex` §5.2, y un revisor puede preguntar qué pasa a 4×.

### 2. Y el sondeo destapó un problema con `d_trans` · F29

`d_trans` **invierte** entre los dos presupuestos: med < max a 5000, med > max a
10000. Y al mirar las dispersiones, el problema es peor que una inversión —
nunca separó nada. Ver F29.

---

## R19 — Semillas 5–9 en las seis celdas que discriminan · TERMINADA (sesión 9, D21)

`bash _devlog/run-seeds-5-9.sh` — tres invocaciones, una por familia, porque el
runner cruza `--families` con `--distances` y las celdas pedidas no son un
producto cartesiano:

| Familia | Niveles | Celdas | Referencias |
| --- | --- | --- | --- |
| minigrid | min, med, max | 75 | 15 |
| gymnasium | max | 25 | 5 |
| dmcontrol | med, max | 50 | 10 |
| | **Total** | **150** | **30** |

360 entrenamientos a 5000 pasos, ~día y medio. Log en `seeds-10.log`.

**Escribe en `results/`**, a diferencia de R18, porque es el mismo protocolo. El
runner salta las semillas 0–4 ya cacheadas y rechazaría cualquier cosa de otro
presupuesto. **Reanudable**: reejecutar el mismo script.

**Las tres celdas de control se quedan en cinco semillas** (D21): no distinguen
entre métodos por definición, así que semillas ahí no compran resolución.

### Qué se espera, escrito antes de leerlo

- **Lo que compra:** el suelo de la permutación exacta emparejada baja de
  2/2⁵ = 0.0625 a 2/2¹⁰ = **0.002**. Las comparaciones que hoy salen «5/5 con
  p = 0.0625, el suelo» pueden por fin cruzar el umbral convencional — o no, y
  eso también sería informativo.
- **Lo que NO se espera que cambie:** F27. El pico en `med` sale 3 de 3 con dos
  agregaciones y dos presupuestos; cinco semillas más no deberían moverlo. Si lo
  mueven, es un resultado más interesante que el que hay.
- **Lo que hay que vigilar:** las 22 celdas marcadas por sesgo (D15). Con 10
  semillas la marca de cola pesada puede aparecer o desaparecer en varias, y el
  paper cita ese conteo.

### Lo que obliga a hacer después, y no es opcional

Cambian las medianas de seis celdas, así que **cambia cada cifra que el paper
cita de ellas**. Regenerar las ocho tablas y hacer la pasada de números sobre
`abstract.tex`, `intro.tex`, `results.tex` y `discussion.tex`. Las frases que
hoy dicen «0.0625, el suelo con n=5» hay que reescribirlas enteras.

**`final-summary.txt` queda obsoleto mientras esto corre.** Regenerarlo al
acabar.

### Resultado (4 ago 2026)

**375 celdas, un protocolo, 0 pasos con NaN**, tres exits en 0. Seis celdas a
n=10, los tres controles a n=5.

**F27 aguanta.** Pico en `med` en las tres familias, y la etiqueta baja de
+0.05 a **+0.00** — de casi nada a exactamente nada.

| | n=5 | n=10 |
| --- | --- | --- |
| RD minigrid min/med/max | 32.3 / **65.4** / 36.4 | 33.73 / **60.88** / 32.11 |
| RD gymnasium | 77.4 / **85.9** / 59.6 | 77.45 / **85.91** / 64.01 |
| RD dmcontrol | 1.2 / **71.8** / 49.0 | 1.15 / **71.64** / 52.61 |
| ρ etiqueta | +0.05 | **+0.00** |
| ρ `d_trans` | +0.53 | **+0.58** |
| ρ dificultad de B | +0.58 | **+0.43** |

### Lo que compró, que era el objetivo

El suelo de p baja a 0.002 y **cuatro celdas lo alcanzan**. Replay contra
finetuning en RD, sobre las seis celdas con olvido:

| Celda | n=5 | n=10 |
| --- | --- | --- |
| minigrid min | 5/5, p=0.0625 | **10/10, p=0.0020**, d_z −2.28 |
| minigrid med | 5/5, p=0.0625 | **10/10, p=0.0020**, d_z −3.11 |
| minigrid max | 5/5, p=0.0625 | **10/10, p=0.0020**, d_z −2.36 |
| gymnasium max | replay peor 5/5 | **replay peor 9/10, p=0.0059**, d_z +1.14 |
| dmcontrol med | 5/5, p=0.0625 | **10/10, p=0.0020**, d_z −3.01 |
| dmcontrol max | contaba como victoria | **p=0.6758, d_z −0.14 — no hay efecto** |

**La última fila es el hallazgo del gasto.** Con cinco semillas la celda contaba
como victoria de replay; con diez no hay dirección. «Cinco de seis» pasa a
«cuatro de seis, una en contra y una nula», y eso está en el paper.

### Lo que se movió y obligó a reescribir, no a retocar

1. **Los dos predictores medidos se intercambian.** Dificultad de B 0.58 → 0.43;
   `d_trans` 0.53 → 0.58. Ahora `d_trans` lidera — el mismo instrumento que F29
   dice que no resuelve dentro de familia. Se reporta como lo que es: los dos
   baten a la etiqueta con holgura y **ninguno está resuelto frente al otro**,
   con la inestabilidad como prueba directa de que n=9 no da para ordenarlos.
2. **EWC ya no es «marginalmente peor» que finetuning en el codificador**
   (753 vs 745), sino **indistinguible** (800 vs 811). Es el mismo argumento sin
   el adorno.
3. **PF de finetuning: negativo en 7 de 9 celdas**, no en 8.
4. **Sesgo: 17 de 180 celdas**, no 22.
5. **`d_trans` de dmcontrol también invierte** med/max (7.58 vs 7.46, rangos
   solapados). Más evidencia para F29, ahora en dos familias.

### Un bug encontrado al leerlo, y arreglado en la raíz

`export_tables.py` se negó a generar nada: `check_runs_consistent` veía dos
protocolos. **El único campo que difería era `seeds`** — la lista que cada
invocación pidió, o sea procedencia, no presupuesto. Falso positivo mío, y el
runner tenía el mismo fallo latente: habría rechazado su propia caché tras
cualquier `--seeds`. Corregido con `protocol_identity()` en
`run_full_benchmark.py`, que es de donde salen ambos. Ver I22.

**Y una consecuencia para el paper:** la Tabla 1 decía «Seeds 5 (0,1,2,3,4)»
leyendo esa lista. Ahora **cuenta las semillas que existen** y dice «5--10», y
la tabla de las nueve casillas gana una columna con el n de cada una.

---

## R20 — k=4 en MiniGrid · TERMINADA (sesión 9, D22, cierra el paso 3 de D20)

`python experiments/run_sequence.py` — 4 tareas × 5 métodos × 5 semillas = 100
entrenamientos a 5000 pasos, en `results-seq/`. Reanudable por semilla y método.

Secuencia: `Empty-5x5 → Empty-8x8 → FourRooms → KeyCorridorS3R1`, las mismas
cuatro de la rejilla emparejada y en el mismo orden.

**Prueba de humo previa (60 pasos, semilla 999):** los cinco métodos atraviesan
la secuencia entera. EWC acumuló 3 diagonales de Fisher, progressive nets 3
columnas, replay los 4 buffers, UG-MTM 4 expertos. La diagonal PF(i,i) y RD(i,i)
salió **0.0 exacto**, que es la comprobación de que cada tarea se compara contra
su propio snapshot.

### Qué se espera, escrito antes de mirar

- **Lo que la rejilla no puede decir:** si el olvido de `T_1` se agrava a medida
  que pasan tareas, o se satura. Con k=2 no hay curva que mirar.
- **Predicción sobre los métodos:** progressive nets y EWC deberían separarse
  aquí más que en k=2, porque las dos acumulan estructura por tarea (columnas y
  penalizaciones) y con un solo cambio de tarea esa acumulación no se ve.
- **Riesgo conocido:** UG-MTM tiene K=4 expertos, o sea que 4 tareas lo agotan
  exactamente. Con k=5 se quedaría sin expertos y el resultado hablaría del
  límite, no del método. Por eso la secuencia es de 4 y no de 5.
- **No cambia nada del paper actual.** Es una subsección nueva; la rejilla de
  9 casillas y sus cifras no se tocan.

### Resultado (5 ago 2026)

25 corridas (5 métodos × 5 semillas), 4 tareas cada una, **0 pasos con NaN**.

#### RD(T1, k) — cuánto se ha perdido de la primera tarea, según avanza la secuencia

| Método | tras T2 | tras T3 | tras T4 |
| --- | --- | --- | --- |
| finetuning | 38.6 | **250.3** | 125.9 |
| replay_infinite | 25.4 | 40.4 | 74.8 |
| ewc | 29.3 | **184.3** | 68.4 |
| progressive_nets | 45.8 | **142.8** | 65.1 |
| ug_mtm | 16.0 | 24.6 | 35.3 |

#### Y el hallazgo: **el olvido tampoco es monótono en la longitud de la secuencia**

El olvido de T1 **hace pico en T3 y retrocede en T4** — en los tres métodos que no
protegen el codificador, y por **unanimidad de las cinco semillas**:

| Método | semillas con RD(1,3) > RD(1,4) |
| --- | --- |
| finetuning | **5/5** |
| ewc | **5/5** |
| progressive_nets | **5/5** |
| replay_infinite | 1/5 |
| ug_mtm | 2/5 |

Que salga 5/5 justo en los tres que mueven el codificador, y ~0 en los dos que no
lo tocan, es consistencia interna: no es ruido, es el efecto apareciendo solo
donde hay algo que mover.

**Y la causa es la misma que en F27.** Dificultad de cada tarea (reconstrucción
reservada tras entrenarla):

| | T1 | T2 | **T3** | T4 |
| --- | --- | --- | --- | --- |
| dificultad | 0.52 | 2.60 | **47.32** | 32.58 |

**El olvido de T1 hace pico exactamente en la tarea más difícil de la secuencia, y
retrocede cuando viene una más fácil.** Es F27 reproducido sobre un eje
**completamente distinto** —posición en la secuencia en vez de nivel de
distancia— con los mismos entornos y el mismo presupuesto.

Eso es la confirmación independiente que a F27 le faltaba: el pico sobre los
niveles podía ser una peculiaridad de cómo se construyeron los niveles. El pico
sobre la posición, dentro de una sola corrida continua, no.

#### La predicción pre-registrada: acertada a medias

Predije que «progressive nets y EWC deberían separarse más aquí que en k=2».
**Se cumple**: al final de la secuencia los dos quedan por debajo de finetuning
(65.1 y 68.4 frente a 125.9) e incluso por debajo de replay (74.8), cuando en la
celda equivalente de k=2 progressive nets era el *peor* de los cuatro RSSM.

**Lo que no predije es la no-monotonía**, que es el resultado de verdad.

#### Lo que replica exacto de la rejilla

- **El codificador.** Factor de degradación de T1 en píxeles tras T4: finetuning
  773, ewc 759, progressive nets 785 — y **replay 3.6**, **ug_mtm 1.00**. Misma
  historia que en k=2, ahora sobre una secuencia.
- **La disociación de EWC.** RD 68.4 (conserva `M` razonablemente) con el
  codificador a 759. F21 otra vez, ahora acumulando tres Fisher.
- **UG-MTM.** Factor 1.00 exacto en las tres etapas: congela el VAE. No olvida
  porque no aprende, también en secuencia.

#### Escrito en el paper (§5.7)

Con generador propio, `experiments/summarize_sequence.py` + `sequence_table()`
en `export_tables.py`, y 8 tests. **Las cifras de arriba salen ahora de ahí**, no
del script suelto con el que se leyeron primero — reproducen exacto.

`tab_sequence` se genera solo si existe `results-seq/`: las ocho tablas de la
rejilla k=2 no dependen de que la corrida de secuencia exista.

La columna de píxeles se formatea como la de `tab_encoder` (entero por encima de
10) para que la prosa pueda citar la tabla **literalmente** en vez de redondearla
otra vez. `check-numbers.py`: 158 respaldadas por tabla, 0 sueltas en §5.7.
