# Paper (`main (5).pdf`, 9 jul 2026) vs. código y datos

Contrastado contra `src/`, `experiments/` y los `metrics.json`.

> **Aviso de vigencia (2 ago 2026).** Las secciones de abajo se escribieron
> contra los **225 resultados originales (R0), que son inválidos** (VAE
> colapsado, F0). Se conservan porque documentan qué decía el paper y de dónde
> salían sus números — no se reescriben en silencio (D4). Lo que ya **no** vale
> de ellas está marcado, y lo nuevo está en la sección final,
> *Discrepancias tras la ejecución válida (R16)*.

---

## Tabla 1 — Protocolo de entrenamiento

Circulan **tres** conjuntos de valores distintos:

| Parámetro | Paper | YAML `configs/` | Código ejecutado |
| --- | --- | --- | --- |
| Rollouts por tarea | 1000 | 1000 | **20** |
| Pasos de gradiente | 3000 | 50000 | **1000** |
| Batch size | 16 | 32 | **8** |
| Longitud de secuencia | 10 | 50 | **5** |
| Semillas | 5 | 5 | 5 ✓ |
| Horizonte H (RD) | 15 | — | 15 ✓ |
| `mc_dropout_T` | 3 | 10 | **3** ✓ |

El paper parece escrito desde los YAML, que a su vez no describen lo que
ejecuta `run_full_benchmark.py`.

---

## Tabla 3 — Celdas que no reproducen desde `results/`

Paper vs. media real sobre 5 semillas (redondeado a 3 decimales):

| Método | Celda | Paper | Real |
| --- | --- | --- | --- |
| Fine-tuning | Gym-med | 0.007 | **0.005** |
| Fine-tuning | Gym-max | 0.008 | **0.009** |
| Fine-tuning | DMC-max | 0.016 | **0.015** |
| Replay | MG-med | 0.041 | **0.040** |
| Replay | Gym-min | 0.006 | **0.007** |
| Replay | DMC-max | 0.040 | **0.042** |
| Prog. Nets | MG-med | −0.027 | **−0.024** |
| Prog. Nets | MG-max | −0.012 | **−0.014** |
| Prog. Nets | Gym-min | 0.009 | **0.008** |
| Prog. Nets | DMC-max | 0.016 | **0.013** |

Con el código publicado la tabla no sale igual. Parece transcrita a mano o
generada desde otra ejecución.

---

## Desviaciones típicas (§5.2)

| Afirmación | Real |
| --- | --- |
| Fine-tuning MG-min `0.081 ± 0.057` | `0.081 ± 0.069` |
| DMControl fine-tuning `0.042 ± 0.004` | `0.042 ± 0.005` |
| DMControl replay `0.039 ± 0.011` | `0.038 ± 0.006` |

---

## §5.5 — Rango de Forward Transfer de UG-MTM

Afirma `FT ≈ −0.5 a −0.8`. Datos reales:

```
media  = -1.207
rango  = -2.103  ..  -0.317
por familia:  MiniGrid -0.44..-0.50 | Gymnasium -1.66..-1.73 | DMControl -1.39..-1.55
```

El rango citado describe solo MiniGrid y subestima la magnitud global ~2×.

---

## §3.4 — DMControl `distance_min`

Descrito como "Cheetah-run → Cheetah-run+wind". El viento no existe: ambas
tareas son `cheetah/run` idénticas (ver F8).

---

## Finding 5 — "Progressive Networks exhibit forward transfer"

Con n=5, **ninguna** de las nueve celdas es distinguible de cero
(t-test de una muestra contra 0):

| Familia | Nivel | Media | sd | p |
| --- | --- | --- | --- | --- |
| MiniGrid | min | −0.0124 | 0.0437 | 0.56 |
| MiniGrid | med | −0.0240 | 0.0396 | 0.25 |
| MiniGrid | max | −0.0141 | 0.0402 | 0.48 |
| Gymnasium | min | +0.0079 | 0.0328 | 0.62 |
| Gymnasium | med | +0.0073 | 0.0319 | 0.63 |
| Gymnasium | max | +0.0083 | 0.0325 | 0.60 |
| DMControl | min | +0.0076 | 0.0327 | 0.63 |
| DMControl | med | +0.0212 | 0.0334 | 0.23 |
| DMControl | max | +0.0132 | 0.0326 | 0.42 |

La desviación típica supera la media en todos los casos: el signo negativo es
ruido.

---

## Finding 2 — Contradicción interna

El abstract dice que el olvido "escala con la complejidad observacional". El
propio texto del hallazgo dice que el WMF es más alto en MiniGrid, la familia
observacionalmente más simple.

Orden real: MiniGrid (0.081) > DMControl (0.042) > Gymnasium (0.007). No es
monótono en ninguna noción de complejidad.

---

## ~~Lo que sí se sostiene~~ · **el Finding 4 ya no** (R12, R16)

**Finding 4 — "Replay is insufficient".** Que replay tenga *más* olvido que
fine-tuning en MiniGrid a distancia media y máxima parecía estadísticamente
claro:

```
MiniGrid distance_med:  ft=+0.0107  replay=+0.0404  p<0.001
MiniGrid distance_max:  ft=+0.0147  replay=+0.0775  p<0.001
```

> **Invertido en R12 y confirmado en R16.** Con el pipeline corregido, replay
> olvida **menos** que fine-tuning en las tres celdas de MiniGrid, y en 6 de las
> 9 celdas del banco por unanimidad de las 5 semillas. Pierde por unanimidad en
> `gymnasium/distance_max`. La p<0.001 original era además imposible: con n=5,
> el suelo de un test exacto emparejado es 0.0625.

**Valores de `d_param`.** 0.283 / 0.586 / 0.622 — reproducen exactamente.

**Valores de `d_param`.** 0.283 / 0.586 / 0.622 — reproducen exactamente.

**Tabla 2 (hiperparámetros de UG-MTM).** Coincide con el código, incluido T=3.

---

## Impacto sobre el paper

Todos los números de la Tabla 3 y de §5.2 provienen de mediciones sobre ruido
(F1) y con el objetivo de NLL equivocado (F2). **Hay que re-ejecutar y
reescribir**, no parchear.

---

## Discrepancias tras la ejecución válida (R16, 2 ago 2026)

Lo de arriba compara el paper con los resultados inválidos. Esto lo compara con
las **225 celdas buenas**. Es la lista de lo que el paper nuevo **no puede
decir** como lo decía el viejo.

### Protocolo (Tabla 1)

| Parámetro | Paper | Ejecutado en R16 |
| --- | --- | --- |
| Rollouts por tarea | 1000 | **20** |
| Pasos de gradiente | 3000 | **5000** |
| Batch size | 16 | **8** |
| Longitud de secuencia | 10 | **5** |

La tabla del paper se genera ahora desde los `metrics.json`, que llevan su
bloque `protocol` entero. Nunca a mano.

### Pares de tareas que el paper declara y no se pueden ejecutar

- **§3.4, DMControl `distance_min`**, "Cheetah-run → Cheetah-run+wind": el viento
  no lo leía nadie y ambas tareas eran idénticas (F8). Ahora es `cheetah/run` con
  **gravedad 9.81 → 7.0**, la misma perturbación que el `distance_min` de
  Gymnasium.
- **DMControl `distance_max`**, `cheetah/run → reacher/easy`: **inejecutable**.
  6 actuadores contra 2 y un solo world model para las dos tareas (F25). Ahora es
  `cheetah/run → walker/stand`, que comparte los 6.

Las dos cosas hay que declararlas: el paper describe un diseño que no corría.

### Métricas que cambian de definición

- **FT** ya no es `NLL(random) − NLL(post-A)`, que no tocaba ni un dato de la
  tarea B y salía idéntica para todos los métodos con la misma arquitectura
  (F20). Ahora es `recon_B(desde cero) − recon_B(preentrenado)`. La cifra vieja
  sobrevive como `task_A_fit_gain`.
- **`d_trans`** (Ec. 9) se calcula por primera vez, con un modelo por entorno
  entrenado desde cero. El paper la asignaba a MiniGrid y DMControl y no la
  calculaba nadie (F15).
- **WMF** deja de ser el número principal (D10). Se publican PF y RD por
  separado, con el reparto a la vista: RD aporta el 78–99.9%.
- **PIS** se retira (D18). El paper la anunciaba como una de las cuatro métricas
  y nunca se implementó (F6); la suite pasa a ser PF, RD y FT, y `pis` se guarda
  como `null` en vez de como un `0.0` que se leería como medición.

### Afirmaciones del abstract que la ejecución no respalda

- **"Controlled dynamic distances".** El eje ordinal `min/med/max` **no acierta
  el orden del olvido en ninguna de las tres familias** (F27): el olvido hace
  pico en el nivel intermedio. `d_trans` ordena mejor (+0.53 frente a +0.05) pero
  tampoco es limpio, y en Gymnasium contradice a `d_param`.
- **Los cinco Findings.** No sobrevive ninguno; el 4 además se invierte.

### Lo que sigue sosteniéndose

- **El hueco.** Nadie mide olvido a nivel del componente de transición `M`.
- **`d_param`**: 0.283 / 0.586 / 0.622, reproducen exactamente.
- **Tabla 2** (hiperparámetros de UG-MTM), incluido `T=3`.
- **El banco discrimina**: cinco métodos separables en PF, RD y FT, con 5
  semillas y comparación emparejada.
