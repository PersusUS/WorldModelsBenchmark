# WMF Benchmark — Traspaso

Documento autocontenido para retomar el proyecto en una sesión nueva.
Última actualización: **14 ago 2026, fin de sesión 17 — ENVIADO a CL4FMAgents. Una auditoría externa corrigió tres cifras rancias y declaró dos elecciones de protocolo antes de enviar. Lo siguiente no es trabajo, es esperar al 29 sep.**

> **Lectura mínima para arrancar:** §0 (estado), §1 (qué es), §3 (el paper) y
> §4 (qué hacer ahora). El resto es referencia que se consulta, no se lee.

## 0. Estado en una pantalla (14 ago 2026)

**Enviado el 14 ago 2026 a CL4FMAgents @ NeurIPS 2026** (deadline era el 29 ago
AoE, no archival). Notificación el **29 sep**; taller el 11–12 dic en Sídney.
No queda trabajo de investigación ni de escritura: queda esperar, y decidir si
se manda también a CWM (deadline 30 ago) antes de que pase la fecha.

**Lo que la auditoría de la sesión 17 cambió, todo antes del envío y sin tocar
un solo número de las tablas** (commits `f822fc7`, `dabe254`, `04beb69`,
`1d12cd2`, `79c2e82`): el PF de finetuning es negativo en **siete** de nueve
celdas, no ocho —lo decían mal la versión corta y `discussion.tex`, y la tabla
generada siempre dijo siete—; el peor crecimiento de replay es **31%**, no 27%;
las parejas de referencia son **75 contra 375 corridas**, no 45 contra 225; un
`\ref` de `discussion.tex` llevaba un byte CR en vez de la barra y el PDF de 26
páginas renderizaba «Section efsec:discussion:methods». Y dos declaraciones que
el paper debía y no daba: **progressive nets se evalúa sin oráculo de tarea**
(con oráculo, la columna A congelada daría PF = 0 por construcción, así que el
PF positivo es una afirmación sobre inferencia task-agnostic), y **UG-MTM
congela también la cabeza μ/σ** además del VAE, que es lo que `switch_task`
hizo siempre. La afiliación pasó a Universidad de Sevilla, que es la del perfil
de OpenReview.

**El entorno estaba roto y se arregló:** había `numpy 2.4.6` contra el
`numpy==1.26.4` que fija `requirements.txt`, y torch 2.3.1 lanzaba «Numpy is
not available» en `from_numpy` — 22 tests caídos, cero celdas reproducibles.
Con la versión fijada vuelven los **399 tests** de la suite rápida. Ojo para el
futuro: los 375 resultados se produjeron con **torch 2.3.1**, y
`requirements.txt` fija **2.1.2**; ningún `metrics.json` guarda versiones de
librería, así que la reproducibilidad bit a bit está verificada en esta máquina
y no garantizada en otra.

**Deadline: 29 ago 2026 AoE — CL4FMAgents @ NeurIPS 2026.** Ocho páginas sin
referencias, no archival, notificación el 29 sep, taller el 11–12 dic en
Sídney. Pide explícitamente *negative results* y *benchmarks*, y entre sus
temas están «catastrophic forgetting» y «continual learning for … world
models». **La versión de 8 páginas ya está escrita** (`paper/main_workshop.tex`,
§3). Segundo destino posible el mismo fin de semana: **Continual World Models**
(CWM), deadline 30 ago, cuyas bases aún no estaban publicadas al cierre de esta
sesión.

**El repositorio es público**: https://github.com/PersusUS/WorldModelsBenchmark
— una sola rama `main`, descripción y trece topics. La bitácora viaja dentro
(los `.md` y los scripts; fuera quedan `archive/`, logs y volcados
regenerables).

**El PDF está compilado y verificado.** Nadie lo había mirado hasta la sesión
12: entonces la Tabla 1 se salía 2,2 pt (siete columnas; sobresalían las reglas
de `booktabs`, no el texto). Corregido bajando el relleno de columna de 4 pt a
3 pt, **recompilado el 13 ago y vuelto a medir**: 26 páginas, **cero líneas y
cero reglas fuera de la caja**, y la Tabla 1 sigue completa —los siete entornos
y las nueve `d_trans`— pese a estrecharla. Las cifras corregidas siguen dentro
(88.5, 58.79, 2.01, 14.8, 8135) y las viejas no (78–97, 1.96, 15×).

**El README está reescrito** para un repo público con un paper dentro. Abría
describiendo el protocolo, decía «225 runs» y «291 tests», y su sección de
resultados rezaba «Not yet published — results are being regenerated». Ahora
abre con los dos hallazgos y sus cifras, enlaza el PDF, y sus recuentos son los
de hoy.

- **375 celdas ejecutadas y commiteadas.** 75 parejas de referencia, 0 pasos con
  NaN, un solo protocolo (5000 pasos), cada resultado con su par de tareas.
  Diez semillas en las seis celdas que discriminan, cinco en los tres controles
  (D21).
- **El paper está completo y compilado**: nueve secciones más `main.tex`, con
  **diez tablas generadas** por `experiments/export_tables.py`, y el PDF de 26
  páginas en `paper/WMF.pdf`. Al lado, **la versión de 8 páginas** para el
  taller, en `paper/main_workshop.tex`. Ver §3.
- **El título ya no promete el método**: «Forgetting in World Models Does Not
  Follow Task Distance: A Component-Level Benchmark and Two Negative Results».
  Es una elección, no un dato; cambiarlo es una línea de cada `main.tex`.
- **Una sola rama, `main`**, en local y en el remoto, árbol limpio. La rama de
  sesión se fusionó por fast-forward y se borró.
- **El hallazgo principal es F27** y no es sobre métodos: el olvido hace pico en
  el nivel intermedio en las tres familias, así que el eje `min/med/max` no
  acierta el orden en ninguna. Sobrevive al cambio de agregación **y al doble de
  presupuesto**.
- **No hay decisiones abiertas sobre lo que se publica.** P12, P13 y P2 se
  cerraron en la sesión 8 (D15, D16, D18). Lo que queda abierto (P1, P5, P6,
  P11) es sobre UG-MTM y sobre el Fisher de EWC, y no bloquea escribir.
- **F27 está comprobado al doble de presupuesto (R18) y al doble de semillas
  (R19).** El pico en `med` aguanta las dos veces, y la objeción «es que no
  llegas a ajustar B» **se refuta**: B es más *fácil* de ajustar en el máximo
  (15.14 frente a 24.65). La brecha se estrecha con el presupuesto (1.96 →
  1.42): la dirección es robusta, la magnitud no.
- **Ojo al escribir sobre los predictores.** A n=10 los dos medidos **se
  intercambian**: dificultad de B 0.58 → 0.43, `d_trans` 0.53 → 0.58. El paper
  ya **no los ordena** — dice que los dos baten a la etiqueta y que n=9 no los
  resuelve, con el propio intercambio como prueba. No lo vuelvas a convertir en
  un ranking.
- **`d_trans` sale tocado por partida doble.** **F28**: puntúa el modelo B en la
  base latente de A, la misma objeción que obligó a medir FT en píxeles (F20).
  **F29**: no separa los niveles dentro de una familia —las medianas de
  Gymnasium abarcan 6 unidades y cada celda abarca 15–19 entre semillas— y al
  doblar el presupuesto dos de las tres se intercambian. **Ninguno toca F27**,
  que es un negativo sobre el eje etiquetado y no pasa por `d_trans`. Lo que se
  cae es la mitad positiva: la recomendación baja a «reportad una distancia
  medida y exigidle cuentas».
- **La bitácora se versiona desde la sesión 10.** Los documentos y los scripts
  de comprobación están en git; siguen fuera `archive/` (resultados de corridas
  invalidadas), los `.log`/`.err` de cada corrida y los volcados regenerables
  como `final-summary.txt`. Ojo: hay remote en GitHub y **el push no está
  hecho**, así que la publicación efectiva sigue siendo una decisión del autor,
  y `paper-plan.md` contiene la valoración interna de viabilidad.

Las dos órdenes que reproducen todo lo publicable, sin entrenar nada:

```bash
cd C:/Users/Usuario/WorldModelsBenchmark/cf_worldmodels && python experiments/summarize_results.py
```

```bash
cd C:/Users/Usuario/WorldModelsBenchmark/cf_worldmodels && python experiments/export_tables.py
```

---

## 1. Qué es el proyecto

**WMF Benchmark** — un banco de pruebas para medir **olvido catastrófico en el
componente de transición `M` de un world model**, no en la política ni en el
sistema completo.

**Tesis: el benchmark ES la contribución** (decisión D8). UG-MTM, el método del
autor, es uno de los cinco métodos evaluados, no el resultado. Que gane o pierda
no compromete el paper. Criterio de éxito: ¿discrimina el benchmark entre métodos
de forma estable, interpretable y reproducible?

**Autor:** Jesús Pérez Bazarot. Paper previo: `_devlog/archive/paper-anterior-main5.pdf` (9 jul 2026),
"Catastrophic Forgetting in World Model Transition Components: A Benchmark and
Structural Mitigation". **Hay que rehacerlo** — ver `paper-plan.md`, que incluye
una valoración honesta de viabilidad.

### Diseño

- **3 familias** × **3 niveles de distancia dinámica** × **5 métodos** × **5
  semillas** = 225 corridas.
- Familias: MiniGrid (discreto), Gymnasium/MuJoCo HalfCheetah (continuo, física
  variable), DMControl (visual).
- Protocolo: entrenar en tarea A → cambiar a tarea B → medir degradación en A.
- Distancias: `d_param` (L2 normalizada entre vectores de física) y `d_trans`
  (KL entre un modelo de transición por entorno, Ec. 9). Desde la sesión 7 las
  dos se calculan; `d_trans` cubre las 9 casillas.

### Métodos

| Método | Rol |
| --- | --- |
| `finetuning` | Secuencial sin mitigación (cota inferior) |
| `replay_infinite` | Buffer ilimitado de todas las tareas (cota superior aprox.) |
| `ewc` | Elastic Weight Consolidation, λ=1000 |
| `progressive_nets` | Una columna GRU por tarea con conexiones laterales |
| `ug_mtm` | Uncertainty-Gated Mixture of Transition Models (del autor) |

### Métricas, y qué mide realmente cada una

**La suite es PF, RD y FT** (D18: PIS se retira, ver abajo), y **PF y RD se
publican por separado** (D10). No hay escalar único: `WMF = 0.4·PF + 0.4·RD` es
un agregado del que RD se lleva el 78–97%, o sea RD con pasos de más. WMF se
sigue calculando y se imprime rotulado como agregado heredado, con la columna que
dice cuánto de él es RD, para poder reproducir el número del paper anterior.

| Métrica | Qué mide | Estado |
| --- | --- | --- |
| **PF** | NLL a un paso sobre `D_A`, entre el modelo post-A y el post-B | Reportada. Alcance declarado: mide `M` en base latente fija (D9) |
| **RD** | KL entre rollouts imaginados a 15 pasos | Reportada. Mismo alcance |
| **FT** | Transferencia hacia delante | **Rehecha (s7)**: `recon_B(desde cero) − recon_B(preentrenado)`, en píxeles. Ya discrimina métodos |
| ~~**PIS**~~ | Impacto en la política | **Retirada de la suite (D18)**: se anunció y nunca se implementó (F6). Se guarda como `null`, no como `0.0` |
| `d_trans` | Distancia entre entornos (Ec. 9) | **Implementada (s7)**, un modelo por entorno entrenado desde cero |

PF y RD se evalúan sobre latentes **congelados**, codificados una sola vez por el
modelo post-tarea-A (decisión D5). Eso aísla `M`, que es el objetivo declarado,
pero implica que la deriva del codificador es invisible para ellas. **El paper lo
declara como alcance y reporta la reconstrucción en píxeles al lado** (D9): cada
corrida guarda las dos escalas, y la diferencia entre ellas es un hallazgo (F18),
no una nota al pie.

Y no es solo cosa de las métricas: **EWC tampoco protege el codificador**, y por
el mismo motivo estructural — su Fisher es exactamente cero ahí (F21). Las dos
mitades de esa predicción están medidas en R16.

---

## 2. Dónde estamos

> Esta sección es **historia y evidencia**: cómo se llegó al estado de §0 y qué
> significan los resultados. Para escribir el paper, lo operativo es §3. Para
> comprobar un número, `final-summary.txt`.

### Salud del banco de pruebas: los bugs que lo invalidaban están cerrados

| | Qué era | Estado |
| --- | --- | --- |
| **F0** | Posterior del VAE colapsado: 0/32 dims activas | CORREGIDO (s4) |
| **F1** | Evaluación sobre ruido gaussiano | CORREGIDO (s4) |
| **F2** | `compute_nll` puntuaba contra `z` en vez de `z'` | CORREGIDO (s4) |
| **F13** | La KL de RD y `d_trans` no era una KL (RD inflada ~12×) | CORREGIDO (s5) |
| **F16** | El pipeline no reproducía con la misma semilla | CORREGIDO (s5) |
| **F17** | No se registraba si el modelo aprendió la tarea A | INSTRUMENTADO (s6) |
| **I1** | Hiperparámetros hardcodeados, configs mintiendo | CORREGIDO (s6) |
| **I3** | El Fisher de EWC era `(E[∇])²`: EWC casi inerte | CORREGIDO (s7) |
| **F8** | DMControl `distance_min` comparaba una tarea consigo misma | CORREGIDO (s7) |
| **F15** | `d_trans` (Ec. 9) no la calculaba nadie | CORREGIDO (s7) |
| **F20** | FT no medía transferencia | CORREGIDO (s7) |

Más F4, F5, F7, F9, F12, I2, I4. **El benchmark mide algo válido, reproducible,
documenta a qué presupuesto se midió y si había algo que olvidar, y todas las
cifras que publica miden lo que dicen medir.** Desde la sesión 8 ya no hay
excepción: PIS no se arregló, se retiró (D18), que era la otra salida honesta.

### La ejecución completa está TERMINADA

**R16** terminó el 2 ago a las 10:08: **225 celdas, 45 referencias, 0 pasos con
NaN, un protocolo, y cada resultado con su par de tareas.** Están commiteadas.

**El resultado principal es F27, y no es sobre métodos.** El olvido hace pico en
el nivel *intermedio* de distancia en las tres familias, así que el eje ordinal
del diseño (`min < med < max`) no acierta el orden en ninguna. Sobre las 9
celdas, lo que predice el olvido por correlación de rangos es la **dificultad de
la tarea B** (+0.58) y `d_trans` (+0.53), no el nivel etiquetado (+0.05). La
lectura: **el olvido escala con lo que la tarea nueva exige del modelo, no con lo
lejos que está de la vieja.** (Con la agregación anterior, medias en vez de
medianas, era +0.62 / +0.57 / +0.13: el resultado no depende de eso.)

La corrida necesitó **tres arranques** y no perdió una sola celda — el runner
salta lo cacheado y rechaza lo que venga de otro protocolo:

1. **F24**, al pasar de gymnasium a dmcontrol: contexto OpenGL. Corregido con un
   subproceso por familia.
2. **F25**, al entrar en `dmcontrol/distance_max`: el par emparejaba tareas de 6
   y 2 actuadores (cheetah y reacher). Corregido con un preflight que lo detecta
   en segundos.
3. **F26**, que no fue una caída sino un resultado falso: el par de repuesto que
   elegí (`walker/stand`) resultó ser **el mismo entorno** que `distance_med`
   (`walker/run`) — en dm_control las tareas de un dominio solo se diferencian en
   la recompensa, que este benchmark no usa. Las 25 celdas eran duplicados y
   están archivadas.

`distance_max` es ahora `cheetah/run → walker/run` con gravedad 4.0, masa ×3 y
fricción ×0.5 (**D14**), la misma perturbación que el nivel máximo de Gymnasium.
Relanzado el 1 ago a las 21:04, log `full-run-dmc-max3.log`, ~5 h.

**Y desde F26 cada `metrics.json` guarda su par de tareas**, que la comprobación
de caché valida igual que el protocolo — el agujero que habría hecho pasar las
celdas viejas por nuevas. Números completos en `runs.md`
(R16) y en `minigrid-summary.txt`; lo que sigue es lo que significan.

**1. F18 deja de ser un diagnóstico y pasa a ser un resultado.** En MiniGrid, tres
de los cinco métodos pierden la reconstrucción de la tarea A por un factor de
~790 (0.51 → ~400 a distancia mínima) mientras **PF sale negativo** para
`finetuning` (−5.70) y **cero** para `ewc` (−0.06). Las métricas del paper dicen
que esos modelos no empeoraron. Tres niveles, cinco semillas.

**2. EWC protege `M` y nada más, y ahora está medido en las dos mitades.** PF de
−0.06 / −0.01 / +0.08 (conserva la transición casi exactamente) con una
reconstrucción de A de 403 / 693 / 1287, indistinguible de `finetuning`. Es F21
confirmado: su Fisher es cero sobre el codificador. Efecto colateral que cierra
el argumento de D10: **para EWC, RD aporta el 99.7–99.9% del WMF**.

**3. El Finding 4 invertido se sostiene, pero no en todas partes.** Sobre las
**seis celdas que producen olvido**, replay tiene menor RD que finetuning en
cinco, por unanimidad de semillas en cuatro (p = 0.0625, el suelo; d_z de −1.99 a
−3.05). La excepción es `gymnasium/distance_max`, donde replay es **peor** en las
cinco semillas (90.61 frente a 58.79, d_z = +1.79). Y lo paga en transferencia:
en las tres celdas no-MiniGrid donde protege el codificador, su FT es el peor de
los cuatro métodos RSSM por un factor de 2–4 (−10.48, −7.84, −8.54) — con la
salvedad de que replay entrena sobre A+B, así que su FT mezcla transferencia con
la mitad de presupuesto efectivo en B.

**4. Tres celdas no producen olvido (F22) · CERRADO por D16.** Son
`gymnasium/distance_min`, `gymnasium/distance_med` y `dmcontrol/distance_min`:
la tarea A no pierde nada del codificador (entre −9.7% y +6.0%) frente a hasta
+75244% en las que sí olvidan. Se reportan como **controles declarados**, así que
la rejilla efectiva son 6 de 9 celdas. Dos sorpresas: el criterio tiene que ser
**relativo** (en absoluto `dmcontrol/distance_min` no saldría), y
`gymnasium/distance_med` **no estaba en ninguna lista previa** — no pierde nada
en píxeles y tiene el RD más alto de su familia, que es F18 por el otro lado.

**5. La casilla que no se podía reportar ya se reporta (F23) · CERRADO por D15.**
`ug_mtm` en `minigrid/distance_max` sale como `+520.4 [17.72, 4364]!`: mediana,
rango y marca de cola pesada, con las cinco semillas listadas debajo. Se cuenta
como resultado sobre UG-MTM. Y no es la única: **22 de las 180 celdas** de
pf/rd/ft/wmf están marcadas por sesgo, que es lo que justifica la política.

**6. `d_trans` ordena mejor que la etiqueta, y aun así no ordena MiniGrid.** Con
medianas son **20.07 / 19.59 / 24.85**: el mínimo sale *por encima* del medio, no
solo solapado. Recupera el orden de RD en Gymnasium exactamente y separa el
mínimo de DMControl del resto. La frase para el paper es que el instrumento
medido del banco ordena mejor que su eje diseñado a mano (+0.53 frente a +0.05),
no que sea un sustituto acabado.

**7. Y el caso más limpio de F27 no necesita comparar familias.** En Gymnasium el
nivel máximo es **estrictamente el medio más perturbación**: los dos llevan
HalfCheetah de gravedad 9.8 a 4.0, y el máximo añade además masa ×3 y fricción
×0.5, sobre el mismo par de tareas. Produce **menos** olvido (59.59 frente a
85.91), y `d_trans` se pone del lado del resultado (24.71 frente a 30.86). Un
superconjunto estricto de una perturbación no puede quedar por debajo si el eje
ordena algo monótono. Salió de leer `configs/benchmark/gymnasium.yaml` en la
sesión 8; no depende de agregación, ni de escalas entre familias, ni de
`d_trans`.

### Los cuatro resultados anteriores, a presupuesto pequeño

> **Aviso de presupuesto.** Todo lo de este bloque es de **1000 pasos**, no de
> los 5000 que ejecuta R16, y de MiniGrid. **No cites estas cifras en el
> paper** — están superadas por las de arriba, salvo la prueba de escalado, que
> es sobre presupuestos por definición. Se conservan porque documentan cómo se
> llegó aquí. Detalle numérico en `runs.md` (R10, R11, R12).

**1. La tarea A se aprende.** Era la objeción capaz de tumbar el paper entero:
con 20 rollouts aleatorios y 1000 pasos, ¿había algo que olvidar? Sí:
reconstrucción de entrenamiento **931.8 → 8.2**, y **6.49 sobre datos
reservados** — error cuadrático sumado sobre 12288 píxeles, o sea 5.3e-04 por
píxel, RMSE ≈ 0.023 en `[0,1]`. NLL sobre `D_A` de 22.28 frente a 38.84 de un
modelo sin entrenar. Y la prueba de escalado (R11) dice dónde está la curva:

| Presupuesto | Recon. reservada | Mejora | Coste de las 225 corridas |
| --- | --- | --- | --- |
| 1× (1000 pasos, hoy) | 6.486 | — | ~19 h |
| 2× | 3.013 | −53.5% | ~31 h |
| 5× | 1.606 | −46.7% | **~3.3 días** |
| 10× | 1.367 | −14.9% | ~6.5 días |

El presupuesto actual está en la parte empinada: la tarea A se reconstruye
**4.7× peor** que a 10×. **5× es el punto dulce.**

**2. Las métricas no ven el olvido donde ocurre (F18).** Misma corrida, tres
métodos:

| Método | Recon. tarea A: tras A → tras B | PF |
| --- | --- | --- |
| `finetuning` | 6.49 → **725.27** (×112) | **−1.78** |
| `ewc` | 6.49 → **718.01** (×111) | +1.33 |
| `ug_mtm` | 7.66 → **7.66** (bit a bit) | +5.58 |

`finetuning` pierde la reconstrucción de la tarea A por un factor de 112 y **PF
sale negativo**: la métrica del paper dice que mejoró. No es un bug —aislar `M`
es el objetivo— pero el paper tiene que declarar el alcance y reportar las dos
escalas, porque si no lo hace él lo hará un revisor. De paso queda cuantificado
UG-MTM: congela el VAE, así que **no olvida porque no aprende** — su
reconstrucción al final de la tarea B es **533.19** frente a **19.66** de
`finetuning`.

**3. El Finding 4 se invierte (R12, 5 semillas).** Era el único de los cinco
Findings del paper que resistía el escrutinio (*"Replay is insufficient in
low-capacity settings"*, p<0.001):

| Métrica | Celda | `finetuning` | `replay_infinite` | Gana en | d_z |
| --- | --- | --- | --- | --- | --- |
| WMF | med | +10.02 ± 3.42 | **+4.94 ± 1.81** | 5/5 | −1.97 |
| WMF | max | +12.70 ± 4.63 | **+4.69 ± 1.99** | 5/5 | −2.27 |
| PF | med | −0.47 ± 1.17 | **−2.32 ± 0.69** | 5/5 | −3.07 |
| PF | max | +2.70 ± 2.31 | **−2.47 ± 0.43** | 5/5 | −2.27 |
| RD | med | +25.51 ± 8.06 | **+14.66 ± 4.03** | 5/5 | −1.66 |
| RD | max | +29.06 ± 9.44 | **+14.19 ± 4.63** | 5/5 | −2.17 |

Replay olvida menos en **las 10 comparaciones emparejadas**, en las tres
métricas, en los dos niveles. `p = 0.0625` en todas, que es el **suelo** de un
test exacto con n=5 — la máxima evidencia que 5 semillas pueden dar.

**De los cinco Findings del paper no sobrevive ninguno.** El consuelo es que la
dirección nueva es la esperable a priori (replay entrena sobre A+B): la
afirmación llamativa era la vieja, y era el artefacto. El mecanismo se ve en la
columna de píxeles — el codificador de replay no se degrada (5.84 → 5.86) y el
de finetuning se va ×127.

**4. EWC estaba siendo evaluado como un `finetuning` con pasos de más (R13).**
Con el Fisher corregido (I3), su PF cae de **+1.33 a +0.003** en la misma celda,
y sin perder ajuste a la tarea B (19.59 frente a 19.66 de `finetuning`). La
penalización era casi inerte porque `(E[∇])²` descarta la varianza del gradiente,
que es donde vive casi toda la señal. Lo que **no** cambia es la reconstrucción en
píxeles de la tarea A: sigue destruida, y F21 dice por qué.

### Las decisiones que bloqueaban están tomadas (sesión 7)

| ID | Decisión | Qué se decidió |
| --- | --- | --- |
| **D9** (P9) | La ceguera al codificador | Declararla como **alcance** y reportar píxeles al lado de PF/RD |
| **D10** (P7) | La agregación | **PF y RD por separado.** WMF baja a agregado heredado, con el reparto a la vista |
| **D11** (P8+P10) | `d_trans` y FT | Se pagan los entrenamientos de referencia: las dos se miden de verdad |
| **D12** (P4) | El presupuesto | **5×** — `n_train: 5000` en las tres familias |

Las cuatro están **implementadas**, no solo decididas. El coste real de D11
resultó ser **+90 entrenamientos**, no +45: la referencia es una *pareja* (un
modelo por entorno) por casilla y semilla, 45 × 2. Sobre los 450 de la ejecución
completa es un +20%.

### Lo que queda abierto

| ID | Qué es |
| --- | --- |
| **P11 / F21** | Si el Fisher de EWC debe estimarse sobre secuencias en vez de transiciones desde `h = 0` |
| **P1, P5, P6** | UG-MTM: umbral, peso de incertidumbre, inversión de la señal |

**P12 y P13 se cerraron en la sesión 8** (D15, D16). Y con ellos F22 y F23, que
pasan de bloquear a ser resultados reportables.

---

## 3. El paper

```
paper/
├── main.tex          raíz: preámbulo, título, \input de todo
├── abstract.tex      escrito (sesión 9)
├── intro.tex         §1 — escrita (sesión 9)
├── conclusion.tex    §7 — escrita (sesión 9)
├── appendix.tex      apéndices A–D (sesión 9)
├── figures/          axis_peak (Fig. 1) · forgetting_vs_distance (apéndice)
├── related.tex       §2 — escrita (sesión 9)
├── method.tex        §3 — escrita (sesión 9)
├── results.tex       §5 — escrita (sesión 8)
├── discussion.tex    §6 — escrita (sesión 8, retocada en la 9 por F28)
├── refs.bib          20 entradas, todas del PDF viejo o ya verificadas
└── tables/tab_*.tex  GENERADAS, no editar a mano
```

**Está todo escrito y verificado.** La pasada de números se hizo en la sesión
10: las 74 cifras que ninguna tabla respalda se comprobaron una a una contra
`results/`, `results-2x/` y `runs.md`. **Cuatro estaban mal** y están
corregidas (ver el commit «Correct four numbers the ten-seed pass had left
behind»). Lo único que queda es **compilarlo** la primera vez que haya
toolchain.

**Comprobación estructural sin toolchain de LaTeX** (llaves, `$` pareados,
entornos, `\input` sin fichero, `\ref` sin `\label`, `\cite` sin entrada):

```bash
cd C:/Users/Usuario/WorldModelsBenchmark && python _devlog/check-paper.py
```

**Las tablas se regeneran, no se tocan.** Cada `tab_*.tex` lleva una cabecera que
lo dice. Si cambia `results/` o la agregación:

```bash
cd C:/Users/Usuario/WorldModelsBenchmark/cf_worldmodels && python experiments/export_tables.py
```

Requiere `booktabs` en el preámbulo. **No hay toolchain de LaTeX en esta
máquina**: los dos `.tex` están comprobados estructuralmente —entornos, llaves,
`$` pareados, los seis `\input` resuelven, cero referencias colgantes— pero
**nunca se han compilado**. Compilarlos es lo primero que hará falta cuando haya
dónde.

### El argumento, en cinco frases

1. No existe un banco de pruebas que mida olvido catastrófico **a nivel del
   componente de transición `M`** de un world model. (Continual World mide
   políticas; Kessler et al. estudian DreamerV2 como sistema integrado.)
2. Construimos uno: 3 familias × 3 distancias × 5 métodos × 5 semillas.
3. **El eje de distancia que le pusimos no ordena el olvido** en ninguna de las
   tres familias (F27). Lo que ordena mejor es lo que se mide —`d_trans` y la
   dificultad de la tarea B— no la etiqueta.
4. **El olvido ocurre casi todo en el codificador**, que es justo donde las
   métricas al uso —incluidas las nuestras— no miran (F18), y EWC no lo protege
   porque su Fisher es exactamente cero ahí (F21).
5. Los métodos quedan **caracterizados**, no rankeados; el del autor incluido, y
   sale caracterizado, no ganador (D8: la tesis es el banco).

### Todas las piezas, y quién las escribió

| Pieza | Estado |
| --- | --- |
| `abstract.tex`, `intro.tex` | Sesión 9 |
| `related.tex` | Sesión 9. El hueco se hereda intacto; §2.4 y §2.5 sitúan F27 y F18 contra las prácticas que enmiendan |
| `method.tex` | Sesión 9. Tabla 1 (protocolo) y la de las nueve casillas se generan desde los `metrics.json` |
| `results.tex` | Sesión 8, ampliada en la 9 (10 semillas, 2×, k=4) y corregida en la 10 |
| `discussion.tex` | Sesión 8, retocada en la 9 por F28 |
| `conclusion.tex`, `appendix.tex` | Sesión 9 |
| `main.tex`, `refs.bib` | Sesión 9. 20 entradas, todas del PDF viejo o verificadas |
| Figuras | `axis_peak` (Fig. 1) y `forgetting_vs_distance` (apéndice) |

**Lo único no hecho es compilar.** No hay LaTeX en esta máquina.

### La versión de 8 páginas (`paper/main_workshop.tex`)

Para **CL4FMAgents @ NeurIPS 2026** (deadline 29 ago 2026 AoE, 8 páginas sin
referencias, no archival). Es un **documento aparte**, no una variante del
largo: 12.306 palabras no sobreviven como subconjunto de 3.600, así que está
reescrito. Lo que sí es idéntico son las cifras y las tablas — hace `\input` de
las mismas `../tables/`.

Qué se cayó, y dónde va el lector: trabajo relacionado (a un párrafo de §1) ·
las ecuaciones de las métricas (descritas en palabras) · las tablas por método
de PF/RD/FT (al paper largo) · amenazas a la validez (a §4.4) · el relato de la
versión superada (a un párrafo de §1).

**Compila y cabe: 8 páginas con bibliografía**, y el límite de CL4FMAgents
excluye las referencias. **No hay nada que recortar ni que añadir** — ver D23.
De la primera compilación, que dio 8 páginas *sin* bibliografía, salieron dos
recortes:

- **Fuera §3.6, la secuencia k=4.** Ganancia **neta cero**: las ~50 palabras que
  libera se van en declarar la limitación que ese resultado cubría — sin él, el
  paper solo enseña un cambio de tarea, y eso hay que decirlo. Lo único que
  ahorra es el título de subsección. Se conserva el crédito: la limitación
  apunta a que el resultado de k=4 existe y está reportado en el paper largo.
- **Fuera la tabla del codificador**, sustituida por la frase que sostenía. Un
  tercio de página, y sus dos números (811 y 800) ya estaban en la prosa; se
  añadió el rango que solo daba la tabla (1.00 a 848) y un puntero a los
  resultados publicados. **Esta sí mueve: −0,4 páginas.**

**La tabla del codificador volvió a entrar** al confirmarse el tamaño real: el
recorte se había hecho sobre una lectura equivocada. **La sección de k=4 no
vuelve** (D23, decisión del autor): devolverla obliga a revertir también la
limitación escrita para cubrirla, el margen es de décimas, y el resultado está
completo en el paper largo.

Quedan la figura del pico y **tres tablas**: eje, predictores y codificador.

### Overleaf: la lección, ya cobrada

**El documento corto vive en `paper/main_workshop.tex`, junto a `refs.bib`,
`tables/` y `figures/`. No lo devuelvas a un subdirectorio.**

Empezó en `paper/workshop/` porque quedaba más ordenado, y esa decisión costó
**tres fallos de compilación seguidos**, todos síntomas de lo mismo: Overleaf
compila **desde la raíz del proyecto**, no desde la carpeta del fichero
principal.

1. **Rutas.** Los `input{../tables/…}` salían fuera del proyecto.
2. **Documento principal.** Con dos `main.tex`, Overleaf no elegía el correcto
   por su cuenta.
3. **La bibliografía no aparecía.** Y esta no se arreglaba con un prefijo: el
   nombre del `.bib` llega a **bibtex** por el `.aux`, y bibtex no expande
   macros de LaTeX ni abre rutas que salgan del directorio de compilación.

Los dos primeros los parcheé con un prefijo autodetectado; el tercero demostró
que el parche no era la solución. Moviendo el fichero desaparecen los tres:
todas las rutas son planas y relativas al directorio desde el que compilan
tanto Overleaf como una corrida local. **Compila, y las referencias salen.**

En Overleaf: subir `paper/` y fijar Menu → Settings → Main document →
`main_workshop.tex`.

### Trampas al escribir (todas se han pisado ya una vez)

- **Ninguna cifra a mano.** `export_tables.py` para las tablas,
  `summarize_results.py` para comprobar cualquier número suelto.
  `paper-vs-code.md` existe porque la versión anterior se desincronizó de sus
  propias corridas.
- **`check-numbers.py` clasifica, no verifica.** Marca TABLE / PROTOCOL /
  CHECK. Hoy: 173 / 9 / **74**. Las 74 se comprobaron a mano en la sesión 10 y
  cuatro estaban mal — o sea que la herramienta encuentra dónde mirar, no si
  está bien.
- **Cualquier número de `runs.md` anterior a R13 es de otro presupuesto** (1000
  pasos). Incluye la convergencia que circulaba como «5.3e-04 por píxel».
  Escalado real: 6.49 → 3.01 → 1.61 → 1.37 en 1000/2000/5000/10000.
- **Al comparar presupuestos, fija también las semillas.** `results-2x/` tiene
  cinco y `results/` diez en las celdas que discriminan. Comparar la mediana de
  cinco contra la de diez fue una de las cuatro correcciones de la sesión 10.
- **`d_trans` no ordena MiniGrid** (22.35 / 20.64 / 25.19: el mínimo por encima
  del medio). No escribir que ordena los niveles dentro de una familia.
- **Los predictores no se ordenan entre sí.** A n=10 se intercambian respecto a
  n=9: dificultad de B 0.58 → 0.43, `d_trans` 0.53 → 0.58. El paper dice que
  los dos baten a la etiqueta (+0.00) y que n=9 no los resuelve. No lo
  conviertas en un ranking.
- **Replay contra finetuning, a diez semillas, no es «cinco de seis».** Es:
  cuatro celdas con replay por debajo a p = 0.0020, una al revés
  (`gymnasium/distance_max`, p = 0.0059, d_z +1.14) y una sin efecto
  (`dmcontrol/distance_max`, p = 0.68).
- **La rejilla efectiva son 6 celdas, no 9** (D16), y **17 de 180** están
  marcadas por sesgo, no 22.
- **El suelo de p ya no es 0.0625.** A diez semillas es 0.002; los tres
  controles siguen a cinco, donde sí es 0.0625.
- **La lectura de F27 falla en DMControl** — la dificultad de B sube de min a
  max mientras RD baja tras el nivel medio. Declarado así en `results.tex` §5.3
  y en `discussion.tex` §6.2; no lo asciendas a mecanismo demostrado.

### Una decisión editorial que dejé tomada y es reversible

`discussion.tex` §6.7 cuenta sin rodeos que **no sobrevive ninguno de los cinco
Findings del paper anterior**, cuáles fueron los cinco defectos de instrumento
que los produjeron, y que el que se invirtió era el más contraintuitivo — que es
el patrón esperable cuando una medición está rota. Creo que suma (la narrativa de
reproducibilidad es la que sostiene el resultado negativo), pero es una decisión
del autor: quitarlo es borrar una subsección.

---

## 4. Qué hacer en la sesión siguiente

**Por orden, y con fecha encima.**

1. **HECHO — enviado a CL4FMAgents el 14 ago 2026.**
   `https://openreview.net/group?id=NeurIPS.cc/2026/Workshop/CL4FMAgents`
   Ocho páginas con bibliografía incluida, y el límite excluye referencias, así
   que cumple con margen. No archival: **no quema el paper para CoLLAs 2027**.
   Notificación el **29 sep 2026**.

2. **Mirar si CWM publicó bases** (deadline 30 ago), y decidir. Encaja por
   título mejor que ninguno; al cierre de la sesión 12 su web no daba páginas
   ni formato. Es la única fecha viva ahora mismo, y se pasa en dos semanas.
   Enviar a los dos sitios es legítimo mientras los dos sean no archival —
   confirmar que CWM lo es antes de mandar nada.

4. **Después: CoLLAs 2027**, que es archival y la sede natural del tema.
   Deadline sin anunciar; por los años anteriores, finales de febrero.

5. **Volver a pasar los números si se toca `results/`.** El orden importa:
   regenerar tablas primero, clasificar después, comprobar las CHECK a mano.

   ```bash
   cd C:/Users/Usuario/WorldModelsBenchmark/cf_worldmodels && python experiments/export_tables.py
   ```

   ```bash
   cd C:/Users/Usuario/WorldModelsBenchmark && python _devlog/check-numbers.py
   ```

### Se puede hacer sin decisiones, y ninguna es necesaria

4. **F28 — el arreglo de `d_trans`** (puntúa el modelo B en la base latente de
   A). **Recomendación: no hacerlo.** Ya está declarado como fallido en el
   paper, y arreglar un instrumento secundario no mueve la valoración.
5. **Prueba de escalado en `dmcontrol`.** La comprobación de que 5000 pasos
   bastan en la familia visualmente más difícil. Una corrida.
6. **Más semillas en los tres controles.** Siguen a cinco, donde el suelo de p
   es 0.0625. Son celdas donde nadie olvida, así que compra poco.
7. **Una figura de calidad-en-A.** El apéndice lleva la tabla; la figura no
   existe. No necesita entrenar nada.

### Lo primero para una sede archival (CoLLAs), y solo eso

8. **F30 — contrabalancear el orden de la secuencia k=4.** El pico en la tercera
   tarea se midió con **un solo orden**, así que «posición 3» y «FourRooms»
   (la tarea más difícil, 47.32) son la misma columna: §5.7 no distingue entre
   el olvido siguiendo a la demanda de la tarea y el olvido dependiendo del
   número de cambios. Se arregla con una config nueva por orden y ningún cambio
   de código; el orden que discrimina es el que pone FourRooms en la **segunda**
   posición, porque solo ahí el pico baja dos veces. 25 corridas por orden, en
   la familia barata. Detalle y predicción escrita en `findings.md` (F30).
   **No afecta a lo enviado al taller** — k=4 no está en las 8 páginas (D23).

### Después, y solo si el paper vuelve con revisiones

8. **F19** — si las métricas de UG-MTM se promedian sobre varias evaluaciones.
   Es el único método cuyas cifras llevan ruido de medición.
9. **P11 / F21** — si el Fisher de EWC debe estimarse sobre secuencias.
10. **P5, P6, P1** — UG-MTM: peso del término de incertidumbre, inversión de la
    señal a distancia media, mecanismo del umbral.
11. **Validación cruzada:** que `d_trans` ordene Gymnasium igual que `d_param`.

---

## 5. Entorno y comandos

```
Repo:        C:\Users\Usuario\WorldModelsBenchmark
Código:      C:\Users\Usuario\WorldModelsBenchmark\cf_worldmodels
Bitácora:    C:\Users\Usuario\WorldModelsBenchmark\_devlog   (en .gitignore)
```

- Windows 11. PowerShell y Git Bash, cada uno con su sintaxis.
- Python 3.11.6, torch **2.3.1+cu121** (`requirements.txt` fija 2.1.2), CUDA
  disponible. `scipy` está instalado pero **no declarado**: no usarlo en código
  del repo sin añadirlo (sería repetir F12).
- **Casi todo se ejecuta desde `cf_worldmodels/`.** El cwd de la herramienta Bash
  se resetea entre llamadas — hacer `cd` explícito.
- `import wandb` falla con `AttributeError`. Todo se ejecuta con `--no_wandb`.
- Los tres simuladores funcionan, pero **cada familia necesita su propio
  proceso** (F24): dm_control no consigue contexto OpenGL en un proceso donde
  MuJoCo ya tiene uno. El runner lo hace solo desde la sesión 7 — lanza un
  subproceso por familia. La nota que decía que I20 "no afecta al runner" quedó
  desmentida por R16, que murió en esa frontera tras 30 horas.
- **Ritmo medido (sesión 7, en aislamiento).** Entrenamiento: **46 s por 1000
  pasos**. Recogida de rollouts: **0.5 s/episodio** en MiniGrid, **2.7 s** en
  Gymnasium (renderiza cada paso). A 5000 pasos, una celda son 2 entrenamientos
  (~8 min) y una pareja de referencia otros 2. La recogida se hace **una vez por
  `(familia, distancia, semilla)`** y la comparten los cinco métodos y la
  referencia: 60 episodios, entre 30 s y 2.7 min según la familia.
  Estimación: ~12 celdas-equivalentes por casilla-semilla → **~2 días** las 225
  corridas con sus referencias.
- **No ejecutes el benchmark a la vez que la suite de integración.** La corrida
  de gymnasium de R15 consumió 4.8 min de CPU en ~40 min de reloj mientras
  `pytest` construía entornos MuJoCo y dm_control en paralelo: los contextos de
  render se serializan (es el vecindario de I20).

```bash
cd C:/Users/Usuario/WorldModelsBenchmark/cf_worldmodels
```

```bash
python -m pytest
```

```bash
python -m pytest -m "not integration and not slow"
```

La ejecución completa. Salta lo ya hecho, así que es seguro interrumpir. Incluye
las parejas de referencia (D11), que se entrenan una vez por casilla y semilla
antes de los cinco métodos:

```bash
python experiments/run_full_benchmark.py
```

Sin las referencias — `ft` y `d_trans` se guardan como `null`, nunca como 0:

```bash
python experiments/run_full_benchmark.py --skip-reference
```

El protocolo sale del config (I1). Ver el efectivo y el plan sin entrenar nada:

```bash
python experiments/run_full_benchmark.py --dry-run
```

Una celda concreta, con override explícito del presupuesto:

```bash
python experiments/run_full_benchmark.py --families minigrid --distances distance_med --methods finetuning --seeds 999 --steps 2000
```

Prueba de escalado de la tarea A (F17): una corrida evaluada por el camino.

```bash
python experiments/convergence_A.py --family minigrid --multipliers 1 2 5 10
```

Todas las tablas salen de aquí, nunca a mano. Avisa si los resultados no
comparten protocolo, y `--compare` da la comparación emparejada por semilla:

```bash
python experiments/summarize_results.py --compare replay_infinite finetuning
```

```bash
python experiments/plot_final.py
```

El caso de dm_control que se salta en la suite completa, aislado:

```bash
python -m pytest tests/test_seeding.py -k dmcontrol
```

---

## 6. Estructura del código

```
cf_worldmodels/
├── configs/
│   ├── benchmark/{minigrid,gymnasium,dmcontrol}.yaml   # 3 distancias + protocolo
│   ├── benchmark/minigrid_sequence.yaml                # k=4 (D22)
│   └── models/{rssm_baseline,ug_mtm}.yaml
├── src/
│   ├── envs/          base_env, minigrid_env, gymnasium_env, dmcontrol_env
│   │                  (normalizan a (64,64,3) float32 en [0,1]; todos con seed())
│   ├── models/        vae.py (ConvVAE), rssm.py (RSSM + BaseWorldModel), ug_mtm.py
│   ├── baselines/     finetuning, replay, ewc, progressive_nets
│   ├── benchmark/     metrics.py (PF/RD/WMF/FT + diag_gaussian_kl), distances.py,
│   │                  protocol.py (dataset de eval + evaluate_reconstruction)
│   └── utils/         buffer.py, checkpointing.py, logging_utils.py,
│                      seeding.py (set_seed + preserve_rng_state)
├── experiments/       run_full_benchmark.py   <- punto de entrada, k=2
│                      run_sequence.py         <- k>2 (D22)
│                      summarize_results.py    <- todas las tablas salen de aquí
│                      summarize_sequence.py   <- ...las de la secuencia
│                      export_tables.py        <- ...y las del paper, en LaTeX
│                      plot_axis.py            <- Fig. 1 (el pico)
│                      plot_final.py           <- PF y RD por familia (apéndice)
│                      convergence_A.py        <- prueba de escalado de la tarea A
│                      run_benchmark.py, train_baseline.py, train_ug_mtm.py
├── tests/             434 tests en 19 archivos + conftest
├── results/           375 celdas + 75 parejas en _reference/. Diez semillas en
│                      las seis que discriminan, cinco en los tres controles
├── results-2x/        el sondeo de doble presupuesto: 10 celdas, 10 parejas
└── results-seq/       la secuencia k=4: 25 corridas

paper/                 main.tex + 8 secciones + refs.bib
├── figures/           axis_peak (Fig. 1) · forgetting_vs_distance (apéndice)
└── tables/            tab_*.tex  <- GENERADAS por export_tables.py
```

Repo versionado: **~1.85 MB**, de los que **0.8 MB son los `metrics.json`**. El
código, el paper y las tablas son el resto.

**Tres scripts producen todo lo publicable y ninguno entrena nada:**
`summarize_results.py` y `summarize_sequence.py` (consola) y `export_tables.py`
(LaTeX del paper). El tercero importa a los dos primeros, así que solo pueden
discrepar si el código discrepa consigo mismo.

### Tres invariantes que conviene no romper

1. **El protocolo vive en el config, no en el código** (I1). `resolve_protocol()`
   lee, castea y valida; un campo que falte es un error, no un valor por defecto.
   Cada `metrics.json` guarda su bloque `protocol` entero.
2. **El runner se niega a mezclar presupuestos.** Salta celdas cacheadas (por eso
   es reanudable) pero rechaza las producidas con otro protocolo, para que no se
   promedien dos presupuestos en la misma casilla de la tabla.
3. **Medir no puede cambiar el resultado.** Toda instrumentación va dentro de
   `seeding.preserve_rng_state()`. No es paranoia: la puerta de UG-MTM mantiene
   MC-dropout activo en evaluación **por diseño**, así que cada `compute_nll`
   consume el flujo aleatorio. Verificado: las 12 cifras de R9 se reproducen con
   delta < 5e-10 tras reescribir el runner entero.

### Qué guarda cada `metrics.json`

Métricas de olvido (`pf`, `rd`, `wmf`, `pis`) · transferencia (`ft`, con sus dos
brazos `heldout_reconstruction_B_from_scratch` y `..._B_after_task_B`) ·
`d_trans` · `task_A_fit_gain` (lo que antes se llamaba FT) · las tres NLL sobre
`D_A` (`nll_A_after_task_A`, `..._after_task_B`, `nll_A_random_init`, así PF queda
descomponible) · calidad en píxeles (`heldout_reconstruction_A_after_task_A`,
`..._A_after_task_B`, `..._B_after_task_B`, `initial/final_reconstruction_loss_A`
y `_B`, dos curvas de 20 puntos) · salud (`n_nan_steps_A/B`,
`n_update_steps_A/B`) · procedencia (`method`, `family`, `distance`, `seed`,
`protocol` completo).

`results/_reference/<familia>_<distancia>_<semilla>.json` guarda aparte lo que
produce la pareja entrenada desde cero: `d_trans`, las dos reconstrucciones
reservadas desde cero, y su protocolo.

---

## 7. Tests

**434 tests, 432 pasan** (399 fuera de `integration` y `slow`). No existía
ninguno antes de la sesión 4. **Dos se saltan en la suite completa** porque el
contexto de render de dm_control ya está tomado cuando les toca — uno por I20
(`test_seeding.py:161`) y otro por F24 (`test_envs.py:315`); los dos pasan
aislados.

| Archivo | Cubre |
| --- | --- |
| `conftest.py` | Fixtures, dimensiones pequeñas para velocidad |
| `test_vae.py` | ConvVAE, recorte 80→64, **regresión de colapso del posterior** |
| `test_rssm.py` | Contrato `BaseWorldModel`, estado GRU, gradientes |
| `test_ug_mtm.py` | Expertos, MC-dropout, ThresholdNet, gating, congelación |
| `test_baselines.py` | 4 baselines + **Fisher por muestra (I3) y sus dos ceros (F21)** + columnas progresivas |
| `test_metrics.py` | PF/RD/WMF + transferencia hacia delante + `task_A_fit_gain` + **KL contra la referencia de torch** |
| `test_distances.py` | `d_param`; `d_trans` contra la KL de torch |
| `test_protocol.py` | Dataset de evaluación + `evaluate_reconstruction` (escala, chunking, determinismo, que entrenar la baja) |
| `test_run_full_benchmark.py` | Protocolo desde el config, overrides, validación, capacidad emparejada, rechazo de cachés de otro protocolo, **caché de la pareja de referencia**, y que no vuelvan las constantes de módulo |
| `test_summarize_results.py` | Agregación: protocolos mezclados, emparejado por semilla, estadística, **mediana y marca de sesgo (D15), la Spearman propia, y el criterio de celda de control (D16)** |
| `test_export_tables.py` | El LaTeX del paper: cabecera de "generado", pico en negrita, daga de control, asterisco de sesgo, celda ausente como `--` |
| `test_seeding.py` | `set_seed`, cuDNN, **entrenar dos veces y comparar pesos**, rollouts reproducibles en las 3 familias, `preserve_rng_state` |
| `test_plot_final.py` | Qué se dibuja (PF/RD, nunca WMF) y qué va en el eje X |
| `test_plot_axis.py` | La figura del pico: que marque el nivel donde RD es máximo |
| `test_run_sequence.py` | El runner de k>2: etapas, caché por etapa, y que una secuencia no se mezcle con una pareja |
| `test_summarize_sequence.py` | Agregación de la secuencia: curva de retención, etapa del pico, dificultad por etapa |
| `test_buffer.py`, `test_checkpointing.py`, `test_configs.py` | Utilidades |
| `test_envs.py` | Integración contra simuladores reales |

Marcadores: `integration` (30), `slow` (5).

**Nota estadística que vive en `summarize_results.py`.** Las comparaciones entre
métodos son **emparejadas por semilla** (dos métodos con la misma semilla
comparten datos e inicialización) y se reportan con una **permutación exacta**,
no una t. Con n=5 la p exacta más pequeña posible es 2/2⁵ = **0.0625**: ninguna
comparación de 5 semillas puede dar p < 0.05 sin apoyarse enteramente en el
supuesto de normalidad. El paper anterior reclamaba p < 0.001 con n=5.

---

## 8. Los 30 problemas encontrados (F0–F29)

Detalle completo con evidencia numérica en `findings.md`. Resumen:

### Corregidos

**F0 — Posterior del VAE colapsado. LA CAUSA RAÍZ.**
`F.mse_loss(..., reduction="mean")` promediaba sobre 12288 píxeles mientras la KL
se sumaba sobre las dims latentes: reconstrucción infraponderada ~12288×. Medido:
**0 de 32 dims activas**, std por dim 3.5e-05. Toda observación se codificaba al
mismo vector. El modelo de transición nunca vio el entorno.
**Invalida las 225 corridas, para los cinco métodos.**

**F1 — Evaluación sobre ruido gaussiano.** `eval_ds = torch.randn(...)`. PF, RD,
FT y el Fisher de EWC se calculaban sobre eso. Corregido con
`protocol.build_latent_eval_dataset()`.

**F2 — `compute_nll` puntuaba contra `z_t` en vez de `z_{t+1}`.** Medía cuánto
*mueve* la transición el estado, no cuánto lo *acierta*.

**F4 — `ExpertPool` destruía la reproducibilidad.** `torch.manual_seed(torch.seed())`
al construirse. Corregido con `Generator` local; verificado bit a bit.

**F5 — Progressive Nets rompía con 3+ columnas.** No afectaba a los resultados
(k=2) pero el protocolo se presenta como general.

**F7 — RD usaba acciones fuera de distribución.** `randn*0.1` sobre acciones
one-hot.

**F9 — Las cinco ablaciones eran inertes.** `run_ablations.py` construía los
overrides y nunca los pasaba. Script archivado y eliminado.

**F12 — `Pillow` no declarada** y faltaba el `--extra-index-url` de cu121.

**F13 — La KL de RD y `d_trans` no era una KL** (s5). `log_sigma` es log-varianza
en todo el código, pero `compute_rd` y `compute_d_trans` lo leían como
log-desviación, y la fórmula manual llevaba un 0.5 de más en el término
logarítmico: no coincidía con ninguna de las dos lecturas. Corregido con
`metrics.diag_gaussian_kl` + `torch.distributions.kl_divergence`.
**RD venía inflada ~12×.**

**F16 — El pipeline no reproducía** (s5, era I5). La causa **no era cuDNN**: las
tres familias entrenaban sobre datos distintos en cada corrida porque
`np.random.seed` no alcanza el RNG del entorno de Gymnasium, el del `action_space`
ni el `RandomState` de dm_control. Corregido con `set_seed()` + `BaseEnv.seed()`.
Verificado bit a bit entre procesos.

**F17 — No se registraba si el modelo aprendió la tarea A** (s6). Se guardaba la
reconstrucción al *inicio* de A y al *final* de B; `final_A` se calculaba y se
tiraba. Ahora se guarda todo lo de §6, más la prueba de escalado
(`convergence_A.py`). **La tarea A se aprende** — ver §2.

### Abiertos — decisiones de diseño, no bugs

**F18 / P9 — PF y RD son ciegas al olvido del codificador · declarado como
alcance (D9).** `compute_nll` no llama nunca a `encode`: opera sobre latentes
congelados. `finetuning` degrada la reconstrucción de la tarea A ×112 mientras PF
sale negativo. No se "arregla" —aislar `M` es el objetivo— pero el paper lo
declara y reporta las dos escalas.

**~~F22 / P13~~ — Hay celdas que no producen olvido · CERRADO (D16, s8).** Son
tres: `gymnasium/distance_min`, `gymnasium/distance_med` y
`dmcontrol/distance_min`. Se reportan como controles declarados; rejilla efectiva
6 de 9. El criterio es **relativo** a lo que el modelo tenía, con umbral 10%, y
no consulta RD: `gymnasium/distance_med` no pierde nada en píxeles y tiene el RD
más alto de su familia.

**~~F23 / P12~~ — RD de UG-MTM estalla en una celda · CERRADO (D15, s8).**
`minigrid/distance_max`, por semilla: 17.7, 40.0, 520, 574, **4364**. No era el
ruido de MC-dropout de F19 ni el rollout divergiendo, sino **colapso de varianza
en el modelo post-B** (R17). Se reporta como `+520.4 [17.72, 4364]!` y se cuenta
como resultado sobre el método.

**F21 — El Fisher de EWC no cubre ni el codificador ni la vía recurrente.** Salió
de arreglar I3. Es **exactamente cero** sobre los 20 parámetros del VAE (no entran
en `log P(z'|z,a)`) y sobre `gru.weight_hh` (el conjunto del Fisher son
transiciones desde `h = 0`). Explica el enigma de R10: EWC no protege el
codificador porque no puede. Los dos ceros están fijados con tests.

**~~F20 / P10~~ — FT no medía transferencia · CORREGIDO (s7).** Ahora es
`recon_B(desde cero) − recon_B(preentrenado)`, en píxeles, con una referencia
entrenada solo en B. Primera medición: +54.69 para `finetuning` frente a −516.10
para UG-MTM, donde la métrica vieja daba delta 0.000 por construcción.

**F14 / P7 — RD domina el WMF · resuelto por decisión (D10).** 78–97% del agregado. Estructural: sale casi igual
antes y después de sembrar los entornos. La explosión de RD (13708 en UG-MTM) era
artefacto de F13 y desapareció; queda un outlier (UG-MTM a distancia máxima,
`RD = 167` frente a 14–32).

**~~F15 / P8~~ — `d_trans` (Ec. 9) no la calculaba nadie · CORREGIDO (s7).** Se
mide como la define el paper: una pareja de RSSM planos por casilla y semilla, uno
por entorno, y `KL(P_A || P_B)` sobre transiciones reservadas de A. Las 9 casillas
tienen distancia numérica.

**F19 — Las métricas de UG-MTM son estocásticas en evaluación.** La puerta
mantiene MC-dropout activo en `eval()` por diseño, así que dos evaluaciones del
mismo modelo sobre el mismo `D_A` difieren (1.4e-02 sobre un PF de 5.58). Es el
único método cuyas cifras llevan ruido de medición, comparado contra cuatro
estimadores deterministas. Declararlo o promediar. **No afecta a la
reproducibilidad**: el ruido es entre evaluaciones, no entre ejecuciones.

**~~F6 / P2~~ — PIS nunca se calcula · RETIRADA (D18, s8).** Se anunció y nunca
se implementó: medirla exige un controlador entrenado en la imaginación del
modelo, que no existe en el repo. La suite es PF, RD y FT, y `pis` se guarda
como `null`, no como `0.0`.

**~~F8 / P3~~ — DMControl `distance_min` comparaba una tarea consigo misma ·
CORREGIDO (s7).** Ahora es `cheetah/run` con gravedad 9.81 → 7.0, el mismo cambio
físico que el `distance_min` de Gymnasium. El wrapper rechaza las claves de física
que no implementa, que es cómo `lateral_wind: true` sobrevivió sin hacer nada.

**F3 / P1 — El gating de UG-MTM.** Con el VAE colapsado, `tau` y `u_t` no eran
conmensurables y el experto nuevo recibía puerta 1.9e-3. Tras F0 las escalas se
acercan. Discriminación tarea A vs B (AUC, 0.5 = nada):

| Señal | med (pre → post F0) | max (pre → post) |
| --- | --- | --- |
| UncertaintyHead MC-dropout | 0.519 → **0.294** | 0.508 → **0.864** |
| MC-dropout sobre la transición | 0.481 → 0.283 | 0.515 → 0.780 |
| Error de predicción a un paso | 0.769 → 0.693 | 0.533 → 0.862 |
| Distancia latente media L2 | 1.4e-03 → **2.89** | 3.7e-04 → **5.24** |

La premisa es viable a distancia grande (0.864) y **se invierte** a distancia
media (0.294). Resultado reportable, no necesariamente un fallo.

**F28 — `d_trans` puntúa el modelo B en la base latente de A · DECLARADO (s9).**
Los dos RSSM de referencia se entrenan por separado, así que `P_B(z'|z,a)` se
evalúa sobre latentes del codificador de `model_A`. Es la objeción de F20
aplicada a la otra métrica: parte de lo que mide es desalineamiento de bases.
No toca F27 (el negativo sobre el eje etiquetado no usa `d_trans`), sí acota la
recomendación de §6.1. Declarado en `method.tex` §3.3 y en §6.1/§6.6.

**F29 — `d_trans` no separa dentro de una familia, y se mueve con el
presupuesto · DECLARADO (s9).** Las medianas de Gymnasium abarcan 6 unidades y
cada celda abarca 15–19 entre semillas; al doblar el presupuesto, med y max se
intercambian (30.86/24.71 → 57.30/71.51) y la magnitud casi se dobla. Buena
parte del +0.53 lo carga la separación *entre* familias, que es justo lo que
F28 dice que no está legitimado a comparar. Ver R18.

**F11 — El escalado de gradiente usa solo el último timestep.**

**F10 — Los configs no coincidían con lo ejecutado.** Resuelto en su mecánica por
I1; ver §10.

---

## 9. Mejoras de código (I1–I21)

Detalle en `improvements.md`. Estado de las que pueden morder al re-ejecutar:

- ~~**I1**~~ — Hiperparámetros hardcodeados. **CORREGIDO (s6)**: el bloque
  `protocol:` del config es la única fuente de verdad. Resuelve también F10 y la
  parte mecánica de P4.
- ~~**I2**~~ — Los pasos con NaN se saltaban en silencio. **CORREGIDO (s6)**: se
  cuentan, se guardan y se avisa. Medido: 0 NaN en MiniGrid.
- ~~**I3**~~ — El Fisher de EWC era `(E[∇])²` en vez de `E[(∇)²]`. **CORREGIDO
  (s7)**: gradientes por muestra. El PF de EWC pasó de +1.33 a **+0.003** sin
  perder ajuste a la tarea B — la penalización estaba siendo casi inerte. Ver R13
  y F21.
- ~~**I4**~~ — `ThresholdNet._ptr` fuera del `state_dict`. **CORREGIDO (s6)**:
  ahora es un buffer registrado. Verificado por re-ejecución que no mueve ningún
  resultado (delta < 5e-11).
- ~~**I5**~~ — Promovido a **F16**, ya corregido.
- ~~**I21**~~ — La comprobación de consistencia del directorio de resultados
  estaba dentro de una sola tabla, y las otras siete habrían promediado la
  mezcla en silencio. **CORREGIDO (s9)**: `check_runs_consistent` junto a
  `load_runs`; el exportador levanta, el resumen avisa.

Robustez: I6 (errores opacos), I7 (`np.int64` rompe OmegaConf), I8 (`policy` sin
validar), I9 (`log_metrics` no escribe en consola), I10 (`build_latent_eval_dataset`
codifica en un solo batch), **I20** (contexto GLFW, ver §5).

Higiene: I11 (código muerto), I12 (sin packaging), I13 (tres nombres de figura),
I14 (directorio huérfano), I15 (`docs/` eran instrucciones para agente), I16 (sin
CI), I17 (`list.pop(0)`), I18 (herencia múltiple frágil), I19 (`wandb` roto).

---

## 10. El paper anterior, y el protocolo sin ambigüedad

Valoración honesta de viabilidad, qué se salva y qué se tira: **`paper-plan.md`**.
Discrepancias numéricas con el PDF: `paper-vs-code.md`.

Resumen de una línea: **el hueco y la infraestructura se sostienen; ninguno de
los cinco Findings sobrevive; qué afirma el benchmark está decidido (D9–D22) y
ejecutado, las 375 celdas están hechas, y el paper entero está escrito alrededor
de F27 y verificado cifra a cifra** (§3). Objetivo realista: workshop, aunque esa
valoración es anterior a que existieran F27 y el resultado del codificador y
merece revisarse.

Valoración actualizada con los datos en la mano: **§0bis de `paper-plan.md`**.

### El protocolo, ya sin ambigüedad

Circulaban tres conjuntos de valores. Ahora el config declara lo que se ejecuta y
cada resultado guarda el suyo, así que **la Tabla 1 se genera desde los
`metrics.json`**. Para el registro, lo que decía cada fuente:

| Parámetro | Paper | YAML (antes) | YAML ahora (D12) |
| --- | --- | --- | --- |
| Rollouts por tarea | 1000 | 1000 | **20** |
| Pasos de gradiente | 3000 | 50000 | **5000** |
| Batch size | 16 | 32 | **8** |
| Longitud de secuencia | 10 | 50 | **5** |

Los pasos de gradiente subieron a 5000 en la sesión 7 (D12, sobre el dato de
R11), y R16 es la primera corrida a ese presupuesto: **cualquier número de
`runs.md` anterior a R13 es de otro presupuesto** y no se mezcla con los de
`results/` — el runner se niega, y `summarize_results.py` avisa.

---

## 11. Decisiones

Tomadas (D1–D8): licencia MIT · eliminar los 7 runners redundantes · excluir
checkpoints de git (713 MB → <1 MB) · no arreglar resultados en silencio ·
codificar `D_i` una sola vez con el modelo post-tarea-A · `mc_dropout_T_eval`
separado · rehacer el paper en vez de parchearlo · **la tesis es el benchmark**.

Sesión 7 (D9–D14): **el alcance se declara** y se reportan las dos escalas ·
**PF y RD por separado**, sin escalar único · **se pagan los entrenamientos de
referencia** y `d_trans` y FT se miden de verdad · **el presupuesto es 5×** ·
**D13**, el nivel máximo de dmcontrol deja de ser `reacher` (inejecutable, 2
actuadores contra 6) · **D14**, y acaba siendo cambio de cuerpo *más*
perturbación física, tras descubrirse que `walker/stand` con la física de
cheetah era el mismo entorno (F26).

Sesión 8 (D15–D18): **cada celda se resume por mediana y rango**, no por media ±
σ, y las colas se marcan en vez de promediarse (cierra P12) · **tres celdas se
declaran controles** con un criterio relativo, y son tres, no las dos que se
asumían (cierra P13) · **el eje de F27 se agrega sobre los cuatro métodos RSSM**
y la regla se imprime, en vez de estar tomada de hecho · **PIS se retira de la
suite** y el banco reporta PF, RD y FT (cierra P2).

Sesión 9 (D19–D20): **el paper se anuncia como hallazgo, no como banco** — el
banco es el instrumento, y el título tiene que cambiar porque todavía promete
UG-MTM · **se compra cómputo antes de terminar de escribir, eligiendo la
pregunta y no el multiplicador**: subir el presupuesto global está descartado
por R11 (15% de mejora al doble de coste), y el orden es sondeo 2× → 10
semillas → k>2 → F28, con la regla de parada escrita antes de mirar.

Pendientes:

| ID | Decisión |
| --- | --- |
| **P11** | Si el Fisher de EWC debe estimarse sobre secuencias (F21) |
| **P1** | Mecanismo del umbral en UG-MTM (menos urgente tras F0) |
| **P5** | Peso del término de incertidumbre en UG-MTM |
| **P6** | Inversión de la señal a distancia media |

Detalle en `decisions.md`.

---

## 12. Estado del repositorio

**Una sola rama: `main`**, en local y en `origin`, árbol limpio. La rama
`session-7-benchmark-decisions` llevaba las sesiones 7 a 11; el 12 ago se
fusionó por fast-forward (31 commits, 0 divergencia, sin merge commit) y se
borró en los dos lados. **El repositorio es público** desde esa misma fecha.

Los ocho commits más recientes:

```
65022d8 Add the eight-page version, and teach the checker about subdirectories
6847426 Pull Table 1's columns in by a point so its rules stay in the block
b95037f Add the compiled paper, and archive the version it replaces
489ae5c Track the development log, minus the bulk and the noise
f199849 Correct four numbers the ten-seed pass had left behind
d86ae18 Give the paper a conclusion, a figure, and an appendix
5609a5e Write the k=4 sequence into the paper, with a generator behind it
f8cfa55 Add the k=4 sequence run: 25 runs, four tasks each
```

**El paper compilado vive en `paper/WMF.pdf`** (26 páginas, 8 ago 2026), junto a
las fuentes que lo producen. Lleva las cuatro correcciones de la sesión 10
—comprobado: 88.5, 58.79, 2.01 y 14.8 están; 78–97, 1.96 y 15× no—, así que es
posterior a ellas.

El PDF de la versión anterior e inválida ya no está en la raíz: se archivó en
`_devlog/archive/paper-anterior-main5.pdf`, fuera de git, porque sigue siendo la
referencia de `paper-vs-code.md` y no debía perderse al sustituirlo.

**`paper/` sí se versiona** (decisión de la sesión 8, y es reversible con un
`git mv`): es fuente nueva, y viaja con el código que genera sus tablas. Lo que
se dejó fuera del repo fue el PDF de la versión inválida, que es otra cosa.

`results/` tiene **las 375 celdas** (protocolo 5000 pasos) más las 75 parejas en
`_reference/`, y al lado están `results-2x/` (el sondeo de doble presupuesto, 10
celdas) y `results-seq/` (la secuencia k=4, 25 corridas). Todo commiteado. Los
`metrics.json` se versionan; los `.pt` no existen (no se guardan checkpoints).

Fuera del repo quedan los diagnósticos previos, medidos a presupuesto pequeño y
por tanto no mezclables con lo anterior. Están en `_devlog/archive/`:

| Directorio | Qué es |
| --- | --- |
| `results-R0/` | Los 226 `metrics.json` inválidos (VAE colapsado) y sus figuras |
| `results-R12-finding4/` | Las 20 celdas del chequeo del Finding 4 (R12) |

Los ~20 checkpoints (713 MB, entrenados con el VAE colapsado) se borraron.

### Bitácora `_devlog/` (gitignored)

| Archivo | Contenido |
| --- | --- |
| `README.md` | Índice y estado |
| `HANDOFF.md` | Este documento |
| `findings.md` | Los 28 problemas (F0–F27) con evidencia numérica |
| `changelog.md` | Cambios por archivo, 8 sesiones |
| `improvements.md` | Mejoras I1–I20 |
| `paper-vs-code.md` | Discrepancias con el PDF |
| `paper-plan.md` | **Valoración honesta** + qué se salva y en qué orden |
| `runs.md` | Ejecuciones R0–R17 con sus números |
| `decisions.md` | D1–D18 tomadas, P1/P5/P6/P11 pendientes |
| `final-summary.txt` | La salida completa de `summarize_results.py`. **OBSOLETO desde que arrancó R19** — regenerar al acabar, y hasta entonces no usarlo como referencia |
| `run-seeds-5-9.sh` + `seeds-10.log` | La corrida de diez semillas (D21/R19). Reanudable: reejecutar el script |
| `seq-k4.log` | k=4 en MiniGrid (D22/R20), el runner de secuencias |
| `probe-2x.log` | El sondeo a 2× presupuesto (R18) |
| `minigrid-summary.txt` | Lo mismo, pero solo MiniGrid y de la corrida parcial |
| `diagnose-p12.py` + `p12-diagnostico.log` | La reproducción de la celda de F23 (R17). El patrón sirve para cualquier diagnóstico futuro que necesite modelos entrenados, ya que no se guardan checkpoints (D3) |
| `check-paper.py` | Comprobación estructural de `paper/*.tex` sin LaTeX: llaves, `$`, entornos, `\input`/`\ref`/`\cite` colgantes |
| `archive/` | Resultados invalidados, conservados como evidencia |

**Desde la sesión 10 estos documentos sí viajan en git** — los `.md` y los
scripts de comprobación, no `archive/` ni los logs. Lo que no está hecho es el
push. Si el repo se hace público, repasar antes `paper-plan.md`, que es una
valoración interna de si esto llega a publicarse y dónde.
