# Decisiones

---

## D1 — Licencia MIT

Elegida por el autor. Estándar para código de investigación.

## D2 — Eliminar los 7 runners redundantes

`run_full_benchmark.py` reproduce todos los números publicados. Mantener 15
scripts obligaba al revisor a adivinar cuál importa. Decidido por el autor.

## D3 — Excluir checkpoints del repositorio

713 MB → <1 MB. Todos los números derivan de los `metrics.json`, que sí se
versionan, así que tablas y figuras reproducen sin los `.pt`.

## D4 — No "arreglar" los resultados en silencio

Los problemas F1/F2 invalidan los números publicados. Se corrigió el código y
se documentó el impacto, en vez de regenerar cifras sin avisar.

## D5 — Codificar `D_i` una sola vez, con el modelo post-tarea-A

PF compara dos modelos sobre el mismo `D_i`. Si cada modelo codificase con su
propio VAE, se estaría comparando cada uno contra su propio espacio latente en
movimiento, no contra `D_i`. Por eso `build_latent_eval_dataset` recibe el
modelo cuyo espacio latente *define* la tarea.

## D6 — `mc_dropout_T_eval` separado de `mc_dropout_T`

Una varianza estimada con 3 pasadas es muy ruidosa. Más muestras estiman mejor
*la misma* cantidad, así que subirlo en evaluación no altera el método pero
reduce varianza en las métricas. Por defecto = `mc_dropout_T` si no se
especifica (retrocompatible).

## D7 — Reescribir el paper en vez de parchearlo

Decidido por el autor. Los números de la Tabla 3 y de §5.2 no son recuperables:
provienen de mediciones sobre ruido y con el objetivo de NLL equivocado.

## D8 — La tesis del paper es el benchmark, no UG-MTM

Decidido por el autor. La contribución es **un benchmark que funcione** para
medir olvido catastrófico a nivel del componente de transición `M`.

**Consecuencias.**

1. La validez del banco de pruebas es lo prioritario: métricas correctas,
   escalas conmensurables, protocolo reproducible. Todo lo demás va después.
2. UG-MTM pasa a ser **uno de los cinco métodos evaluados**, no el resultado.
   Que gane o pierda deja de ser un riesgo para el paper.
3. Un resultado del tipo "el enrutamiento por incertidumbre funciona a
   distancia grande y falla a distancia moderada" es **material válido**: es
   justo lo que un benchmark debe ser capaz de revelar.
4. El criterio de éxito pasa a ser: ¿discrimina el benchmark entre métodos de
   forma estable, interpretable y reproducible?

Esto elimina el riesgo señalado en `paper-plan.md`.

---

## D9 — La ceguera al codificador se declara como alcance (P9, sesión 7)

Decidido por el autor. Opción 1: el paper declara que **mide `M` en una base
latente fija**, y reporta la reconstrucción en píxeles **al lado** de PF y RD, no
en su lugar.

Coste cero de cómputo: los dos números ya estaban en cada `metrics.json`.
Implementado en `summarize_results.py`, que ahora encabeza la sección de métricas
con el alcance y saca una sección de calidad en píxeles con las tres columnas
(recon A tras A, recon A tras B, recon B tras B).

La consecuencia buena: *dónde* ocurre el olvido dentro de un world model pasa a
ser una pregunta empírica que este banco de pruebas puede responder, en vez de
una crítica que el paper no ve venir.

## D10 — Se publican PF y RD por separado, no un escalar único (P7, sesión 7)

Decidido por el autor. Opción 2: se renuncia al titular de "una sola cifra".

Con PIS a 0 fijo, `WMF = 0.4·PF + 0.4·RD` es un agregado del que RD se lleva el
78–97%: publicarlo como número principal es publicar RD con pasos de más.

`summarize_results.py` encabeza con PF, RD y FT; WMF sigue imprimiéndose, en una
sección rotulada **"LEGACY AGGREGATE (Eq. 6)"** y acompañada de la columna que
dice qué fracción de él aporta RD. La cifra del paper anterior sigue siendo
reproducible; lo que cambia es qué se afirma con ella.

## D11 — Se pagan los entrenamientos de referencia (P8 + P10, sesión 7)

Decidido por el autor. Se entrena un modelo por entorno desde cero, y con el
mismo cómputo se arreglan las dos debilidades:

- `d_trans` fiel a la Ec. 9 — KL entre dos modelos, cada uno entrenado en su
  entorno (F15).
- **FT de verdad** — la tarea B aprendida con y sin preentrenamiento en A (F20).

Coste real: **+90 entrenamientos**, no +45. La referencia es una *pareja*
(un modelo por entorno), una por `(familia, distancia, semilla)`: 45 parejas × 2.
Sobre los 450 entrenamientos de la ejecución completa es un +20%.

Se cachea en `results/_reference/`, se comparte entre los cinco métodos de la
casilla y se rechaza si viene de otro protocolo.

## D12 — El presupuesto es 5× (P4, sesión 7)

Decidido por el autor sobre el dato de R11: `n_train: 5000` en las tres familias.
La reconstrucción reservada de la tarea A queda a un 17% del valor de 10× por la
mitad de cómputo, y a 1000 pasos era 4.7× peor.

Coste estimado: ~3.3 días para las 225 corridas, más el +20% de las referencias.
`results/` estaba vacío, así que no hubo nada que archivar.

---

## Pendientes de decidir

### P1 — Mecanismo del umbral en UG-MTM (F3, bloqueante)

`tau` (sigmoide, acotado a 0–1) y `u_t` (varianza sin normalizar) no son
conmensurables. Opciones:

1. **Sustituir la sigmoide por softplus.** Deja que `tau` alcance el rango real
   de `u_t`. Riesgo: en la inicialización `softplus(~0) ≈ 0.69`, todavía por
   encima de `u_t ≈ 0.13`, y no está claro que el gradiente lo empuje hacia
   abajo.
2. **Umbral relativo a la historia:** `tau = μ_hist + k·σ_hist`, con `k`
   aprendido. Adaptativo a la escala por construcción y codifica directamente
   "incertidumbre inusualmente alta". Compatible con §4.2 del paper, que
   describe `tau` como aprendido por un MLP sobre la ventana reciente y **no
   menciona la sigmoide**.
3. **Estandarizar `u_t`** contra la ventana antes de compararlo.

**Antes de elegir hay que medir si `u_t` discrimina tarea A de tarea B.** Si no
separa, ninguna regla de umbral funciona y la premisa del método falla.

### ~~P7~~ — Cómo agregar métricas de escalas distintas · **RESUELTO (D10)**

`WMF = 0.4·PF + 0.4·RD + 0.2·PIS` presupone que las tres son conmensurables.
Re-medido con la KL corregida (R6): RD es 3–32× mayor que PF y se lleva el
**75–97%** del agregado; PIS sigue siendo 0. (Con la KL rota era 1–3 órdenes de
magnitud y el agregado era de facto `0.4·RD`.)

Opciones:

1. **Normalizar cada componente** antes de agregar (por ejemplo, dividir por su
   valor en el baseline de finetuning, o estandarizar por familia). Mantiene el
   agregado pero lo hace interpretable.
2. **Reportar las tres por separado** y renunciar al escalar único. Más honesto,
   pierde el titular "una sola cifra".
3. **Acotar RD** — normalizar por horizonte, o truncar el rollout cuando
   diverge, o usar una divergencia acotada (Jensen-Shannon en vez de KL).

**Estado (sesión 5).** F13 y F16 corregidos, y los porcentajes re-medidos sobre el
pipeline reproducible (R9): PF se lleva el **2.8–22.4%** del agregado, RD el
**77.6–97.2%**. Casi idéntico a la medición previa con los entornos sin sembrar,
lo que dice que **F14 es estructural**, no un artefacto de unos datos concretos.

Qué cambia esto respecto a las tres opciones:

- **La opción 3 pierde casi todo su peso.** "Acotar RD" existía sobre todo para
  contener la explosión de 13708, y esa desapareció sola al arreglar la KL. Queda
  un solo outlier (UG-MTM a distancia máxima, `RD = 167` frente a 14–32) — real,
  pero no es un desbordamiento numérico y no justifica cambiar la definición de
  la métrica.
- **La tensión real es entre 1 y 2.** Y es una decisión sobre qué afirma el
  paper, no sobre qué es más cómodo: la opción 1 conserva el titular de "una sola
  cifra" a cambio de que esa cifra dependa de una normalización elegida por el
  autor; la opción 2 renuncia al titular y gana en honestidad.

**Ya no hay nada que medir antes de decidir.** Lo único que sigue pendiente es la
dispersión entre semillas, que no cambia el reparto PF/RD (es estructural, no
ruido) y que de todos modos se obtiene en la ejecución completa.

### P5 — Peso del término de incertidumbre en UG-MTM (nuevo, tras F0)

Con la reconstrucción bien escalada (~170 en vez de ~0.014), el término
`total_uncertainty` de `UG_MTM.compute_loss` queda proporcionalmente
insignificante. Es la pérdida que calibra `u_t` contra el error de predicción
real, así que si no se entrena, la puerta se degrada.

Opciones: darle un peso explícito en el config (`beta_uncertainty`), o
normalizarlo contra la escala de la reconstrucción. Medir primero cuánto
importa en la práctica.

### P6 — La señal de incertidumbre se invierte a distancia media (nuevo)

AUC 0.294 en `Empty-8x8 -> FourRooms`: la tarea B parece *menos* incierta que
la A, así que la puerta enruta al revés. A distancia máxima funciona bien
(0.864).

No es necesariamente algo que "arreglar" — es un resultado honesto sobre
cuándo funciona el enrutamiento por incertidumbre. Decidir si se reporta como
hallazgo o se intenta mitigar.

### ~~P2~~ — Qué hacer con PIS (F6) · **RESUELTO por D18 (sesión 8)**

Implementarlo (requiere un controlador, que no existe en el repo) o sacarlo del
paper y redefinir WMF con los pesos realmente usados.

**Decidido por el autor: se saca.** Ver D18.

### ~~P3~~ — DMControl `distance_min` (F8) · **RESUELTO (sesión 7)**

Implementar la perturbación de viento, o sustituir el par por dos tareas
realmente distintas, o eliminar ese nivel de la familia.

**Resuelto (sesión 7)** por una cuarta vía que las tres opciones no contemplaban:
el viento de MuJoCo no funciona sin densidad de fluido, así que se implementó el
mismo mecanismo de perturbación física que ya usa Gymnasium y `distance_min` pasa
a ser `cheetah/run` con gravedad 9.81 → 7.0 — el mismo cambio físico que el
`distance_min` de Gymnasium. Ver F8.

### ~~P4~~ — Alinear los configs con lo ejecutado (F10) · **RESUELTO en parte (sesión 6)**

La **mecánica** está resuelta (I1): el bloque `protocol:` del config es la única
fuente de verdad, el runner lee de ahí los 14 campos que antes tenía escondidos,
cada `metrics.json` guarda su protocolo, y el runner se niega a mezclar
presupuestos en un mismo directorio de resultados. La Tabla 1 del paper se genera
desde los resultados.

Los YAML declaran ahora **lo que se ejecuta hoy** (20 rollouts, 1000 pasos,
batch 8, `seq_len` 5), no lo que se aspiraba a ejecutar. Eso deja el config
honesto pero no decide la parte que sigue siendo del autor:

**Lo que queda por decidir: el presupuesto de la ejecución completa.** Ya no es
una decisión a ciegas — R11 mide dónde se estabiliza la calidad en la tarea A al
subir los pasos. Con ese dato en la mano, subir la escala es cambiar cuatro
números en tres YAML y archivar `results/`.

---

## ~~P8~~ — Qué es `d_trans` · **RESUELTO (D11, opción 1)**

La Ec. 9 define `d_trans(E_A, E_B) = E[KL(P_A(z'|z,a) || P_B(z'|z,a))]`: entre
dos modelos, **cada uno entrenado en su propio entorno**. El protocolo actual no
produce eso — solo tiene `model_i` (post-A) y `model_k` (post-A-luego-B), y la KL
entre esos dos mide cuánto se movió el modelo, casi lo mismo que RD.

Opciones:

1. **Medirla como la define la Ec. 9.** Entrenar un modelo en B desde cero por
    celda y compararlo con `model_i` sobre los mismos `(z, a)`. Coste: +45
    entrenamientos sobre los 225. Es la lectura fiel del paper y da la validación
    cruzada contra `d_param` en Gymnasium.
2. **Redefinir la Ec. 9** como distancia entre el modelo antes y después del
    cambio de tarea. Gratis, pero deja de ser una distancia entre *entornos* y se
    solapa con RD — habría que justificar por qué son dos métricas y no una.
3. **Sacar `d_trans` del paper** y admitir que solo Gymnasium tiene distancia
    cuantificada, con MiniGrid y DMControl como niveles ordinales. Honesto pero
    debilita el "controlled dynamic distances" del abstract.

La opción 1 es la única que sostiene la afirmación del paper. 45 entrenamientos
extra a escala de smoke run son ~1 h; a la escala de la ejecución final, depende
del presupuesto que se decida en la Fase 3.

---

## ~~P9~~ — Qué hacer con la ceguera al codificador · **RESUELTO (D9, opción 1)**

**La decisión más importante de las que quedan**, porque no es un arreglo: define
qué afirma el paper.

`compute_nll` opera sobre latentes que codificó **una sola vez** el modelo
post-tarea-A (decisión D5, que es correcta por lo que buscaba: que `model_i` y
`model_k` se puntúen sobre entradas y objetivos idénticos). El efecto no buscado
es que PF, RD, WMF y FT solo ven el GRU y `stoch_fc`. El codificador puede
degradarse por completo sin que ninguna métrica del paper lo note. Medido:
`finetuning` pierde la reconstrucción de la tarea A ×112 mientras **PF sale
negativo**.

Opciones:

1. **Declararlo como alcance y reportar las dos escalas.** "Medimos `M` en una
   base latente fija, y reportamos la calidad en píxeles al lado." Coste: cero
   cómputo — los dos números ya se guardan en cada `metrics.json`. Es lo mínimo
   defendible, y convierte una crítica potencial en un resultado: *dónde* ocurre
   el olvido dentro de un world model pasa a ser una pregunta empírica.
2. **Añadir una métrica que reencode con el codificador vigente.** Mide la
   degradación del sistema, no de un componente en coordenadas obsoletas. Es una
   **definición nueva**, no un parche, y rompe la comparabilidad con la Ec. 3 del
   paper. Entra en el mismo saco que P7: hay que decidirla junto con la
   agregación, no por separado.
3. **No tocar nada y no decirlo.** Ni siquiera está en la lista, pero conviene
   escribirlo para descartarlo: la disociación es visible con dos líneas de
   análisis por cualquiera que reejecute el código, y aparecer en un review es
   mucho peor que aparecer en la sección de limitaciones.

Recomendación: **la 1 sí o sí** (es gratis y ya está implementada), y decidir la 2
al mismo tiempo que P7 porque las dos preguntan lo mismo: qué es exactamente el
número que el benchmark publica.

---

## ~~P10~~ — Qué hacer con FT · **RESUELTO (D11, opción 2)**

`FT = NLL(M_random, D_A) − NLL(M_i, D_A)` no toca ni un dato de la tarea B. Los
métodos que comparten arquitectura solo se diferencian en el cambio de tarea y
después, así que **FT es idéntico para todos ellos por construcción** — medido:
delta exactamente 0.000 entre `finetuning` y `replay_infinite` en las 10 semillas
de R12. Y no mide lo que dice su docstring ("prior knowledge helped learn new task
faster"): mide el ajuste a la tarea A.

Opciones:

1. **Renombrarlo a lo que mide.** Es la evidencia de F17 —¿había algo que
   olvidar?— y en ese papel es útil y ya se guarda descompuesto
   (`nll_A_after_task_A`, `nll_A_random_init`). Coste cero. Se pierde una de las
   cuatro métricas anunciadas.
2. **Medirlo de verdad:** entrenar un modelo solo en B y comparar sobre `D_B` el
   aprendizaje con y sin preentrenamiento en A. Es la definición correcta de
   transferencia hacia delante. Coste: +45 entrenamientos — **el mismo modelo que
   pide la opción 1 de P8 para `d_trans`**. Si se decide hacer una, la otra sale
   casi gratis, y ese es el argumento más fuerte para hacer las dos.
3. **Sacarlo del paper.** Honesto, pero deja la suite en PF + RD, con PIS a 0 y
   RD llevándose el 78–97% del agregado. Es mucho encoger.

Recomendación: **1 si el presupuesto aprieta, 2 si se decide pagar los +45
entrenamientos de P8** — en cuyo caso hacer las dos cosas con el mismo cómputo
convierte dos debilidades en dos métricas bien definidas.

---

## Nota sobre el orden (resuelta, sesión 5)

Esta nota decía que P7 no se podía decidir hasta arreglar F16, porque el ruido de
ejecución (~0.45 en PF) era comparable a |PF| en varias celdas y los números
tendrían una barra de error desconocida.

**F16 ya está corregido y los números re-medidos (R9).** P7 está desbloqueado: ver
el estado actualizado en la propia entrada de P7 más arriba.

Un hallazgo nuevo que salió de R9 y que conviene no perder: **`replay_infinite`
sale con el WMF más bajo en los dos niveles** (4.35 y 5.12), con PF negativo. En
la Tabla 3 original pasaba lo contrario — replay *peor* que finetuning en
MiniGrid era el **Finding 4**, el único hallazgo del paper que resistió el
escrutinio (p<0.001). Con una sola semilla no concluye nada, pero es **lo primero
que hay que mirar en la ejecución completa**: si el Finding 4 se invierte, cambia
el resultado más sólido que tenía el paper.

**Cerrado en R16 (sesión 7), y con matiz.** En MiniGrid la inversión se sostiene:
replay olvida menos en RD en **las 15 comparaciones emparejadas** de los tres
niveles, con la p en su suelo de 0.0625 y d_z de −1.99 a −2.64. Pero en
`gymnasium/distance_min` **no se replica**: replay iguala a finetuning en PF
(−8.16 vs −8.20) y sale peor en RD (82.8 vs 75.3). La inversión puede ser un
resultado de MiniGrid, no del banco. Decidir cómo se afirma cuando estén las tres
familias.

---

## ~~P12~~ — Qué hacer con la explosión de RD en UG-MTM (R16, ver F23) · **CERRADO por D15**

`ug_mtm` en `minigrid/distance_max` da RD por semilla de 17.7, 40.0, 520, 574 y
**4364**. Media 1103 ± 1647. Bloquea reportar esa casilla.

Opciones:

1. **Diagnosticar primero, decidir después.** La curva de KL por paso dice en qué
   paso del rollout diverge; si es en los últimos, el problema es el horizonte.
   Requiere reejecutar esa celda con instrumentación: los checkpoints no se
   guardan (D3).
2. **Acotar RD** — truncar el rollout cuando diverge, o usar Jensen-Shannon en vez
   de KL. Era la opción 3 de P7, descartada porque la explosión de entonces
   resultó ser el bug F13. Esta no lo es, así que la opción vuelve a estar viva.
3. **Reportar la mediana y declarar la dispersión.** Honesto y gratis, pero
   esconde que el método es inestable en esa celda — que es justo lo que un banco
   de pruebas debería sacar a la luz.

Recomendación: 1 y luego 3. El diagnóstico es una corrida.

### Diagnosticado (sesión 7) — ver F23

**Es colapso de varianza en el modelo post-B, no divergencia del rollout.** En la
semilla mala hay dimensiones con `σ = 0.007`; el término `(mu_i − mu_k)² / var_k`
de la KL se dispara con diferencias de medias corrientes. La KL ya vale 80 en el
paso 0, así que **truncar el horizonte no arregla nada** y la opción 2 queda
reducida a "usar una divergencia acotada".

Lo que queda por decidir, ahora con la causa en la mano:

1. **Reportar mediana y dispersión en vez de media** (opción 3 original). RD tiene
   cola pesada **por construcción** cuando un método produce modelos
   sobreconfiados, y `KL(P_A‖P_B)` es no acotada justo en ese caso. Promediar
   cinco semillas de las que una vale 4364 no describe nada. Coste cero.
2. **Añadir una divergencia acotada** (Jensen-Shannon) al lado de RD. Es una
   métrica nueva y hay que justificarla, pero acota el problema de raíz.
3. **Poner un suelo a `log_var` en evaluación.** Descartada: cambia la métrica
   para tapar un comportamiento real del método.

**Recomendación: 1, y reportar el hecho como resultado sobre UG-MTM.** Que
congelar el codificador produzca modelos de transición sobreconfiados es un
hallazgo del banco de pruebas, no un estorbo: coincide con su PF de +39.98 en esa
misma semilla — seguro y equivocado.

---

## ~~P13~~ — Qué hacer con las celdas que no producen olvido (R16, ver F22) · **CERRADO por D16**

En `gymnasium/distance_min` ningún método olvida: la reconstrucción de la tarea A
mejora tras entrenar en B, para los cinco. Es lo esperable de un nivel de
distancia mínima —es el control del eje— pero una celda donde nadie olvida
tampoco discrimina.

Opciones:

1. **Reportarla como control declarado.** "A distancia mínima no hay olvido que
   medir, y el banco lo detecta." Coste cero, y convierte un hueco en evidencia de
   que el eje de distancia funciona.
2. **Subir la perturbación** del nivel mínimo de Gymnasium. Invalida esas 5 celdas
   y hay que reejecutarlas.

Esperar a que termine R16 antes de decidir: hay que saber cuántas de las nueve
celdas son controles, no solo una.

**Decidido en D16 (opción 1, con el criterio en relativo).** Con las nueve
celdas delante son **tres**, no dos, y la tercera —`gymnasium/distance_med`— no
estaba en ninguna lista previa.

---

## D13 — El nivel máximo de dmcontrol pasa a `cheetah/run → walker/stand` (sesión 7, ver F25)

El par declarado en el paper, `cheetah/run → reacher/easy`, es inejecutable: 6
actuadores contra 2, y un solo world model para las dos tareas.

Se eligió entre tres opciones con el criterio que fijó el autor — *resultados
correctos y publicables* — y no por coste, que es el mismo en dos de las tres:

1. **Rellenar con ceros** hasta 6 y conservar el par del paper. **Descartada:**
   cuatro dimensiones de acción quedarían constantes durante toda la tarea B y
   RD las reactiva al desplegar acciones de `D_A`, así que parte del olvido
   medido sería respuesta a entradas muertas. Confound no cuantificable barato.
2. **`walker/stand`** — 6 actuadores, cambia cuerpo y objetivo. **Elegida.**
3. **Quitar el nivel.** Coste cero pero deja la familia visual con dos niveles y
   debilita el "controlled dynamic distances".

**Riesgo declarado por adelantado:** `distance_med` (`walker/run`) mide
`d_trans = 6.99 ± 3.00`. Si `walker/stand` sale por debajo, el nivel está mal
nombrado y hay que decirlo y reordenarlo. La comprobación se hace en cuanto
terminen las 5 parejas de referencia.

### El riesgo se materializó, y peor de lo previsto (D14)

`walker/stand` dio `d_trans = 6.91 ± 2.92` contra los `6.99 ± 3.00` de
`walker/run`: indistinguibles. Y la causa no era que el cambio fuese pequeño sino
que **no había cambio ninguno** — las dos "tareas" son el mismo entorno, ver
**F26**. Las 25 celdas salieron bit a bit idénticas a las de `distance_med`.

Lección sobre el propio D13: elegí `walker/stand` razonando sobre las etiquetas
("cambia el objetivo") sin comprobar qué produce el simulador. El compromiso de
declarar la predicción por adelantado sirvió para lo que tenía que servir — el
resultado la falsó y no hubo margen para racionalizarlo — pero la predicción
correcta habría sido innecesaria si hubiera mirado los datos primero. **D14
sustituye a D13.**

---

## D14 — El nivel máximo de dmcontrol es cambio de cuerpo + perturbación física (sesión 7, ver F26)

`cheetah/run → walker/run` **con gravedad 4.0, masa ×3 y fricción ×0.5**, la
misma perturbación que aplica el `distance_max` de Gymnasium.

Es la única construcción disponible que es estrictamente más cambio que
`distance_med`:

- Por **dominio** no se puede: solo `cheetah` y `walker` tienen 6 actuadores
  (F25) y el cambio de dominio ya está gastado en `distance_med`.
- Por **tarea** no se puede: en dm_control las tareas de un dominio solo se
  diferencian en la recompensa, que este benchmark no usa (F26).
- Queda la **física**, que es además el mismo eje que ordena la familia
  Gymnasium, así que las dos familias continuas quedan comparables en su nivel
  máximo.

Y esta vez no hay que fiarse de la etiqueta: `distance_med` mide
`d_trans = 6.99 ± 3.00` y `distance_max` tiene que salir por encima. Si no sale,
la familia se reporta con dos niveles, no tres.

**Y hay que declararlo en el paper**, no cambiarlo en silencio (D4): el par que
aparecía en §3.4 no se podía ejecutar, y esa es información sobre el diseño
original que un lector merece.

---

## D15 — Las celdas se resumen por mediana y rango, no por media ± desviación (sesión 8, cierra P12)

**Aplicada la recomendación de P12.** `summarize_results.py` y `plot_final.py`
resumen cada celda como `mediana [min, max]` sobre las cinco semillas, y las
barras de la figura son **asimétricas** por el mismo motivo.

El argumento no es que la media esté mal calculada, sino que RD es una KL sin
cota superior: un método que acaba sobreconfiado produce cola pesada **por
construcción** (F23, diagnosticado en R17 como colapso de varianza). La media de
17.7, 40.0, 520, 574 y 4364 es 1103, que no describe ninguna de las cinco.

**Y la cola no se esconde: se marca.** Las celdas cuya mitad superior se estira
cinco veces más que la inferior salen con `!` y se listan debajo con sus cinco
semillas y su media, porque es un resultado sobre el método, no un estorbo.

Dos detalles que salieron al implementarlo y conviene no reabrir:

- **El criterio es de estadísticos de orden**, no de media/mediana ni de
  coeficiente de variación: esos dos se disparan en celdas que simplemente
  cruzan el cero, como el PF de EWC (−0.005 con dispersión 0.083, perfectamente
  simétrico).
- **El marcador no se aplica a las columnas de píxeles.** Ahí la magnitud está
  acotada por los datos y marcar el 25–35 habitual de una línea base de
  Gymnasium sería señalarlo todo. El rango se imprime igual.

Efecto medido sobre las tablas: **22 celdas marcadas** de las 180 de pf/rd/ft/wmf.
O sea que la cola pesada no es exclusiva de la casilla de F23 — es frecuente con
n=5, y eso es en sí la justificación empírica de la política.

## D16 — `gymnasium/distance_min`, `gymnasium/distance_med` y `dmcontrol/distance_min` se declaran controles (sesión 8, cierra P13)

**Son tres, no dos.** El handoff y `paper-plan.md` §0bis daban por hecho
`gymnasium/distance_min` y `dmcontrol/distance_min`. Con el criterio aplicado a
las nueve celdas salen otras tres, y la lista no coincide con la asumida:

| Celda | Pérdida de la tarea A (rango sobre métodos) |
| --- | --- |
| `gymnasium/distance_min` | −9.7% .. +0.0% |
| `gymnasium/distance_med` | −8.2% .. **+6.0%** |
| `dmcontrol/distance_min` | +0.0% .. **+1.6%** |
| — el resto — | hasta **+75244%** (`minigrid/distance_min`, ewc) |

**El criterio es relativo, y ese es el cambio.** Medido en píxeles absolutos,
`dmcontrol/distance_min` degrada +0.23 y *no* sería un control; pero +0.23 sobre
una base de 14.74 es un 1.6%, mientras que el mismo +0.23 sobre los 0.52 de
MiniGrid sería un 44%. El umbral es el **10%** de lo que el modelo tenía, y no es
un filo de navaja: entre los controles (≤6%) y la celda que menos olvida de las
que olvidan (+1359%) hay un factor 200.

**El hallazgo real está en `gymnasium/distance_med`**, que nadie esperaba en esta
lista: no pierde nada del codificador y a la vez tiene **el RD más alto de su
familia** (85.9). Es F18 visto desde el otro lado — la transición olvida donde el
codificador no— y por eso la sección imprime los números en vez de un veredicto,
y el criterio **no consulta RD**.

Consecuencia para el paper: la rejilla efectiva son **6 de 9 celdas**, y las tres
restantes se reportan como controles declarados. En Gymnasium eso significa que
el olvido a nivel de codificador **solo aparece en el nivel máximo**.

## D17 — El eje de distancia (F27) se agrega sobre los cuatro métodos RSSM, y se dice (sesión 8)

Las cifras de F27 se habían calculado a mano promediando **cuatro** métodos: la
exclusión de `ug_mtm` estaba tomada de hecho, sin pasar por aquí ni por el
script. Ahora la regla vive en la constante `AXIS_METHODS`, se imprime en la
cabecera de la sección y se puede mover con `--axis-methods`.

**Motivo declarado:** `ug_mtm` congela el codificador, lo que en dmcontrol pone
su RD dos órdenes de magnitud por debajo del resto (0.002 frente a 1.2).
Agregarlo con los otros cuatro sería promediar dos escalas, no leer el eje.

Y la agregación es **mediana de medianas**: sobre semillas por D15, y sobre
métodos porque los cinco métodos no son cinco muestras de una cantidad — son
cinco cantidades distintas, y una media dejaría que el método más extremo fijara
el número de la familia.

## D18 — PIS se retira de la suite; el banco reporta PF, RD y FT (sesión 8, cierra P2)

**Decidido por el autor.** PIS se anunció en el paper anterior como una de las
cuatro métricas y **nunca se implementó**: valía `0.0` fijo. Medirla de verdad
exige entrenar un controlador dentro de la imaginación del modelo y evaluarlo en
el entorno real, que es un proyecto, no una métrica, y no hay controlador en el
repositorio.

Se retira en vez de reportarse. **La suite es PF, RD y FT.**

Qué cambia en el código, que no es cosmético:

- **`pis` se guarda como `null`, no como `0.0`.** Un cero almacenado se lee como
  "medido, y salió cero"; `null` es la convención que ya usan `ft` y `d_trans`
  cuando su pareja de referencia no se entrenó, y está cubierta por un test que
  impide promediarla como si fuera un valor. Los tres runners (`run_full_benchmark`,
  `train_baseline`, `train_ug_mtm`) escriben ahora `null`.
- **WMF se queda como está**, y esto es deliberado: es la Ec. 6 del paper
  anterior y se calcula para poder reproducir aquellos números, con su término
  `gamma` evaluado a cero — que es exactamente con lo que se calcularon. D10 ya
  lo había bajado a agregado heredado; retirar PIS no lo cambia, solo obliga a
  decir por qué el término está a cero. El argumento `pis_list` de `compute_wmf`
  sobrevive por el mismo motivo.
- Docstrings, README del repo (sección *Metrics* y limitación nueva nº 8) y la
  cabecera del bloque de WMF en `summarize_results.py` dicen las tres cosas: que
  la suite son tres, que PIS se anunció, y que no se implementó.

**Lo que cuesta:** la contribución encoge sobre el papel — "proponemos PF, RD y
FT" es menos que "una suite de cuatro métricas". Con D10 el coste es pequeño,
porque WMF ya no era el titular y el titular es F27, que no es una métrica sino
un hallazgo.

**Lo que se gana:** ninguna cifra del paper es una métrica sin implementar. Era
la última que quedaba (F6, abierto desde la sesión 1).

**Nota sobre los 225 resultados ya guardados:** conservan `"pis": 0.0`. No se
tocan —D4, no se reescriben resultados— y nada los lee: `summarize_results.py` no
tabula `pis`, y su `wmf` se calculó con el mismo cero. La diferencia solo
aparecería en corridas nuevas.

---

## D19 — El paper deja de anunciarse como banco de pruebas y se anuncia como hallazgo (sesión 9)

**El problema.** Después de D8 («la tesis es el banco») el resumen interno del
proyecto quedó en *«tenemos un benchmark que mide el olvido»*. Con las 225
celdas leídas eso ya no describe lo que hay, y describe algo más difícil de
publicar: un paper de herramienta se acepta cuando la herramienta se adopta
—`pip install`, CI, gente compitiendo— y este proyecto no está ahí.

**Lo decidido.** El banco es el **instrumento**; la contribución es el
**hallazgo**:

> El olvido en un world model no escala con la distancia entre tareas sino con
> lo que la tarea nueva exige del modelo (F27), y ocurre casi todo en el
> codificador, que es donde las métricas al uso —incluidas las nuestras— no
> miran (F18 + F21).

**Qué cambia en la práctica:**

- **El título tiene que cambiar.** El actual —*«…: A Benchmark and Structural
  Mitigation»*— promete UG-MTM, que ya no es el resultado. Un revisor que lee
  eso y encuentra el método caracterizado-no-ganador se siente engañado antes
  de llegar a §5.
- La introducción presenta tres contribuciones en este orden: el hallazgo del
  eje, el hallazgo del codificador, y el banco que los produce. No al revés.
- Decide en qué se gasta el cómputo: no en hacer la rejilla más grande, sino en
  hacer el hallazgo más difícil de tumbar. Ver D20.

**No cambia** D8: la tesis sigue sin ser UG-MTM. D19 es más fino — la tesis
tampoco es el banco.

---

## D20 — Se extienden las pruebas antes de terminar de escribir, y el multiplicador no es el presupuesto (sesión 9)

**Lo decidido.** Parar de escribir introducción y abstract, y comprar cómputo
primero. **Pero eligiendo la pregunta, no el multiplicador.**

**Lo que NO se hace, y es la versión obvia:** relanzar la rejilla a 10×. R11 lo
descarta con datos propios — de 5000 a 10000 pasos la reconstrucción reservada
mejora un **15%** al doble de coste. El presupuesto es el único eje donde ya
estamos en rendimientos decrecientes: cuatro días para un 15% y cero preguntas
respondidas.

**Lo que sí, por orden de retorno:**

| # | Qué | Coste | Qué compra |
| --- | --- | --- | --- |
| 1 | **Sondeo 2× en el par anidado de Gymnasium** (`med` + `max`, `finetuning`, 5 semillas) | ~4 h | Mata «el pico es tu presupuesto». La única que puede cambiar el paper |
| 2 | **10 semillas en las 6 celdas que discriminan** | ~día y medio | El suelo de p pasa de 0.0625 a 0.002 |
| 3 | **k>2 en MiniGrid** (4 tareas, 3 cambios) | ~9 h | Cierra la objeción de diseño más previsible |
| 4 | **F28: referencia con codificador compartido** | +20% | `d_trans` deja de medir desalineamiento de bases |

**El orden importa.** El sondeo va **primero y solo**: es el único que puede
invalidar `results.tex` y `discussion.tex`, que están construidos alrededor de
F27, y la celda que interroga es justo el caso limpio del hallazgo.

**Regla de parada, fijada antes de mirar el resultado** —para no repetir el
patrón que produjo los cinco Findings caídos:

- Si el pico **aguanta** a 2×: F27 pasa de sólido a blindado y se sigue con 2, 3, 4.
- Si el pico **se cae** a 2×: el orden del olvido **depende del presupuesto**, o
  sea que el eje etiquetado no indexa una propiedad del par de tareas. Sigue
  siendo el mismo paper —el eje no ordena lo que dice ordenar— con un mecanismo
  distinto y más incómodo. **No se rediseñan los niveles para recuperar el pico**;
  eso es elegir el eje para que dé la respuesta esperada, y ya está descartado
  en `discussion.tex` §6.6.

Ningún resultado deja el proyecto sin paper. Eso es lo que hace que la apuesta
valga.

**Mientras corre se escribe trabajo relacionado**, que es la única sección
inmune a cualquier resultado. Introducción y abstract **no**: los dos dependen
del sondeo.

**Aislamiento.** El sondeo escribe en `results-2x/`, no en `results/`. El runner
ya se negaría a mezclar presupuestos (`check_protocol_consistency`), pero un
directorio aparte hace que el sondeo sea borrable de una pieza.

---

## D21 — Las diez semillas se pagan solo en las seis celdas que discriminan (sesión 9, ejecuta D20 paso 2)

**Lo decidido.** Semillas 5–9 en `minigrid` (los tres niveles), `gymnasium/max`
y `dmcontrol/med` + `dmcontrol/max`. Las **tres celdas de control se quedan en
cinco semillas**.

**Por qué.** Una celda de control no distingue entre métodos por definición
(D16): ningún método pierde nada de la tarea A ahí. Semillas gastadas en ellas
no compran resolución sobre ninguna comparación del paper. 360 entrenamientos
en vez de 540, sin perder nada de lo que se reporta.

**Lo que compra.** Con n=10 el suelo de una permutación exacta emparejada baja
de **0.0625 a 2/2¹⁰ = 0.002**. Es el arreglo del punto más débil del paper —el
único que no se puede argumentar, solo comprar— y no necesita ninguna decisión
más.

**El coste que hay que tener presente antes de leerlo:** las medianas de las
seis celdas cambian, así que **cambia cada cifra que el paper cita de ellas**.
Hay que regenerar las ocho tablas y hacer una pasada de números sobre
`abstract.tex`, `intro.tex`, `results.tex` y `discussion.tex`. Eso no es un
motivo para no hacerlo; es un motivo para no leer el paper como si estuviera
terminado hasta haberla hecho.

**Consecuencia menor y aceptada:** `results/` queda con seis celdas de diez
semillas y tres de cinco. La agregación es por celda, así que no mezcla nada;
las comparaciones emparejadas ya descartan semillas que un método no corrió.

**Aislamiento:** esto sí escribe en `results/`, a diferencia del sondeo de 2×,
porque es el **mismo protocolo**. El runner salta lo cacheado y rechazaría
cualquier cosa de otro presupuesto. Reanudable: `bash _devlog/run-seeds-5-9.sh`.

---

## D22 — k>2 se implementa como un runner aparte, no generalizando el de parejas (sesión 9)

**El problema.** El plan de D20 daba k>2 por «lanzable, ~9 h». **No lo era.** El
cómputo sí son ~9 h, pero `run_full_benchmark.py` está cableado a exactamente
dos tareas: el esquema del config (`task_A`/`task_B`), `create_env_pair`, el
bucle de entrenamiento, `switch_task`, y **todas las claves de `metrics.json`**
(`nll_A_after_task_B`, `heldout_reconstruction_B_after_task_B`, …). La
estimación era de GPU y se me olvidó la fontanería.

**Lo bueno:** los cinco métodos ya soportan k>2 sin tocarlos. EWC acumula una
lista de Fisher y suma las penalizaciones, progressive nets añade columna (F5),
replay acumula por `task_id`, y UG-MTM activa un experto más — con K=4 expertos,
justo los que necesita una secuencia de 4.

**Lo decidido: `experiments/run_sequence.py`, nuevo y separado.**

Generalizar el runner de parejas habría significado meter un índice donde hoy
pone `task_A` — en el config, el bucle, las métricas y el nombre de cada fichero
de resultados. **Eso pone en riesgo las 375 corridas que el paper reporta para
responder una pregunta lateral.** No compensa.

El script nuevo comparte modelos, bucle de entrenamiento, métricas y lector de
protocolo con el de parejas, y **no comparte estado**. Escribe en
`results-seq/`, con su propio esquema.

**Qué mide, y es más de lo que hacía falta:** la matriz de olvido completa. Para
cada tarea `i` y cada etapa posterior `k`, PF(i,k), RD(i,k) y la reconstrucción
en píxeles de `T_i`. Eso es literalmente el formalismo `PF(i,k)` que el paper
enuncia y nunca había ejecutado.

**Fidelidad al alcance declarado (D9):** `D_i` se construye una sola vez, con el
modelo tal como quedó justo tras entrenar `T_i`, y ese snapshot es la referencia
de PF y RD para esa tarea en todas las etapas posteriores. La diagonal
PF(i,i)=RD(i,i)=0 **exactamente**, y hay un test que lo fija: si no fuera cero,
el snapshot no sería el que corresponde y toda la matriz sería ruido.

**Las cuatro tareas son las mismas cuatro de la rejilla emparejada**
(Empty-5x5 → Empty-8x8 → FourRooms → KeyCorridorS3R1), en ese orden, y al mismo
presupuesto. Así, cualquier diferencia entre k=4 y k=2 es sobre longitud de
secuencia y nada más.

**Lo que NO hace, a propósito:** no entrena parejas de referencia, así que no
reporta FT ni `d_trans`. La pregunta es si el olvido se compone a lo largo de
una secuencia; la transferencia hacia delante ya está medida en la rejilla y
pagarla aquí multiplicaría el coste sin responder nada nuevo.

## D23 — El resultado de k=4 no entra en la versión de 8 páginas (sesión 16)

**Decidido por el autor**, con el documento ya compilado en 8 páginas
(bibliografía incluida) y el límite en 8 **excluyendo** referencias.

La subsección se había quitado durante los recortes, y al confirmarse que sobraba
sitio propuse devolverla. El autor decidió que no. Las razones se sostienen:

- **No eran dos frases.** Devolverla obliga a revertir también la limitación que
  se escribió para cubrirla —«los experimentos reportan un solo cambio de
  tarea»— porque las dos cosas no pueden convivir sin contradecirse. El coste
  real es un párrafo, no una línea.
- **El margen es de décimas.** El cuerpo ronda las 7,8 de 8 con la tabla del
  codificador dentro, y mis estimaciones de página han fallado dos veces por
  cerca de una página, siempre por debajo.
- **El resultado no se pierde**: está completo en el paper largo, y la
  limitación de la versión corta apunta a él.

La versión corta queda con **la figura del pico y tres tablas** (eje,
predictores, codificador), y **k=2 declarado como alcance**.
