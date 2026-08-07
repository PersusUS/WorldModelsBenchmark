# Plan para rehacer el paper — y valoración honesta

Estado: **escrito de principio a fin** (sesión 9). Abstract, introducción,
trabajo relacionado, método, resultados y discusión, más `main.tex`, con las
ocho tablas generadas por `experiments/export_tables.py`. Nunca compilado: no
hay toolchain de LaTeX en esta máquina.

**Lo que queda no es redactar.** Es (a) la pasada de números cuando termine R19,
(b) los pasos 3 y 4 del plan de cómputo de D20, y (c) compilarlo.

Última actualización: 3 ago 2026 (sesión 9: paper escrito, D19–D21, R18, F28–F29).

---

## 0ter. La valoración, con el paper escrito y el sondeo leído (sesión 9)

| Destino | Nota | Lectura |
| --- | --- | --- |
| **Workshop** (CL o world models) | **7/10** | Aceptado, probablemente en la mitad alta |
| **Main track / Datasets & Benchmarks** | **4/10** | Borderline reject con el alcance de hoy |
| **Preprint / registro** | **8/10** | Publicable ya |

Desglose: hueco 7 · fuerza de F27 **7** · soporte estadístico **3** (sube a ~6
cuando acabe R19) · escala **3** · rigor y reproducibilidad 9 · honestidad 9 ·
escritura 8 · contribución de método 3 (irrelevante por D8/D19).

**Qué cambió respecto a §0bis, para bien:**

- F27 **está comprobado a dos presupuestos** (R18), que es lo que separa un
  hallazgo de un artefacto de escala. Y la objeción principal no solo
  sobrevivió: se refutó, porque el mecanismo que proponía va al revés.
- El argumento de Gymnasium (superconjunto estricto que olvida menos) **no
  depende de `d_trans` ni de comparar familias**. Es el párrafo más fuerte del
  paper y es inmune a F28 y F29.

**Qué cambió para mal:**

- **F29.** La mitad *positiva* del eje se cae: `d_trans` no resuelve dentro de
  familia y se mueve con el presupuesto. Buena parte de su ρ=+0.53 lo carga la
  separación entre familias, que es justo lo que F28 dice que no puede comparar.
  El paper ya no lo vende; recomienda «reportad una distancia medida y exigidle
  cuentas». Es más débil como titular y es lo que los datos sostienen.

**Los tres ataques que quedan, por orden:**

1. **Escala.** 5000 pasos, 20 episodios de política aleatoria, latente 32.
   «¿Sobrevive a escala Dreamer?» Sin respuesta sin cómputo serio. Sigue siendo
   el riesgo nº 1.
2. **Dificultad de B y RD salen de las mismas corridas.** Una celda con datos
   duros infla las dos. No es circular, pero admite la lectura «esta casilla es
   dura». **Añadido a amenazas a la validez en la sesión 9.**
3. **k = 2.** Paso 3 de D20, no ejecutado.

Referencia: `main (5).pdf`, 9 jul 2026, "Catastrophic Forgetting in World Model
Transition Components: A Benchmark and Structural Mitigation".

---

## 0bis. Dónde estamos de verdad (sesión 7, con los datos en la mano)

Lo de §0 se escribió sin resultados válidos. Ahora hay 225 celdas y la
valoración cambia en un punto importante, así que va primero.

**El paper ya no es "un benchmark que discrimina métodos". Es un hallazgo.**

> El olvido catastrófico en un world model **no escala con la distancia entre
> tareas, escala con lo que la tarea nueva exige del modelo**. Y ocurre casi
> todo en el codificador, que es justo el componente que las métricas al uso
> —las del propio paper— no miran.

Las dos mitades están medidas, en tres familias y cinco semillas:

1. **F27.** El olvido hace pico en el nivel *intermedio* de distancia en las
   tres familias. El eje `min/med/max` no acierta el orden en ninguna
   (Spearman **+0.05**). Lo que sí lo predice: la dificultad de la tarea B
   (**+0.58**) y `d_trans` (**+0.53**). Cifras de la política D15; con medias
   eran +0.13 / +0.62 / +0.57, o sea que el resultado no depende de eso.
2. **F18 + F21.** `finetuning` pierde la reconstrucción de la tarea A por un
   factor de ~790 mientras **PF sale negativo**; EWC conserva `M` casi
   perfectamente (PF −0.06) con el codificador igual de destruido, porque su
   Fisher es exactamente cero sobre el VAE. La predicción era estructural y se
   escribió antes de medirla.

Eso es más publicable que cualquier ranking de métodos, y es exactamente lo que
un banco de pruebas debería producir: **una corrección a lo que el campo mide**.

### Lo que ha mejorado desde §0

- Los cinco Findings del paper viejo siguen sin sobrevivir, pero ahora hay
  **hallazgos nuevos que los sustituyen**, no un hueco.
- Infraestructura: 385 tests, reproducibilidad bit a bit **verificada ocho días
  y tres familias después** (R17 reprodujo una celda con delta 0 al cuarto
  decimal), un subproceso por familia, preflight de configuración.
- Las cuatro decisiones que bloqueaban (D9–D12) están tomadas **e
  implementadas**, más D13–D18: ya no queda ninguna decisión abierta sobre lo
  que se publica.

### Lo que ha empeorado, y hay que mirarlo de frente

- **"Controlled dynamic distances" no se sostiene** tal cual. Es la afirmación
  del abstract y el eje del diseño. Hay que reescribirla como lo que la
  medición dice, no como se pretendía.
- **Tres de las nueve celdas son controles sin olvido** (D16, sesión 8):
  `gymnasium/distance_min`, `gymnasium/distance_med` y `dmcontrol/distance_min`.
  Defendible como control declarado, pero deja la **rejilla efectiva en 6 de 9**,
  y en Gymnasium el olvido a nivel de codificador solo aparece en el nivel
  máximo. Eran dos en la lista que había aquí: `gymnasium/distance_med` apareció
  al aplicar el criterio a las nueve, y es el caso interesante — no pierde nada
  en píxeles y tiene el RD más alto de su familia.
- **Las colas pesadas son frecuentes, no una casilla** (F23/P12): **22 de las 180
  celdas** de pf/rd/ft/wmf salen marcadas por sesgo a la derecha. Aplicada la
  política de mediana y rango (D15), la casilla de F23 ya se reporta.
- **PIS se retira** (D18, sesión 8). Se anunció como una de las cuatro métricas y
  nunca se implementó, así que el paper anuncia **tres**: PF, RD y FT. Encoge la
  contribución sobre el papel y es la salida honesta; con D10 el coste es
  pequeño, porque el titular es F27, que no es una métrica.

### Lo que priorizaría ahora

1. ~~**Escribir la sección de resultados alrededor de F27.**~~ Hecha
   (`paper/results.tex`). El hallazgo **es robusto al cambio de agregación**:
   con mediana el pico en `med` sigue saliendo 3 de 3 y la etiqueta baja a +0.05.
   Escrita también la discusión. Lo que queda: introducción, trabajo
   relacionado, método y abstract — este último con la enmienda a «controlled
   dynamic distances» ya redactada en §6.1 de la discusión.
2. ~~**Aplicar la política de reporte de P12 y decidir P13.**~~ Hecho en la
   sesión 8: D15 y D16.
3. ~~**Decidir qué se hace con PIS.**~~ Hecho en la sesión 8: D18, se retira. La
   suite que anuncia el paper son PF, RD y FT.
4. **k>2 en una familia**, si sobra tiempo. Convierte "un cambio de tarea" en
   "una secuencia" y el arreglo de F5 ya lo permite.

### Lo que sigue sin poder decirse

Que UG-MTM funciona. Los datos lo caracterizan bien —no olvida porque congela el
codificador, y lo paga no aprendiendo la tarea B (FT −556 en MiniGrid), salvo
cuando las tareas se parecen visualmente (Gymnasium, donde el coste casi
desaparece)— y esa caracterización es material publicable. El titular no.

---

## 0. Valoración honesta: ¿tiene futuro esto? · *escrita en la sesión 6, sin resultados válidos*

Sin adornos, porque una valoración amable no sirve de nada aquí.

### Sí, tiene futuro — pero no como el paper que es ahora

**Lo que es sólido de verdad:**

1. **El hueco existe y sigue abierto.** No hay benchmark que mida olvido
   catastrófico a nivel del componente de transición `M`. Continual World mide
   políticas; Kessler et al. estudian DreamerV2 como sistema integrado. Esta
   afirmación del paper resiste, y es la base de todo.
2. **La infraestructura ya es de verdad.** 309 tests, reproducibilidad bit a bit
   verificada entre procesos, la KL correcta y contrastada contra torch, el VAE
   aprendiendo, el protocolo leído del config y guardado en cada resultado. Eso es exactamente el sustrato que un paper de benchmark
   necesita, y es lo que más cuesta construir. Hoy está.
3. **Hay resultados matizados y honestos.** El enrutamiento por incertidumbre
   funciona a distancia grande (AUC 0.864) y **se invierte** a distancia media
   (0.294). Eso es un hallazgo real sobre la premisa de UG-MTM, y es material
   publicable — de hecho es más interesante que "funciona siempre".

4. **Hay un resultado estructural nuevo, y es de los buenos** (sesión 6, F18).
   Al instrumentar la calidad en la tarea A apareció una disociación que nadie
   había medido: `finetuning` pierde la reconstrucción de la tarea A por un
   **factor 112** mientras **PF sale negativo**. Las métricas del paper operan
   sobre latentes congelados y no ven al codificador. Eso convierte "¿dónde
   ocurre el olvido en un world model?" en una pregunta empírica con respuesta
   medible, y es exactamente el tipo de cosa que justifica que exista un
   benchmark a nivel de componente. Bien contado, es un argumento a favor del
   paper, no en contra.

**Lo que está genuinamente débil, por orden de gravedad:**

1. **La escala. Este es el riesgo número uno, con diferencia.**
   20 rollouts de política aleatoria, 1000 pasos de gradiente, batch 8,
   `seq_len` 5. Cualquier revisor preguntará lo obvio: ¿el modelo había aprendido
   la tarea A? Porque si no, PF y RD miden la distancia entre dos modelos malos,
   y llamar a eso olvido catastrófico no se sostiene.
   **Actualización (sesión 6): esta parte ya tiene respuesta.** La tarea A se
   aprende y ahora se mide y se guarda: reconstrucción de entrenamiento 931.8 →
   8.2, y **6.49 sobre datos reservados** — 5.3e-04 por píxel, RMSE ≈ 0.023 en
   `[0,1]`. Más la prueba de escalado de R11. Lo que sigue abierto es la decisión
   de presupuesto para la ejecución completa, que es distinto de no tener
   evidencia.
2. **k = 2.** El formalismo del paper es `T_1 … T_k` y las métricas se definen
   como `PF(i,k)`, pero solo se ejecuta un cambio de tarea. Los benchmarks de
   continual learning suelen ir a 10–20 tareas. Que F5 (Progressive Nets rompía
   con 3+ columnas) llevara meses sin detectarse es la prueba de que nunca se
   ejecutó `k>2`. Con dos tareas, "secuencial" es generoso.
3. **Dos de las cuatro métricas tienen problemas de definición, no de
   implementación.** PIS nunca se implementó (abajo), y FT no mide transferencia
   (F20): se calcula sobre el modelo *previo* al cambio de tarea y sobre datos de
   la tarea A, así que es method-blind por construcción. De las cuatro métricas
   anunciadas, las que discriminan métodos son **dos**: PF y RD. Y RD se lleva el
   78–97% del agregado (punto 4). Merece la pena decirlo junto: el paper anuncia
   una suite de cuatro métricas y lo que realmente discrimina es, sobre todo, RD.

4. **PIS nunca se implementó.** El WMF se presenta como agregado de tres
   métricas y una vale 0 constante. O se implementa —lo que exige un controlador
   que no existe: entrenar una política en la imaginación del modelo y evaluarla
   en el entorno real, que es trabajo serio— o se redefine. Redefinir es honesto
   pero encoge la contribución: "proponemos PF y RD" es menos que "proponemos una
   suite de tres métricas".
5. **RD se lleva el 78–97% del WMF.** Incluso después de arreglar la KL. Si la
   respuesta a P7 es "reportarlas por separado", el paper pierde la cifra única —
   que es buena parte de lo que hace que un benchmark se adopte.
6. **6 de las 9 celdas no tienen distancia cuantificada.** `d_trans` no se
   calcula (F15), así que el eje de "controlled dynamic distance" —el principal
   argumento de diseño más allá de "medimos `M`"— solo es real en Gymnasium. Y
   en DMControl el nivel `min` compara `cheetah/run` consigo mismo (F8):
   distancia literalmente cero etiquetada como mínima.
7. **UG-MTM ya no vende nada.** Su reclamo era `WMF ≈ 0` por aislamiento
   estructural; eso era un modelo congelado que apenas se movía (F3). Ahora es
   mediocre a distancia media y el peor a máxima (`RD = 167` frente a 14–32).
   Como contribución de método es flojo. La decisión D8 (la tesis es el
   benchmark) fue acertada, pero implica que el paper tiene **una** contribución,
   no dos.
   **Y ahora está cuantificado** (R10): su reconstrucción de la tarea A no se
   mueve ni un bit tras la tarea B (7.663551597595215 en los dos lados) porque
   congela el VAE, pero su reconstrucción al final de la tarea B es **533.19**
   frente a **19.66** de `finetuning`. **No olvida porque no aprende.** Esa frase,
   con esos dos números al lado, es lo que hay que escribir.

8. **Las métricas no ven el olvido donde ocurre** (F18, sesión 6). Es el otro
   lado de la moneda del punto 4 de arriba: bien contado es un hallazgo, mal
   contado es una crítica letal. El paper tiene que declarar el alcance —
   "medimos `M` en una base latente fija" — y reportar la degradación en píxeles
   al lado de PF y RD. Si no lo hace el paper, lo hará un revisor.
9. **Cero resultados ahora mismo.** Todo lo anterior está invalidado y la
   ejecución completa no se ha hecho.

### Dónde encaja realistamente

- **Workshop** (continual learning o world models, en NeurIPS/ICML/ICLR):
  **viable de verdad**, y creo que es el objetivo correcto para la siguiente
  iteración. El hueco es real, el tooling es sólido y reproducible, y los
  hallazgos matizados son justo lo que un workshop valora. Con las 225 corridas
  hechas a escala decente y P7/P8 decididos, hay paper.
- **Conferencia principal, track de benchmarks/datasets:** no con el alcance
  actual. Ahí un benchmark entra cuando es adoptable: escala creíble, más
  tareas, baselines a un tamaño que la gente reconozca, `pip install`, CI,
  resultados que inviten a competir. Eso es otra vuelta de trabajo y bastante
  cómputo — no imposible, pero no es donde está el proyecto hoy.
- **Journal / preprint como registro sólido:** perfectamente defendible ya, y
  tiene valor: el análisis de por qué la versión anterior estaba mal es en sí
  mismo instructivo.

### Lo que yo priorizaría si el objetivo es que esto se publique

Actualizado tras la sesión 6. Los dos primeros de la lista anterior están hechos:
la convergencia en la tarea A se demuestra (F17 + R11) y el protocolo se reporta
de verdad (I1).

1. **Decidir qué mide el benchmark**, que ahora son tres preguntas y no dos:
   P9/F18 (la ceguera al codificador), P7/F14 (la agregación) y F20 (qué se hace
   con FT). Las tres deciden qué *afirma* el paper y ninguna necesita cómputo.
2. **Subir la escala** hasta donde llegue el presupuesto. R11 dice que 5× es el
   punto dulce y cuánto cuesta.
3. **Decidir P8** (`d_trans`). Ojo: la medición fiel de la Ec. 9 y una medición
   honesta de FT necesitan **el mismo** modelo entrenado en B desde cero. Si se
   hace una, la otra sale casi gratis.
4. **Una familia con k>2**, aunque solo sea una. Convierte "un cambio de tarea"
   en "una secuencia" y usa el arreglo de F5 que ya está hecho.
5. Ejecutar, y regenerar tabla y figura desde los `metrics.json` con
   `summarize_results.py`.

### Lo que no diría en el paper

Que UG-MTM funciona. Los datos actuales no lo sostienen, y forzarlo es lo que
llevó a la versión anterior a tener cinco Findings de los que ninguno se
sostiene. La versión honesta —"aquí hay un banco de pruebas, esto es lo que
discrimina, y el método que propusimos funciona solo en un régimen"— es más
débil como titular y mucho más fuerte como paper.

---

## 1. Por qué hay que rehacerlo, no parchearlo

Los números de la Tabla 3, de §5.2 y de la Figura 1 se produjeron con:

- el posterior del VAE colapsado (F0) — el modelo de transición recibía un
  latente **constante**, así que no había dinámica que aprender ni que olvidar;
- la evaluación sobre ruido gaussiano en vez de datos de la tarea A (F1);
- el objetivo de la NLL equivocado (F2);
- la KL de RD mal formulada (F13), que inflaba RD ~12×;
- entornos sin sembrar (F16), así que las 5 semillas no eran 5 réplicas.

No es cuestión de corregir cifras: **las cinco filas de la Tabla 3 miden ruido
alrededor de un modelo degenerado**. Los cinco métodos, las tres familias, las
tres distancias, las 225 corridas.

---

## 2. Qué se salva del paper actual

| Elemento | Estado |
| --- | --- |
| Motivación e introducción (§1) | Se salva |
| Background y trabajo relacionado (§2) | Se salva |
| Hueco identificado (no hay benchmark a nivel de `M`) | Se salva — sigue siendo cierto |
| Definiciones de PF y RD (§3.2) | Se salvan como definiciones, con el alcance declarado (F18) |
| Definición de FT (§3.2) | **Rehacer o renombrar** — no mide transferencia (F20) |
| Definición de PIS (§3.2) | **Tirar** — se retira de la suite (D18); nunca se implementó |
| Fórmula de agregación WMF (Ec. 6) | **Rehacer** — F14/P7 |
| `d_param` (Ec. 8) y sus valores | Se salva — reproduce exacto (0.283/0.586/0.622) |
| `d_trans` (Ec. 9) | **Rehacer** — nadie la calcula (F15/P8) |
| Familias de entornos (§3.4) | Se salva salvo DMControl `distance_min` (F8) |
| Tabla 1 (protocolo) | **Rehacer** — no describe lo ejecutado (I1) |
| Arquitectura UG-MTM (§4.2) | Se salva la idea; revisar el umbral (P1) |
| Tabla 2 (hiperparámetros) | Se salva — coincide con el código |
| Tabla 3 (resultados) | **Tirar** |
| Figura 1 | **Tirar** |
| Findings 1–5 (§5.4) | **Tirar todos** |
| §5.5 (limitaciones de UG-MTM) | **Tirar** — el FT negativo tenía otra causa (F3) |
| Discusión (§6) | Reescribir en función de los resultados nuevos |

### Nuevo: lo que hay que **añadir** y no estaba

- Evidencia de convergencia en la tarea A (F17). Sin esto el resultado principal
  es atacable. **Ya se mide y se guarda** en cada `metrics.json`; la figura de
  apéndice sale de `experiments/convergence_A.py` (R11).
- **Calidad en la tarea A al lado de las métricas de olvido**, en píxeles y en
  latente. Un benchmark de olvido tiene que demostrar que había algo que olvidar,
  y la disociación entre las dos escalas es un resultado en sí (F18).
- **Declaración de alcance de las métricas** (F18): PF y RD miden `M` en una base
  latente congelada y no ven la deriva del codificador. Decirlo antes de que lo
  diga un revisor.
- Sección de reproducibilidad. Ya redactada en el README del repo; hay que
  llevarla al paper.
- El régimen en que el enrutamiento por incertidumbre falla (F3, P6). Es un
  resultado, no una limitación a esconder.
- **"No olvida porque no aprende"** para UG-MTM, con las dos columnas de R10.

---

## 3. Afirmaciones del paper a re-verificar desde cero

Ninguna se sostiene con los datos actuales:

- "catastrophic forgetting is measurable in world model transition components
  across all three families"
- "forgetting magnitude scales with observational complexity" — además es
  internamente contradictoria (ver `paper-vs-code.md`)
- "structural isolation (UG-MTM) and regularization (EWC) both suppress
  forgetting to near zero" — el ≈0 de UG-MTM era un artefacto de F3
- "Replay is insufficient in low-capacity settings" — era el resultado más sólido
  del paper (p<0.001). **Medido con 5 semillas y se invierte** (R12):
  `replay_infinite` olvida menos que `finetuning` en las 10 comparaciones
  emparejadas, en WMF, PF y RD, en los dos niveles. `p` en su suelo de 0.0625 y
  `d_z` entre −1.66 y −3.07. **Ninguno de los cinco Findings sobrevive.**
  Consuelo: la dirección nueva es la esperable a priori — replay entrena sobre
  A+B. La afirmación llamativa era la vieja, y era un artefacto.
- "Progressive Networks exhibit forward transfer" — no era significativo (n=5,
  p entre 0.23 y 0.63), y ahora hay además un motivo **estructural**: FT no puede
  distinguir métodos que compartan arquitectura, porque no entra ningún dato de la
  tarea B en su cálculo (F20). Delta exactamente 0.000 en 10/10 semillas entre
  finetuning y replay.
- UG-MTM `FT ≈ -0.5 a -0.8` — el rango real era otro, y la causa también

---

## 4. El riesgo de tesis, ya resuelto

Se temía que con la medición corregida UG-MTM dejara de ganar. **Se confirma:**
en R9 es mediocre a distancia media y el peor a máxima.

Y hay un segundo riesgo del mismo tipo, ya materializado: **el Finding 4 se
invierte** con 5 semillas (R12). Era el único de los cinco que resistía. Con D8
en la mano no es un problema para la tesis —el benchmark discrimina, que es lo que
tiene que hacer— pero conviene tenerlo asumido antes de escribir: **de los cinco
Findings del paper anterior no sobrevive ninguno**, y el que parecía más sólido
resultó ser el más llamativo justamente porque era un artefacto.

Eso ya no es un problema, porque D8 fijó que la tesis es el benchmark. La
contribución se sostiene sola — es la que el propio paper presenta como
contribución nº 1. Pero conviene tenerlo asumido **antes** de mirar los
resultados de las 225 corridas, para no sesgar la interpretación.
