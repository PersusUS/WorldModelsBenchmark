# Bitácora de desarrollo — WMF Benchmark

Carpeta interna de trabajo. **No se publica** (está en `.gitignore`).
Registra todo lo hecho, lo encontrado y lo pendiente mientras preparamos el
código y los resultados para reescribir el paper.

## Índice

| Archivo | Contenido |
| --- | --- |
| [HANDOFF.md](HANDOFF.md) | **Empieza por aquí.** Documento autocontenido para retomar el proyecto |
| [findings.md](findings.md) | Los 30 problemas encontrados (F0–F29), con evidencia y estado |
| [changelog.md](changelog.md) | Cambios de código, archivo por archivo |
| [improvements.md](improvements.md) | Deuda técnica y mejoras (I1–I22) |
| [paper-vs-code.md](paper-vs-code.md) | Discrepancias entre el PDF viejo y el código/datos |
| [paper-plan.md](paper-plan.md) | **Valoración honesta de viabilidad**, qué se salvó y qué se tiró |
| [runs.md](runs.md) | Ejecuciones R0–R19, con sus números |
| [decisions.md](decisions.md) | Decisiones tomadas (D1–D21) y las pendientes (P1, P5, P6, P11) |
| [check-paper.py](check-paper.py) | Comprobación estructural de `paper/**/*.tex` sin LaTeX |
| [check-numbers.py](check-numbers.py) | Cada cifra de la prosa y si una tabla generada la respalda |
| [run-seeds-5-9.sh](run-seeds-5-9.sh) | La corrida de diez semillas (D21). Reanudable |

## Estado actual

**Fase: el paper está escrito y verificado.** 375 celdas commiteadas (diez
semillas en las seis celdas que discriminan, cinco en los tres controles), más
el sondeo de 2× y la secuencia k=4. Las nueve secciones y las diez tablas están
en `paper/`, las tablas generadas por `experiments/export_tables.py`.

**Sesión 10:** verificadas a mano las 74 cifras de la prosa que ninguna tabla
respalda. Cuatro estaban mal y están corregidas (cuota de RD, la comparación de
presupuestos que mezclaba cinco semillas con diez, y un 15× que era 14.8×). Lo
único que queda del paper es **compilarlo**: no hay toolchain de LaTeX aquí.

**Antes:** Corregidas la causa raíz (F0), la KL (F13), la
reproducibilidad (F16), instrumentada la convergencia (F17) y el protocolo leído
del config (I1). En la sesión 7 se tomaron **y se implementaron** las cuatro
decisiones que definían qué publica el benchmark (D9–D12): alcance declarado, PF
y RD por separado, `d_trans` y FT medidas de verdad, y presupuesto 5×.
**No queda ninguna decisión bloqueando la corrida completa.**

Corregido:

- [x] Suite de tests desde cero (432 tests, todos pasan)
- [x] **F0 — colapso del posterior del VAE** (0/32 dims activas → 32/32)
- [x] F1 — evaluación sobre rollouts reales de la tarea A en vez de ruido
- [x] F2 — `compute_nll` puntúa contra `z'` en vez de `z`
- [x] F4 — reproducibilidad en `ExpertPool` (RNG global)
- [x] F5 — Progressive Nets con 3+ columnas
- [x] F7 — RD usa acciones de la distribución real
- [x] F12 — dependencias (`Pillow`, índice de wheels cu121)
- [x] **F13 — la KL de `compute_rd` y `d_trans`** (convención log-varianza +
      `torch.distributions.kl_divergence`; RD venía inflada ~12×)
- [x] **F16 — reproducibilidad** (era I5). La causa no era cuDNN sino los
      entornos sin sembrar: las tres familias entrenaban sobre datos distintos
      en cada corrida. `set_seed()` + `BaseEnv.seed()`. Verificado bit a bit en
      tres métodos (R8).
- [x] **F17 — instrumentada la calidad en la tarea A** (sesión 6). Se aprende:
      reconstrucción 931.8 → 8.2 en entrenamiento y **6.49 sobre datos
      reservados** (5.3e-04 por píxel). Más la prueba de escalado de R11. Toda
      medición nueva va dentro de `preserve_rng_state`, así que no movió ningún
      resultado: las 12 cifras de R9 se reproducen con delta < 5e-10.
- [x] **I1 — el protocolo se lee del config** (sesión 6). Una sola fuente de
      verdad, validada al leerla, impresa antes de entrenar y guardada en cada
      `metrics.json`. Resuelve F10 y la parte mecánica de P4.
- [x] I2 — los pasos con NaN se cuentan y se guardan (0 en R10)
- [x] I4 — `ThresholdNet._ptr` viaja en el `state_dict` (verificado que no mueve
      ningún resultado)
- [x] Gating de UG-MTM activo en modo eval
- [x] Limpieza para publicación (`.gitignore`, README, LICENSE)

Sesión 7 — decidido e implementado:

- [x] **I3 — el Fisher de EWC** era `(E[∇])²`. Con `E[g²]` su PF pasa de +1.33 a
      **+0.003** sin perder ajuste a la tarea B: la penalización estaba casi
      inerte (R13).
- [x] **F8 — DMControl `distance_min`** comparaba `cheetah/run` consigo mismo.
      Ahora es gravedad 9.81 → 7.0, el mismo cambio físico que el `distance_min`
      de Gymnasium.
- [x] **F15 / P8 — `d_trans` (Ec. 9)** se mide con un modelo por entorno
      entrenado desde cero. Las 9 casillas tienen distancia numérica.
- [x] **F20 / P10 — FT** se mide de verdad: `recon_B(desde cero) −
      recon_B(preentrenado)`. Ya discrimina (+54.7 vs −516.1 en el primer smoke),
      donde antes daba delta 0.000 por construcción.
- [x] **F18 / P9 — la ceguera al codificador** se declara como alcance y se
      reportan las dos escalas (D9).
- [x] **F14 / P7 — RD domina el WMF**: se publican PF y RD por separado; WMF pasa
      a agregado heredado con el reparto a la vista (D10).

Sesión 8 — cerrado lo que quedaba de reporte:

- [x] **F23 / P12 — RD de UG-MTM estalla** en `minigrid/distance_max`: 17.7,
      40.0, 520, 574 y 4364 según la semilla. Diagnosticado en R17 (colapso de
      varianza en el modelo post-B) y **reportable**: sale como
      `+520.4 [17.72, 4364]!` con la política de D15.
- [x] **F22 / P13 — celdas sin olvido**: son **tres** (`gymnasium/distance_min`,
      `gymnasium/distance_med`, `dmcontrol/distance_min`), con criterio relativo,
      y se declaran controles (D16). Rejilla efectiva 6 de 9.
- [x] **F6 / P2 — PIS** se retira de la suite (D18). El banco reporta PF, RD y
      FT, y `pis` se guarda como `null`, no como `0.0`.

Sesión 9 — escrito el paper y comprobado el hallazgo:

- [x] **Abstract, introducción, trabajo relacionado y método**, más `main.tex`.
      Nunca compilado: no hay toolchain aquí, `check-paper.py` cubre lo que puede.
- [x] **Tabla 1 y la tabla de las nueve casillas se generan** desde los
      `metrics.json`, no desde el YAML. Ocho tablas generadas en total.
- [x] **`check_runs_consistent` junto a `load_runs`**: el candado que solo
      protegía una tabla ahora lo heredan las ocho y el resumen de consola.
- [x] **R18 — F27 comprobado al doble de presupuesto.** Aguanta.

Sesión 10 — verificado el paper cifra a cifra:

- [x] **Las 74 cifras que ninguna tabla respalda, comprobadas a mano** contra
      `results/`, `results-2x/` y `runs.md`. **Cuatro estaban mal.**
- [x] **La cuota de RD** ya no dice «78–97%» (rango de una corrida vieja) sino
      la mediana real, 88.5%, con las excepciones nombradas.
- [x] **La comparación de presupuestos fija también las semillas**: 58.79, no
      60.18, y la razón entre celdas es 2.01.
- [x] Lección: `check-numbers.py` **clasifica, no verifica**. Las cuatro
      erratas estaban en su lista CHECK y ninguna se detecta sin recalcular.

Abiertos:

- [ ] **F27 — el olvido no sigue al eje de distancia.** Hallazgo principal.
      Robusto al cambio de agregación **y al doble de presupuesto** (R18).
      Escrito en `results.tex` §5.2–5.3 y `discussion.tex` §6.1–6.2.
- [ ] **F29 — `d_trans` no resuelve dentro de familia y se mueve con el
      presupuesto.** Declarado en el paper. Arreglos no ejecutados: más semillas,
      y la pareja de referencia con codificador compartido (que es también F28).
- [ ] **F28 — `d_trans` puntúa el modelo B en la base latente de A.** La
      objeción de F20 aplicada a la otra métrica. Declarado; no arreglado.
- [ ] **F21 — el Fisher de EWC es cero sobre el codificador y sobre
      `gru.weight_hh`.** Documentado y con tests; queda decidir si debe estimarse
      sobre secuencias.
- [ ] **F19 — las métricas de UG-MTM son estocásticas en evaluación** (MC-dropout
      activo en la puerta por diseño). Menor, pero es el único método cuyas
      cifras llevan ruido de medición.

**El Finding 4 se invierte, pero no en todas partes** (diez semillas): replay
olvida menos en cuatro de las seis celdas con olvido (p = 0.0020), más en una
(`gymnasium/distance_max`) y empata en otra. Era el único de los cinco Findings
del paper viejo que resistía. Ya no sobrevive ninguno.

Decisiones pendientes: ver [decisions.md](decisions.md) (P11, P1, P5, P6).
Ninguna afecta a lo que se publica.

Después:

- [x] **Ejecutadas las 225 corridas** a 5000 pasos (R16), con sus 45 parejas de
      referencia
- [x] **Reescrito el paper** — ver [paper-plan.md](paper-plan.md), que incluye la
      valoración honesta de si esto llega a publicarse y dónde
- [ ] **La pasada de números** cuando termine R19. No es opcional
- [ ] k>2 en al menos una familia (paso 3 de D20)
- [ ] F28: pareja de referencia con codificador compartido (paso 4 de D20)
- [ ] Faltan la figura de calidad-en-A y la de convergencia. Cero cómputo
- [ ] Compilar el paper la primera vez que haya una máquina con LaTeX

## El hallazgo que lo explica todo

`F.mse_loss(recon, obs, reduction="mean")` promediaba sobre los 12288 píxeles
mientras la KL se sumaba sobre las dimensiones latentes: reconstrucción
infraponderada ~12288×. El VAE colapsaba a **0 de 32 dimensiones activas** —
toda observación se codificaba al mismo latente constante.

Es decir: el modelo de transición nunca vio información sobre el entorno. Las
225 corridas medían ruido alrededor de un modelo degenerado, para los cinco
métodos. Corregido el escalado: 32/32 dimensiones activas, reconstrucción 30×
mejor, y la incertidumbre por fin discrimina tareas (AUC 0.86 a distancia
máxima, frente a 0.51 antes).

## Objetivo

Era: resultados correctos y reproducibles con los que reescribir el paper desde
cero. Hecho — las 225 celdas están ejecutadas y el paper está escrito alrededor
de ellas.

Ahora: **que el hallazgo aguante lo que un revisor le va a hacer.** Eso ya no es
escribir, es comprar resolución donde el paper es débil, en el orden de D20:
presupuesto (hecho, R18), semillas (corriendo, R19), longitud de secuencia
(k>2), y el arreglo de `d_trans` (F28/F29).
