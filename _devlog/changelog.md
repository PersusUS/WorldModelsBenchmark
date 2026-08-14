# Changelog de código

Orden cronológico. `F*` remite a [findings.md](findings.md).

---

## Sesión 1 — Tests, bugs y limpieza para publicación

### Añadido: suite de tests (no existía ninguna)

`cf_worldmodels/tests/` — 212 tests, todos pasan.

| Archivo | Cubre |
| --- | --- |
| `conftest.py` | Fixtures compartidas, dims pequeñas para velocidad |
| `test_vae.py` | ConvVAE: formas, recorte 80→64, rango sigmoide, train/eval, peso β |
| `test_rssm.py` | Contrato `BaseWorldModel`, arrastre de estado GRU, gradientes |
| `test_ug_mtm.py` | Expertos, MC-dropout, ThresholdNet, gating, congelación |
| `test_baselines.py` | 4 baselines parametrizados + Fisher/penalty de EWC, columnas |
| `test_metrics.py` | PF/RD/WMF/FT: identidad, antisimetría, validación de pesos |
| `test_distances.py` | `d_param` con valores conocidos, monotonía; `d_trans` |
| `test_protocol.py` | Construcción del dataset de evaluación |
| `test_buffer.py` | Desalojo, muestreo, formas |
| `test_checkpointing.py` | Esquema de checkpoint, roundtrip |
| `test_configs.py` | Consistencia config ↔ código |
| `test_envs.py` | 20 tests de integración contra simuladores reales |

`pytest.ini` — `pythonpath`, marcador `integration`.

### `src/models/ug_mtm.py` — F4

`ExpertPool.__init__`: `torch.Generator` local por experto en vez de
`torch.manual_seed` global. Verificado bit a bit contra la inicialización
original: no cambia ningún peso.

### `src/baselines/progressive_nets.py` — F5

`transition`: reproduce la cadena de columnas de abajo arriba. Comportamiento
idéntico para 1–2 columnas; deja de romper con 3+.

### `src/models/vae.py`

Docstring del decoder corregido: produce 80×80 y recorta a 64×64, no 64×64
directo.

### Limpieza para publicación

- `.gitignore` — excluye ~20 checkpoints: **713 MB → <1 MB**
- `README.md` — instalación, reproducción, tabla, métricas, limitaciones
- `LICENSE` — MIT
- Eliminados 7 runners redundantes (`run_all_*`, `run_dmcontrol_*`,
  `run_gymnasium_*`). Sin referencias colgando; `plot_final.py` sigue
  regenerando la figura.
- `requirements.txt` / `environment.yml` — F12: añadido `Pillow` y
  `--extra-index-url`; eliminados `torchvision`, `hydra-core`, `scipy`,
  `seaborn`, `einops` (verificado por grep que nadie los importa)

---

## Sesión 2 — Corrección de la medición

### `src/benchmark/protocol.py` — F1

Nuevo `build_latent_eval_dataset(model, buffer, device, n_transitions, seed)`.
Devuelve tripletas `(z, a, z')` desde rollouts reales. Dos propiedades
importantes, documentadas en el docstring:

1. Vienen de rollouts reales de la tarea i, no de ruido.
2. Se codifican **una sola vez**, por un solo modelo, para que todos los
   modelos evaluados vean entradas y objetivos idénticos.

### `src/benchmark/metrics.py` — F2, F7

- `compute_nll`: objetivo `next_obs` (= `z'`). `KeyError` explícito si falta.
- `compute_rd`: acciones remuestreadas del dataset en vez de `N(0, 0.1)`.

### `src/baselines/ewc.py` — F2

`consolidate`: la verosimilitud usa `next_obs` como objetivo. `KeyError`
explícito si falta.

### `src/models/ug_mtm.py` — gating en eval

- La rama de eval ya no fuerza `u_t = 0`; calcula la incertidumbre por
  MC-dropout.
- Nuevo `self.T_eval` (`mc_dropout_T_eval`, por defecto = `mc_dropout_T`).
- La historia de incertidumbre **no** se muta durante la evaluación.

### `configs/models/ug_mtm.yaml`

`mc_dropout_T_eval: 20`.

### Runners — F1

`run_full_benchmark.py`, `train_baseline.py`, `train_ug_mtm.py`:

- Buffer `buf_A_heldout` recogido aparte, nunca entra en entrenamiento.
- `model_i` se construye en el cambio de tarea y codifica `D_i`.
- EWC consolida sobre datos reales de la tarea A.

---

## Sesión 3 — Colapso del posterior (F0)

### `src/models/vae.py`

Nueva función pública:

```python
def reconstruction_loss(recon, target):
    """Squared error summed over pixels, averaged over the batch."""
    return F.mse_loss(recon, target, reduction="sum") / target.shape[0]
```

El docstring explica por qué `reduction="mean"` colapsa el posterior. Fuente
única de verdad para los cuatro modelos.

### Sustituido `F.mse_loss(..., reduction="mean")` por `reconstruction_loss()`

- `src/models/vae.py` — `ConvVAE.compute_loss`
- `src/models/rssm.py` — `RSSM.compute_loss`
- `src/models/ug_mtm.py` — `UG_MTM.compute_loss`
- `src/baselines/progressive_nets.py` — `ProgressiveNetWorldModel.compute_loss`

### Tests nuevos

- `test_reconstruction_loss_sums_over_pixels_and_averages_over_batch`
- `test_reconstruction_loss_is_not_the_pixelwise_mean`
- `test_reconstruction_term_dominates_kl_at_initialisation`
- `test_training_does_not_collapse_the_posterior` (marcado `slow`) — entrena
  300 pasos sobre 4 imágenes distintas y exige ≥4 dimensiones latentes activas

Marcador `slow` añadido a `pytest.ini`.

**216 tests pasan.**

### Pendiente de revisar tras el cambio de escalado

La pérdida de reconstrucción pasa de ~0.014 a ~170. En `UG_MTM.compute_loss`:

```python
total_loss = total_recon + self.beta_kl * total_kl + total_uncertainty
```

`total_uncertainty` queda ahora proporcionalmente mucho más pequeño. Habrá que
decidir si necesita un peso propio — ver P5 en `decisions.md`.

---

## Sesión 4 — Limpieza: eliminar lo inválido y lo que confunde

El repo pasa de **713 MB a 346 KB**. Nada se ha borrado sin copia: todo lo que
tenía algún valor está en `_devlog/archive/` (gitignored).

### Archivado en `_devlog/archive/`

| Destino | Qué | Por qué |
| --- | --- | --- |
| `results-R0/` | 226 `metrics.json` + figuras (233 ficheros) | Resultados inválidos (F0). Se conservan como registro histórico; el resumen ya está en `runs.md` |
| `docs/` | `START_HERE.md`, `CLAUDE.md`, `PHASES.md`, `RULES.md`, `SPECS.md` | Instrucciones para agente, no documentación. `SPECS.md` además contradice al código (arquitectura del decoder, gating) |
| `notes/` | `FIX_GRADIENT_BLENDING.md`, `NEXT_STEP.md` | Notas de trabajo. La primera documenta por qué existe `register_gradient_scaling_hooks` — tiene valor histórico |
| `scripts/` | `plot_results.py`, `train_single_task.py`, `run_ablations.py` | Superados o no funcionales |

### Borrado sin copia (ruido puro)

- `results/**/*.pt` — 20 checkpoints, ~713 MB, entrenados con el VAE colapsado.
  Sin valor alguno.
- `full_benchmark_output.txt` — 355 KB de stdout de las corridas inválidas.
- Todos los `__pycache__/`.

### Código muerto eliminado

- `ug_mtm.apply_gradient_masking()` — nunca se llamaba; `UG_MTM.transition` usa
  `register_gradient_scaling_hooks`. La sustitución fue deliberada y está
  documentada en la nota archivada `FIX_GRADIENT_BLENDING.md`.
- `UG_MTM.self.threshold_grad` — su único consumidor era la función anterior.
- Clave `threshold_grad` de `configs/models/ug_mtm.yaml` — ya no la lee nadie.

### Tests

- Sustituidos los 3 tests de `apply_gradient_masking` por 2 de
  `register_gradient_scaling_hooks`, que es el mecanismo real: verifican que el
  gradiente se escala por la puerta y que los hooks se pueden retirar.
- Nuevo `test_ug_mtm_config_has_no_keys_the_model_ignores`: falla si el YAML
  gana una clave que el modelo no lee. Evita que se repita lo de
  `threshold_grad`.

**216 tests pasan.**

### README

- Tabla de resultados inválida sustituida por una nota que explica por qué no
  hay resultados publicados todavía.
- Recuento de tests actualizado (194 → 216).
- Limitación 4 reescrita: las cinco ablaciones eran inertes, no tres.
- `docs/` eliminado del layout.

---

## Sesión 5 — F13: la KL de RD y d_trans

### `src/benchmark/metrics.py` — F13

Nueva función `diag_gaussian_kl(mu_p, log_var_p, mu_q, log_var_q)`: KL entre dos
gaussianas diagonales, sumada sobre la dimensión de características. Fija la
convención **log-varianza** — la que ya usaban `sample_stoch`, `compute_nll` y
los cuatro `compute_loss` — y delega en `torch.distributions.kl_divergence` en
vez de escribir la fórmula a mano.

`compute_rd`: sustituida la KL manual por la nueva función. Variables renombradas
`log_sigma_*` → `log_var_*` para que la convención se lea en el nombre.

### `src/benchmark/distances.py` — F13

`compute_d_trans` (Ec. 9 del paper): mismo cambio, importa `diag_gaussian_kl`.

### Efecto numérico

Con los valores de referencia de findings.md, la implementación nueva devuelve
**1.7841**, que es exactamente la KL exacta bajo log-varianza (la fórmula
antigua daba 1.7997, que no era ninguna de las dos lecturas posibles).

Sobre entradas del tamaño real (latente 32-dim, `log_sigma ~ N(0,1)`): antigua
**752.97** vs correcta **81.18**, un factor **9×**. RD venía inflada ~9× frente
a PF — agrava F14 pero no lo explica entero.

### Tests

- `test_metrics.py`: 4 nuevos — igualdad con la referencia de torch, valor
  analítico `KL(N(0,1) || N(0,e²))` que distingue log-varianza de log-desviación,
  identidad, no-negatividad y asimetría.
- `test_distances.py`: 1 nuevo — `d_trans` completo contra la KL de torch sobre
  el mismo dataset.

**221 tests pasan.**

### Re-medición y hallazgos nuevos (sesión 5)

Sin cambios de código. Dos corridas de diagnóstico, `runs.md` R6 y R7:

- **R6** reproduce R5 con la KL corregida. La explosión de RD desaparece
  (13708 → 196.9) y PF pasa de aportar ~1% del WMF a 3–25%. **F14 revisado**:
  se reduce mucho pero RD sigue con el 75–97% del agregado.
- **R7** aísla dos cosas que R6 no podía responder: la inflación real de la
  fórmula antigua sobre modelos entrenados (**~12×**, consistente entre corridas)
  y la reproducibilidad del pipeline.

Hallazgos nuevos, ambos en `findings.md`:

- **F16 (bloqueante)** — el pipeline **no es reproducible**: misma semilla, PF
  cambia de signo (+0.237 vs −0.210). Promovido desde I5, que estaba como deuda
  técnica. Pasa a ser lo primero de la Fase 1.
- **F15** — `d_trans` (Ec. 9) no la calcula ningún runner, solo los tests. Deja
  6 de las 9 celdas sin valor numérico de distancia. Y medirla como la define la
  Ec. 9 exige un modelo entrenado en B desde cero: +45 entrenamientos. Nueva
  decisión pendiente **P8**.

---

## Sesión 5 (cont.) — F16: reproducibilidad

### Diagnóstico primero

Antes de tocar código, medido de dónde venía el no-determinismo. **No era cuDNN**,
o no principalmente: las tres familias entrenaban sobre datos distintos en cada
corrida, porque `np.random.seed` no alcanza el RNG del entorno de Gymnasium, el
del `action_space`, ni el `RandomState` de la tarea de dm_control. Tabla por
familia en `findings.md`, F16.

### Nuevo: `src/utils/seeding.py`

`set_seed(seed, deterministic=True)` — `random`, `numpy`, `torch`,
`cuda.manual_seed_all`, `PYTHONHASHSEED`, `cudnn.deterministic = True`,
`cudnn.benchmark = False`. Verificado que basta para que el entrenamiento sea bit
a bit idéntico, así que `use_deterministic_algorithms` queda desactivado a
propósito: no aporta y lanza en ops sin implementación determinista.

### `src/envs/base_env.py`

Nuevo método abstracto `seed(seed)`. Documenta que es de una sola vez por
corrida, no por episodio, para que los `reset()` sucesivos den estados iniciales
distintos de forma determinista.

### `src/envs/minigrid_env.py`, `src/envs/gymnasium_env.py`

`seed()`: `self._env.reset(seed=seed)` + `self._env.action_space.seed(seed)`, que
es la vía propia de Gymnasium.

### `src/envs/dmcontrol_env.py`

`seed()`: resembra `self._env.task.random` en sitio — evita recargar el entorno y
su contexto de render. `sample_action` pasa a usar un `Generator` propio del
wrapper en vez del `np.random` global.

### `experiments/run_full_benchmark.py`, `train_baseline.py`, `train_ug_mtm.py`

Los cuatro puntos de sembrado (`torch.manual_seed` + `np.random.seed`) sustituidos
por `set_seed()`, más `env_A.seed(seed)` / `env_B.seed(seed + 1)`.

### Tests

`tests/test_seeding.py`, 13 nuevos: los tres RNG globales, los flags de cuDNN, el
opt-out de determinismo, entrenar dos veces y comparar **los pesos**, rollouts
reproducibles en las tres familias, y que sembrar no colapse todos los episodios
al mismo estado inicial (si lo hiciera, el conjunto reservado de la tarea A sería
una copia del de entrenamiento y se desharía la corrección de F1).

**234 tests pasan** (1 se salta en la suite completa por I20; pasa aislado).

### README

Nueva sección "Determinism" en Reproducing the results — punto 20 de la Fase 4.
Recuentos de tests actualizados (216 → 234, integración 20 → 25).

### Hallazgo lateral: I20

El contexto GLFW no sobrevive a `dm_control → abrir/cerrar MuJoCo → dm_control`.
Acotado por eliminación; **verificado que no afecta a `run_full_benchmark.py`**,
cuyo orden de familias (minigrid ×3 → gymnasium ×3 → dmcontrol ×3) pasa. No hay
riesgo para la ejecución de las 225 corridas.

### R9 — línea base nueva (sesión 5)

Sin cambios de código. Re-medición de F14 sobre el pipeline ya reproducible.
Resultado: **F14 es estructural.** Los porcentajes salen prácticamente iguales que
antes de sembrar los entornos (PF 2.8–22.4% vs 3–25%), así que no dependían de
unos datos concretos.

Dos cosas que R9 aporta además:

- **Reproducibilidad entre procesos.** Las tres celdas que R9 comparte con R8 dan
  valores idénticos, y fueron procesos distintos. Más fuerte que R8 solo.
- **Señal de alarma sobre el Finding 4.** `replay_infinite` sale con el WMF más
  bajo de los cinco métodos en los dos niveles, con PF negativo — lo contrario de
  lo que decía el único hallazgo del paper que resistió el escrutinio. Anotado en
  `decisions.md` y en el HANDOFF como lo primero que mirar en la ejecución
  completa.

### Cierre de sesión 5

- **F17 documentado** (nuevo): `final_A` se calcula en `run_full_benchmark.py`
  líneas 116 y 195 y se descarta. `metrics.json` guarda la reconstrucción al
  inicio de A y al final de B, y tira la única cifra que diría si el modelo
  aprendió la tarea A antes de medirle el olvido. Es el riesgo más serio para el
  paper y es barato de desarmar. Sin cambios de código todavía.
- **`paper-plan.md` reescrito** con una valoración honesta de viabilidad (§0):
  qué es sólido, qué está débil por orden de gravedad, y dónde encaja
  realistamente el paper.
- **`HANDOFF.md` reescrito** y reestructurado. Había crecido por parches; ahora
  arranca con "dónde estamos" y "qué hacer ahora", y el resto es referencia.
- Contadores actualizados en todos los documentos (216 → 234 tests, 15 → 18
  problemas, 19 → 20 mejoras, 7 → 8 decisiones pendientes).

---

## Sesión 6 — F17 (instrumentar la convergencia) e I1 (protocolo desde el config)

Ningún cambio en las métricas. Verificado: las 12 cifras de R9 se reproducen con
delta < 5e-10.

### Nuevo: `src/utils/seeding.py::preserve_rng_state()`

Gestor de contexto que restaura los cuatro RNG globales (`random`, `numpy`,
`torch`, `torch.cuda`) al salir. Es lo que permite instrumentar una corrida sin
cambiarla: la puerta de UG-MTM mantiene MC-dropout activo en evaluación por
diseño, así que cada `compute_nll` consume el flujo aleatorio. Sin el guardián,
medir la convergencia de la tarea A habría movido los resultados de UG-MTM.

### `src/benchmark/protocol.py` — F17

Nueva `evaluate_reconstruction(model, buffer, device, n_frames, seed, chunk_size)`:
error cuadrático por fotograma **en píxeles**, sobre fotogramas reservados, en
modo `eval` (z = mu, sin muestrear). Devuelve la misma cantidad que el componente
`reconstruction` de `compute_loss`, así que las dos se pueden leer juntas.

Es la **única señal de calidad comparable entre presupuestos de entrenamiento**:
la NLL latente se mide contra latentes que produce el propio modelo, y ese
objetivo se mueve mientras el codificador entrena. Los píxeles no se mueven.
Codifica por trozos (`chunk_size=64`) para acotar memoria. I10 sigue abierto: es
`build_latent_eval_dataset` la que codifica en un solo batch, no esta.

### `experiments/run_full_benchmark.py` — reescrito (I1, I2, F17)

- Constantes de módulo eliminadas. `resolve_protocol()` lee el bloque `protocol:`
  del config, castea, valida y devuelve un dict serializable que se guarda en
  cada `metrics.json`. Campo que falta = error, no default.
- `run_baseline` y `run_ug_mtm` unificadas en `run_cell` + `create_model` +
  `switch_task`. Eran casi idénticas y la instrumentación habría que haberla
  duplicado. Orden de llamadas preservado exactamente.
- Interfaz de línea de comandos: `--families/--methods/--distances/--seeds`,
  overrides explícitos de protocolo, `--results-dir`, `--dry-run`.
- `check_protocol_consistency()`: se niega a reutilizar cachés de otro protocolo.
- `train_task` devuelve un dict con curva submuestreada, primer y último paso
  **aceptado**, NaN descartados y actualizaciones efectivas.
- `metrics.json` gana: las tres NLL sobre `D_A`, reconstrucción inicial/final de
  A y de B, reconstrucción reservada en píxeles antes y después de la tarea B,
  dos curvas, contadores de NaN, y procedencia (`method`, `family`, `distance`,
  `seed`, `protocol`). Se eliminan las claves ambiguas
  `initial_reconstruction_loss` (era el inicio de A) y `final_reconstruction_loss`
  (era el final de B) en favor de nombres con sufijo `_A`/`_B`.

### Nuevo: `experiments/convergence_A.py` — la prueba de escalado de F17

Entrena **una** corrida de la tarea A al presupuesto mayor pedido y evalúa en
cada múltiplo de `n_train` por el camino. Como los lotes se extraen de un flujo
sembrado, el estado en el punto `m` es exactamente el de una corrida
independiente de `m × n_train` pasos: cuesta un 10× en vez de 1×+2×+5×+10×. Las
evaluaciones van dentro de `preserve_rng_state`, que es lo que hace válida esa
equivalencia. Verificado contra la celda real: la NLL y la reconstrucción del
punto 1× coinciden con las de R10 (misma semilla, misma familia).

### `configs/benchmark/{minigrid,gymnasium,dmcontrol}.yaml`

Bloque `protocol:` reescrito con lo que de verdad se ejecuta, con comentario por
campo. Añadidos los que el runner tenía escondidos (`learning_rate`,
`n_eval_*`, `n_fisher_transitions`, `n_recon_frames`, `rd_*`, `ewc_lambda`,
`mc_dropout_T_train`, `wmf_weights`, `curve_points`). Eliminado `eval_every`, que
no lo leía nadie.

### Tests

**291 pasan** (eran 234; 1 se salta por I20, como antes).

- `tests/test_run_full_benchmark.py` — nuevo, 30 tests: resolución del protocolo,
  overrides, validación, capacidad emparejada en los cinco métodos, consistencia
  config↔runner (incluido "ninguna clave que el runner ignore"), rechazo de
  cachés con otro protocolo, y un test que **falla si alguien vuelve a meter una
  constante de protocolo en el módulo**.
- `tests/test_seeding.py` — 5 nuevos: `preserve_rng_state` sobre los tres flujos,
  restauración ante excepción, y confirmación de que el dropout en modo `train`
  es un consumidor del flujo (el motivo de que el guardián exista).
- `tests/test_protocol.py` — 8 nuevos: `evaluate_reconstruction` en la misma
  escala que la pérdida de entrenamiento, independencia del `chunk_size`,
  determinismo, restauración del modo, tope de fotogramas, error claro con buffer
  vacío, y que **entrenar la baja**.

### Nuevo: `experiments/summarize_results.py`

Agrega los `metrics.json` y es de donde deben salir todas las tablas del paper
(punto 13 del handoff: nunca a mano). Imprime el protocolo compartido — y avisa
en grande si los resultados **no** comparten protocolo, porque entonces cualquier
media mezcla presupuestos —, las métricas de olvido con media ± desviación, las
columnas de calidad en la tarea A que añadió F17, y con `--compare A B` una
comparación **emparejada por semilla** entre dos métodos.

Sobre la estadística: reporta una **p de permutación exacta** (los 2^n cambios de
signo) y el tamaño de efecto emparejado `d_z`, no una t. Motivo: con n = 5 la p
exacta más pequeña posible es 2/2^5 = **0.0625**, así que *ninguna* comparación de
5 semillas puede dar p < 0.05 sin apoyarse enteramente en el supuesto de
normalidad. Es un dato relevante para el Finding 4 del paper, que reclamaba
p < 0.001 con n = 5, y un argumento concreto para subir el número de semillas.

Sin dependencias nuevas: la permutación exacta no necesita `scipy` (que está
instalado pero **no declarado** en `requirements.txt`, y usarlo repetiría F12).

De paso: las cabeceras de tabla abreviaban `minigrid` a `min`, así que una columna
se titulaba `min_med` — que se lee como "mínimo mediano". Ahora `mgrid_med`.

### Tests (cont.)

- `tests/test_summarize_results.py` — nuevo, 17 tests: carga, detección de
  protocolos mezclados (incluido que el orden de claves no cuente), selección de
  celda, emparejado por semilla y descarte de semillas sin pareja, y la
  estadística (suelo de la p a 2/2^n, dos colas, varianza cero, n = 1).

### `src/models/ug_mtm.py` — I4

`ThresholdNet._ptr` pasa a ser un buffer registrado, así que el cursor del buffer
circular viaja en el `state_dict` junto a su contenido. Verificado que no mueve
ningún resultado: la celda `ug_mtm / minigrid / distance_med / 999` reproduce las
cuatro métricas de R9/R10 con delta < 5e-11.

### Hallazgos nuevos

- **F18 (afecta a la tesis)** — PF y RD son **ciegas al olvido del codificador**.
  `compute_nll` nunca llama a `encode`: opera sobre latentes congelados. Medido:
  `finetuning` degrada la reconstrucción de la tarea A **×112** (6.49 → 725.27)
  mientras PF sale **negativo** (−1.78). UG-MTM sale 7.6636 → 7.6636 bit a bit
  porque congela el VAE, y el precio está en la otra columna: reconstrucción al
  final de la tarea B de **533.19** frente a **19.66** de `finetuning`.
  Cuantifica lo que F3 decía en cualitativo: **no olvida porque no aprende**.
- **F19 (menor)** — las métricas de UG-MTM son estocásticas en evaluación
  (MC-dropout activo por diseño en la puerta). Dos evaluaciones de lo mismo
  difieren en 1.4e-02 sobre un PF de 5.58. Es el único método cuyas métricas
  llevan ruido de medición.

### R12 — el Finding 4 se invierte, y sale F20

20 celdas (`finetuning` vs `replay_infinite`, MiniGrid, dos niveles, 5 semillas),
en `_devlog/archive/results-R12-finding4/`, fuera de `results/` para no
comprometer la decisión de presupuesto.

- **El Finding 4 se invierte.** Replay olvida **menos** que finetuning en las 10
  comparaciones emparejadas, en WMF, PF y RD. `p` en su suelo de 0.0625, `d_z`
  entre −1.66 y −3.07. Era el único de los cinco Findings del paper que resistía
  el escrutinio; ya no sobrevive ninguno.
- El mecanismo se ve en la columna de píxeles que añadió F17: el codificador de
  replay no se degrada (5.84 → 5.86) y el de finetuning se va ×127 (5.84 →
  739.81). La separación real entre métodos es **×126 mayor que la que ve PF**.
- **F20 (nuevo)** — FT no mide transferencia hacia delante. No entra ningún dato
  de la tarea B en su cálculo, así que es idéntico para todos los métodos que
  comparten arquitectura: delta **exactamente 0.000** en 10/10 semillas. Tumba el
  Finding 5 por vía estructural. Nueva decisión pendiente **P10**.

---

## Sesión 7 — Las decisiones del paper, ejecutadas (D9–D12) + I3, F8, F15, F20

Sesión sin bugs que arreglar en la infraestructura: lo que se cierra aquí son las
tres decisiones que definían **qué publica el benchmark**, más los dos arreglos
que ya no dependían de nadie.

### `src/baselines/ewc.py` — I3, el Fisher

Gradientes **por muestra**: una pasada hacia atrás por transición, `E[g²]` en vez
de `(E[g])²`. Con `n_fisher_transitions = 50` son 50 backward sobre una GRU
pequeña. Se quita el `train()` intermedio: no hay dropout en esta ruta y el
modelo se queda en `eval()` de principio a fin.

Efecto medido (R13): el PF de EWC pasa de +1.33 a **+0.003**, sin perder ajuste a
la tarea B. `finetuning` no se mueve ni 1e-11. Ver también **F21**: el Fisher es
exactamente 0 sobre el VAE y sobre `gru.weight_hh`, y ahora hay un test que lo
dice con su porqué.

### `src/benchmark/metrics.py` — F20

`compute_ft` desaparece. En su lugar:

- `compute_task_A_fit_gain(nll_after_A, nll_random)` — la cifra vieja con el
  nombre de lo que mide.
- `compute_forward_transfer(error_desde_cero, error_preentrenado)` —
  transferencia hacia delante de verdad, en píxeles sobre la tarea B.

### `experiments/run_full_benchmark.py` — D11, el grueso

- `collect_cell_buffers()` — las cuatro recogidas (A, B, A reservada, **B
  reservada**) en un orden fijo, compartidas por las celdas y por la referencia.
  Las tres familias muestrean acciones de un RNG propio del entorno, así que dos
  pasadas con la misma semilla ven los mismos episodios sin depender del flujo
  global.
- `run_reference_cell()` — entrena **una pareja de RSSM planos por
  `(familia, distancia, semilla)`**, uno por entorno, y guarda `d_trans` (Ec. 9)
  y la reconstrucción reservada de B desde cero. Se cachea en
  `results/_reference/` y `load_reference()` la rechaza si viene de otro
  protocolo.
- `run_cell()` recibe la referencia y calcula `ft` y `d_trans`; la evaluación
  nueva de la tarea B va dentro de `preserve_rng_state`, así que no desplaza el
  flujo aleatorio de nada anterior.
- Bandera `--skip-reference`; sin ella la referencia forma parte de la corrida.
  Las celdas hechas sin ella guardan `ft` y `d_trans` como `null`, nunca 0.
- La tabla final imprime PF y RD, no WMF (D10).

### `experiments/summarize_results.py` — D9 y D10

- Encabeza con **PF, RD y FT**. WMF baja a una sección rotulada *legacy
  aggregate*, con una columna nueva: **qué fracción de |WMF| aporta RD**.
- Sección de **calidad en píxeles** con la columna nueva de la tarea B, precedida
  del alcance: PF y RD miden `M` en una base latente fija y no ven la deriva del
  codificador.
- Tabla de `d_trans` por casilla.
- `cell_values` trata un `null` como ausente: promediar un `null` como 0
  inventaría una medición.

### `src/envs/dmcontrol_env.py` y `configs/benchmark/dmcontrol.yaml` — F8

`DMControlEnv` acepta `physics_params` con las mismas tres perturbaciones que
Gymnasium, y **rechaza** las claves que no implementa. `distance_min` pasa de
`cheetah/run` contra sí mismo a `cheetah/run` con gravedad 9.81 → 7.0.

### `configs/benchmark/*.yaml` — D12

`n_train: 1000 → 5000` en las tres familias.

### `metrics.json` guarda **qué tareas** ejecutó (F26)

Guardaba el protocolo pero no el par de tareas, así que cambiar un par en el
YAML dejaba celdas viejas que el runner reutilizaba bajo el nombre del par
nuevo. Ahora cada resultado y cada referencia llevan su bloque `tasks`, y
`check_protocol_consistency` y `load_reference` lo validan igual que el
protocolo. Los 240 resultados de R16 anteriores al cambio se rellenaron con
`_devlog/backfill-tasks.py`, que documenta por qué es legítimo hacerlo (el único
par cambiado desde el lanzamiento está archivado, verificable con `git log`) y
se niega a tocar precisamente ese.

`configs/benchmark/dmcontrol.yaml`: `distance_max` pasa a ser
`cheetah/run → walker/run` **con gravedad 4.0, masa ×3 y fricción ×0.5** (D14).

### `tests/test_envs.py` — los dos entornos de un par tienen que producir datos distintos (F26)

El test que había comparaba los **diccionarios** de configuración, y
`walker/run` contra `walker/stand` son diccionarios distintos que producen el
mismo entorno. El nuevo compara **las observaciones**: misma semilla, mismas
acciones, y los dos entornos tienen que divergir. Cubre las tres familias, y en
dmcontrol se salta dentro de la suite completa por F24 (pasa aislado).

### `experiments/run_full_benchmark.py` — comprobar los anchos de acción antes de entrenar (F25)

`preflight_action_dims()` valida **las dos** tareas de cada par contra el
`action_dim` declarado, antes de que se entrene nada, y lista todos los pares
rotos de una vez. Se validaba solo `task_A`, así que
`cheetah/run → reacher/easy` (6 actuadores contra 2) reventaba dentro de la GRU
doce horas después de arrancar la familia. Ahora falla en segundos.

`configs/benchmark/dmcontrol.yaml`: el nivel máximo pasa a
`cheetah/run → walker/stand` (D13), con el porqué escrito en el propio YAML.

### `experiments/run_full_benchmark.py` — un subproceso por familia (F24)

La corrida completa murió a las 30 horas al pasar de gymnasium a dmcontrol:
`mujoco.FatalError: Default framebuffer is not complete`. Los dos simuladores
quieren un contexto OpenGL y el segundo en pedirlo dentro del mismo proceso no lo
consigue; `close()` no lo libera, salir del proceso sí.

`main()` detecta que se le piden varias familias y lanza **un intérprete por
familia**, reenviando todas las banderas; `--dry-run` se queda en un proceso
porque solo imprime. Un fallo en una familia deja de costar la corrida entera.

El riesgo del arreglo es el de F9 —montar una línea de comandos y no pasarla
completa—, así que el test la **parsea de vuelta** y compara campo por campo,
incluidos los overrides que deben seguir siendo `None`.

### `experiments/run_full_benchmark.py` — una sola recogida por casilla

El bucle pasa a anidarse **por semilla** en vez de por método, y
`collect_cell_buffers()` siembra los entornos y recoge los cuatro buffers **una
vez**; los comparten los cinco métodos y la pareja de referencia.

Antes cada uno de los seis re-sembraba y volvía a recoger **los mismos**
episodios: son idénticos por construcción, porque los entornos se siembran por
celda y la recogida no consume el flujo global. Medido, es la mayor partida de
tiempo de las familias que renderizan: 60 episodios × 2.7 s en Gymnasium, seis
veces, son 13.5 min tirados por casilla y semilla.

Numéricamente inerte, verificado dos veces: `finetuning` sigue reproduciendo R10
con delta < 1.5e-11 y las cifras de EWC coinciden con delta **exactamente 0**.

### `experiments/plot_final.py` — reescrito (D10)

Era un script de arriba abajo que dibujaba **una fila de WMF** y volvía a
imprimir la tabla de resultados. Ahora:

- **Una fila por métrica reportada (PF y RD)**, una columna por familia. No hay
  panel de WMF a propósito: con PIS a cero, un panel de WMF es un panel de RD con
  otra etiqueta.
- **El eje X es la distancia medida.** `d_trans` existe ahora en las tres
  familias, así que se usa donde las corridas la traen; si a una familia le falta
  en algún nivel, se cae entero a Min/Med/Max y **el rótulo del eje lo dice** —
  mezclar dos escalas en el mismo eje sería peor que perder la información.
- `argparse` con `--results-dir`, `--out` y `--stem` (antes todo hardcodeado, y
  el nombre del fichero era `figurasfinalv2`, uno de los tres nombres de I13).
- Deja de imprimir tablas: las tablas salen de `summarize_results.py`.

### Tests: 309 → 341

Nuevos: Fisher por muestra reconstruido desde cero · regresión de I3 (la varianza
del gradiente sobrevive) · los dos ceros del Fisher (F21) · caché de la
referencia y rechazo por protocolo · `--skip-reference` · transferencia hacia
delante · `task_A_fit_gain` · `null` tratado como ausente · reparto de |WMF| ·
ninguna casilla empareja una tarea consigo misma (las tres familias) · dm_control
aplica y valida su bloque de física · `test_plot_final.py` entero (10): qué se
dibuja, el eje X y su fallback, que una columna parcial no mezcle escalas, y que
se escriban los dos formatos · que la misma semilla recoja los mismos episodios
y otra semilla no · que los episodios reservados no sean los de entrenamiento ·
que la recogida no toque los flujos aleatorios globales · un subproceso por
familia, la familia que falla corta la corrida, `--dry-run` no lo hace, y la
línea de comandos del hijo reparseada campo por campo (6) · el preflight de
anchos de acción: par que discrepa rechazado, todos los pares rotos reportados
juntos, y los entornos cerrados al comprobarlos (3).

### `_devlog/diagnose-p12.py` — herramienta de diagnóstico (F23)

Fuera del repo, en la bitácora, porque es de un solo uso. Reproduce una celda
entera —mismo protocolo, misma semilla, mismos datos— parcheando `compute_rd`
para capturar la KL descompuesta paso a paso dentro de `preserve_rng_state`, y
**comprueba contra el `metrics.json` guardado** antes de interpretar nada: si la
reproducción no coincide, el diagnóstico no es sobre el objeto real. Coincidió
con las dos semillas hasta el cuarto decimal.

El patrón vale para cualquier diagnóstico futuro que necesite modelos entrenados,
ya que los checkpoints no se guardan (D3).


---

## Sesión 8 — La política de reporte, ejecutada (D15–D17: P12 y P13 cerrados)

Sin cómputo nuevo: las 225 celdas son las mismas, cambia cómo se resumen.

### `experiments/summarize_results.py` — de media ± desviación a mediana y rango

- **`is_right_skewed(values, factor=5)`** — criterio de sesgo sobre estadísticos
  de orden: la mitad superior se estira `factor` veces más que la inferior. No
  sobre media/mediana ni coeficiente de variación, que se disparan en celdas que
  cruzan el cero (el PF de EWC: −0.005 con dispersión 0.083 y simétrico).
- **`cell_summary(values, flag_skew=True)`** — `mediana [min, max]` con `!` si
  hay sesgo. Cuatro cifras significativas, porque la misma columna tiene que
  sostener −0.062 y 4364. `flag_skew=False` en las columnas de píxeles: ahí la
  magnitud está acotada y marcar el 25–35 de una línea base sería señalarlo todo.
- **`print_metric_table`** lista debajo cada celda marcada con sus cinco semillas
  y su media, para que la cola se lea como resultado y no como estorbo.
- **`spearman` y `rank`** escritas a mano (empates promediados). `scipy` está
  instalado pero no declarado: importarlo sería repetir F12.
- **`axis_value` y `print_distance_axis_section`** — el bloque de F27: RD por
  familia y nivel como mediana de medianas, dónde hace pico, y las tres
  correlaciones de rangos. La regla de inclusión de métodos vive en
  `AXIS_METHODS`, se imprime en la cabecera y se mueve con `--axis-methods`.
- **`task_a_loss`, `control_cells(threshold=0.10)` y `print_control_section`** —
  el bloque de P13: pérdida de la tarea A **relativa** a lo que el modelo tenía,
  por celda y como rango sobre métodos. Imprime números, no un veredicto, porque
  una celda puede no perder nada en píxeles y mover RD igual (F18).
- `print_distance_table` da ahora mediana y rango **junto a** la media y la
  desviación: lo que `d_trans` tiene que sostener es si dos niveles se separan, y
  eso es una afirmación sobre dispersión.

### `experiments/plot_final.py` — la figura sigue la misma política

- `cell()` devuelve `(mediana, (abajo, arriba))` y las barras son **asimétricas**:
  RD es una KL sin cota superior y la cola va solo hacia arriba (F23). Una
  desviación simétrica la dibujaba también hacia abajo.
- El eje X usa la mediana de `d_trans`.
- La heurística de escala logarítmica mira ahora **la extensión dibujada**,
  barras incluidas, no los centros. Con medianas, el panel de MiniGrid/RD se
  quedaba en lineal y la cola de UG-MTM aplastaba a los otros cuatro métodos
  contra el cero. Los tres paneles de RD salen en log.

### Tests — 353 pasan (`not integration and not slow`), 21 nuevos

`test_summarize_results.py`: que una celda simétrica no se marque y una con una
semilla un orden de magnitud fuera sí · que el sesgo sea de un solo lado · que
`cell_summary` no imprima la media · que la exclusión de F27 esté en una
constante y no dispersa por el dibujo · que la mediana de medianas no la arrastre
un método extremo · Spearman contra un valor calculado a mano, con empates,
constante → NaN, y longitudes distintas → error · que la pérdida de la tarea A
sea relativa · y que una celda de control lo siga siendo aunque su RD sea alto,
que es el caso de `gymnasium/distance_med`.

`test_plot_final.py`: mediana y rango en vez de media y desviación, y barras
asimétricas cuando las semillas lo son.

### PIS retirada de la suite (D18, cierra P2/F6)

- **`src/benchmark/metrics.py`** — la cabecera del módulo declara la suite: PF,
  RD y FT. PIS pasa a párrafo que dice que se anunció, que no se implementó y
  por qué (hace falta un controlador entrenado en la imaginación del modelo).
  `compute_wmf` conserva `pis_list` porque la Ec. 6 tiene ese término, y lo dice.
- **`run_full_benchmark.py`, `train_baseline.py`, `train_ug_mtm.py`** — los tres
  guardan `"pis": None` en vez de `0.0`. Un cero almacenado se lee como "medido,
  y salió cero"; `null` es la convención de `ft` y `d_trans` sin referencia. El
  `0.0` que entra en `compute_wmf` se queda, con un comentario que aclara que es
  el término gamma de la Ec. 6 y no una medición.
- **`summarize_results.py`** — la cabecera de la sección WMF dice las tres cosas:
  suite de tres, PIS retirada, término a cero como en el paper anterior.
- **`plot_final.py`** — misma corrección en el docstring que explica por qué no
  hay panel de WMF.
- **`README.md` del repo** — sección *Metrics* encabezada por la suite, entrada
  de PIS eliminada de la lista de métricas, y limitación nueva **nº 8** que
  cuenta que se anunció y no se implementó.
- **Tests (+1, 384 pasan)** — `test_the_reported_suite_is_pf_rd_and_ft` fija
  `DEFAULT_METRICS == ["pf", "rd", "ft"]`, y el test de gamma en `test_metrics.py`
  documenta que lo retirado es la métrica reportada, no el argumento de la Ec. 6.

Los 225 `metrics.json` guardados conservan `"pis": 0.0` y no se tocan (D4). Nada
los lee: `summarize_results.py` no tabula `pis` y su `wmf` se calculó con ese
mismo cero.

### La sección de resultados, y sus tablas generadas (sesión 8)

- **`paper/results.tex`** (nuevo, fuera de `cf_worldmodels/`) — sección de
  resultados montada sobre F27. Estructura: cómo se reporta (D15) → el eje no
  ordena el olvido → qué sí lo predice → las tres celdas de control → dónde
  ocurre el olvido (F18/F21) → lectura por método → amenazas a la validez.
- **`experiments/export_tables.py`** (nuevo) — emite las seis tablas del paper
  como LaTeX (`paper/tables/tab_*.tex`), importando las funciones de
  `summarize_results.py`. Es la misma regla de la casa aplicada al paper: una
  tabla transcrita a mano se desincroniza de las corridas que dice describir, y
  `paper-vs-code.md` existe precisamente por eso. Marca lo mismo que la consola:
  nivel del pico, daga de celda de control, asterisco de sesgo. La tabla del
  codificador pasa a **factor** (×745) en vez de porcentaje (+74434.6%), porque
  1.00 y 745 tienen que caber en la misma columna.
- **`summarize_results.py`** — imprime la tabla de predictores por celda (las
  nueve filas de las que salen las correlaciones), y la comparación emparejada
  muestra **medianas** junto a la diferencia media, que es el estadístico sobre
  el que se calculan la permutación y `d_z`.
- **Tests (+8, 392 pasan)** — `test_export_tables.py`: que cada tabla lleve la
  cabecera de "generado", que el pico salga en negrita y nombrado, que las
  celdas de control lleven daga, que el sesgo lleve asterisco, y que una celda
  ausente sea `--` y no un cero.

**Cuatro afirmaciones que no sobrevivieron al contraste con las tablas** y se
corrigieron antes de dar la sección por buena:

1. «una tarea lejana pero pobre enseña poco» **no se sostiene en DMControl**: la
   dificultad de B sube de 15.43 a 39.00 de min a max mientras RD baja tras el
   nivel medio. La sección lo dice: el predictor explica dos familias de tres, y
   lo que las nueve celdas comparten es el resultado negativo.
2. El salto entre controles y celdas con olvido es de **×13**, no de tres
   órdenes de magnitud.
3. El FT del resto de métodos no está «dentro de ±3.4»: el rango real es
   **[−10.48, +3.40]**.
4. Las cifras de convergencia de la tarea A que circulaban (5.3e-04 por píxel)
   son del presupuesto de **1000** pasos, no de los 5000 ejecutados. Se citan
   ahora las tres de la prueba de escalado: 6.49 → 1.61 → 1.37.

### La discusión, y el par anidado de Gymnasium (sesión 8)

- **`paper/discussion.tex`** (nuevo) — qué significa «controlled dynamic
  distance» y qué no · las dos cosas que se llaman «distancia» y se separan en el
  extremo alto · el olvido no está donde miraban las métricas · qué compra cada
  familia de mitigación · lecciones de construcción del banco · limitaciones ·
  y un apartado sobre por qué no sobrevive ninguno de los cinco Findings
  anteriores.
- **`paper/results.tex`** — añadido el argumento que no necesita comparar
  familias: en Gymnasium el nivel **máximo es estrictamente el medio más
  perturbación** (los dos van de gravedad 9.8 a 4.0; el máximo añade masa ×3 y
  fricción ×0.5, sobre el mismo par de tareas) y **produce menos olvido**
  (59.59 frente a 85.91). `d_trans` se pone del lado del resultado, no de la
  construcción: 24.71 frente a 30.86.

Eso salió de verificar `configs/benchmark/gymnasium.yaml` mientras contrastaba
una frase de la discusión, y es el caso más limpio de F27: un superconjunto
estricto de una perturbación no puede quedar por debajo si el eje ordena algo
monótono. No depende de agregación, ni de escalas entre familias, ni de
`d_trans`.

---

## Sesión 9 — El método del paper, y las dos tablas que faltaban por generar

### `experiments/export_tables.py` — dos tablas nuevas, ambas leídas de los resultados

- **`tab_protocol.tex`** (Tabla 1). Sale del bloque `protocol` que guarda cada
  `metrics.json`, no del YAML. **Levanta `ValueError` si los resultados cargados
  traen más de un protocolo**: una Tabla 1 no puede describir dos presupuestos.
  Verificado sobre las 225 celdas — un solo protocolo, idéntico bit a bit.
- **`tab_tasks.tex`** (las nueve casillas). Sale del bloque `tasks`, por el
  mismo motivo elevado al cuadrado: los niveles se editaron tres veces (F8,
  F25, F26) y el config no es evidencia de qué produjo los números. **Levanta
  `ValueError` si una casilla se ejecutó sobre dos parejas de tareas
  distintas** — que es exactamente la forma que tuvo F26.
- **`d_param_of`**: emite la Ec. 8 solo donde diferenciar un vector de física
  dice algo. En los pares que cambian cheetah por walker devolvería la
  distancia entre dos vectores por defecto —cero— y describiría el cambio más
  grande de la familia como ningún cambio. Se deja la celda vacía. Los tres
  valores de Gymnasium salen **0.283 / 0.586 / 0.622**, que reproducen exacto
  los del paper viejo.
- `task_label` suprime las escalas ×1 (no son perturbación) y conserva la
  gravedad siempre que esté declarada.

### `tests/test_export_tables.py` — 5 tests nuevos (12 → 13 archivos sin tocar)

Lo que se fija es lo que puede volver a morder: que la Tabla 1 salga del run y
no del config, que se niegue a describir dos presupuestos, que `d_param` quede
vacío donde mentiría, y que una casilla con dos parejas de tareas reviente en
vez de promediar. 367 tests fuera de integration/slow, todos pasan.

### `paper/method.tex` — §3, escrita

Formalismo (`E`, `M`, `D`; qué componente se mide) · PF, RD y FT con sus
ecuaciones · las dos distancias · las tres familias con sus dos restricciones
de diseño (una única anchura de acción por pareja; en dm_control dos tareas del
mismo dominio son el mismo entorno) · los cinco métodos, con los dos ceros del
Fisher de EWC enunciados **antes** de los resultados · el protocolo y la
reproducibilidad.

Declara dos límites de alcance que solo vivían en comentarios del código:

1. Las transiciones de `D_A` se puntúan desde `h = 0`, así que PF mide el mapa
   a un paso, no el arrastre recurrente.
2. **F28**: `d_trans` evalúa el modelo B en la base latente de A.

### `paper/refs.bib` — nuevo, 10 entradas

Solo lo que se cita hoy. Cotejar con la bibliografía del PDF viejo cuando se
escriba trabajo relacionado.

### `paper/discussion.tex` — retocada por F28

§6.1 ya no recomienda `d_trans` sin la salvedad, y dice cuál sería el arreglo
(una pareja de referencia con codificador compartido). §6.6 lo lista entre las
limitaciones.

### `_devlog/check-paper.py` — nuevo

Comprobación estructural de `paper/**/*.tex` sin toolchain de LaTeX: llaves,
`$` pareados, entornos, y `\input`/`\ref`/`\cite` que no resuelven. Incluye
`tables/`, porque un generador que emita un `\textbf{` sin cerrar rompe la
compilación igual que una errata. Hoy: 11 ficheros, 37 labels, 10 entradas de
bib, OK.

### Limpieza tras la revisión (sesión 9)

Cuatro revisiones en paralelo (reuso, simplificación, eficiencia, altitud)
sobre el diff de la sesión. La de eficiencia no encontró nada —a 225 dicts y
una ejecución por línea de comandos, las otras tres coincidieron en que
pre-agrupar haría el código peor de leer sin ganar nada—. Las otras tres
convergieron en lo mismo, y era un problema de capa, no de estilo.

**El arreglo de fondo: `check_runs_consistent` junto a `load_runs`.** El
`ValueError` que metí en `tasks_table` protegía solo a `tasks_table`. Las
**otras cuatro tablas medianizan sobre las mismas celdas**, así que una celda
que mezclara dos presupuestos o dos parejas de tareas se habría promediado en
silencio en `axis`, `predictors`, `encoder`, `pf`, `rd` y `ft` — y
`summarize_results.py` no lo habría notado siquiera. Ahora la comprobación
—protocolo único + una pareja de tareas por celda— vive junto al cargador y se
ejecuta una vez:

- `export_tables.py` **levanta**: alimenta el paper.
- `summarize_results.py` **avisa y sigue**: sus números son cómo se diagnostica
  el directorio del que se queja, y negarse a imprimirlos sería negarse a
  ayudar.

Es la tercera capa que vigila el mismo invariante, y ahora las tres coinciden
por construcción: el runner (`check_protocol_consistency`, antes de entrenar),
el cargador (antes de leer) y el exportador (antes de escribir el paper).

**Tres duplicaciones eliminadas:**

- `protocol_table` tenía su propia noción de identidad de protocolo al lado de
  `shared_protocol` — y **más laxa**: la coerción con `str()` hacía que `5000` y
  `"5000"` fueran el mismo presupuesto aquí y dos distintos en la consola, en un
  fichero cuyo docstring promete que los dos «solo pueden discrepar si el código
  discrepa consigo mismo».
- La mediana de `d_trans` de una celda estaba escrita tres veces
  (`tasks_table`, `predictor_table`, `print_distance_table`), cada una con su
  copia de la regla que importa: un `null` guardado es ausencia, no cero. Ahora
  es `cell_d_trans`.
- El idiom de agrupar por familia (`\addlinespace` + etiqueta solo en la primera
  fila) iba por su cuarta copia. Ahora es `family_cell`.

**Dos listas que derivaban:**

- `PHYSICS_PARAMS` en `distances.py` es ahora la única declaración del vector
  phi de la Ec. 8, y hay una comprobación **en tiempo de importación** de que
  las etiquetas del paper nombran esas mismas claves. Antes, añadir un parámetro
  al wrapper lo metía en la distancia y lo dejaba fuera de la tabla de tareas.
- `PROTOCOL_ROWS` se valida contra `PROTOCOL_FIELDS | MODEL_FIELDS` del runner,
  con `PROTOCOL_OMITTED` para lo que se deja fuera a propósito. Un campo que el
  runner empiece a registrar ya no puede faltar de la Tabla 1 en silencio: era
  un whitelist opt-in, que es justo lo que la función existía para evitar.

**Movido de capa:** la política de «dónde está definida la Ec. 8» era mía en el
emisor de LaTeX y ahora es `d_param_for_pair` junto a `compute_d_param`.

**Verificación:** las ocho tablas salen **byte a byte idénticas** antes y
después. 376 tests fuera de integration/slow (antes 367); los tres que probaban
funciones movidas se movieron con ellas, a `test_distances.py` y
`test_summarize_results.py`.

### `paper/related.tex` — §2, escrita (sesión 9)

Se rescata del §2 del PDF viejo, que era lo que `paper-plan.md` daba por
salvable, y se actualiza en dos sitios porque el paper ya no afirma lo mismo
(D19).

**Lo que se hereda intacto:** el hueco. Continual World mide políticas; Kessler
et al. estudian DreamerV2 como sistema integrado con métricas de política.
Ninguno aísla `M` ni define una métrica de calidad de las dinámicas imaginadas.
Es la única afirmación del paper anterior que sobrevivió al escrutinio.

**Lo que se corrige:** el RSSM se atribuye a PlaNet, no a DreamerV1 como decía
el PDF.

**Dos subsecciones nuevas, una por hallazgo:**

- **§2.4, la distancia entre tareas como factor experimental.** Sitúa F27 contra
  la práctica que enmienda: montar secuencias por un juicio de similitud y
  tratar ese orden como variable independiente. Dice explícitamente que nuestro
  propio diseño hace eso, que es lo que hace la enmienda creíble.
- **§2.5, dónde se mide el olvido dentro de un modelo.** Sitúa F18 como lo que
  es —una métrica tiene que congelar algo para atribuir un cambio, y lo que
  congela acota lo que puede ver— en vez de como un defecto de estimador.

**Y la comparación con Kessler et al. sale reforzada, no debilitada.** Ellos
reportan que L2 sobre el DreamerV2 entero funciona mal; nosotros que EWC
restringido a `M` lo conserva casi exacto y no protege nada más, porque su
Fisher es idénticamente cero fuera de la pérdida de transición (F21). Juntos
sugieren que el problema de la regularización en world models no es la
penalización sino **dónde tiene soporte su señal de importancia**. Esa lectura
no estaba disponible antes de F21.

`refs.bib` pasa de 10 a 20 entradas, todas de la bibliografía del PDF viejo (ya
verificada por el autor) salvo las que ya estaban.

### Sondeo de 2× leído, y el paper corregido (sesión 9, R18 → F29)

**`paper/results.tex` §5.2** — el argumento de Gymnasium gana el párrafo que le
faltaba: se reejecutaron las dos celdas al doble de presupuesto, el orden
aguanta (118.04/58.79 → 144.70/102.18, solo `finetuning` para comparar el mismo
estimador), y **la objeción se refuta en vez de solo sobrevivir** — la tarea B
es más fácil de ajustar en el nivel máximo, no más difícil. Declarado también
que la brecha se estrecha (2.01 → 1.42): dirección robusta, magnitud no.

**`paper/results.tex` §5.3** — reescrito. Decía que `d_trans` «recupera el orden
de RD en Gymnasium exactamente». No lo hace: las medianas abarcan 6 unidades y
cada celda abarca 15–19 entre semillas, y al doblar el presupuesto dos de las
tres se intercambian. Ahora dice de dónde sale realmente el +0.53 —de la
separación entre familias, que es lo que F28 dice que no está legitimado a
comparar— y qué queda en pie.

**`paper/discussion.tex` §6.1** — la recomendación baja de «reportad `d_trans`»
a «reportad una distancia medida y exigidle cuentas», con las dos deficiencias
y sus arreglos nombrados.

**`paper/discussion.tex` §6.2** — el mecanismo deja de ser especulación en
Gymnasium. Decía «a distancia máxima el modelo a menudo no llega a ajustar B».
Medido, es al revés: B es *más fácil* ahí. El cheetah a masa ×3 y gravedad 4.0
apenas se mueve, así que la perturbación más fuerte de la familia produce la
tarea más pobre. Las dos formas de «no hay nada que aprender» —demasiado difícil
(dmcontrol, sospecha) y demasiado pobre (gymnasium, medido)— quedan como fallos
opuestos del mismo empaquetado.

**Amenazas a la validez** — acotado qué establece el sondeo: dos celdas de
nueve, la dirección del eje en la familia con niveles anidados, y `d_trans` no.

### k=4 escrito en el paper (sesión 9, R20 → §5.7)

**`experiments/summarize_sequence.py`** — nuevo. Los resultados de secuencia
tienen su propio esquema (una matriz de retención por corrida, no las métricas
de una celda), así que llevan su propio cargador y su propia agregación, y
comparten la política de reporte —medianas sobre semillas— con la rejilla.

Dos decisiones que estaban en el código y ahora están fijadas por tests:

- **La diagonal se excluye de la curva.** RD(i,i) es cero por construcción;
  dejarlo dentro pondría un cero estructural al principio de cada curva e
  **inventaría una subida que nadie midió**.
- **El pico se cuenta por semilla, no se lee de la mediana.** Un pico que solo
  existe después de promediar es una propiedad del promedio. Por eso la tabla
  dice `T3 (5/5)` y no solo `T3`.

**`export_tables.py`** — `sequence_table()`. Se genera **solo si existe**
`results-seq/`: las ocho tablas de k=2 no dependen de que la corrida de
secuencia exista. Sin `\multirow` — un generador no debería añadir un paquete al
preámbulo de un documento que no es suyo.

La columna de píxeles se formatea como `tab_encoder` (entero por encima de 10)
para que la prosa cite la tabla **literalmente**. Antes decía 773/759/784
mientras la tabla emitía 773.29/759.06/784.47, y `check-numbers.py` lo marcaba
como no respaldado — que es exactamente para lo que se escribió esa herramienta.

**`paper/results.tex` §5.7** — la subsección. El argumento: el relato del k=2
hace una predicción sobre secuencias largas que la rejilla no puede probar, y
el pico en T3 con 5/5 semillas en los tres métodos que mueven el codificador
—y en ninguno de los dos que no— la cumple.

**Declarado como lo que es**, en la propia subsección y en amenazas a la
validez: una familia, una secuencia, cinco semillas. Cierra el hueco entre el
formalismo y los experimentos; **es una comprobación de consistencia, no una
confirmación independiente**. Y UG-MTM tiene 4 expertos, así que k=4 agota su
capacidad exactamente — con k=5 el resultado hablaría del límite.

392 tests fuera de integration/slow.

### Los tres huecos del documento (sesión 9, cierre)

Al preguntarse «¿queda algo?» aparecieron tres agujeros que no eran de
investigación sino de documento, y que un revisor ve antes de leer una palabra.

**1. No había conclusión.** `paper/conclusion.tex`. Dice qué se encontró y qué
autoriza, sin una sola cifra que no esté ya en una tabla generada. Termina
nombrando los dos siguientes pasos reales: las mismas mediciones a escala
Dreamer, y una distancia entre entornos que sobreviva a su propio instrumento.

**2. No había ni una figura**, con ocho tablas. Y la que existía
(`forgetting_vs_distance`) estaba **sin usar y obsoleta** — del 2-ago, anterior
a las diez semillas y a k=4.

Peor: **jugaba en contra del paper**. Pone `d_trans` en el eje X, que era lo
correcto cuando se creía que era el mejor eje. Reordenar las celdas por
`d_trans` es exactamente lo que hace **invisible el pico**.

- **`experiments/plot_axis.py`** — nueva, Figura 1. Eje X = nivel etiquetado,
  un panel por familia, escala log, y la agregada sobre los cuatro RSSM dibujada
  encima con la anotación `peak`. UG-MTM en gris y fuera de la agregada, por lo
  mismo que en la tabla. 6 tests: que el eje X sea la etiqueta y no la distancia
  medida, que el pico se anote donde la agregada realmente pica, y que UG-MTM se
  dibuje pero no entre en la agregada.
- La vieja **pasa al apéndice**, donde lo que enseña *es* el argumento: los
  niveles tampoco se separan en el eje medido, y en DMControl dos de las tres
  etiquetas se solapan literalmente.

**3. Faltaba el apéndice planificado.** `paper/appendix.tex`: la tabla de
calidad de la tarea A (**generada**, `tab_quality` — ¿había algo que olvidar?),
dónde cae el presupuesto en la curva de convergencia, la reproducibilidad, y la
figura de `d_trans`.

**Y un fallo de compilación cazado antes de que ocurriera.** `tab_tasks` tenía
una fila de 138 caracteres, que se sale del ancho de texto de un `article` a una
columna. Se quita el boilerplate de los nombres (`MiniGrid-`, `-v0`) — la
columna de familia ya dice de qué suite es — y la tabla va en `\footnotesize`
con `tabcolsep` reducido. De 138 a 132 caracteres de fuente, y el contenido
restante es irreducible.

**`check-paper.py` ahora valida `\includegraphics`**, que no lo hacía. Probado
rompiendo una ruta a propósito: lo detecta.

399 tests. `check-numbers.py`: 174 respaldadas por tabla; las únicas sueltas de
las secciones nuevas son las cuatro del sondeo de escalado, que el propio
apéndice declara que no vienen de la rejilla.

---

## Sesión 10 — La verificación cifra a cifra del paper

Sin cómputo. `check-numbers.py` clasifica 173 cifras como respaldadas por una
tabla generada, 9 como constantes de protocolo y **74 sin respaldo**. Las 74 se
comprobaron una a una contra `results/`, `results-2x/` y `runs.md`. **Cuatro
estaban mal.**

### `paper/results.tex` y `paper/intro.tex`

1. **La cuota de RD en el agregado heredado seguía en «78–97%»**, que era el
   rango de una corrida anterior y hoy no es el rango de nada: sobre las 45
   celdas va de **2.5% a 100%**, mediana **88.5%**, y ≥75% en **37 de 45**. Las
   celdas bajas no contradicen la afirmación, la ilustran: son aquellas donde RD
   ya es casi cero (la celda de control de DMControl y las tres de DMControl del
   método que congela el codificador). La frase reporta ahora la mediana y dice
   cuáles son las excepciones. Aparecía dos veces, en resultados y en la
   introducción.

2. **La comparación de presupuestos decía fijar el estimador y luego mezclaba
   cinco semillas contra diez.** La celda del nivel máximo vale **58.79** sobre
   las cinco semillas que los dos presupuestos comparten; **60.18** es su
   mediana a diez y pertenece a la comparación emparejada de §5.5. La razón
   entre las dos celdas es **2.01**, no 1.96.

3. **La celda con menos olvido es 14.8×, no 15×.** La frase sigue diciendo que
   cualquier corte entre 1.1 y 14 selecciona los mismos tres controles, que es
   cierto, y ahora también lo es del número impreso al lado.

### Lo que la verificación confirmó

Todo lo demás. En particular las diez semillas de RD de `ug_mtm` en
`minigrid/distance_max` (17.7 … 8135, media 2075, mayor que siete de las diez),
las 17 celdas sesgadas de 180, los tres controles, `p = 0.0020` en cuatro
celdas con `d_z` de −2.28 a −3.11, el intercambio de `d_trans` al doblar el
presupuesto (30.86/24.71 → 57.30/71.51) y la prueba de escalado
(6.49 / 3.01 / 1.61 / 1.37).

### Lección de método

`check-numbers.py` **clasifica, no verifica**: dice dónde mirar, no si está
bien. Las cuatro erratas estaban las cuatro en su lista CHECK, y ninguna se
habría detectado sin recalcular. Conviene recordarlo la próxima vez que se
regeneren tablas y se dé por buena la prosa.

---

## Sesión 13 — PDF recompilado y verificado, README reescrito

### El desborde de la Tabla 1, cerrado

Recompilado el paper (13 ago) con el relleno de columna a 3 pt y vuelto a medir
sobre el PDF, caja por caja: **26 páginas, cero líneas y cero reglas de
`booktabs` fuera del bloque de texto**. La Tabla 1 sigue completa pese a
estrecharla —los siete entornos y las nueve `d_trans`— y las cifras corregidas
en la sesión 10 siguen dentro (88.5, 58.79, 2.01, 14.8, 8135), mientras que las
viejas no aparecen (78–97, 1.96, 15×).

### `README.md` reescrito

Estaba escrito para un repo privado con la corrida recién invalidada: abría
describiendo el protocolo, decía «225 runs» y «291 tests», y su sección de
resultados rezaba **«Not yet published — results are being regenerated»**. El
repo lleva dos días público con un paper de 26 páginas dentro.

- Abre con **los dos hallazgos y sus cifras**, y enlaza `paper/WMF.pdf` en la
  primera pantalla.
- Recuentos al día: 375 corridas, 75 parejas de referencia, diez semillas en las
  seis celdas que discriminan, 434 tests. El layout menciona `results-2x/`,
  `results-seq/` y `paper/`.
- Tres correcciones dentro de lo que ya había: **los checkpoints no se
  escriben** (decía que se escribían y se ignoraban en git), el suelo del test
  exacto es **0.002** a diez semillas y no 0.0625, y la cita lleva el título
  actual.
- Se conservan la sección de determinismo, las definiciones de métricas con su
  alcance declarado, y las ocho limitaciones conocidas.

### Un fallo de método que conviene no repetir

El commit del README **se subió afirmando que la cita llevaba el título nuevo, y
no era cierto**: la edición falló por un error de escapado, el script murió
después de escribir el resto del fichero, y el `git commit` se ejecutó igual
porque iba encadenado en la misma orden.

Corregido en el commit siguiente, que lo dice explícitamente. La lección no es
el escapado sino el encadenado: **un mensaje de commit es una afirmación sobre
el árbol, y este se verificó después de publicarlo**. En un repo público eso
deja un mensaje incorrecto en la historia para siempre. Separar la escritura de
la verificación y del commit.

---

## Sesión 14 — La versión corta compila, y dos recortes

### Overleaf: tres cosas que la rompían

Overleaf compila **desde la raíz del proyecto**, no desde la carpeta del fichero
principal, y con `paper/` subida entera eso rompía la versión de 8 páginas.

1. **Rutas.** `../tables/…` apuntaba fuera del proyecto. Resuelto con un prefijo
   autodetectado: `\IfFileExists{tables/tab_axis.tex}` decide si `\wmfroot` es
   vacío o `../`. El mismo fichero compila desde `paper/` y desde
   `paper/workshop/`, **sin duplicar las tablas** en el subdirectorio.
2. **Documento principal.** Hay dos `main.tex`; hay que fijarlo a mano en
   Menu → Settings → Main document.
3. **La bibliografía no salía**, y fue error mío: `\bibliography{\wmfroot refs}`
   no puede funcionar. Ese nombre llega a **bibtex** por el `.aux`, y bibtex no
   expande macros de LaTeX ni abre rutas que salgan del directorio de
   compilación. Queda `\bibliography{refs}`, correcto compilando desde `paper/`.
   La alternativa —copiar `refs.bib` dentro de `workshop/`— arreglaba lo mismo
   creando dos bibliografías que pueden divergir.

`check-paper.py` aprende a seguir el macro: prueba cada expansión que el
documento define antes de dar por rota una ruta.

### Los recortes, y lo que costó cada uno

La primera compilación dio **8 páginas**, justo en el límite y sin sitio para las
referencias.

- **§3.6, la secuencia k=4: ganancia neta cero.** Libera ~50 palabras y obliga a
  gastar ~46 en declarar la limitación que ese resultado cubría — sin él el
  paper solo enseña un cambio de tarea, y callarlo en un workshop de continual
  learning no es una opción. Lo único que ahorra es el título de subsección.
- **La tabla del codificador: −0,4 páginas.** Un tercio de página con su
  caption, y sus dos números (811 y 800) ya estaban en la prosa. Sustituida por
  la frase que sostenía, más el rango que solo daba la tabla (1.00 a 848) y un
  puntero a los resultados publicados.

Quedan la figura del pico y dos tablas (eje, predictores). Estimación **6,97
páginas** sin referencias, contra 7,38 antes: **~1 página de margen**.

**Lección para el próximo recorte:** quitar una sección que responde a una
objeción no ahorra lo que ocupa, porque la objeción vuelve y hay que declararla.
Lo que ahorra de verdad es material redundante — una tabla cuyos números ya
están en el texto.

---

## Sesión 15 — El subdirectorio se va, y con él tres fallos

La versión corta **compila en Overleaf con su bibliografía**. Lo que hizo falta
no fue un arreglo más, sino deshacer la decisión que los causaba: estaba en
`paper/workshop/` porque quedaba ordenado.

Overleaf compila **desde la raíz del proyecto**, no desde la carpeta del fichero
principal, y de ahí salieron tres fallos seguidos: los `\input{../tables/…}`
apuntaban fuera del proyecto, con dos `main.tex` no elegía el correcto, y la
bibliografía no aparecía. Los dos primeros los parcheé con un prefijo
autodetectado (`\wmfroot`); el tercero demostró que el parche no valía, porque
el nombre del `.bib` llega a **bibtex**, que no expande macros de LaTeX ni abre
rutas que salgan del directorio de compilación.

`paper/workshop/main.tex` pasa a **`paper/main_workshop.tex`**, junto a
`refs.bib`, `tables/` y `figures/`. Todas las rutas quedan planas, el prefijo
desaparece, y la cabecera del fichero dice que no lo devuelvan a un
subdirectorio y por qué.

**Lección:** tres síntomas distintos con la misma causa no piden tres parches,
piden quitar la causa. Lo propuse al segundo fallo y seguí parcheando; debí
moverlo entonces.

---

## Sesión 16 — Cabe, y se cierra el trabajo

La versión corta compila en **8 páginas en total, bibliografía incluida**, con
la tabla del codificador puesta. El límite de CL4FMAgents es de 8 **excluyendo**
referencias, así que el cuerpo queda por debajo: **cumple con margen y no hay
nada que recortar**.

Recorrido del tamaño, junto porque mis estimaciones fallaron dos veces en la
misma dirección:

| Momento | Estimado | Real |
| --- | --- | --- |
| Primera escritura | 7,3–7,6 | 8 sin bibliografía |
| Calibrado contra el PDF largo | 7,38 | — |
| Sin k=4 ni tabla del codificador | 6,97 | 8 con bibliografía |
| Con la tabla devuelta | ~7,8 de cuerpo | 8 con bibliografía |

**El sesgo era sistemático, no casual:** contaba prosa por densidad más el alto
medido de los flotantes, y nunca el espacio vertical de encabezados de sección,
`\paragraph`, saltos de párrafo ni la recolocación de flotantes. En un documento
de nueve secciones eso es cerca de una página. Para la próxima: **sumar ~1
página a cualquier estimación hecha así**, o compilar y no estimar.

De los dos recortes que provocó ese error, la tabla se revirtió y k=4 no — y
k=4 no ahorraba nada de todos modos, porque obligaba a declarar la limitación
que ese resultado cubría.
