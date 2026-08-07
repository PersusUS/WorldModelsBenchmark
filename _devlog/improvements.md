# Mejoras pendientes en el código

Cosas observadas que **no** son los bugs de `findings.md` (aquellos falsean
resultados). Aquí van riesgos latentes, deuda técnica y ergonomía. Ordenado por
prioridad.

Leyenda: `[!]` puede afectar a resultados · `[~]` robustez · `[·]` higiene

---

## Prioridad alta — pueden morder al re-ejecutar

### ~~`[!]` I1 — Los hiperparámetros están hardcodeados~~ · CORREGIDO (sesión 6)

Era el origen de F10 y de la Tabla 1 imposible: `STEPS = 1000`, `SEQ_LEN = 5`,
`BATCH_SIZE = 8`, `N_COLLECT = 20` como constantes de módulo, más `32`/`512`
mágicos en `create_baseline_model`, mientras los YAML declaraban
`n_collect: 1000`, `n_train: 50000`, `batch_size: 32`, `seq_len: 50` y el paper
una tercera cosa.

**Cómo quedó.**

- El bloque `protocol:` de `configs/benchmark/<familia>.yaml` es la única fuente
  de verdad, y ahora declara **lo que de verdad se ejecuta**. Se le añadieron los
  campos que el runner tenía escondidos: `learning_rate`, `n_eval_episodes`,
  `n_eval_transitions`, `n_fisher_transitions`, `n_recon_frames`, `rd_horizon`,
  `rd_samples`, `ewc_lambda`, `mc_dropout_T_train`, `wmf_weights`, `curve_points`.
  Se quitó `eval_every`, que no lo leía nadie.
- `resolve_protocol()` lee, castea y valida. **Un campo que falte es un error, no
  un valor por defecto** — un default silencioso es exactamente cómo el protocolo
  se separó de los configs.
- Las dimensiones del modelo salen del protocolo para los cinco métodos, así que
  la capacidad emparejada entre UG-MTM y las bases se cumple **por
  construcción**, no por coincidencia. El `mc_dropout_T = 3` que el runner metía
  a mano es ahora `mc_dropout_T_train` en el config, y se imprime.
- Los overrides son flags explícitos (`--steps`, `--batch-size`, `--seq-len`,
  `--n-collect`, `--seeds`) y se imprime qué se sobreescribió. `--dry-run`
  muestra el protocolo efectivo y el plan sin entrenar nada.
- Cada `metrics.json` guarda su bloque `protocol` completo, y el runner **se
  niega a reutilizar resultados producidos con otro protocolo**. Saltar celdas
  cacheadas es lo que lo hace reanudable; promediar dos presupuestos en la misma
  casilla de la tabla es lo que esta comprobación impide.

**Consecuencia para el paper:** la Tabla 1 se genera desde los `metrics.json`.
Nunca a mano.

Cubierto por `tests/test_run_full_benchmark.py` (incluido un test que falla si
alguien vuelve a meter una constante de protocolo en el módulo).

---

### ~~`[!]` I2 — Los pasos con NaN se saltan en silencio~~ · CORREGIDO (sesión 6)

`train_task` cuenta ahora los pasos descartados y `metrics.json` guarda
`n_nan_steps_A`, `n_nan_steps_B`, `n_update_steps_A` y `n_update_steps_B`; si hay
alguno, el runner avisa por consola. También se eliminó el `comps` que se
devolvía tras el bucle y podía venir de un paso descartado (con `steps=0` daba
`UnboundLocalError`): las cifras que salen ahora son del último paso **aceptado**.

Medido en R10 (MiniGrid, 3 métodos, 2000 pasos cada uno): **0 NaN**. Con F0
corregido no aparecen, al menos en esta familia.

---

### ~~`[!]` I3 — El Fisher de EWC es el gradiente del batch al cuadrado~~ · CORREGIDO (sesión 7)

`src/baselines/ewc.py`:

```python
loss = -log_prob.mean()
loss.backward()
fisher[name] += param.grad.data.pow(2) / n_batches
```

Eso es `(E[∇])²`, el cuadrado del gradiente **promediado sobre el batch**.
La definición de Kirkpatrick et al. es `E[(∇)²]`, la media de los gradientes
por muestra al cuadrado.

Como `(E[X])² ≤ E[X²]`, esto **infraestima** el Fisher de forma sistemática, y
el sesgo crece con el tamaño de batch (aquí 32). Es una aproximación que se ve
en el mundo real, pero hay que documentarla o corregirla — no dejarla implícita.

**Qué hacer.** O bien batches de tamaño 1 para el cálculo del Fisher, o bien
declararlo explícitamente en el README y en el paper.

**Corregido (sesión 7).** Una pasada hacia atrás por transición: `E[g²]` de
verdad. Con `n_fisher_transitions = 50` son 50 backward sobre una GRU pequeña,
o sea nada. `E[g²] = (E[g])² + Var[g]`, y el término que faltaba —la varianza—
es donde vive casi toda la señal.

**Qué mueve.** Misma celda que R10 (`minigrid/distance_med`, semilla 999), dos
procesos independientes, mismo resultado:

| | R10 (Fisher roto) | s7 (Fisher correcto) |
| --- | --- | --- |
| PF | +1.3306 | **+0.0029** |
| RD | 19.3906 | **17.8688** |
| WMF | 8.2885 | 7.1487 |
| Recon. entrenamiento al final de B | 20.06 | 19.59 |
| Recon. reservada de A tras B | 718.01 | 725.76 |

**PF cae a 0.003: EWC pasa a conservar la NLL latente de la tarea A casi
exactamente**, y sin perder plasticidad — su ajuste final a la tarea B (19.59) es
el mismo que el de `finetuning` (19.66). Con el Fisher infraestimado, la
penalización era prácticamente inerte y EWC estaba siendo evaluado como un
`finetuning` con pasos de más.

Lo que **no** cambia es la reconstrucción de la tarea A en píxeles: sigue
destruida. El motivo es estructural y ahora está medido — ver **F21**.

`finetuning` en la misma corrida reproduce R10 con delta < 1.5e-11, así que el
cambio toca a EWC y solo a EWC.

---

### ~~`[!]` I4 — `ThresholdNet._ptr` no se guarda en el checkpoint~~ · CORREGIDO (sesión 6)

`_ptr` era un atributo normal, así que quedaba fuera del `state_dict`: tras
`load_state_dict` volvía a 0 aunque `_history` se restaurara bien, y el buffer
circular reanudaba escribiendo por el extremo equivocado de la ventana.

Corregido con `register_buffer("_ptr", torch.zeros((), dtype=torch.long))`, más un
test de regresión que guarda, recarga, escribe en los dos y compara las ventanas.

**No mueve ningún resultado, verificado y no argumentado**: se re-ejecutó la celda
`ug_mtm / minigrid / distance_med / 999` y las cuatro métricas salen idénticas a
R9/R10 (delta < 5e-11). El motivo es que `_ptr` solo se lee en `update_history`,
que solo se llama en modo `train`, y `model_i` —el único que recarga pesos— se usa
únicamente en evaluación.

---

### `[!]` I5 — No hay control de determinismo en CUDA · PROMOVIDO A F16

Los runners hacen `torch.manual_seed` y `np.random.seed`, pero nunca fijan
`torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic` ni
`torch.use_deterministic_algorithms`. Con cuDNN eligiendo algoritmos por
heurística, dos corridas con la misma semilla pueden divergir en GPU.

**Ya no es hipotético: medido en R7.** Misma celda, misma semilla, dos veces —
`PF = +0.2373` vs `PF = −0.2101`. Cambia de signo, y el signo de PF es la
interpretación de la métrica.

**Deja de ser deuda técnica y pasa a ser bloqueante.** Movido a
[findings.md](findings.md) como **F16**, que es donde está el detalle y el plan.

---

## Prioridad media — robustez

### `[~]` I6 — Mensajes de error opacos

`ReplayBuffer.sample()` sobre un buffer vacío:

```
ValueError: high <= 0
```

Viene de `np.random.randint`. No dice qué pasa ni dónde.

**Qué hacer.** Comprobación explícita con mensaje útil. Ídem en
`collect_rollouts` si el entorno devuelve episodios más cortos que `seq_len`
de forma sistemática — ahora se descartan en silencio y el buffer queda vacío.

---

### `[~]` I7 — `MiniGridEnv.action_dim` devuelve `np.int64`

Rompe la asignación a un config de OmegaConf:

```
UnsupportedValueType: Value 'int64' is not a supported primitive type
```

Los runners lo parchean con `int(...)` en el punto de uso. Debería devolver
`int` desde la propiedad.

---

### `[~]` I8 — `collect_rollouts` acepta un `policy` que no valida

```python
def collect_rollouts(env, buffer, n_rollouts, policy="random", max_steps=500):
```

Cualquier valor distinto de `"random"` se ignora en silencio y se usa política
aleatoria igualmente. O se implementa, o se elimina el parámetro, o se valida.

---

### `[~]` I9 — `log_metrics` no registra por consola

El docstring dice "Log metrics to wandb and/or console" pero la función solo
escribe en wandb; sin `wandb_run` no hace nada. Con `--no_wandb` (que es como
se ejecutó todo) no queda ninguna traza de la evolución del entrenamiento.

---

### `[~]` I10 — `build_latent_eval_dataset` codifica en un solo batch

Con `n_transitions` grande son N×3×64×64 imágenes de golpe por el encoder.
Con los 100 actuales no hay problema; si se sube, conviene trocear.

---

## Prioridad baja — higiene

### `[·]` I11 — Código muerto: `apply_gradient_masking`

Definida y testeada, pero `UG_MTM` nunca la llama: usa
`register_gradient_scaling_hooks`. Junto con `self.threshold_grad`, que se
guarda y no se usa, es lo que deja inerte la ablación `no_gradient_masking`
(ver F9). O se conecta, o se borra.

### `[·]` I12 — `sys.path.insert` en cada script de `experiments/`

No hay `pyproject.toml` ni paquete instalable. Cada script manipula `sys.path`
a mano. Un `pip install -e .` eliminaría el truco y haría los imports
predecibles.

### `[·]` I13 — Tres convenciones de nombre para la figura

En `results/figures/` conviven `wmf_vs_distance.*`, `wmf_final.*` y
`figurasfinalv2.*`. `plot_final.py` escribe la tercera; `NEXT_STEP.md` mencionaba
una cuarta (`wmf_vs_distance_final`). Elegir una y borrar el resto.

### `[·]` I14 — Directorio huérfano en `results/`

`results/single_task_minigrid_MiniGrid-Empty-5x5-v0_42/` lo dejó
`train_single_task.py`. No encaja con el esquema
`results/{método}/{familia}_{distancia}_{semilla}/` que leen los scripts de
figuras.

### `[·]` I15 — `docs/` contiene instrucciones para agente, no documentación

`START_HERE.md`, `CLAUDE.md`, `PHASES.md` son instrucciones de construcción
dirigidas a un agente ("You are implementing a research project…"). `SPECS.md`
sí tiene valor como documentación técnica, aunque está desactualizado respecto
al código en varios puntos (la arquitectura del decoder, el gating).

**Qué hacer.** Decidir qué se publica. Recomendación: sacar los tres primeros
del repositorio publicado y actualizar `SPECS.md` o fusionarlo en el README.

### `[·]` I16 — Sin integración continua

No hay workflow de GitHub Actions. Con 234 tests ya escritos, un CI que los
corra en cada push es barato y da señal de calidad al revisor.

### `[·]` I17 — `ReplayBuffer` usa `list.pop(0)`

O(n) en cada desalojo. Con 20 episodios da igual; con los 1000 que declaran los
configs, no tanto. `collections.deque(maxlen=...)` lo resuelve.

### `[·]` I18 — Herencia múltiple frágil

`class RSSM(BaseWorldModel, nn.Module)` con `super().__init__()` depende del MRO
para acabar llamando a `nn.Module.__init__`. Funciona, pero es sutil. Un
`nn.Module.__init__(self)` explícito, o hacer de `BaseWorldModel` un
`Protocol`, sería menos frágil.

### `[·]` I19 — `wandb` está en las dependencias pero no funciona en este entorno

`import wandb` lanza `AttributeError`. Todo se ejecutó con `--no_wandb`, así que
no bloqueó nada, pero es una dependencia pesada que nadie usa. Considerar
hacerla opcional (`extras_require`).

### `[~]` I20 — El contexto GLFW no sobrevive a `dm_control → MuJoCo → dm_control` · **la conclusión de abajo quedó desmentida: ver F24**

Encontrado al añadir los tests de F16. Secuencia que falla, en un mismo proceso:

1. Un `DMControlEnv` renderiza.
2. Se crea y se cierra un `GymnasiumEnv` (MuJoCo con `render_mode="rgb_array"`).
3. Un `DMControlEnv` nuevo intenta renderizar →
   `mujoco.FatalError: Default framebuffer is not complete, error 0x0`

Acotado por eliminación: `dmc` solo, `minigrid → dmc`, `gym → dmc`,
`gym abierto → dmc`, `dmc → gym` y `dmc × 3` **todos funcionan**. Solo falla
cuando dm_control ya ha renderizado *antes* de que un entorno MuJoCo se abra y se
cierre. Parece que `GymnasiumEnv.close()` invalida el contexto GLFW compartido y
dm_control no lo reconstruye.

**~~No afecta a `run_full_benchmark.py`.~~** Verificado explícitamente con su
orden real de familias (minigrid ×3 → gymnasium ×3 → dmcontrol ×3): pasa. Como
dm_control se crea por primera vez *después* de que se cierren todos los
gymnasium, nunca entra en la secuencia mala. La ejecución de las 225 corridas no
está en riesgo por esto.

> **Desmentido por R16 (sesión 7).** La corrida completa murió exactamente en esa
> frontera, con `Default framebuffer is not complete`, tras 30 horas. La
> comprobación en seco pasaba; bajo carga —15 celdas de gymnasium, tres parejas de
> entornos abiertas y cerradas, decenas de miles de fotogramas— no. Corregido
> ejecutando **cada familia en su propio subproceso**. Ver **F24**.

**Sí afecta a la suite de tests**, donde `test_envs.py` renderiza dm_control y
`test_seeding.py` crea un MuJoCo después. El test de reproducibilidad de
dm_control se salta con un mensaje explícito en ese caso, y da cobertura real
aislado:

```
python -m pytest tests/test_seeding.py -k dmcontrol
```

**Qué hacer.** Si en el futuro hace falta mezclar familias en un proceso,
aislarlas en subprocesos. Alternativa más limpia: renderizar dm_control por
EGL/osmesa en vez de GLFW (`MUJOCO_GL=egl`), que no depende de un contexto de
ventana.

Nota menor asociada: al cerrar el intérprete, `dm_control` lanza
`AttributeError: 'MjrContext' object has no attribute '_ptr'` desde
`MjrContext.__del__`. Es un fallo de limpieza de la propia librería durante el
recolector de basura ("Exception ignored in"), inofensivo.

---

### ~~`[!]` I21 — La comprobación de consistencia estaba en la capa equivocada~~ · CORREGIDO (sesión 9)

Salió de la revisión de limpieza de la sesión 9. En la sesión anterior metí un
`ValueError` en `tasks_table` para que una casilla que mezclara dos parejas de
tareas no se reportara en silencio. **Protegía solo a esa tabla.**

Las otras cuatro constructoras de tablas —`axis`, `predictors`, `encoder`,
`method`— medianizan sobre las mismas celdas, así que una casilla que mezclara
dos presupuestos o dos parejas se habría promediado sin ruido en las ocho
tablas del paper. Y `summarize_results.py` no lo habría notado siquiera: solo
avisaba de protocolos mezclados, no de parejas.

**Corregido:** `check_runs_consistent(runs)` vive junto a `load_runs` y se
ejecuta una vez por consumidor.

- `export_tables.py` **levanta**. Alimenta el paper; un número malo ahí es un
  desastre.
- `summarize_results.py` **avisa y sigue**. Sus números son cómo se diagnostica
  el directorio del que se queja, y negarse a imprimirlos sería negarse a
  ayudar.

Es la tercera capa que vigila el mismo invariante y ahora las tres coinciden por
construcción: el runner antes de entrenar (`check_protocol_consistency`), el
cargador antes de leer, y el exportador antes de escribir el paper.

De paso, tres duplicaciones y dos listas que derivaban — detalle en el
`changelog.md` de la sesión 9. Verificación: las ocho tablas salen byte a byte
idénticas antes y después.

---

### ~~`[!]` I22 — `seeds` formaba parte de la identidad del protocolo~~ · CORREGIDO (sesión 9)

Salió al leer R19. Cada `metrics.json` guarda el bloque `protocol` entero, y las
tres capas que vigilan que no se mezclen presupuestos comparaban **el dict
completo**. Pero `protocol["seeds"]` es la lista que *esa invocación* pidió
correr: las celdas hechas con `--seeds 5 6 7 8 9` guardan `[5,6,7,8,9]` y las
originales `[0,1,2,3,4]`.

**Es procedencia, no presupuesto.** Dos celdas con hiperparámetros idénticos que
solo difieren en qué lista nombró la línea de órdenes son perfectamente
comparables — de hecho combinarlas es exactamente el objetivo de D21.

Consecuencias, una vista y otra latente:

- **Vista:** `export_tables.py` se negó a generar las tablas del paper tras
  R19, con «2 different protocols across 375 runs».
- **Latente, y peor:** `check_protocol_consistency` en el runner tenía la misma
  comparación. Habría **rechazado su propia caché** en cuanto alguien
  reejecutara con las semillas del config tras haber usado `--seeds`. No mordió
  porque cada invocación solo mira las celdas que le tocan.

**Corregido** con `PROTOCOL_IDENTITY_EXCLUDED` y `protocol_identity()` en
`run_full_benchmark.py`, que es el módulo del que dependen los otros dos, usado
en los dos sitios del runner y en `shared_protocol`.

**Y arrastró una corrección al paper.** La Tabla 1 imprimía «Seeds 5
(0,1,2,3,4)» leyendo esa misma lista: habría dicho que el estudio corrió cinco
semillas cuando seis celdas corrieron diez. Ahora `protocol_table` **cuenta las
semillas que existen** en los resultados y la tabla de las nueve casillas lleva
una columna `Seeds` por celda. `summarize_results` imprime el desglose cuando no
son uniformes.
