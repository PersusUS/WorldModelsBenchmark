# Problemas encontrados

Ordenados por gravedad. Estado: `ABIERTO` / `CORREGIDO` / `DECISIÓN PENDIENTE`.

---

## F0 — El posterior del VAE estaba completamente colapsado · CORREGIDO

**LA CAUSA RAÍZ.** Invalida las 225 corridas de todos los métodos, no solo
UG-MTM.

**Evidencia.** VAE tras 2000 pasos sobre MiniGrid-Empty-8x8:

```
per-dim std across data = 3.4671e-05   (max 6.0502e-05)
active dims (std > 1e-2) = 0 / 32
KL -> 0.000342  (decreciendo hacia 0)
```

**Cero de 32 dimensiones latentes activas.** El encoder mapeaba *toda*
observación al mismo vector latente constante. El modelo de transición recibía
una `z` constante independientemente de lo que viera el agente.

**Causa.** En `compute_loss` de `ConvVAE`, `RSSM`, `UG_MTM` y
`ProgressiveNetWorldModel`:

```python
recon_loss = F.mse_loss(recon, obs_t, reduction="mean")   # promedia 3*64*64
kl = 0.5 * torch.sum(...)                                  # suma sobre dims latentes
```

La reconstrucción se promedia sobre los 12288 elementos mientras la KL se suma
sobre las dimensiones latentes. Eso infrapondera la reconstrucción ~12288×, y
la forma más barata de bajar la pérdida total pasa a ser llevar la KL a cero:
ignorar la entrada. Colapso del posterior de manual.

**Verificación controlada.** Dos VAE idénticos sobre los mismos datos, con la
única diferencia del escalado:

```
current   active dims= 0/32   mean per-dim std=1.217e-04   recon MSE=1.6585e-02
fixed     active dims=32/32   mean per-dim std=2.890e-01   recon MSE=5.4610e-04
```

32/32 dimensiones activas y reconstrucción **30× mejor**.

**Corrección.** Nuevo `vae.reconstruction_loss()` — suma sobre píxeles, media
sobre batch — usado en los cuatro `compute_loss`. Test de regresión
`test_training_does_not_collapse_the_posterior`.

**Consecuencia.** Todo lo que se midió antes de esto era ruido alrededor de un
modelo degenerado. Explica por qué ninguna señal separaba las tareas (AUC≈0.5)
y por qué todos los métodos daban WMF cercano a cero.

---

## F1 — La evaluación se hacía sobre ruido gaussiano · CORREGIDO

**Qué pasaba.** Todos los runners construían el dataset de evaluación así:

```python
eval_ds = {"obs": torch.randn(100, 32), "actions": torch.randn(100, action_dim)}
```

PF, RD y FT se calculaban sobre eso. `EWCWorldModel.consolidate` estimaba la
diagonal de Fisher sobre el mismo tipo de ruido.

**Por qué importa.** Las métricas medían divergencia entre funciones de
transición sobre entradas latentes aleatorias, no sobre el `D_i` que definen
las Ec. 3–7 del paper. Afectaba a las 225 corridas y a todas las filas de la
Tabla 3.

**Corrección.** `protocol.build_latent_eval_dataset()` recoge rollouts de la
tarea A reservados (fuera del buffer de entrenamiento) y los codifica una sola
vez con el modelo post-tarea-A, de forma que `model_i` y `model_k` se puntúan
sobre entradas y objetivos idénticos.

---

## F2 — `compute_nll` puntuaba contra el estado equivocado · CORREGIDO

**Qué pasaba.** El objetivo de la verosimilitud era `obs` (= `z_t`), no
`next_obs` (= `z_{t+1}`):

```python
h_next = model.transition(h, obs, actions)
mu, log_sigma = model.predict_stoch(h_next)
nll = ... (obs - mu).pow(2) ...        # objetivo = z_t
```

**Por qué importa.** Medía cuánto *mueve* la transición el estado, no cuánto lo
*acierta*. La Ec. 3 define `NLL = -E[log P(z'|z,a)]`. Mismo fallo en el Fisher
de EWC.

**Corrección.** El dataset lleva `next_obs` y es el objetivo. Sin esa clave,
`compute_nll` y `consolidate` lanzan `KeyError` en vez de degradar en silencio.

---

## F3 — El experto nuevo de UG-MTM nunca gana la puerta · ABIERTO / BLOQUEANTE

**Evidencia.** Modelo UG-MTM entrenado sobre MiniGrid-Empty-8x8, medido sobre
100 transiciones reservadas:

```
tau               = 0.446077
u_t  mean/max     = 1.327e-01 / 1.498e-01
u_t > tau         : 0 / 100 transiciones
mean gate expert0 = 0.998084    expert1 = 1.9e-03
```

**Causa raíz.** `ThresholdNet` acota `tau` a `(0, 1)` con una sigmoide, pero
`u_t` es una varianza de MC-dropout **sin normalizar**, cuya escala depende del
dropout rate y del ancho de capa. Nada en el método hace conmensurables las dos
cantidades.

**Consecuencia en cadena.** `register_gradient_scaling_hooks` escala los
gradientes de cada experto por su puerta. Con puerta 1.9e-3, el experto nuevo
tampoco aprende durante la tarea B. Y el runner congela todo lo demás. Así que
la tarea B no cambia prácticamente nada:

- `WMF = 0.0000` exacto en 6 de 9 configuraciones (PF crudo `0.000e+00`)
- `FT = -1.21` de media (−2.75 en el smoke test tras las correcciones)

**Lectura.** El `WMF ≈ 0` de UG-MTM **no es aislamiento estructural**: es un
modelo que apenas se mueve. Verificado perturbando el experto nuevo con
`N(0, 1000)` — PF se mueve 3.8e-6, por debajo de la precisión reportada.

**Nota.** El paper (§4.2) describe `tau` como "aprendido por un MLP sobre una
ventana de valores recientes de `u_t`". **No menciona la sigmoide** — solo
aparece en el código y en `docs/SPECS.md`. Cambiarla no contradice el texto.

**Actualización tras corregir F0.** Medido si `u_t` discrimina tarea A de B,
como AUC = P(señal_B > señal_A). 0.5 = ninguna discriminación.

Con el VAE colapsado (pre-F0) — ninguna señal servía:

| Señal | med | max |
| --- | --- | --- |
| (a) UncertaintyHead MC-dropout | 0.519 | 0.508 |
| (b) MC-dropout sobre la transición (Ec. 11) | 0.481 | 0.515 |
| (c) error de predicción a un paso | 0.769 | 0.533 |
| distancia latente media L2 | 1.4e-03 | 3.7e-04 |

Con el VAE corregido (post-F0):

| Señal | med | max |
| --- | --- | --- |
| (a) UncertaintyHead MC-dropout | **0.294** | **0.864** |
| (b) MC-dropout sobre la transición | 0.283 | 0.780 |
| (c) error de predicción a un paso | 0.693 | 0.862 |
| distancia latente media L2 | 2.89 | 5.24 |

**Lecturas.**

1. La premisa de UG-MTM **es viable**: a distancia máxima la incertidumbre
   discrimina bien (AUC 0.864). El fallo era la representación, no la idea.
2. A distancia media la señal **se invierte** (0.294): la tarea B parece
   *menos* incierta que la A, así que la puerta dispararía más en la tarea A
   que en la B — enrutamiento activamente perjudicial.
3. Las escalas de `u_t` (≈0.43–0.65) y `tau` (sigmoide, ≈0.45) ahora son
   comparables, así que el problema de conmensurabilidad se reduce mucho al
   corregir F0. Falta confirmarlo en una corrida completa.

La inversión a distancia media es un resultado **reportable e interesante**:
la puerta por incertidumbre funciona cuando la distancia dinámica es grande y
falla cuando es moderada. Mucho más honesto y útil que "UG-MTM suprime el
olvido a cero en todas partes".

---

## F4 — `ExpertPool` destruía la reproducibilidad · CORREGIDO

**Qué pasaba.** Reseeding no determinista durante la construcción:

```python
for k in range(K):
    torch.manual_seed(k * 1337 + 42)
    self.experts.append(nn.GRUCell(...))
torch.manual_seed(torch.seed())      # <-- semilla aleatoria
```

El runner llama `torch.manual_seed(seed)` y acto seguido construye el modelo,
así que la semilla de la corrida se descartaba. Las 5 "semillas" de UG-MTM eran
5 corridas independientes no reproducibles.

**Corrección.** Un `torch.Generator` local por experto. Verificado que
reproduce los pesos originales bit a bit: no altera ningún resultado.

---

## F5 — Progressive Nets rompía con 3+ columnas · CORREGIDO

`transition` pasaba la entrada cruda a la columna `k-1`, pero toda columna más
allá de la primera espera además el lateral concatenado:

```
RuntimeError: input has inconsistent input_size: got 11 expected 27
```

No afecta a los resultados publicados (el benchmark usa 2 tareas, un solo
`add_column`), pero el protocolo se presenta como general. Corregido
reproduciendo la cadena de columnas de abajo arriba.

---

## F6 — PIS nunca se calcula · CERRADO por D18 (sesión 8)

> **Se retira de la suite.** El banco reporta **PF, RD y FT**. Implementarla
> exigía un controlador entrenado en la imaginación del modelo y evaluado en el
> entorno real, que no existe en el repositorio.
>
> - `pis` se guarda como **`null`**, no como `0.0`: un cero almacenado se lee
>   como "medido, y salió cero". Es la misma convención de `ft` y `d_trans`
>   cuando falta su referencia, ya cubierta por tests. Los tres runners escriben
>   `null`.
> - **WMF no cambia.** Es la Ec. 6 del paper anterior y se conserva para poder
>   reproducir aquellos números, con su término `gamma` a cero — que es con lo
>   que se calcularon. Lo que cambia es que ahora se dice por qué.
> - Los 225 resultados guardados conservan `"pis": 0.0` y no se tocan (D4). Nada
>   los lee.
>
> Era el último número del paper que no correspondía a una medición.

`pis = 0.0` hardcodeado en las 225 corridas (verificado: valores únicos
`[0.0]`). No hay ningún controlador en el repositorio — ni política, ni
entrenamiento en imaginación, ni evaluación de retorno.

El WMF reportado es `0.4·PF + 0.4·RD`, no la Ec. 6.

El paper lo presenta en el abstract y en Contributions como una de las tres
métricas complementarias.

**Opciones:** implementarlo, o sacarlo del paper y redefinir WMF con los pesos
que se usan de verdad.

---

## F7 — RD usaba acciones fuera de distribución · CORREGIDO

`a = torch.randn(N, action_dim) * 0.1` — off-manifold para espacios continuos
acotados y sin sentido para las acciones one-hot de MiniGrid. Ahora se
remuestrean acciones del propio dataset.

---

## F8 — DMControl `distance_min` compara una tarea consigo misma · CORREGIDO (s7)

`task_A` y `task_B` son ambos `cheetah/run`. El parámetro `lateral_wind: true`
que debía diferenciarlos no lo lee nadie: `DMControlEnv` solo recibe
`domain_name` y `task_name`.

El paper (§3.4) lo describe como "Cheetah-run → Cheetah-run+wind".

**Corregido (sesión 7).** `DMControlEnv` acepta ahora `physics_params` con las
mismas tres perturbaciones que la familia Gymnasium (`gravity`, `mass_scale`,
`friction_scale`), aplicadas sobre el mismo `MjModel`. `distance_min` pasa a ser
`cheetah/run` con **gravedad 9.81 → 7.0**, que es exactamente el cambio físico
del `distance_min` de Gymnasium: los niveles mínimos de las dos familias son la
misma perturbación, y `d_param` queda definida también aquí.

**El viento no.** `opt.wind` de MuJoCo solo produce fuerza si `opt.density` o
`opt.viscosity` son distintos de cero, y ambos valen 0 por defecto en estos
modelos. Ponerlo habría sido una segunda perturbación que no perturba nada. El
wrapper ahora **rechaza** las claves que no implementa, así que un
`lateral_wind: true` en el YAML es un error y no un silencio de dos sesiones.
Verificado: dos entornos con la misma semilla divergen tras 20 pasos idénticos.

Dos tests nuevos lo fijan, uno de ellos sobre las tres familias: ningún nivel de
distancia puede emparejar una tarea consigo misma.

---

## F9 — Las CINCO ablaciones eran inertes · RESUELTO (script eliminado)

Corregido al alza respecto al diagnóstico inicial (que decía tres).

`run_ablations.py` construía un `ug_cfg` con los overrides, lo imprimía por
pantalla… y **nunca se lo pasaba a `run_single`**, que recarga
`configs/models/ug_mtm.yaml` del disco:

```python
# run_ablations.py
ug_cfg = OmegaConf.load("configs/models/ug_mtm.yaml")
for key, value in ablation_cfg.items():
    if hasattr(ug_cfg.model, key):          # además: salta en silencio si no existe
        OmegaConf.update(ug_cfg, f"model.{key}", value)
print(f"Config overrides: {ablation_cfg}")
run_single(cfg, dist, seed, args.steps, args.no_wandb, ...)   # ug_cfg no viaja

# train_ug_mtm.run_single
ug_cfg = OmegaConf.load("configs/models/ug_mtm.yaml")          # recarga sin overrides
```

Así que **las cinco** ablaciones ejecutaban UG-MTM sin modificar, imprimiendo
por consola unos overrides que no se aplicaban. Salida activamente engañosa.

Aparte, `fixed_tau` y `content_gate` fijan claves que `UG_MTM` no lee, y
`no_gradient_masking` fijaba `threshold_grad`, que se guardaba en el modelo sin
usarse (`transition` usa `register_gradient_scaling_hooks`, no
`apply_gradient_masking`).

**Resuelto.** Script archivado en `_devlog/archive/scripts/`. Rehacerlo desde
cero si se quieren ablaciones — requiere además implementar `fixed_tau` y
`content_gate` en el modelo. Eliminados también `apply_gradient_masking`
(código muerto) y la clave `threshold_grad` del config.

---

## F10 — Los configs no coinciden con lo ejecutado · ABIERTO

Circulan tres conjuntos de hiperparámetros distintos. Ver
[paper-vs-code.md](paper-vs-code.md).

---

## F11 — El escalado de gradiente usa solo el último timestep · ABIERTO

`UG_MTM.transition` limpia y re-registra sus hooks en cada llamada, así que
tras desenrollar una secuencia los gradientes de toda la secuencia se escalan
con las puertas calculadas en el último paso.

---

## F13 — La KL de `compute_rd` no es una KL · CORREGIDO (sesión 5)

Descubierto al ver que RD explotaba tras corregir F0.

**Convenciones incompatibles.** `stoch_fc` devuelve `mu, log_sigma`. Dos partes
del código lo interpretan de forma distinta:

| Sitio | Código | Interpreta `log_sigma` como |
| --- | --- | --- |
| `compute_nll` | `exp(0.5 * log_sigma)` | log-**varianza** |
| `RSSM.sample_stoch` | `exp(0.5 * log_sigma)` | log-varianza |
| `RSSM.compute_loss` (KL) | `log_sigma.exp()` como varianza | log-varianza |
| `compute_rd` | `exp(log_sigma)` | log-**desviación** |
| `compute_d_trans` | `exp(log_sigma)` | log-desviación |

**Y la fórmula no cuadra con ninguna de las dos.** Verificado numéricamente
contra `torch.distributions.kl_divergence` con
`mu_i=[0,1], mu_k=[0.5,-1], log_sigma_i=[0.3,-0.7], log_sigma_k=[-0.2,0.4]`:

```
formula del codigo (compute_rd)       = 1.7997
KL exacta si log_sigma = log-std      = 2.0997
KL exacta si log_sigma = log-varianza = 1.7841
```

La causa: usa `exp(log_sigma)` (lectura log-std) para el término logarítmico
pero lo eleva al cuadrado en el término cuadrático. Mezcla las dos lecturas.

**Impacto.** Con el VAE colapsado todo valía ~0 y no se notaba. Ahora RD domina
el WMF, así que el error se propaga a la métrica principal. Afecta también a
`d_trans`, la distancia dinámica que el paper propone como universal (Ec. 9).

**Corregido.** Nueva función `metrics.diag_gaussian_kl(mu_p, log_var_p, mu_q,
log_var_q)`: fija la convención **log-varianza** (la mayoritaria en el código) y
delega en `torch.distributions.kl_divergence` en vez de escribir la fórmula a
mano. La usan `compute_rd` y `compute_d_trans`. Con los valores de referencia de
arriba devuelve **1.7841**, exactamente la KL exacta bajo log-varianza.

Cinco tests nuevos: igualdad con la referencia de torch, valor analítico
conocido `KL(N(0,1) || N(0,e²))` que distingue log-varianza de log-desviación,
identidad, no-negatividad y asimetría. Más uno en `test_distances.py` que
compara `d_trans` completo contra la KL de torch sobre el mismo dataset.

**Magnitud del error.** Sobre `log_sigma ~ N(0,1)`, latente 32-dim: la fórmula
antigua daba **752.97** donde la KL correcta da **81.18** — un factor **9.3×**.
Casi todo venía de leer `log_sigma` como log-desviación (la KL exacta bajo esa
lectura errónea da 752.89); el 0.5 sobrante en el término logarítmico aportaba
lo poco restante.

Medido después sobre **modelos reales entrenados**, calculando las dos fórmulas
sobre el mismo par de modelos (R7): inflación **12.82×** y **11.96×** en dos
corridas. Consistente con el 9.3× sintético. Es decir: **RD venía inflada ~12×
frente a PF**, lo que agrava F14 pero no lo explica entero — ver F14 revisado.

---

## F14 — RD domina el WMF · RESUELTO POR DECISIÓN (D10) · re-medido en R16

### Medición original (R5, con la KL rota)

```
                    PF        RD          WMF
finetuning       -0.7696   +248.75      +99.19
replay_infinite  -2.1532   +262.82     +104.27
ewc              +0.6768   +148.39      +59.62
progressive_nets +5.0160   +311.75     +126.71
ug_mtm          +12.3122    +87.86      +40.07
```

Conclusión de entonces: RD 1–3 órdenes de magnitud mayor, `WMF ≈ 0.4·RD`, PF
aporta ~1%. Y RD inestable: UG-MTM a distancia máxima daba `RD = 13708.83`.

### Re-medición (R6, con la KL corregida)

**Dos de las tres conclusiones cambian.**

1. **La explosión desaparece.** UG-MTM a distancia máxima: `13708` → `196.9`.
   El rango de RD queda en 11–197 en vez de 75–13708. El rollout a lazo abierto
   de 15 pasos sigue sin cota, así que la inestabilidad estructural sigue ahí en
   principio, pero con la KL correcta no se manifestó en ninguna de las 10 celdas.
2. **PF ya no es irrelevante.** Pasa de ~1% a **3–25%** del agregado, mediana
   ~15%. La brecha `RD/|PF|` baja de 1–3 órdenes de magnitud a **3–32×**.
3. **Pero RD sigue dominando**: 75–97% del WMF en las 10 celdas. La frase
   "`WMF ≈ 0.4·RD`" ya no es exacta; "RD manda y PF matiza" sí.

Reparto por celda en `runs.md` (R6).

**El problema de fondo no se ha ido.** `WMF = 0.4·PF + 0.4·RD + 0.2·PIS` sigue
presuponiendo escalas conmensurables, y con PIS = 0 (F6) los pesos declarados no
son los pesos efectivos. Sigue siendo una decisión de diseño — ver P7.

### Cifras definitivas (R9, con F13 y F16 corregidos)

R6 se midió con los entornos sin sembrar, así que se repitió sobre el pipeline
reproducible. **El resultado es el mismo, lo que dice que F14 es estructural y no
un artefacto de unos datos concretos:**

| | R6 (pre-F16) | R9 (post-F16) |
| --- | --- | --- |
| % PF en el WMF | 3–25% | **2.8–22.4%** (mediana 11.7%) |
| % RD en el WMF | 75–97% | **77.6–97.2%** |
| RD/&#124;PF&#124; | 3–32× | **3.5–34.4×** |

Sembrar los entornos movió RD entre ×0.72 y ×1.45 — los órdenes de magnitud de R6
eran correctos.

**El caso raro que queda:** UG-MTM a distancia máxima, `RD = 167` frente a 14–32
del resto de métodos, `RD/|PF| = 34.4`. Ya no explota (13708 con la KL rota) pero
es un outlier de 5–10×. Conviene tenerlo delante al decidir P7.

Tabla completa por celda en `runs.md` (R9). Sigue siendo **una sola semilla**: la
dispersión entre semillas está sin medir.

---

## F12 — `Pillow` era dependencia no declarada · CORREGIDO

Los tres wrappers de entorno hacen `from PIL import Image`. Funcionaba solo
como dependencia transitiva de matplotlib. Además faltaba el
`--extra-index-url` sin el cual los pines `+cu121` no instalan.

---

## F15 — `d_trans` (Ec. 9) no la calcula nadie · CORREGIDO (s7)

Descubierto al arreglar F13: `compute_d_trans` existe, ahora es correcta, y
**ningún runner la llama**. Los únicos consumidores son los tests.

```
$ grep -rn "compute_d_trans" --include=*.py .
./src/benchmark/distances.py:35:def compute_d_trans(...)
./tests/test_distances.py: (4 llamadas)
```

**Por qué importa.** El paper presenta la distancia dinámica como el eje
controlado del benchmark, y reparte el trabajo entre dos métricas:

- Ec. 8, `d_param`, explícitamente "for Gymnasium variable-physics environments".
- Ec. 9, `d_trans`, explícitamente **"for other families"** — es decir, MiniGrid
  y DMControl.

`d_param` sí se calcula y reproduce (0.283 / 0.586 / 0.622). `d_trans` no se
calcula nunca. Así que **6 de las 9 celdas del benchmark no tienen ningún valor
numérico de distancia**: sus niveles min/med/max son una ordenación a ojo de los
pares de entornos, no una medida.

Lo confirma `plot_final.py`, que rotula el eje X de Gymnasium como
`Dynamic distance $d_{param}$` y el de MiniGrid y DMControl como
`Dynamic distance level` (ordinal). La Figura 1 del paper dice ser "WMF as a
function of dynamic distance"; para dos de los tres paneles es WMF frente a una
etiqueta ordinal.

**Consecuencia para el paper.** La afirmación de "controlled dynamic distances"
solo está respaldada en Gymnasium. Y F8 lo empeora: en DMControl el nivel
`distance_min` compara `cheetah/run` consigo mismo, o sea distancia real 0
etiquetada como "min".

**Ojo: no es solo fontanería.** La Ec. 9 define
`d_trans(E_A, E_B) = E[KL(P_A(z'|z,a) || P_B(z'|z,a))]` — entre **dos modelos,
cada uno entrenado en su propio entorno**. Lo que el runner tiene a mano son
`model_i` (entrenado en A) y `model_k` (entrenado en A y luego en B), y la KL
entre esos dos no es una distancia entre entornos: es una medida de cuánto se
movió el modelo, que es prácticamente lo que ya mide RD.

### Corregido (sesión 7): fiel a la Ec. 9

Se mide como la define el paper, con **un modelo por entorno entrenado desde
cero** (D11, opción 1 de P8). El runner entrena una pareja de referencia por
`(familia, distancia, semilla)` —RSSM planos, porque una distancia entre
entornos no puede depender del método de continual learning— y calcula
`KL(P_A || P_B)` sobre los mismos `(z, a)`.

Decisiones concretas, por si un revisor pregunta:

- Se evalúa sobre transiciones **de la tarea A** reservadas, en la base latente
  de `model_A`. La KL ya es asimétrica; puntuarla sobre la tarea de origen la
  deja en el mismo soporte `(z, a)` que PF y RD.
- Es una propiedad **del par de tareas y la semilla**, no del método, así que se
  calcula una vez y las cinco celdas de esa casilla guardan el mismo valor. El
  directorio de resultados queda autocontenido.
- La pareja de referencia se cachea en `results/_reference/` y se rechaza si
  viene de otro protocolo, por el mismo motivo que las celdas: el brazo "desde
  cero" solo es un contrafactual al mismo presupuesto.

Coste real: **+90 entrenamientos** sobre los 450 de la ejecución completa (dos
por referencia, no uno), o sea un +20%. El handoff decía +45 contando modelos, no
entrenamientos.

Primeras cifras (smoke, no comparables entre sí por presupuesto):
`minigrid/distance_med` a 200 pasos → `d_trans = 13.03`;
`dmcontrol/distance_min` a 50 pasos → `d_trans = 5.75`.

Medirlo bien exige **un modelo entrenado en B desde cero** por celda, comparado
con `model_i` sobre los mismos `(z, a)`. Eso es un entrenamiento extra por celda
(+45 corridas sobre las 225), no una llamada gratis.

Como referencia de lo que sale con la versión barata (R7, `model_i` vs `model_k`,
MiniGrid med): **3.73** y **5.28** en dos corridas de la misma semilla. Los
primeros valores de la Ec. 9 que produce el proyecto — y de paso otra muestra de
F16, porque difieren un 40% entre sí.

**Qué hacer.** Decidir si `d_trans` se mide como la define la Ec. 9 (con el
modelo entrenado en B desde cero, coste +45 corridas) o si se redefine la Ec. 9
para lo que el protocolo permite medir. Luego calcularla en las tres familias y
guardarla en `metrics.json`. Da además la validación cruzada del punto 12 de la Fase 2
—que `d_trans` ordene Gymnasium igual que `d_param`— que es material fuerte
para el paper: si las dos distancias coinciden donde ambas son medibles, `d_trans`
queda justificada como la métrica universal en las familias donde `d_param` no
existe.

---

## F16 — El pipeline no era reproducible con la misma semilla · CORREGIDO (sesión 5)

Promoción de I5, que estaba en `improvements.md` como deuda técnica. Ya no es
deuda técnica: es un bloqueante para la tesis del paper.

**Evidencia (R7).** La misma celda, dos veces, misma semilla 999:

```
Device: cuda    torch.backends.cudnn.deterministic = False

run 1:  PF= +0.2373   RD= 19.9702   d_trans= 3.7260
run 2:  PF= -0.2101   RD= 17.7509   d_trans= 5.2808

delta PF = 0.447    delta RD = 2.219
```

**A PF le cambia el signo entre dos corridas idénticas.** Y el signo de PF es la
interpretación entera de la métrica: positivo = olvido, negativo = transferencia
hacia delante (Ec. 3). El delta de 0.447 supera |PF| en varias celdas de R6
(med/ewc = 0.883, max/ewc = 1.564).

**Causa — medida, y no era la que suponíamos.** Diagnóstico por familia antes de
tocar nada, comparando el hash de todas las observaciones y acciones que ve el
bucle de entrenamiento:

| Fuente | ¿Reproducible con la semilla anterior? |
| --- | --- |
| MiniGrid — datos de entrenamiento | **NO** (longitudes `[101,256,256]` vs `[256,256,194]`) |
| Gymnasium — datos de entrenamiento | **NO** (observaciones y acciones) |
| DMControl — datos de entrenamiento | **NO** (acciones sí, estado inicial no) |
| Camino de cómputo, con datos idénticos | **NO** (delta 8.6e-02 en la loss tras 50 pasos) |

Las tres familias **entrenaban sobre datos distintos en cada corrida**. Eso pesa
mucho más que el ruido de cuDNN. `np.random.seed` no alcanza:

- el RNG del entorno de Gymnasium (se siembra en `reset(seed=...)`),
- el RNG del `action_space` (se siembra en `action_space.seed(...)`),
- el `RandomState` de la tarea de dm_control (`task.random`).

Los tres se sembraban con entropía del sistema en `make()` / `suite.load()`.
El no-determinismo de cuDNN existe además: los runners no fijaban
`cuda.manual_seed_all` ni `cudnn.deterministic`, y el backward de GRU en cuDNN no
es determinista por defecto.

**Corregido en cuatro piezas.**

1. `src/utils/seeding.py::set_seed(seed, deterministic=True)` — `random`,
   `numpy`, `torch`, `cuda.manual_seed_all`, `PYTHONHASHSEED`,
   `cudnn.deterministic = True`, `cudnn.benchmark = False`. Verificado que eso
   basta para que el entrenamiento sea bit a bit idéntico, así que
   `use_deterministic_algorithms` se deja **deliberadamente desactivado**: no
   aporta nada aquí y lanza en ops sin implementación determinista.
2. `BaseEnv.seed(seed)` como método abstracto, implementado en las tres
   familias. MiniGrid y Gymnasium por la vía propia de Gymnasium
   (`reset(seed=)` + `action_space.seed()`); dm_control resembrando
   `task.random` en sitio, lo que evita reconstruir el entorno y su contexto de
   render.
3. Los tres runners llaman `set_seed()` y siembran ambos entornos.
   **Se siembra una vez por corrida, no por episodio**: así los `reset()`
   sucesivos recorren un flujo determinista de estados iniciales *distintos*.
   Sembrar por episodio haría que el conjunto reservado de la tarea A fuese una
   copia del de entrenamiento, y PF se mediría sobre datos de entrenamiento —
   deshaciendo la corrección de F1. Hay un test que lo cubre.
4. `tests/test_seeding.py`, 13 tests: `set_seed` sobre los tres RNG globales, los
   flags de cuDNN, entrenar dos veces y comparar **los pesos** (no solo la loss),
   rollouts reproducibles en las tres familias, y que sembrar no colapse todos
   los episodios al mismo estado inicial.

**Verificación end-to-end (R8).** La misma celda dos veces por los runners
reales, en tres métodos con rutas de código distintas:

```
finetuning   WMF= 8.5205296834  PF=-1.7783794403  RD=23.0797036489  FT=16.5521640778
ewc          WMF= 8.2884732437  PF= 1.3305854797  RD=19.3905976295  FT=16.5521640778
ug_mtm       WMF=10.5560239029  PF= 5.5788383484  RD=20.8112214088  FT=19.0437374115
```

Los 12 valores con `delta = 0.000e+00`. Incluye `ug_mtm`, la ruta más difícil
(MC-dropout y hooks de escalado de gradiente).

**Nota.** Sembrar los entornos cambia qué episodios se recogen, así que los
números de R6 no son comparables con los posteriores a este arreglo. Los
porcentajes de F14 se re-midieron en R9.

**Por qué es bloqueante y no una mejora.** La tesis del paper es que **el
benchmark es la contribución**, y el criterio de éxito declarado incluye
"reproducible". Un benchmark cuyas celdas no se reproducen al re-ejecutarlas con
la misma semilla no cumple ese criterio. Además:

- Las "5 semillas" no son 5 réplicas controladas: la varianza que reportan mezcla
  varianza entre semillas con varianza de la propia implementación.
- Cualquier diferencia entre métodos por debajo de ~0.45 en PF es indistinguible
  del ruido de ejecución. Eso afecta directamente a los tests de significancia
  (recordar que el Finding 5 original ya no era significativo con n=5).
- Nadie puede replicar la Tabla 3 desde el repo, ni el autor.

**Lo que esto arregla del paper.** Las 5 semillas pasan a ser 5 réplicas
controladas: su varianza ya es varianza entre semillas y nada más, así que los
tests de significancia vuelven a tener sentido. Y la Tabla 3 pasa a ser
replicable desde el repo.

**Hallazgo lateral: I20.** Al añadir los tests apareció que el contexto GLFW no
sobrevive a la secuencia `dm_control → abrir/cerrar MuJoCo → dm_control`.
Acotado por eliminación y **verificado que no afecta a `run_full_benchmark.py`**,
cuyo orden de familias nunca entra en esa secuencia. Detalle en
[improvements.md](improvements.md).

---

## F17 — No se registraba si el modelo aprendió la tarea A · INSTRUMENTADO (sesión 6)

Encontrado al valorar la credibilidad del paper (sesión 5). No falsea ningún
número: era una **ausencia** en lo que se guardaba, y es la que más expuesto
dejaba el resultado principal.

> **Estado tras la sesión 6.** La instrumentación está hecha y la objeción tiene
> respuesta numérica: **la tarea A se aprende**. En MiniGrid `distance_med`,
> semilla 999 (R10):
>
> | Señal | Valor |
> | --- | --- |
> | Reconstrucción de entrenamiento, primer paso de A | 931.76 |
> | Reconstrucción de entrenamiento, último paso de A | 8.17 |
> | **Reconstrucción sobre `D_A` reservado, en píxeles** | **6.49** |
> | NLL sobre `D_A` tras la tarea A | 22.28 |
> | NLL sobre `D_A` de un modelo sin entrenar | 38.84 |
>
> 6.49 es error cuadrático **sumado sobre 12288 píxeles**: 5.3e-04 por píxel,
> RMSE ≈ 0.023 en `[0,1]`. No es un modelo degenerado, y ya no hay que inferirlo
> del FT: se guarda medido sobre datos que nunca entraron al buffer.
>
> Lo que la instrumentación destapó al medirlo es más grave que la ausencia
> original: **F18**.

**Qué se guarda ahora** en cada `metrics.json`, además de PF/RD/WMF/FT:

`nll_A_after_task_A`, `nll_A_after_task_B`, `nll_A_random_init` (PF y FT quedan
descomponibles a posteriori) · `initial/final_reconstruction_loss_A` y `_B` ·
`heldout_reconstruction_A_after_task_A` y `..._after_task_B` (píxeles, sobre
episodios reservados) · `recon_curve_A` y `recon_curve_B` (20 puntos) ·
`n_nan_steps_A/B` y `n_update_steps_A/B` (I2) · `protocol` completo.

**Cómo se garantiza que instrumentar no cambia el resultado.** Toda medición
nueva va dentro de `seeding.preserve_rng_state()`. No es paranoia: la puerta de
UG-MTM mantiene MC-dropout activo en evaluación **por diseño**, así que cada
`compute_nll` consume el flujo aleatorio. Sin el guardián, medir la convergencia
habría movido los resultados de UG-MTM. Verificado: las 12 cifras de R9
(WMF/PF/RD/FT × 3 métodos) se reproducen con delta < 5e-10, que es la precisión
con la que estaban anotadas.

**La prueba de escalado** (`experiments/convergence_A.py`, R11) cierra la otra
mitad. MiniGrid `distance_med`, una corrida de 10000 pasos evaluada por el camino:

| Presupuesto | Recon. reservada | Mejora sobre el anterior |
| --- | --- | --- |
| 1× (1000, el actual) | 6.486 | — |
| 2× | 3.013 | −53.5% |
| 5× | 1.606 | −46.7% |
| 10× | 1.367 | **−14.9%** |

El presupuesto actual está en la parte empinada: a 1000 pasos la tarea A se
reconstruye **4.7× peor** que a 10000. Los rendimientos decrecientes se ven a
partir de 5×. Dato para decidir el presupuesto, no una objeción abierta.

Y una lección de método: `NLL(own)` cae 22.28 → 8.59 **sin plateau**, y la
separación contra un modelo sin entrenar *crece*. Usar la NLL latente para
argumentar convergencia argumentaría lo contrario de lo que se pretende — el
codificador sigue moviendo el objetivo. Por eso la señal que se reporta es la de
píxeles.

### El problema original, para el registro

`run_full_benchmark.py` calculaba `final_A` y lo descartaba (también `init_B`):

```python
init_A, final_A, _ = train_task(model, buf_A, opt, device, STEPS)
...
"initial_reconstruction_loss": init_A,     # inicio de la tarea A
"final_reconstruction_loss": final_B,      # final de la tarea B
```

Se guardaba la reconstrucción al **empezar** la tarea A y al **acabar** la tarea
B, y se tiraba la única cifra que decía si el modelo había aprendido A antes de
medirle el olvido. La única evidencia de aprendizaje era indirecta: FT entre +16
y +19 en todas las celdas de R9, que dice "mejor que un modelo aleatorio" pero no
dice **cuánto mejor en términos absolutos**.

Las tres cosas que pedía el hallazgo están hechas: (1) guardar la reconstrucción
final de A y la NLL sobre `D_A`; (2) la prueba de escalado
(`experiments/convergence_A.py`, R11); (3) reportar la calidad en A **junto a**
las métricas de olvido, no en lugar de.

---

## F18 — PF y RD son ciegas al olvido del codificador · DECLARADO COMO ALCANCE (s7, D9)

Encontrado en la sesión 6, al instrumentar F17. Es el hallazgo más importante de
la sesión y no es un bug: es una propiedad estructural de cómo están definidas
las métricas, invisible hasta que hubo una medición en píxeles con la que
contrastarlas.

**La medición.** MiniGrid `distance_med`, semilla 999, misma corrida (R10):

| Método | Recon. `D_A` tras A | Recon. `D_A` tras B | Factor | PF |
| --- | --- | --- | --- | --- |
| `finetuning` | 6.49 | **725.27** | ×111.8 | **−1.78** |
| `ewc` | 6.49 | **718.01** | ×110.7 | +1.33 |
| `ug_mtm` | 7.66 | **7.66** | ×1.000 | +5.58 |

En `finetuning` el modelo pierde la capacidad de reconstruir la tarea A por un
factor de **112**, y PF sale **negativo** — es decir, la métrica del benchmark
dice que *mejoró*.

**Por qué.** `compute_nll` nunca llama a `model.encode`:

```python
obs    = dataset["obs"]        # latentes CONGELADOS, codificados por model_i
target = dataset["next_obs"]   # idem
h_next = model.transition(h, obs, actions)
mu, log_sigma = model.predict_stoch(h_next)
```

`D_i` se codifica una sola vez con el modelo post-tarea-A (decisión D5, y es
correcta para que `model_i` y `model_k` se puntúen sobre entradas idénticas). La
consecuencia no buscada es que **PF, RD, WMF y FT solo ven el GRU y `stoch_fc`**,
en una base latente congelada. El codificador puede derivar hasta ser inútil sin
que ninguna métrica del paper lo registre.

**Por qué afecta a la tesis y no solo a un número.**

1. El alcance declarado del paper es medir olvido en el componente de transición
   `M`, así que aislar `M` es *el objetivo*. Defendible — pero el paper tiene que
   **decirlo explícitamente** y reportar la degradación del codificador al lado,
   porque ahí está el desastre real.
2. Hay una objeción más incómoda: si el codificador se mueve, un `M` "intacto" lo
   está en un sistema de coordenadas que ya nadie produce. En despliegue `M`
   recibe latentes del codificador **nuevo**, no del de `model_i`. Medir olvido de
   `M` en una base congelada mide algo, pero no la degradación del sistema.
3. Explica los **PF negativos** de R9 (`finetuning` −1.78, `replay_infinite`
   −3.25), que como "olvido negativo" no tenían lectura.
4. `ug_mtm` da 7.663551597595215 → 7.663551597595215, **idéntico bit a bit**:
   congela el VAE al activar el experto nuevo, así que su aislamiento del
   codificador es total *por construcción*. El precio está en la otra columna: su
   reconstrucción al final de la tarea B es **533.19** frente a **19.66** de
   `finetuning`. Prácticamente no aprende la tarea B. Es la cuantificación de lo
   que F3 describía en cualitativo.

**Qué hacer.** Es material de decisión, no de arreglo:

- Como mínimo, reportar `heldout_reconstruction_A_after_task_{A,B}` en el paper
  junto a PF y RD, y declarar el alcance ("medimos `M` en una base latente fija").
  Ya se guarda en cada `metrics.json`.
- Considerar una métrica compuesta que reencode con el codificador vigente. Es
  una definición nueva, no un parche, y entra en el mismo saco que P7.
- Al interpretar UG-MTM: distinguir "no olvida" de "no aprende". Con estos dos
  números la distinción es medible.

---

## F19 — Las métricas de UG-MTM son estocásticas en evaluación · ABIERTO / MENOR

Corolario de F18, medido en la misma corrida. La puerta de UG-MTM llama a
`compute_uncertainty(..., T=T_eval)` también en `eval()`, y eso pone la cabeza de
incertidumbre en `train()` para que el dropout siga activo (es deliberado, ver el
comentario en `ug_mtm.transition`). Consecuencia: **dos evaluaciones idénticas
del mismo modelo sobre el mismo `D_A` no dan el mismo número.**

Medido: `pf` frente a `nll_A_after_task_B − nll_A_after_task_A`, dos evaluaciones
independientes de la misma cosa.

| Método | Discrepancia |
| --- | --- |
| `finetuning` | 0.000e+00 |
| `ewc` | 0.000e+00 |
| `ug_mtm` | **−1.398e-02** |

Sobre un PF de 5.58 es un 0.25%: pequeño, pero **UG-MTM es el único método cuyas
métricas llevan ruido de medición**, y no se promedia sobre repeticiones. Al
comparar contra cuatro estimadores deterministas, eso es una asimetría que hay
que declarar o eliminar (promediando varias evaluaciones).

No invalida nada: las corridas siguen siendo reproducibles bit a bit, porque el
flujo aleatorio está sembrado. El ruido es entre *evaluaciones*, no entre
*ejecuciones*.

---

## F20 — FT no mide transferencia hacia delante · CORREGIDO (s7)

Encontrado en R12 (sesión 6), al comparar `finetuning` con `replay_infinite` con
5 semillas. La pista fue una regularidad demasiado limpia:

```
### minigrid / distance_med  (n=5)
  ft   replay_infinite=+18.723  finetuning=+18.723  delta=+0.000  perm p=1.0000
### minigrid / distance_max  (n=5)
  ft   replay_infinite=+18.326  finetuning=+18.326  delta=+0.000  perm p=1.0000
```

**Delta exactamente 0.000 en las 10 semillas.** Ya se veía en R9 (los tres métodos
con arquitectura RSSM daban FT idéntico hasta el último decimal), pero se leyó
como una curiosidad en vez de como lo que es.

### Por qué es estructural

```python
nll_A_after_task_A = compute_nll(model_i, eval_ds, device)   # post-tarea-A
nll_A_random_init  = compute_nll(model_rand, eval_ds, device)
ft = nll_A_random_init - nll_A_after_task_A
```

`model_i` es el modelo **antes** del cambio de tarea y `eval_ds` es `D_A`. **En el
cálculo no entra ni un dato de la tarea B.** Y los métodos de continual learning
—finetuning, replay, EWC— solo se diferencian *en el cambio de tarea y después*.
Con la misma semilla comparten datos, inicialización y entrenamiento de la tarea
A, así que `model_i` es el mismo modelo bit a bit y FT es el mismo número.

**FT no puede distinguir métodos que compartan arquitectura. Por construcción, no
por falta de potencia estadística.**

### Y además no mide lo que dice medir

El docstring dice: *"Positive = prior knowledge helped learn new task faster"*.
Pero se evalúa sobre `D_A`, con el modelo que aún no ha visto la tarea B. Lo que
mide es **cuánto mejor que un modelo aleatorio es el modelo en la tarea A tras
entrenar en la tarea A**. Eso es ajuste a la tarea A — de hecho es exactamente la
señal que F17 necesitaba, y es útil como tal — pero no es transferencia.

Transferencia hacia delante sería: ¿aprende la tarea B más rápido / mejor un
modelo preentrenado en A que uno desde cero? Eso exige medir sobre `D_B` y
comparar contra una referencia entrenada solo en B. No se calcula en ningún sitio.

### Consecuencias

1. **El Finding 5 del paper** ("Progressive Networks exhibit forward transfer") no
   se sostiene, y ahora hay un motivo estructural además del estadístico que ya
   señalaba `paper-vs-code.md` (p entre 0.23 y 0.63). Cualquier diferencia de FT
   entre métodos de la **misma** arquitectura es cero; entre arquitecturas
   distintas mide la arquitectura, no la transferencia. Progressive Nets y UG-MTM
   dan FT distinto porque su modelo es distinto desde la tarea A, no porque
   transfieran.
2. **Cualquier diferencia pequeña y no nula de FT entre métodos de la misma
   arquitectura es un artefacto del flujo aleatorio**, no señal: `model_rand` se
   construye después de entrenar la tarea B, así que un método que consuma RNG de
   forma distinta durante el cambio de tarea desplaza los pesos de la referencia.
3. **Con k=2 no hay nada que rescatar** renombrando. O se mide sobre `D_B` contra
   una referencia entrenada solo en B (coste: +45 entrenamientos, el mismo que
   pide P8 para `d_trans` — probablemente se pueden compartir), o se renombra a lo
   que de verdad es (calidad en la tarea A) y se saca de la lista de métricas de
   transferencia.

**Recomendación:** renombrarlo. Ya se guarda como `nll_A_after_task_A` y
`nll_A_random_init` en cada `metrics.json`, así que la cifra sigue estando y sigue
siendo útil — es la evidencia de F17. Lo que hay que quitar es la etiqueta
"forward transfer" y las conclusiones que se apoyaban en ella.

### Corregido (sesión 7): se mide de verdad

El autor eligió pagar los +45 entrenamientos (D11), así que no se renombró y ya
está: se hicieron **las dos cosas**.

- La cifra vieja sobrevive con su nombre real, `task_A_fit_gain`, y con un
  docstring que dice que no entra ningún dato de la tarea B.
- `ft` pasa a ser **transferencia hacia delante de verdad**:
  `recon_B(desde cero) − recon_B(preentrenado en A)`, sobre episodios reservados
  de la tarea B, con el mismo presupuesto y los mismos datos en los dos brazos.

**En píxeles, no en NLL latente, y el motivo importa.** Los dos modelos se
entrenaron por separado, así que sus espacios latentes son bases distintas y sin
relación: una NLL latente puntuaría a uno de ellos en coordenadas del otro. Los
píxeles son la única escala que ambos comparten. (Es el mismo razonamiento de
D5, aplicado entre modelos en vez de entre instantes.)

**Y ahora discrimina.** Primera medición (smoke, 200 pasos, MiniGrid
`distance_med`, semilla 7):

| Método | FT | Lectura |
| --- | --- | --- |
| `finetuning` | **+54.69** | preentrenar en A ayudó a aprender B |
| `ug_mtm` | **−516.10** | interferencia masiva: congela el VAE y no puede aprender los píxeles de B |

Frente al delta **exactamente 0.000** que daba la métrica vieja en 10/10
semillas. La diferencia entre las dos cifras de UG-MTM es el mismo hecho que
cuenta F18 visto desde el otro lado: lo que le da inmunidad al olvido es lo que
le impide aprender.

**Salvedad que hay que declarar:** para `replay_infinite` el brazo preentrenado
entrena sobre A+B durante la fase B, así que gasta la mitad de sus pasos en datos
de A. Su FT mezcla transferencia con la mitad de presupuesto efectivo sobre B.

---

## F21 — El Fisher de EWC no cubre ni el codificador ni la vía recurrente · ABIERTO (documentado)

Salió de arreglar I3 (sesión 7). No invalida nada, pero explica un resultado que
en R10 se había quedado como enigma: **EWC no protege el codificador en
absoluto** (718.01 de reconstrucción de la tarea A frente a 725.27 de
`finetuning`), aunque su PF sí saliera mejor.

El Fisher se define sobre `log P(z'|z, a)`. Medido, con 20 parámetros de VAE y
6 de transición:

```
parámetros del VAE:        20,  max(fisher) = 0.0
gru.weight_ih  = 9.1693     gru.bias_ih = 0.9024
gru.weight_hh  = 0.0        gru.bias_hh = 0.2434
stoch_fc.weight = 8.3276    stoch_fc.bias = 11.4645
```

**Dos ceros exactos, cada uno por su motivo.**

1. **El codificador.** Los parámetros del VAE no entran en `log P(z'|z, a)`, así
   que su Fisher es exactamente 0 y la penalización nunca los toca. EWC protege
   `M` y solo `M`: la ceguera de F18 no está solo en las métricas, está también
   en la mitigación. No es un bug —el benchmark mide `M` a propósito— pero
   conviene decirlo, porque "EWC no protegió el codificador" suena a fallo de
   EWC y es una consecuencia de cómo está definido aquí.
2. **`gru.weight_hh`.** El conjunto del Fisher son transiciones sueltas que
   arrancan de `h = 0`, así que los pesos recurrentes no contribuyen a la
   verosimilitud y no reciben masa de Fisher: **262144 parámetros sin proteger**.
   `compute_nll` también puntúa desde `h = 0`, así que PF tampoco los ve — pero
   RD despliega 15 pasos y sí los usa. Esa asimetría es reportable: RD puede
   medir olvido en una vía que ni la métrica de un paso ni la penalización de EWC
   tocan.

Los dos ceros están fijados con un test cada uno, con su explicación, para que no
cambien en silencio.

**Confirmado a escala (R16).** Con 3 niveles × 5 semillas en MiniGrid, EWC da
PF de **−0.06 / −0.01 / +0.08** — conserva `M` casi exactamente, que es lo que su
Fisher cubre — y una reconstrucción de la tarea A de **403 / 693 / 1287**,
indistinguible de la de `finetuning` (399 / 683 / 1339), que es lo que su Fisher
no cubre. La predicción estructural y la medición coinciden en las dos mitades.
De paso, es el caso extremo de F14: como su PF es cero, **RD aporta el 99.7–99.9%
de su WMF**. **Qué decidir:** si el Fisher debe estimarse sobre
secuencias (con `h` arrastrado) en vez de sobre transiciones desde cero. Cambia
lo que EWC protege, así que es una decisión de diseño, no una corrección.

---

## F22 — Hay celdas que no producen olvido · CERRADO por D16 (sesión 8)

> **Contadas las nueve celdas, son tres, y la lista no es la que se asumía.**
> `gymnasium/distance_min` (−9.7%..+0.0% de pérdida de la tarea A sobre los
> métodos), `gymnasium/distance_med` (−8.2%..+6.0%) y `dmcontrol/distance_min`
> (+0.0%..+1.6%), frente a hasta **+75244%** en las que sí olvidan.
>
> 1. **`dmcontrol/distance_min` no es un control en píxeles absolutos** —
>    degrada +0.23— sino en relativo: +0.23 sobre su base de 14.74 es un 1.6%,
>    y el mismo +0.23 sobre los 0.52 de MiniGrid sería un 44%. El criterio tiene
>    que ser relativo o las familias no se pueden comparar.
> 2. **`gymnasium/distance_med` sí lo es, y no estaba en ninguna lista previa.**
>    No pierde nada del codificador y tiene a la vez **el RD más alto de su
>    familia** (85.9). Es F18 por el otro lado: la transición olvida donde el
>    codificador no. Por eso el criterio de D16 **no consulta RD** y la sección
>    imprime los números en vez de un veredicto.
>
> Rejilla efectiva: **6 de 9 celdas**. En Gymnasium el olvido a nivel de
> codificador **solo aparece en el nivel máximo**.

Salió de R16, con `gymnasium/distance_min` completo: **ninguno de los cinco
métodos olvida nada**. La reconstrucción de la tarea A en píxeles **mejora** tras
entrenar en la tarea B, para los cinco:

| Método | recon A: tras A → tras B | PF |
| --- | --- | --- |
| `finetuning` | 27.77 → **25.25** | −8.20 |
| `replay_infinite` | 27.77 → **24.69** | −8.16 |
| `ewc` | 27.77 → 25.24 | +0.02 |
| `progressive_nets` | 27.77 → 25.71 | −0.16 |
| `ug_mtm` | 27.16 → 27.16 | −0.06 |

**No es un bug, y en parte es lo correcto.** Las dos tareas son el mismo
HalfCheetah con gravedad 9.8 y 7.0: visualmente casi idénticos, así que entrenar
en B sigue enseñando a reconstruir A. Un nivel de distancia mínima *debería*
producir poco olvido — es el control del eje de distancia.

**Lo que sí es un problema** es que una celda donde nadie olvida tampoco
discrimina: PF sale a −8.2 para dos métodos y a 0 para otros dos, pero eso mide
cuánto sigue mejorando cada uno, no cuánto olvida. Con 9 celdas, que una sea un
control es defendible; hay que ver cuántas lo son cuando termine R16.

Y hay un efecto colateral interesante: **el handicap de UG-MTM desaparece cuando
las tareas se parecen**. Su reconstrucción de B es 31.52 frente a ~25 del resto,
no los 557 frente a 2 de MiniGrid. Congelar el codificador cuesta poco si el
codificador viejo sirve.

**Qué decidir (P13).** Si estas celdas se reportan como control declarado —"a
distancia mínima no hay olvido que medir, y el banco lo detecta"— o si el nivel
mínimo de Gymnasium necesita una perturbación mayor. Ojo: cambiarla invalida esas
celdas y hay que reejecutarlas.

---

## F23 — RD de UG-MTM estalla en una celda · CERRADO por D15 (sesión 8)

> Diagnosticado en R17 (colapso de varianza en el modelo post-B) y **reportable
> desde D15**: la celda sale como `+520.4 [17.72, 4364]!`, con el `!` y las cinco
> semillas listadas debajo. Ya no bloquea nada; se cuenta como resultado sobre
> UG-MTM.
>
> Lo que sí destapó al implementarlo: **22 de las 180 celdas de pf/rd/ft/wmf
> están marcadas por sesgo a la derecha**, no solo esta. La cola pesada es
> frecuente con n=5, y eso es la justificación empírica de la política, más allá
> de esta casilla.

`ug_mtm` en `minigrid/distance_max`, R16, RD por semilla:

```
574.5   520.4   4364.5   17.7   40.0        media 1103.4 ± 1647.0
```

**Tres órdenes de magnitud entre semillas del mismo método en la misma celda.**
El resto de métodos en esa casilla van de 22.9 a 52.7 con desviaciones de 2–8.

Es el outlier de F14 (que era `RD = 167` con una semilla) crecido con datos
reales, y se solapa con F19 (las métricas de UG-MTM llevan ruido de medición
porque la puerta mantiene MC-dropout activo en evaluación). Pero 4364 frente a
17.7 **no es ruido de MC-dropout**: es que en algunas semillas el rollout
imaginado diverge y la KL a 15 pasos se dispara.

No se puede publicar la media de esa celda sin entenderlo. Tres cosas que mirar,
ninguna cara: la curva de KL por paso del rollout (¿en qué paso explota?), si la
semilla mala coincide con una puerta que enruta al experto equivocado (P1/F3), y
si RD necesita truncarse o acotarse — que era la opción 3 de P7, descartada en su
día precisamente porque la explosión de entonces resultó ser el bug F13. Esta no
lo es.

### Diagnosticado (sesión 7): es colapso de varianza, no divergencia del rollout

`_devlog/diagnose-p12.py` reproduce la celda entera y descompone la KL paso a
paso. **Reproduce ambas semillas exactamente** — RD 4364.4502 contra 4364.4502
guardado, PF 39.9796 contra 39.9796 — así que el diagnóstico es sobre el objeto
real, no sobre una reejecución parecida.

Semilla 2 (RD = 4364) contra semilla 3 (RD = 17.7), términos de la KL:

| | s2 paso 0 | s2 paso 4 | s2 paso 14 | s3 paso 0 | s3 paso 14 |
| --- | --- | --- | --- | --- | --- |
| KL | 80.3 | **7872.0** | 2966.5 | 20.6 | 15.2 |
| término de medias | 58.2 | **7690.8** | 2926.4 | 17.5 | 5.1 |
| término de traza | 41.5 | 202.9 | 57.4 | 15.9 | 8.6 |
| término logarítmico | −3.4 | −5.8 | −1.3 | 3.1 | 17.5 |
| `min log_var_k` | −7.26 | **−9.70** | −8.41 | −4.16 | −0.99 |
| \|mu_i\| / \|mu_k\| | 0.28/0.27 | 0.26/0.43 | 0.29/0.57 | 0.34/0.33 | 0.35/0.60 |

**Las medias no divergen.** `|mu_i|` se queda clavado en ~0.27 los quince pasos y
`|mu_k|` sube de 0.27 a 0.57: nada que explique un factor de 250. El rollout no
se va a infinito.

**Lo que explota es el denominador.** En la semilla mala, el modelo post-B tiene
dimensiones latentes con `log_var = −9.9`, o sea **σ = 0.007**; en la semilla
buena el mínimo es −4.2 (σ = 0.12) y va *subiendo* hasta −1.0 (σ = 0.61). El
término de medias es `(mu_i − mu_k)² / var_k`, así que con `var_k ≈ 5e-05` una
diferencia de medias corriente da miles.

**Y la KL es asimétrica en la dirección que castiga esto.** `KL(P_A ‖ P_B)`
penaliza que el modelo **post-B** esté muy seguro donde el post-A no lo estaba.
UG-MTM congela el VAE y entrena un experto nuevo sobre un espacio latente que no
se mueve; ese experto se vuelve **sobreconfiado**, y RD lo cobra sin cota.

Tres consecuencias:

1. **No es un bug numérico ni un fallo del horizonte.** La KL ya vale 80 en el
   paso 0 y 1685 en el paso 1: truncar el rollout no lo arregla.
2. **RD tiene cola pesada por construcción** cuando un método produce modelos
   sobreconfiados. Promediar sobre semillas una cantidad así no está justificado
   — la media de esa celda la fija una semilla.
3. **Es también un resultado sobre UG-MTM**, no solo sobre la métrica: congelar
   el codificador y añadir un experto produce modelos de transición
   sobreconfiados. Eso es reportable y es coherente con su PF alto (+39.98 en esa
   misma semilla): seguro y equivocado.

---

## F24 — Un solo proceso no puede ejecutar las tres familias · CORREGIDO (s7)

**Mató la ejecución completa 30 horas después de lanzarla.** MiniGrid y Gymnasium
terminaron (150 celdas, intactas en disco); al crear el primer entorno de
dm_control:

```
File "src/envs/dmcontrol_env.py", line 74, in reset
    rgb = self._env.physics.render(height=64, width=64, camera_id=0)
  ...
  File "dm_control/mujoco/wrapper/core.py", line 608, in __init__
    ptr = ctx.call(mujoco.MjrContext, model.ptr, font_scale)
mujoco.FatalError: Default framebuffer is not complete, error 0x0
```

MuJoCo (vía Gymnasium) y dm_control quieren los dos un contexto OpenGL, y el
segundo en pedirlo en el mismo proceso no lo consigue. `env.close()` no libera el
contexto; solo lo libera salir del proceso.

**Esto ya se conocía a medias, y ahí está la lección.** Era I20, catalogado como
molestia de la suite de tests, con esta nota: *"No afecta a
`run_full_benchmark.py`. Verificado explícitamente con su orden real de familias
(minigrid ×3 → gymnasium ×3 → dmcontrol ×3): pasa."* La comprobación se hizo y en
su momento pasó — el análisis de I20 incluso registra que `gym → dmc` funciona
aislado.

Lo que la comprobación no reprodujo fue la **escala**: en R16 la frontera se cruza
después de 15 celdas de gymnasium, con tres parejas de entornos abiertas y
cerradas y decenas de miles de fotogramas renderizados. Con eso, el mismo paso que
pasaba en seco falla. No sé decir cuál de esos factores es el detonante, y no
merece la pena averiguarlo: el arreglo elimina la categoría entera.

La lección para la bitácora no es "la verificación estaba mal hecha", es que
**verificar una secuencia en seco no verifica la misma secuencia bajo carga**, y
la nota debería haber dicho a qué escala se probó.

**Corregido.** `run_full_benchmark.py` ejecuta **cada familia en un subproceso**
cuando se le piden varias. Un intérprete nuevo por familia garantiza el contexto
limpio, y de paso un fallo en una familia cuesta esa familia y no la corrida.
`--dry-run` se queda en un solo proceso: solo imprime.

El riesgo del arreglo es el de F9 —construir una línea de comandos y olvidarse de
pasar la mitad de las banderas—, así que el test que importa **parsea de vuelta**
la línea generada y compara campo por campo contra la del padre, incluidos los
overrides que deben seguir siendo `None`.

**Coste real del incidente: cero resultados.** Las 150 celdas de MiniGrid y
Gymnasium estaban en disco con su protocolo, el runner las salta, y dmcontrol se
relanzó en proceso propio.

---

## F25 — `dmcontrol/distance_max` emparejaba tareas con distinto número de actuadores · CORREGIDO (s7)

Segundo fallo de R16, ocho horas después de F24 y con 200 celdas ya en disco.
`cheetah/run` tiene **6 actuadores** y `reacher/easy` **2**. El world model es
uno solo para las dos tareas, así que su GRU se construye con un único ancho de
entrada, `latent_dim + action_dim = 32 + 6 = 38`. Al entrenar la tarea B:

```
File "src/models/rssm.py", line 78, in transition
    h_next = self.gru(x, h)
RuntimeError: input has inconsistent input_size: got 34 expected 38
```

34 = 32 + 2. El par es **inejecutable por construcción**, y lo había sido desde
el primer día: nunca se había llegado a ejecutar esa celda porque las 225
corridas anteriores se hacían con el VAE colapsado y nadie miró el traceback, o
directamente no se llegó a esa casilla.

**Por qué no saltó antes.** El runner validaba `cfg.model.action_dim` contra
`env_A` y **solo contra `env_A`**. La tarea B no se comprobaba nunca.

### Lo que se ha hecho

**1. Comprobación previa, antes de entrenar nada.** `preflight_action_dims()`
construye los pares de la familia, valida **las dos** tareas contra el valor
declarado, y aborta listando todos los pares rotos de golpe. Tarda segundos:

```
configs/benchmark/dmcontrol.yaml declares action_dim=6 but its tasks do not agree:
  distance_max/task_B: exposes action_dim=2, config declares 6
```

**2. El par cambia a `cheetah/run → walker/stand`** (D13).

La alternativa era **rellenar las acciones de reacher con ceros** hasta 6, que
mantiene el par que declara el paper. Se descartó por un motivo concreto, no por
comodidad: durante los 5000 pasos de la tarea B, cuatro dimensiones de acción
estarían fijas a cero; después RD despliega rollouts de 15 pasos con acciones
tomadas de `D_A`, que sí las usan. Parte del "olvido" medido en esa celda sería
la respuesta del modelo a entradas que dejó de ver, no una diferencia de
dinámica — y sería distinta por método, porque cada uno entrena B de otra forma.
Un revisor lo pregunta y no hay respuesta barata.

`walker/stand` comparte los 6 actuadores de cheetah y cambia **el cuerpo y el
objetivo**, donde `distance_med` (`walker/run`) cambia solo el cuerpo.

**Y el orden de los niveles no se afirma, se mide.** `d_trans` de dmcontrol da
2.02 ± 0.54 en `min` y 6.99 ± 3.00 en `med`. Si `max` sale por encima de 6.99, el
eje de distancia queda validado por el propio instrumento del banco de pruebas.
**Si sale por debajo, hay que decirlo y renombrar el nivel** — está escrito aquí
antes de conocer el resultado, a propósito.

---

## F26 — En dm_control, dos "tareas" del mismo dominio son el mismo entorno · CORREGIDO (s7)

El más instructivo de la sesión, y salió de **equivocarme yo**.

Tras F25 puse `distance_max` en `cheetah/run → walker/stand`, razonando que
cambia cuerpo y objetivo donde `distance_med` (`walker/run`) cambia solo el
cuerpo. Las celdas salieron **bit a bit idénticas a las de `distance_med`** en
4 de las 5 semillas:

```
finetuning  semilla 0   pf med=-18.344229460  max=-18.344229460   delta 0
ug_mtm      semilla 3   pf med=  0.605660439  max=  0.605660439   delta 0
finetuning  semilla 4   pf med=-18.465326548  max=-19.089778900   delta 6.2e-01
```

**En dm_control las tareas de un dominio comparten el modelo físico y la
distribución de estado inicial; lo único que cambia es la función de
recompensa.** Este benchmark entrena un world model **sin recompensa** sobre
rollouts de política aleatoria, así que la recompensa no entra en ningún sitio:
`walker/run` y `walker/stand` son el mismo entorno para todo lo que aquí se mide.

### La semilla 4, y lo que enseña de propina

Medido sobre las 20 trayectorias completas de esa semilla: **un episodio de 20
difiere, en un solo paso de 500**, con las acciones idénticas y la trayectoria
volviendo a coincidir después. La diferencia es de **3.9e-03 ≈ 1/255**: un nivel
de gris de la cuantización a 8 bits del renderizador. No es una diferencia de
tarea, es redondeo.

Lo interesante es el efecto río abajo: **un píxel, un nivel de gris, un
fotograma de 10 000 → PF se mueve 0.62** (de −18.47 a −19.09) tras 5000 pasos de
entrenamiento. Es una medida de la sensibilidad caótica del pipeline, y acota
cuánta precisión tiene sentido reclamar: una diferencia de PF de ese orden entre
dos métodos no es señal. Las diferencias entre métodos que sí importan aquí son
de 6 a 20 unidades, un orden por encima — pero conviene tenerlo escrito.

Y confirma la reproducibilidad por el otro lado: recomputada la referencia de
`distance_med` semilla 4 en un proceso nuevo, sale **10.307263374 contra
10.307263374**, exacta. El pipeline reproduce; lo que no reproduce es un
renderizador que redondea distinto.

### Lo que esto implica para el diseño

Un nivel de distancia en dm_control **solo puede diferir por dominio o por
física**. Y de los dominios disponibles solo `cheetah` y `walker` tienen 6
actuadores (restricción de F25), con lo que el cambio de dominio ya está gastado
en `distance_med`.

`distance_max` pasa a ser el cambio de cuerpo **más** la perturbación física que
aplica el `distance_max` de Gymnasium (gravedad 4.0, masa ×3, fricción ×0.5). Es
estrictamente más cambio que `distance_med`, por construcción y no por argumento.

### El test que faltaba, y por qué el que había no bastaba

Existía `test_no_distance_level_compares_a_task_with_itself`, que compara los
**diccionarios** de configuración. `walker/run` y `walker/stand` son diccionarios
distintos. El test nuevo compara **los datos**: misma semilla, mismas acciones,
y los dos entornos tienen que divergir. Cubre las tres familias.

### Y el agujero que destapó

`metrics.json` guardaba el protocolo pero **no qué tareas se habían ejecutado**.
Cambiar el par en el YAML habría dejado 25 celdas viejas que el runner habría
reutilizado tan contento bajo el nombre del par nuevo. Ahora cada resultado
guarda su bloque `tasks`, y la comprobación de caché lo valida igual que el
protocolo.

Las 25 celdas duplicadas están en `archive/results-F26-dmc-max-duplicado/`, no
borradas: son la evidencia.

---

## F27 — El eje de distancia no predice el olvido, y la dificultad de la tarea B sí · ABIERTO (es el resultado principal)

Con las 225 celdas terminadas, la pregunta que el benchmark existía para
responder — *¿el olvido crece con la distancia dinámica?* — tiene respuesta, y no
es la que asume el diseño.

**El olvido hace pico en el nivel intermedio en las tres familias.** RD de los
cuatro métodos con arquitectura RSSM (mediana de medianas, D15/D17):

| Familia | min | med | max |
| --- | --- | --- | --- |
| minigrid | 33.3 | **62.3** | 35.6 |
| gymnasium | 77.5 | **85.9** | 59.6 |
| dmcontrol | 1.2 | **70.9** | 52.7 |

Tres de tres. Las etiquetas `min < med < max` **no aciertan el orden en ninguna
familia**.

### Qué sí lo predice

Correlación de rangos con RD sobre las 9 celdas:

| Predictor | Spearman |
| --- | --- |
| Nivel etiquetado (min/med/max) | **+0.05** |
| `d_trans` medida (Ec. 9) | **+0.53** |
| Reconstrucción de la tarea B (lo difícil que es B) | **+0.58** |

### El resultado sobrevive al cambio de agregación (sesión 8)

Las cifras de arriba son las de la política de reporte D15. Las primeras, con
media sobre semillas y media sobre métodos, eran 32.3/65.4/36.4 · 70.2/89.2/65.0
· 1.2/71.8/49.0 y Spearman +0.13 / +0.57 / +0.62.

**Nada cualitativo se mueve, y una cosa mejora:** el pico en `med` sigue saliendo
3 de 3, el orden de los tres predictores es el mismo, y el nivel etiquetado baja
de +0.13 a **+0.05** — o sea que la etiqueta predice todavía menos de lo que
parecía. Que el hallazgo principal no dependa de cómo se resuma cada celda es
justo lo que hacía falta comprobar antes de escribirlo.

Y ahora **sale del script**, no de una hoja aparte: `summarize_results.py`
imprime la tabla y las tres correlaciones, con la regla de inclusión de métodos
declarada (D17) y una Spearman escrita a mano, sin `scipy`, que sigue sin estar
en `requirements.txt` (F12).

`d_trans` acierta el orden exacto en gymnasium (24.4 < 28.8 < 29.8 frente a RD
65 < 70 < 89) y separa `min` del resto en dmcontrol; falla en minigrid. Las
etiquetas no aciertan en ninguna. **El instrumento del banco de pruebas ordena
mejor que el diseño a mano del banco de pruebas**, que es a la vez un aval de
`d_trans` y una enmienda al eje de distancias.

### La lectura, que es la interesante

Mirando la reconstrucción de la tarea B al lado de RD:

| Familia | recon B (min/med/max) | RD (min/med/max) |
| --- | --- | --- |
| minigrid | 2.6 / **44.1** / 21.7 | 32.3 / **65.4** / 36.4 |
| gymnasium | 25.0 / **28.2** / 18.9 | 70.2 / **89.2** / 65.0 |
| dmcontrol | 14.8 / 37.7 / **41.1** | 1.2 / **71.8** / 49.0 |

En minigrid y gymnasium las dos columnas suben y bajan juntas. **El olvido no lo
determina cuán lejos está la tarea B, sino cuánto tiene que cambiar el modelo
para ajustarla** — y eso escala con la dificultad intrínseca de B, no con la
distancia A→B. Una tarea B lejana pero pobre (el walker perturbado de dmcontrol
se derrumba y se arrastra; el reacher habría sido peor) enseña poco, así que
sobrescribe poco.

**Qué significa para el paper.** El abstract prometía "controlled dynamic
distances" y el eje controlado no controla lo que se creía. Hay tres salidas y
solo la primera es honesta:

1. **Reportarlo como el hallazgo que es**: el olvido en un world model escala con
   lo que la tarea nueva exige, no con lo lejos que está; el eje ordinal del
   diseño no lo captura y `d_trans` lo captura a medias. Es una afirmación
   comprobable, cuesta cero cómputo, y es más interesante que "el método X gana".
2. Rediseñar los niveles para que sean monótonos en `d_trans`. Caro y circular:
   se estaría eligiendo el eje para que salga la respuesta esperada.
3. No mencionarlo. Descartada: sale de las tablas que el propio paper publica.

**Salvedad de tamaño**: 9 celdas, 3 por familia. Las correlaciones de rangos con
n=9 y familias de escalas muy distintas son indicativas, no concluyentes; lo
robusto es el pico en `med`, que se repite 3 de 3.


---

## F28 — `d_trans` evalúa el modelo B en la base latente de A · ABIERTO (declarado en el paper)

Salió al escribir el método (sesión 9). `run_reference_cell` entrena dos RSSM
planos **independientes**, uno por entorno, y luego llama a `compute_d_trans`
sobre un `shared_ds` construido con `build_latent_eval_dataset(model_A, ...)`.
Es decir: `P_B(z'|z,a)` se puntúa sobre latentes que produjo el codificador de
`model_A`, una base que `model_B` no ha visto nunca.

**Es exactamente la objeción que llevó a rehacer FT en píxeles** (F20): "los dos
modelos se entrenaron por separado, así que sus espacios latentes son bases
distintas y sin relación; una NLL latente puntuaría a uno en coordenadas del
otro". El razonamiento se aplicó a FT y no a `d_trans`. El comentario del runner
justifica la elección de la tarea A ("deja la KL en el mismo soporte `(z,a)` que
PF y RD") pero no aborda que el segundo modelo esté en otra base.

**Qué implica, y qué no.**

- Parte de lo que mide `d_trans` es desalineamiento de bases, no diferencia de
  dinámica. Explica por qué las magnitudes no son comparables entre familias
  (minigrid ~20, dmcontrol ~5) y es una causa candidata de que no ordene
  MiniGrid (20.07 / 19.59 / 24.85, invertido y dentro del ruido entre semillas).
- **No invalida F27.** El titular es el negativo —el eje etiquetado no ordena
  (ρ=+0.05)— y eso no depende de `d_trans`. El caso de Gymnasium (superconjunto
  estricto que olvida menos) tampoco lo usa. Lo que sí se debilita es la
  recomendación de §6.1 de la discusión, que propone `d_trans` como el
  instrumento a reportar.
- Tampoco es reparable en píxeles como FT: una distancia entre *dinámicas* vive
  en el espacio de transición. Las salidas plausibles son (a) entrenar la pareja
  de referencia con un codificador compartido, congelado, entrenado sobre A+B,
  o (b) declararlo y usar `d_trans` solo como orden dentro de familia.

**Decidido para esta versión: (b), declararlo.** Está escrito en
`paper/method.tex` §3.3 y el paper solo usa `d_trans` para ordenar. (a) es una
corrida de 45 parejas y no bloquea escribir. **Pendiente**: llevar la salvedad
también a §6.1 de la discusión, que hoy recomienda `d_trans` sin ella.

---

## F29 — `d_trans` no separa los niveles dentro de una familia, y se mueve con el presupuesto · ABIERTO (declarado en el paper)

Salió del sondeo R18. Empezó como «invierte al doblar el presupuesto» y al mirar
las dispersiones resultó ser algo peor: **nunca separó nada dentro de una
familia.**

### Las tres celdas de Gymnasium, a 5000 pasos

| Nivel | mediana | rango sobre 5 semillas |
| --- | --- | --- |
| min | 28.75 | [21.54, 37.05] |
| med | 30.86 | [18.52, 37.24] |
| max | 24.71 | [17.66, 30.13] |

Las medianas abarcan **6 unidades**; cada celda abarca **15–19 entre semillas**.
El orden de rangos que el paper reportaba es un orden sobre medianas que están
dentro del ruido unas de otras.

### Y a 10000 pasos las dos de arriba se intercambian

| Nivel | 5000 | 10000 |
| --- | --- | --- |
| med | 30.86 | **57.30** [41.85, 72.66] |
| max | 24.71 | **71.51** [41.81, 72.05] |

Dos cosas a la vez, y las dos malas:

1. **El orden se da la vuelta.** A 5000 `d_trans` se ponía del lado del
   resultado (max por debajo de med, como RD); a 10000 se pone del lado de la
   construcción y en contra del resultado. La Spearman de esas dos celdas pasa
   de +1.00 a −1.00.
2. **La magnitud casi se dobla.** Una distancia entre dos *entornos* no debería
   depender de cuánto se entrenen los modelos con los que se mide. Depende. Y
   los rangos a 10000 son prácticamente idénticos entre las dos celdas
   ([41.85, 72.66] contra [41.81, 72.05]): no es que ordene mal, es que no
   resuelve.

### Qué implica

- **F27 no se toca.** El titular es el negativo sobre el eje etiquetado
  (ρ = +0.05), y eso no pasa por `d_trans`. El caso de Gymnasium tampoco: es un
  argumento de construcción (superconjunto estricto) más RD medido.
- **Lo que se cae es la mitad positiva.** La frase «`d_trans` ordena mejor que
  la etiqueta (+0.53 frente a +0.05)» sigue siendo cierta como número, pero
  ahora sabemos que **buena parte de ese +0.53 lo carga la separación ENTRE
  familias** —dmcontrol ~5 contra minigrid ~20—, que es justo la comparación que
  F28 dice que no está legitimado a hacer. Las dos objeciones se refuerzan.
- La recomendación de §6.1 baja de «reportad `d_trans`» a «reportad una
  distancia medida y **exigidle cuentas**» — que es más débil como titular y lo
  que los datos sostienen.

### Lo curioso, y va en el paper

**Fue la cantidad medida la que expuso sus propios límites al cambiar el
presupuesto, y la etiqueta la que no pudo**, porque una etiqueta no tiene de qué
equivocarse. Eso es un argumento a favor de medir, aunque el instrumento
concreto salga tocado.

### Arreglos, ninguno ejecutado

1. **Más semillas.** Con n=5 y esas dispersiones no hay nada que resolver.
2. **Codificador compartido y congelado** para la pareja de referencia (el
   arreglo de F28). Quitaría el desalineamiento de bases, que es candidato a
   causa de la varianza y de la deriva con el presupuesto.

Corregido en `results.tex` §5.2 y §5.3, `discussion.tex` §6.1 y §6.2, y en las
amenazas a la validez.

---

## F30 — El pico de k=4 confunde posición con identidad de tarea · ABIERTO (no toca el paper enviado; es el experimento para CoLLAs)

Salió de la auditoría externa de la sesión 17, leyendo §5.7 con ojos de revisor.
No es un defecto de instrumento como F0–F26 ni una limitación declarada como
F28/F29: es **un experimento que falta**, y es el más barato que queda.

### Qué dice hoy el paper

`results.tex` §5.7 reporta que el olvido de `T_1` **hace pico en la tercera
tarea y remite en la cuarta**, en las cinco semillas de cada uno de los tres
métodos que no protegen el codificador, y no lo hace en los dos que sí. El
argumento textual es:

> «A peak across levels can be attributed to how the levels were constructed. A
> peak across position, inside one sequence, cannot.»

La explicación que se ofrece es la fila de dificultad: `T_3` (FourRooms) es la
tarea más difícil de ajustar de la secuencia — **47.32**, frente a 32.58 de
`T_4` (KeyCorridor) y 2.60 de `T_2` (Empty-8x8).

### El agujero

R20 ejecutó **un solo orden**: `Empty-5x5 → Empty-8x8 → FourRooms →
KeyCorridorS3R1`. Con un único orden, «posición 3» y «FourRooms» son la misma
columna de datos. La frase de arriba está mal enunciada: un pico sobre posición
en una sola secuencia **tampoco** distingue entre

1. el olvido sigue a la demanda de la tarea que se aprende (la cuenta de F27), y
2. el olvido depende del número de cambios de tarea, con un tercer cambio
   especial por lo que sea (saturación del buffer, del Fisher, de las columnas).

La partición entre métodos —que el pico aparezca solo donde el codificador está
desprotegido— **descarta ruido, no el confundido**: dice que algo se mueve, no
qué lo indexa.

### El experimento que lo resuelve, y su predicción

Contrabalancear el orden. `T_1` se queda fijo en `Empty-5x5`, porque es la tarea
cuya retención se mide y cambiarla haría incomparables las curvas; se permutan
las posiciones 2–4 para que la tarea difícil caiga en sitios distintos:

| Orden | Secuencia | FourRooms en |
| --- | --- | --- |
| A (=R20) | Empty-5x5 → Empty-8x8 → FourRooms → KeyCorridor | posición 3 |
| B | Empty-5x5 → FourRooms → Empty-8x8 → KeyCorridor | posición 2 |
| C | Empty-5x5 → Empty-8x8 → KeyCorridor → FourRooms | posición 4 |

**Predicción escrita antes de correr nada:** si la cuenta de F27 es correcta, el
pico de RD(`T_1`) **sigue a FourRooms** — aparece tras la etapa 2 en el orden B y
tras la 4 en el C. Si se queda clavado en la posición 3 en los tres órdenes, la
lectura de §5.7 es falsa y hay que retirarla.

**El orden B es el que discrimina; el C solo no vale.** Con tres puntos de
medida (tras `T_2`, `T_3`, `T_4`), un pico en la última etapa es
indistinguible de una acumulación monótona. El orden B produce la forma que
ningún relato de acumulación puede imitar: sube en la etapa 2 y **baja** dos
veces seguidas.

### Coste

Una permutación = lo mismo que R20: 25 corridas (5 métodos × 5 semillas), 100
entrenamientos a 5000 pasos, en MiniGrid, que es la familia barata. El runner ya
acepta todo lo necesario y **no hay que tocar código**: una config nueva por
orden y un directorio de resultados por orden.

```bash
python experiments/run_sequence.py --config configs/benchmark/minigrid_sequence_B.yaml --results-dir results-seq-B
```

La config nueva se copia de `minigrid_sequence.yaml` reordenando la lista
`tasks:`, y el bloque `protocol:` se deja **idéntico** — es la razón por la que
ese bloque está duplicado y no compartido (ver la nota en el propio fichero).

### Por qué importa para CoLLAs y no para el taller

En la versión de 8 páginas la secuencia k=4 no aparece (D23), así que **esto no
afecta a lo enviado**. En el paper largo la subsección ya se presenta como
comprobación de consistencia y no como confirmación independiente, así que
tampoco es una afirmación falsa hoy. Lo que cambia con el contrabalanceo es la
categoría del resultado: de «el mecanismo también predice lo que pasa a k=4» a
«el olvido sigue a la tarea, no a la posición», que es un resultado por derecho
propio y sobre un eje —la posición dentro de una secuencia— que no se puede
achacar a cómo se construyeron los niveles. Es lo más barato que sube el paper
de un resultado y medio a dos.
