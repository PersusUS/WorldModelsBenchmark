"""
Relleno único del campo `tasks` en los resultados de R16 anteriores a F26.

Las 200 celdas de R16 y sus 40 referencias se produjeron antes de que
`metrics.json` guardara qué par de tareas se había ejecutado, así que la
comprobación de caché no puede verificarlas y las rechaza — correctamente.

Rellenarlas es legítimo aquí y sólo aquí, porque el par que ejecutaron es
verificable: de los nueve pares del benchmark, el único que ha cambiado desde
que arrancó R16 el 30 jul es `dmcontrol/distance_max`, y sus celdas están
archivadas en `archive/results-F26-dmc-max-duplicado/`, fuera de `results/`.
Para los ocho restantes, el bloque `sequences` del config de hoy es literalmente
el que corrió.

    git log --oneline -- cf_worldmodels/configs/benchmark/

confirma que los cambios posteriores al lanzamiento tocan sólo ese par.

Alternativa descartada: reejecutar 200 celdas (~40 h de GPU) para añadir un
campo de procedencia.

Uso:
    python _devlog/backfill-tasks.py --dry-run
    python _devlog/backfill-tasks.py
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cf_worldmodels"))

from omegaconf import OmegaConf  # noqa: E402

from experiments.run_full_benchmark import (  # noqa: E402
    FAMILY_CONFIGS,
    task_spec,
)

RESULTS = ROOT / "cf_worldmodels" / "results"
# El par que cambió después de que R16 arrancara. Sus celdas están archivadas;
# si alguna sigue en results/, este script se niega a tocarla.
CHANGED_SINCE_THE_RUN = {("dmcontrol", "distance_max")}


def specs():
    out = {}
    for family, path in FAMILY_CONFIGS.items():
        cfg = OmegaConf.load(ROOT / "cf_worldmodels" / path)
        for distance, seq in cfg.benchmark.sequences.items():
            out[(family, str(distance))] = task_spec(seq)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    by_cell = specs()
    paths = sorted(RESULTS.glob("*/*/metrics.json")) + \
        sorted(RESULTS.glob("_reference/*.json"))

    filled, already, refused = 0, 0, []
    for path in paths:
        stored = json.loads(path.read_text())
        key = (stored.get("family"), stored.get("distance"))
        if stored.get("tasks") is not None:
            already += 1
            continue
        if key in CHANGED_SINCE_THE_RUN:
            refused.append(path)
            continue
        if key not in by_cell:
            refused.append(path)
            continue
        stored["tasks"] = by_cell[key]
        if not args.dry_run:
            path.write_text(json.dumps(stored, indent=2))
        filled += 1

    print(f"{filled} rellenados, {already} ya lo tenían, "
          f"{len(refused)} rechazados")
    for path in refused:
        print(f"  RECHAZADO (par cambiado desde la corrida): {path}")
    if args.dry_run:
        print("--dry-run: no se ha escrito nada.")
    return 1 if refused else 0


if __name__ == "__main__":
    raise SystemExit(main())
