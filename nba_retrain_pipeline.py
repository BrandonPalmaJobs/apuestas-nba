"""
Reentrenamiento periodico de los 6 modelos de NBA (equipo: total/margen/
money line, props: puntos/rebotes/asistencias) - el "aprendizaje continuo"
del sistema: cada vez que corres esto, los modelos se re-entrenan con
TODOS los juegos jugados hasta hoy (mas datos que la vez anterior).

Que hace, en orden:
  1. Recolecta el dataset historico de equipos (nba_train_data.py) - la
     PRIMERA vez tarda ~1.5-2h (liga completa), pero el cache en disco
     (cache/game_summaries.json, se sube a GitHub junto con los modelos)
     hace que las corridas siguientes sean mucho mas rapidas: solo se
     piden los juegos NUEVOS desde la ultima vez.
  2. Reentrena los 3 modelos de equipo (nba_train.py): puntos totales,
     margen local (spread), money line.
  3. Recolecta el dataset historico de props de jugador (nba_props_data.py)
     - reusa el mismo cache de equipo del paso 1, solo agrega roster +
     estadisticas por jugador (lo nuevo y mas lento, ~1.5-2h la primera vez).
  4. Reentrena los 3 modelos de props (nba_props_train.py): puntos,
     rebotes, asistencias.

Uso:
    python nba_retrain_pipeline.py
    python nba_retrain_pipeline.py --force          # ignora el freno de dias
    python nba_retrain_pipeline.py --skip-props      # solo equipo, mas rapido
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime


def days_since_last_retrain(history_path="training_history_nba.csv"):
    """Dias transcurridos desde el timestamp de la ULTIMA fila de
    training_history_nba.csv - None si el archivo no existe o esta vacio
    (nunca se ha reentrenado, no hay razon para saltarse nada)."""
    try:
        with open(history_path, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        return None
    if not rows:
        return None
    last_ts = datetime.fromisoformat(rows[-1]["timestamp"])
    return (datetime.now() - last_ts).total_seconds() / 86400


def run(cmd, description):
    print(f"\n{'='*72}\n{description}\n{'='*72}")
    result = subprocess.run([sys.executable, "-u"] + cmd)
    if result.returncode != 0:
        print(f"ADVERTENCIA: '{description}' termino con codigo {result.returncode}, "
              f"revisa el error arriba antes de confiar en el resultado.")
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Reentrena los modelos de NBA con los datos mas recientes")
    parser.add_argument("--season", type=int, default=None,
                         help="Anio en que TERMINA la temporada. Default: temporada actual")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-prior", type=int, default=5)
    parser.add_argument("--skip-props", action="store_true",
                         help="Solo reentrena los modelos de equipo (total/margen/money line), "
                              "omite la recoleccion de props de jugador (mucho mas lenta)")
    parser.add_argument("--min-days-between", type=float, default=3,
                         help="No reentrena si el ultimo reentrenamiento (segun training_history_nba.csv) "
                              "fue hace menos de N dias (default 3). Se usa junto con un disparador DIARIO "
                              "en GitHub Actions en vez de un cron de 'cada 3 dias' exacto - GitHub a veces "
                              "retrasa o salta corridas programadas sin avisar, y con un disparador diario + "
                              "este freno el peor caso es reentrenar un dia tarde, no varios dias de mas "
                              "(ver el mismo problema ya resuelto en apuestas_mlb/retrain_pipeline.py).")
    parser.add_argument("--force", action="store_true",
                         help="Ignora --min-days-between y reentrena de todos modos")
    args = parser.parse_args()

    if not args.force:
        days = days_since_last_retrain()
        if days is not None and days < args.min_days_between:
            print(f"Ultimo reentrenamiento hace {days:.1f} dia(s) (< {args.min_days_between}) - "
                  f"se omite esta corrida. Usa --force para reentrenar de todos modos.")
            return

    import nba_data as n
    season = args.season or n.current_nba_season()
    season_args = ["--season", str(season), "--window", str(args.window), "--min-prior", str(args.min_prior)]

    ok = run(["nba_train_data.py", *season_args, "--out", "training_data_nba.csv"],
             "PASO 1/4: Recolectando dataset historico de equipos (liga completa)")
    if not ok:
        print("\nSe detiene el reentrenamiento: la recoleccion de equipos fallo.")
        return

    run(["nba_train.py", "--data", "training_data_nba.csv"],
        "PASO 2/4: Reentrenando los modelos de equipo (total/margen/money line)")

    if args.skip_props:
        print("\nPASO 3-4/4: omitidos (--skip-props)")
    else:
        ok = run(["nba_props_data.py", *season_args, "--out", "training_data_props.csv"],
                 "PASO 3/4: Recolectando dataset historico de props de jugador (liga completa)")
        if ok:
            run(["nba_props_train.py", "--data", "training_data_props.csv"],
                "PASO 4/4: Reentrenando los modelos de props (puntos/rebotes/asistencias)")
        else:
            print("\nSe omite PASO 4/4: la recoleccion de props fallo.")

    print("\n" + "=" * 72)
    print("Listo. Revisa training_history_nba.csv / training_history_props.csv para ver "
          "como va cambiando el desempeno de los modelos cada vez que reentrenas.")
    print("=" * 72)


if __name__ == "__main__":
    main()
