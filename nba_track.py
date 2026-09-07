"""
Guarda cada prediccion de ML que hace nba_predict.py en un log
(predictions_log_nba.csv), y despues permite evaluarlas contra el resultado
REAL del juego una vez que termino - para juzgar a los modelos con muchos
juegos en vez de con una sola observacion. Mismo patron que
apuestas_mlb/ml_track.py.

Uso:
    # Se llama automaticamente desde nba_predict.py, no hace falta correrlo
    # a mano para loguear.

    # Para evaluar lo logueado hasta ahora:
    python nba_track.py --evaluate
"""

import argparse
import os
from datetime import date

import pandas as pd

import nba_data as n

LOG_COLUMNS = [
    "logged_date", "event_id", "game_date", "home_team", "away_team",
    "home_espn_id", "away_espn_id",
    "pred_total", "model_total", "baseline_total",
    "pred_margin", "model_margin", "baseline_margin",
    "pred_prob_home_win", "model_win",
    "actual_home_points", "actual_away_points", "actual_total", "actual_margin", "actual_home_win",
    "evaluated",
]


def log_prediction(row, log_path="predictions_log_nba.csv"):
    """Agrega una fila al log, evitando duplicados por event_id."""
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        if (df["event_id"].astype(str) == str(row["event_id"])).any():
            return
    else:
        df = pd.DataFrame(columns=LOG_COLUMNS)

    full_row = {col: row.get(col) for col in LOG_COLUMNS}
    df = pd.concat([df, pd.DataFrame([full_row])], ignore_index=True)
    df.to_csv(log_path, index=False)


def evaluate_log(log_path="predictions_log_nba.csv"):
    if not os.path.exists(log_path):
        print(f"No existe {log_path} todavia - corre nba_predict.py para algunos juegos primero.")
        return

    df = pd.read_csv(log_path)
    pending = df[df["evaluated"].fillna(False) != True]
    updated = 0
    for idx, row in pending.iterrows():
        event_id = str(row["event_id"])
        try:
            comp = n.espn_competition_detail(event_id)
        except Exception as e:
            print(f"  (no se pudo revisar el juego {event_id}: {e})")
            continue
        if not comp.get("completed"):
            continue
        try:
            home_stats, home_quarters = n.team_game_boxscore(event_id, str(int(row["home_espn_id"])))
            away_stats, away_quarters = n.team_game_boxscore(event_id, str(int(row["away_espn_id"])))
        except Exception as e:
            print(f"  (no se pudo bajar el resultado del juego {event_id}: {e})")
            continue
        home_pts = sum(home_quarters)
        away_pts = sum(away_quarters)
        df.loc[idx, "actual_home_points"] = home_pts
        df.loc[idx, "actual_away_points"] = away_pts
        df.loc[idx, "actual_total"] = home_pts + away_pts
        df.loc[idx, "actual_margin"] = home_pts - away_pts
        df.loc[idx, "actual_home_win"] = int(home_pts > away_pts)
        df.loc[idx, "evaluated"] = True
        updated += 1
    df.to_csv(log_path, index=False)
    n.flush_cache()
    print(f"Juegos recien evaluados: {updated}")

    done = df[df["evaluated"] == True]
    if done.empty:
        print("Todavia no hay juegos terminados en el log para evaluar.")
        return

    print(f"\n--- Resultados acumulados: {len(done)} juegos evaluados ---\n")

    for label, pred_col, base_col, actual_col in [
        ("Puntos totales", "pred_total", "baseline_total", "actual_total"),
        ("Margen local (spread)", "pred_margin", "baseline_margin", "actual_margin"),
    ]:
        d = done.dropna(subset=[pred_col, actual_col])
        if len(d):
            mae_model = (d[pred_col] - d[actual_col]).abs().mean()
            d_base = done.dropna(subset=[base_col, actual_col])
            mae_base = (d_base[base_col] - d_base[actual_col]).abs().mean() if len(d_base) else None
            base_txt = f" | MAE formula: {mae_base:.2f}" if mae_base is not None else ""
            print(f"{label}: {len(d)} juegos | MAE modelo: {mae_model:.2f}{base_txt}")

    dwin = done.dropna(subset=["pred_prob_home_win", "actual_home_win"])
    if len(dwin):
        pred_class = (dwin["pred_prob_home_win"] >= 0.5).astype(int)
        acc = (pred_class == dwin["actual_home_win"]).mean()
        print(f"Money line (home_win): {len(dwin)} juegos | accuracy {acc:.1%}")

    print("\n--- Detalle ---")
    cols = ["game_date", "home_team", "away_team", "pred_total", "actual_total",
            "pred_margin", "actual_margin", "pred_prob_home_win", "actual_home_win"]
    print(done[cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Evalua las predicciones logueadas de NBA contra el resultado real")
    parser.add_argument("--log-path", default="predictions_log_nba.csv")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        evaluate_log(args.log_path)
    else:
        print("Usa --evaluate para comparar el log contra los resultados reales.")


if __name__ == "__main__":
    main()
