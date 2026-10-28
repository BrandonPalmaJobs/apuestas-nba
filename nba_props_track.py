"""
Guarda cada prediccion de props (puntos/rebotes/asistencias) que se hace
para un juego REAL proximo, en un log (props_predictions_log.csv), y
despues permite evaluarla contra el resultado REAL del juego una vez que
termino - para juzgar a los modelos con muchas predicciones en vez de con
una sola observacion (mismo patron que ml_track.py en apuestas_mlb,
adaptado a props de jugador NBA - no se comparte codigo entre proyectos,
solo el patron).

Uso:
    # Se llama automaticamente desde nba_props_predict.py y streamlit_app.py
    # (pestana Props) cuando la prediccion es para un partido real
    # proximo - no hace falta correrlo a mano para loguear.

    # Para evaluar lo logueado hasta ahora:
    python nba_props_track.py --evaluate
"""

import argparse
import os
from datetime import date

import pandas as pd

import nba_data as n

LOG_COLUMNS = [
    "logged_date", "event_id", "game_date", "player_id", "player_name",
    "team", "team_espn_id", "opponent",
    "pred_points", "model_points", "baseline_points",
    "pred_rebounds", "model_rebounds", "baseline_rebounds",
    "pred_assists", "model_assists", "baseline_assists",
    "actual_points", "actual_rebounds", "actual_assists", "evaluated",
]


def log_prediction(row, log_path="props_predictions_log.csv"):
    """Agrega una fila al log, evitando duplicados por (event_id, player_id)."""
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        already = ((df["event_id"].astype(str) == str(row["event_id"])) &
                    (df["player_id"].astype(str) == str(row["player_id"]))).any()
        if already:
            return
    else:
        df = pd.DataFrame(columns=LOG_COLUMNS)

    full_row = {col: row.get(col) for col in LOG_COLUMNS}
    df = pd.concat([df, pd.DataFrame([full_row])], ignore_index=True)
    df.to_csv(log_path, index=False)


def build_log_row(event_id, game_date, player_id, player_name, team, team_espn_id, opponent,
                   pred_points, model_points, baseline_points,
                   pred_rebounds, model_rebounds, baseline_rebounds,
                   pred_assists, model_assists, baseline_assists):
    return {
        "logged_date": date.today().isoformat(),
        "event_id": event_id, "game_date": game_date,
        "player_id": player_id, "player_name": player_name,
        "team": team, "team_espn_id": team_espn_id, "opponent": opponent,
        "pred_points": pred_points, "model_points": model_points, "baseline_points": baseline_points,
        "pred_rebounds": pred_rebounds, "model_rebounds": model_rebounds, "baseline_rebounds": baseline_rebounds,
        "pred_assists": pred_assists, "model_assists": model_assists, "baseline_assists": baseline_assists,
        "actual_points": None, "actual_rebounds": None, "actual_assists": None,
        "evaluated": False,
    }


def is_game_final(event_id):
    try:
        comp = n.espn_competition_detail(event_id)
    except Exception:
        return False
    return bool(comp.get("completed"))


def evaluate_log(log_path="props_predictions_log.csv"):
    if not os.path.exists(log_path):
        print(f"No existe {log_path} todavia - corre predicciones de props para partidos reales primero.")
        return

    df = pd.read_csv(log_path)
    pending = df[df["evaluated"].fillna(False) != True]
    updated = 0
    for idx, row in pending.iterrows():
        event_id = row["event_id"]
        if not is_game_final(event_id):
            continue
        try:
            stats = n.espn_player_game_stats(event_id, row["team_espn_id"], row["player_id"])
        except Exception:
            stats = {}
        # Si el jugador no jugo ese partido (descanso/lesion de ultima hora),
        # stats sale vacio - se marca evaluado de todas formas (para no
        # reintentar por siempre) pero con actual_* en None, y el dropna()
        # de mas abajo lo excluye del MAE agregado sin sesgarlo.
        df.loc[idx, "actual_points"] = stats.get("points")
        df.loc[idx, "actual_rebounds"] = stats.get("rebounds")
        df.loc[idx, "actual_assists"] = stats.get("assists")
        df.loc[idx, "evaluated"] = True
        updated += 1
    df.to_csv(log_path, index=False)
    print(f"Predicciones recien evaluadas: {updated}")

    done = df[df["evaluated"] == True]
    if done.empty:
        print("Todavia no hay predicciones evaluadas.")
        return

    print(f"\n--- Resultados acumulados: {len(done)} predicciones evaluadas ---\n")
    for label, pred_col, baseline_col, actual_col in [
        ("Puntos", "pred_points", "baseline_points", "actual_points"),
        ("Rebotes", "pred_rebounds", "baseline_rebounds", "actual_rebounds"),
        ("Asistencias", "pred_assists", "baseline_assists", "actual_assists"),
    ]:
        d = done.dropna(subset=[pred_col, actual_col])
        if not len(d):
            continue
        mae_model = (d[pred_col] - d[actual_col]).abs().mean()
        db = done.dropna(subset=[baseline_col, actual_col])
        linea = f"{label}: {len(d)} predicciones | MAE modelo {mae_model:.2f}"
        if len(db):
            mae_baseline = (db[baseline_col] - db[actual_col]).abs().mean()
            mejor_o_peor = "mejor" if mae_model < mae_baseline else "peor"
            linea += f" | MAE promedio propio (baseline) {mae_baseline:.2f} -> el modelo sale {mejor_o_peor}"
        print(linea)

    print("\n--- Detalle ---")
    cols = ["game_date", "player_name", "pred_points", "actual_points",
            "pred_rebounds", "actual_rebounds", "pred_assists", "actual_assists"]
    print(done[cols].to_string(index=False))


# ---------------------------------------------------------------------------
# Apuestas registradas (linea real + momio): la prueba de fuego del modelo.
# El MAE dice si el modelo atina el numero; esto dice si GANA DINERO contra
# las lineas reales de tu casa.
# ---------------------------------------------------------------------------

BETS_PATH = "props_bets_log.csv"
BET_COLUMNS = [
    "logged_at", "event_id", "game_date", "player_id", "player_name", "team", "team_espn_id", "opponent",
    "stat", "line", "side", "odds", "pred", "model", "p_model", "p_needed", "edge",
    "actual", "result", "profit_units", "evaluated",
]
STAT_LABELS = {"points": "Puntos", "rebounds": "Rebotes", "assists": "Asistencias"}


def log_bet(row, log_path=BETS_PATH):
    """Registra una apuesta. Si ya existe la misma (juego, jugador, stat), se
    reemplaza: cuenta la ultima linea/lado que elegiste."""
    from datetime import datetime
    row = {**row, "logged_at": datetime.now().isoformat(timespec="seconds"),
           "actual": None, "result": None, "profit_units": None, "evaluated": False}
    if os.path.exists(log_path):
        df = pd.read_csv(log_path)
        same = ((df["event_id"].astype(str) == str(row["event_id"])) &
                (df["player_id"].astype(str) == str(row["player_id"])) &
                (df["stat"] == row["stat"]))
        df = df[~same]
    else:
        df = pd.DataFrame(columns=BET_COLUMNS)
    df = pd.concat([df, pd.DataFrame([{c: row.get(c) for c in BET_COLUMNS}])], ignore_index=True)
    df.to_csv(log_path, index=False)


def bet_profit(result, odds):
    """Ganancia en UNIDADES apostando 1 unidad con momio americano."""
    if result == "W":
        return odds / 100 if odds > 0 else 100 / -odds
    if result == "L":
        return -1.0
    return 0.0


def settle(actual, line, side):
    if actual is None or pd.isna(actual):
        return "DNP"  # no jugo: la casa normalmente anula la apuesta
    if actual == line:
        return "P"
    over_won = actual > line
    return "W" if over_won == (side == "OVER") else "L"


def evaluate_bets(log_path=BETS_PATH):
    """Liquida las apuestas de juegos ya terminados. Regresa (df, resumen)."""
    if not os.path.exists(log_path):
        print("Todavia no hay apuestas registradas.")
        return None, None
    df = pd.read_csv(log_path, dtype={"result": "object", "evaluated": "object"})
    df["actual"] = df["actual"].astype("float64")
    df["profit_units"] = df["profit_units"].astype("float64")
    df["evaluated"] = df["evaluated"].map(lambda v: str(v).lower() == "true")
    for idx, row in df[~df["evaluated"]].iterrows():
        if not is_game_final(row["event_id"]):
            continue
        try:
            stats = n.espn_player_game_stats(row["event_id"], row["team_espn_id"], row["player_id"])
        except Exception:
            stats = {}
        actual = stats.get(row["stat"]) if stats and stats.get("minutes") else None
        result = settle(actual, float(row["line"]), row["side"])
        df.loc[idx, "actual"] = actual
        df.loc[idx, "result"] = result
        df.loc[idx, "profit_units"] = bet_profit(result, float(row["odds"]))
        df.loc[idx, "evaluated"] = True
    df.to_csv(log_path, index=False)
    summary = summarize_bets(df)
    print_bets_summary(summary)
    return df, summary


def summarize_bets(df):
    evaluated = df["evaluated"].map(lambda v: str(v).lower() == "true")
    done = df[evaluated & df["result"].isin(["W", "L", "P"])].copy()
    if done.empty:
        return None

    def block(d):
        w, l, p = (d["result"] == "W").sum(), (d["result"] == "L").sum(), (d["result"] == "P").sum()
        units = d["profit_units"].astype(float).sum()
        return {"apuestas": len(d), "ganadas": int(w), "perdidas": int(l), "push": int(p),
                "acierto": w / (w + l) if (w + l) else None, "unidades": round(units, 2),
                "roi": units / len(d) if len(d) else None}

    out = {"total": block(done)}
    for stat, label in STAT_LABELS.items():
        d = done[done["stat"] == stat]
        if len(d):
            out[label] = block(d)
    d = done[done["edge"].astype(float) > 0]
    if len(d):
        out["Con ventaja segun el modelo"] = block(d)
    d = done[done["edge"].astype(float) <= 0]
    if len(d):
        out["Sin ventaja segun el modelo"] = block(d)
    return out


def print_bets_summary(summary):
    if not summary:
        print("Todavia no hay apuestas liquidadas.")
        return
    print("\n--- Apuestas registradas (1 unidad cada una) ---")
    for name, b in summary.items():
        acierto = f"{b['acierto']:.1%}" if b["acierto"] is not None else "N/D"
        print(f"{name}: {b['ganadas']}-{b['perdidas']}-{b['push']} ({acierto}) | "
              f"{b['unidades']:+.2f} u | ROI {b['roi']:+.1%}")


def main():
    parser = argparse.ArgumentParser(description="Evalua las predicciones de props logueadas contra el resultado real")
    parser.add_argument("--log-path", default="props_predictions_log.csv")
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if args.evaluate:
        try:
            evaluate_log(args.log_path)
            evaluate_bets()
        finally:
            n.flush_cache()
    else:
        print("Usa --evaluate para comparar el log contra los resultados reales.")


if __name__ == "__main__":
    main()
