"""
Usa los modelos entrenados (nba_train.py) para predecir el proximo juego
real entre dos equipos: puntos totales, margen del local (spread), y quien
gana (money line) - comparado siempre contra la formula transparente de
nba_report.py (Fase 1), para que quede claro si el modelo de verdad aporta
algo o no.

Calcula las features EXACTAMENTE con las mismas funciones que usa el
reporte en vivo (nba_report.team_side_report / team_rolling_report), para
que no haya diferencia entre como se calculan en entrenamiento y en
prediccion real (mismo patron que apuestas_mlb/ml_predict.py).

Uso:
    python nba_predict.py "Lakers" "Celtics"
    python nba_predict.py "Lakers" "Celtics" --season 2025 --last-n 10

Si no hay un juego real programado entre los dos equipos en los proximos 14
dias (ej. en receso de temporada), se cae a modo PRACTICA: usa las metricas
mas recientes de cada equipo tal cual (sin fecha real, sin fatiga/B2B) y
asume que el Equipo A es local salvo que se pase --local B. Sirve para
probar los modelos ya entrenados sin esperar a que arranque la temporada -
esa prediccion NO se guarda en el log de evaluacion porque no hay un juego
real contra el cual compararla despues.
"""

import argparse
import sys

import joblib
import pandas as pd

import nba_data as n
import nba_report as r
import nba_track as track


def build_matchup_row(team_home, team_away, season, last_n, as_of_date):
    rep_home = r.team_side_report(team_home, team_away, season, last_n, as_of_date=as_of_date)
    rep_away = r.team_side_report(team_away, team_home, season, last_n, as_of_date=as_of_date)
    if not rep_home["advanced"] or not rep_away["advanced"]:
        return None, rep_home, rep_away

    row = {}
    for k in rep_home["advanced"].keys():
        row[f"home_{k}"] = rep_home["advanced"].get(k)
        row[f"away_{k}"] = rep_away["advanced"].get(k)
    row["home_days_rest"] = rep_home.get("days_rest")
    row["away_days_rest"] = rep_away.get("days_rest")
    row["home_b2b"] = int(bool(rep_home.get("is_b2b")))
    row["away_b2b"] = int(bool(rep_away.get("is_b2b")))
    return row, rep_home, rep_away


def predict_with_bundle(bundle, row):
    X = pd.DataFrame([row])[bundle["features"]]
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[0, 1]
    return model.predict(X)[0]


def main():
    parser = argparse.ArgumentParser(description="Predice con ML: puntos totales, spread y money line de NBA")
    parser.add_argument("equipo_a")
    parser.add_argument("equipo_b")
    parser.add_argument("--season", type=int, default=None,
                         help="Anio en que TERMINA la temporada. Default: temporada actual")
    parser.add_argument("--last-n", type=int, default=10)
    parser.add_argument("--model-total", default="model_nba_total.joblib")
    parser.add_argument("--model-margin", default="model_nba_margin.joblib")
    parser.add_argument("--model-win", default="model_nba_win.joblib")
    parser.add_argument("--local", choices=["A", "B"], default="A",
                         help="Solo en modo practica (sin juego real programado): cual equipo se trata "
                              "como local, A o B (default A)")
    args = parser.parse_args()

    season = args.season or n.current_nba_season()
    team_a = n.resolve_team(args.equipo_a)
    team_b = n.resolve_team(args.equipo_b)

    try:
        matchup = n.find_next_matchup(team_a, team_b)
    except Exception as e:
        print(f"(no se pudo buscar el proximo juego: {e})", file=sys.stderr)
        matchup = None

    if matchup:
        home_is_a = matchup["home_espn_id"] == team_a["espn_id"]
        team_home, team_away = (team_a, team_b) if home_is_a else (team_b, team_a)
        as_of_date = matchup["date"]
        print(f"Local: {team_home['full_name']}  |  Visitante: {team_away['full_name']}  |  "
              f"Fecha: {matchup['date']}\n")
    else:
        team_home, team_away = (team_a, team_b) if args.local == "A" else (team_b, team_a)
        as_of_date = None
        print("No encontre un juego programado entre estos dos equipos en los proximos 14 dias "
              "(ej. receso de temporada) - modo PRACTICA: uso las metricas mas recientes de cada "
              f"equipo, asumiendo local a {team_home['full_name']} (usa --local B para invertirlo). "
              "Esta prediccion NO se guarda para evaluar despues, porque no hay un juego real.\n")

    try:
        row, rep_home, rep_away = build_matchup_row(team_home, team_away, season, args.last_n, as_of_date)
    finally:
        n.flush_cache()

    if row is None:
        print("No hay suficientes datos recientes de alguno de los dos equipos para predecir.")
        return

    baseline_home = r.project_points(rep_home, rep_away)
    baseline_away = r.project_points(rep_away, rep_home)
    baseline_total = baseline_margin = None
    if baseline_home and baseline_away:
        baseline_total = baseline_home["projected_points"] + baseline_away["projected_points"]
        baseline_margin = baseline_home["projected_points"] - baseline_away["projected_points"]

    bundle_total = joblib.load(args.model_total)
    bundle_margin = joblib.load(args.model_margin)
    bundle_win = joblib.load(args.model_win)
    print(f"Modelo puntos totales: {bundle_total['model_name']}")
    print(f"Modelo spread (margen local): {bundle_margin['model_name']}")
    print(f"Modelo money line: {bundle_win['model_name']}\n")

    pred_total = predict_with_bundle(bundle_total, row)
    pred_margin = predict_with_bundle(bundle_margin, row)
    prob_home_win = predict_with_bundle(bundle_win, row)

    print("=" * 60)
    print("PREDICCIONES (modelo entrenado vs. formula Fase 1)")
    print("=" * 60)
    print(f"Puntos totales: {pred_total:.1f}"
          + (f"  (formula: {baseline_total:.1f})" if baseline_total is not None else ""))
    print(f"Margen local ({team_home['abbreviation']} - {team_away['abbreviation']}): {pred_margin:+.1f}"
          + (f"  (formula: {baseline_margin:+.1f})" if baseline_margin is not None else ""))
    favorito = team_home if prob_home_win >= 0.5 else team_away
    prob_favorito = prob_home_win if prob_home_win >= 0.5 else 1 - prob_home_win
    print(f"Money line: {favorito['full_name']} favorito con {prob_favorito*100:.1f}% "
          f"(prob. de que gane el local: {prob_home_win*100:.1f}%)")

    if matchup:
        track.log_prediction({
            "logged_date": n.date.today().isoformat(), "event_id": matchup["event_id"],
            "game_date": matchup["date"], "home_team": team_home["abbreviation"],
            "away_team": team_away["abbreviation"],
            "home_espn_id": team_home["espn_id"], "away_espn_id": team_away["espn_id"],
            "pred_total": pred_total, "model_total": bundle_total["model_name"], "baseline_total": baseline_total,
            "pred_margin": pred_margin, "model_margin": bundle_margin["model_name"],
            "baseline_margin": baseline_margin,
            "pred_prob_home_win": prob_home_win, "model_win": bundle_win["model_name"],
            "actual_home_points": None, "actual_away_points": None, "actual_total": None,
            "actual_margin": None, "actual_home_win": None, "evaluated": False,
        })


if __name__ == "__main__":
    main()
