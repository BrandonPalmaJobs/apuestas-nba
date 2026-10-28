"""
Usa los modelos entrenados (nba_props_train.py) para predecir puntos,
rebotes y asistencias de UN jugador en su proximo juego real - comparado
contra su propio promedio movil (el baseline mas obvio) y contra un umbral
"linea" calculado de sus propios datos (mediana de sus ultimos juegos, ya
que no se scrapea ninguna casa de apuestas - mismo criterio ya usado para
los totales de equipo).

Si le pasas la linea real de tu casa (--linea-puntos 24.5, etc.) calcula
la probabilidad de over/under y la ventaja contra el momio (--momio).

Uso:
    python nba_props_predict.py "LeBron James" "Lakers" "Celtics"
    python nba_props_predict.py "LeBron James" "Lakers" "Celtics" --linea-puntos 24.5 --momio -115
    python nba_props_predict.py "Tatum" "Celtics" "Lakers" --season 2026
"""

import argparse
import statistics

import joblib
import pandas as pd

import nba_data as n
import nba_props_train as ptr
import nba_props_track as pt
import nba_report as r

WINDOW = 10


def player_recent_games(team_espn_id, player_id, season, window=WINDOW, buffer=6):
    """Ultimos `window` juegos REALES en los que el jugador SI jugo (no
    DNP), mas recientes primero en la busqueda pero regresados en orden
    cronologico. `buffer` juegos extra por si el jugador se salto alguno
    (lesion, descanso) dentro de la ventana de juegos del equipo."""
    games = n.espn_recent_games(team_espn_id, season=season, last_n=window + buffer)
    out = []
    for g in reversed(games):
        if len(out) >= window:
            break
        try:
            roster = n.espn_game_roster(g["id"], team_espn_id)
        except Exception:
            continue
        entry = next((e for e in roster if str(e["player_id"]) == str(player_id)), None)
        if not entry or entry["did_not_play"]:
            continue
        try:
            stats = n.espn_player_game_stats(g["id"], team_espn_id, player_id)
        except Exception:
            continue
        if not stats or stats.get("minutes") is None:
            continue
        out.append({
            "date": g["date"], "points": stats.get("points"), "rebounds": stats.get("rebounds"),
            "assists": stats.get("assists"), "minutes": stats.get("minutes"),
        })
    out.reverse()
    return out


def build_props_row(recent, rep_team, rep_opp, is_home=None):
    """Fila de features para los modelos de props - la MISMA para el CLI y la
    app de Streamlit, y con las mismas columnas que arma nba_props_data.py
    al entrenar (las que un modelo viejo no use simplemente se ignoran)."""
    last3 = recent[-3:]
    team_adv, opp_adv = rep_team["advanced"], rep_opp["advanced"]
    proj_margin = r.matchup_margin(team_adv, opp_adv)
    row = {
        "team_off_rtg": team_adv["off_rtg"], "team_pace": team_adv["pace"],
        "opp_def_rtg": opp_adv["def_rtg"], "opp_pace": opp_adv["pace"],
        "days_rest": rep_team.get("days_rest"), "b2b": int(bool(rep_team.get("is_b2b"))),
        "is_home": None if is_home is None else int(bool(is_home)),
        # Version en vivo: titulares habituales reportados lesionados hoy.
        "teammates_out": rep_team.get("missing_regulars"),
        "proj_margin": proj_margin,
        "proj_blowout": abs(proj_margin) if proj_margin is not None else None,
    }
    for stat in ("points", "rebounds", "assists", "minutes"):
        row[f"avg_{stat}"] = r._avg([g[stat] for g in recent])
        row[f"avg_{stat}_last3"] = r._avg([g[stat] for g in last3])
    return row


def american_to_prob(odds):
    """Momio americano -> probabilidad implicita (con el margen de la casa)."""
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def over_under(bundle, pred, line, odds=-110):
    """Probabilidad de over/under contra una linea real y ventaja contra el
    momio. None si el modelo es viejo (entrenado antes de guardar su curva de
    error) o no se dio linea."""
    curve = bundle.get("sigma_curve")
    if line is None or not curve:
        return None
    p_over = ptr.prob_over(pred, line, curve)
    p_need = american_to_prob(odds)
    best = "OVER" if p_over >= 0.5 else "UNDER"
    p_best = max(p_over, 1 - p_over)
    return {"line": line, "p_over": p_over, "p_under": 1 - p_over, "pick": best,
            "p_pick": p_best, "p_needed": p_need, "edge": p_best - p_need}


def main():
    parser = argparse.ArgumentParser(description="Predice props de jugador NBA (puntos/rebotes/asistencias)")
    parser.add_argument("jugador", help="Nombre (o parte del nombre) del jugador")
    parser.add_argument("equipo_jugador", help="Equipo del jugador")
    parser.add_argument("equipo_rival", help="Equipo rival")
    parser.add_argument("--season", type=int, default=None,
                         help="Anio en que TERMINA la temporada. Default: temporada actual")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--linea-puntos", type=float, default=None, help="Linea de tu casa, ej. 24.5")
    parser.add_argument("--linea-rebotes", type=float, default=None)
    parser.add_argument("--linea-asistencias", type=float, default=None)
    parser.add_argument("--momio", type=float, default=-110,
                         help="Momio americano del over/under (default -110)")
    parser.add_argument("--model-points", default="model_props_points.joblib")
    parser.add_argument("--model-rebounds", default="model_props_rebounds.joblib")
    parser.add_argument("--model-assists", default="model_props_assists.joblib")
    args = parser.parse_args()

    season = args.season or n.current_nba_season()
    team = n.resolve_team(args.equipo_jugador)
    opponent = n.resolve_team(args.equipo_rival)

    player_id, player_name = n.resolve_player_espn(team["espn_id"], args.jugador, season=season)
    if not player_id:
        print(f"No encontre a '{args.jugador}' en el roster de {team['full_name']} (temporada {season}).")
        return
    print(f"Jugador: {player_name} ({team['full_name']}) vs {opponent['full_name']}\n")

    try:
        matchup = n.find_next_matchup(team, opponent)
    except Exception:
        matchup = None
    as_of_date = matchup["date"] if matchup else None

    try:
        recent = player_recent_games(team["espn_id"], player_id, season, window=args.window)
        rep_team = r.team_side_report(team, opponent, season, args.window, as_of_date=as_of_date)
        rep_opp = r.team_side_report(opponent, team, season, args.window, as_of_date=as_of_date)
    finally:
        n.flush_cache()

    if len(recent) < 3:
        print(f"Muy pocos juegos recientes registrados ({len(recent)}) para predecir con confianza.")
        return
    if not rep_team["advanced"] or not rep_opp["advanced"]:
        print("No hay suficientes datos de equipo recientes para calcular el contexto del rival.")
        return

    is_home = str(matchup.get("home_espn_id")) == str(team["espn_id"]) if matchup else None
    row = build_props_row(recent, rep_team, rep_opp, is_home=is_home)

    game_summaries = [f"{g['points']:.0f}p/{g['rebounds']:.0f}r/{g['assists']:.0f}a" for g in recent]
    print(f"Ultimos {len(recent)} juegos: {', '.join(game_summaries)}\n")

    print("=" * 60)
    print("PREDICCIONES (modelo entrenado vs. promedio propio vs. mediana reciente)")
    print("=" * 60)
    preds = {}
    lines = {"points": args.linea_puntos, "rebounds": args.linea_rebotes, "assists": args.linea_asistencias}
    for stat, model_path, label in [
        ("points", args.model_points, "Puntos"),
        ("rebounds", args.model_rebounds, "Rebotes"),
        ("assists", args.model_assists, "Asistencias"),
    ]:
        bundle = joblib.load(model_path)
        X = pd.DataFrame([row])[bundle["features"]]
        pred = bundle["model"].predict(X)[0]
        avg = row[f"avg_{stat}"]
        mediana = statistics.median(g[stat] for g in recent)
        over_under = "OVER" if pred > mediana else "UNDER" if pred < mediana else "IGUAL"
        print(f"{label} ({bundle['model_name']}): {pred:.1f}  |  promedio propio: {avg:.1f}  |  "
              f"mediana ultimos {len(recent)}: {mediana:.1f}  ->  {over_under} esa mediana")
        ou = over_under(bundle, pred, lines[stat], args.momio)
        if ou:
            veredicto = "VALOR" if ou["edge"] > 0 else "sin ventaja"
            print(f"    Linea {ou['line']}: over {ou['p_over']:.0%} / under {ou['p_under']:.0%}  ->  "
                  f"{ou['pick']} {ou['p_pick']:.0%} vs {ou['p_needed']:.0%} que pide el momio "
                  f"({ou['edge']:+.1%}, {veredicto})")
        elif lines[stat] is not None:
            print("    (este modelo es anterior a las probabilidades - reentrena para verlas)")
        preds[stat] = {"pred": pred, "model_name": bundle["model_name"], "baseline": avg}

    if matchup:
        pt.log_prediction(pt.build_log_row(
            event_id=matchup["event_id"], game_date=matchup["date"],
            player_id=player_id, player_name=player_name,
            team=team["full_name"], team_espn_id=team["espn_id"], opponent=opponent["full_name"],
            pred_points=preds["points"]["pred"], model_points=preds["points"]["model_name"],
            baseline_points=preds["points"]["baseline"],
            pred_rebounds=preds["rebounds"]["pred"], model_rebounds=preds["rebounds"]["model_name"],
            baseline_rebounds=preds["rebounds"]["baseline"],
            pred_assists=preds["assists"]["pred"], model_assists=preds["assists"]["model_name"],
            baseline_assists=preds["assists"]["baseline"],
        ))

    print("\nNOTA: no se usa una linea real de casa de apuestas (no se scrapea ninguna) - "
          "la 'mediana reciente' es un umbral calculado de los propios datos del jugador, "
          "para comparar. Ajusta el criterio con la linea real que te ofrezca tu casa de apuestas.")


if __name__ == "__main__":
    main()
