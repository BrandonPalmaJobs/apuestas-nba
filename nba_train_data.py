"""
Recolecta un dataset historico de juegos de NBA, con features "point-in-time"
(calculadas SOLO con los juegos anteriores de cada equipo, ventana movil de
--window juegos) para entrenar modelos de:
  - puntos totales del juego (regresion)
  - margen del equipo local, home - away (regresion, para el spread)
  - quien gana el juego, home_win 0/1 (clasificacion, para el money line)

Cada fila = UN juego real ya jugado, desde la perspectiva del equipo LOCAL
(se evita duplicar el mismo juego dos veces - una por cada equipo - emitiendo
la fila solo cuando el equipo que se esta recorriendo jugo de local en ese
juego).

Ademas de las features, cada fila trae columnas "baseline_*": la misma
proyeccion de formula transparente que ya usa nba_report.py (Fase 1), para
poder comparar en nba_train.py si el modelo entrenado de verdad le gana a la
formula simple - igual que en apuestas_mlb/ml_train.py.

Uso:
    python nba_train_data.py --season 2025 --out training_data_nba.csv
    python nba_train_data.py --season 2025 --teams "Lakers,Celtics,Nuggets" --out sample.csv   # prueba rapida

Simplificacion conocida (documentada, no oculta):
  - Los lesionados NO se meten como feature de entrenamiento (a diferencia
    del reporte en vivo de nba_report.py, que si los muestra). Un jugador
    lesionado HOY no dice nada sobre si lo estaba en un juego de hace meses
    del dataset historico, y reconstruir el estado de lesionados
    dia-por-dia de la temporada pasada no tiene una fuente publica
    confiable. El impacto de lesionados se sigue aplicando en el reporte
    EN VIVO (nba_report.py), no en el reentrenamiento - mismo patron que
    "enriquecimiento solo en vivo" que ya se uso en el proyecto de MLB.
"""

import argparse
import sys
from datetime import datetime

import pandas as pd

import nba_data as n
import nba_report as r

ADV_KEYS = ["off_rtg", "def_rtg", "net_rtg", "pace", "efg_pct", "ts_pct",
            "tov_pct", "oreb_pct", "dreb_pct", "ast_pct", "fast_break_pts", "points_in_paint"]


def build_team_season_series(team, season, max_games=None):
    """Historial COMPLETO (sin ventana) de un equipo en una temporada: un
    registro point-in-time por juego con sus metricas avanzadas de ESE
    juego, puntos propios/rival, si jugo de local, y dias de descanso desde
    su juego anterior. Se calcula UNA vez por equipo y se reusa para
    construir todas las filas de esa temporada, en vez de volver a pedir
    boxscores por cada fila.

    max_games limita a los ultimos N juegos del equipo (solo para pruebas
    rapidas con --max-games - en una corrida real se deja en None para
    tomar la temporada completa)."""
    try:
        games = n.espn_played_games(team["espn_id"], season=season, last_n=max_games)
    except Exception as e:
        print(f"  ERROR recolectando temporada de {team['full_name']}: {e}", file=sys.stderr)
        return []
    series = []
    prev_date = None
    for e in games:
        opp_espn_id = e["opponent_espn_id"]
        try:
            my_stats, quarters = n.team_game_boxscore(e["id"], team["espn_id"])
            opp_stats, opp_quarters = n.team_game_boxscore(e["id"], opp_espn_id)
            comp = n.espn_competition_detail(e["id"])
        except Exception as ex:
            print(f"    [salteado] juego {e['id']} ({team['abbreviation']}): {ex}", file=sys.stderr)
            continue
        if not my_stats or not opp_stats or not quarters or not opp_quarters:
            continue
        my_points = sum(quarters)
        opp_points = sum(opp_quarters)
        adv = r.game_advanced(my_stats, opp_stats, my_points, opp_points, len(quarters))
        if not adv:
            continue
        my_home = next((c["home_away"] == "home" for c in comp["competitors"]
                         if str(c["espn_id"]) == str(team["espn_id"])), None)
        game_date = e.get("date")
        days_rest = None
        if prev_date and game_date:
            d1 = datetime.fromisoformat(prev_date.replace("Z", "+00:00")).date()
            d2 = datetime.fromisoformat(game_date.replace("Z", "+00:00")).date()
            days_rest = (d2 - d1).days
        series.append({
            "event_id": e["id"], "date": game_date, "opponent_espn_id": opp_espn_id,
            "advanced": adv, "points": my_points, "opp_points": opp_points,
            "is_home": my_home, "days_rest": days_rest,
        })
        prev_date = game_date
    return series


def _rolling_features(series, upto_idx, window):
    """Promedio de las metricas avanzadas de los `window` juegos ANTERIORES
    al indice upto_idx (nunca incluye el juego actual - point-in-time)."""
    prior = series[max(0, upto_idx - window):upto_idx]
    if not prior:
        return None
    return {k: r._avg([g["advanced"][k] for g in prior]) for k in ADV_KEYS}


def _opponent_rolling_asof(opp_series, before_date, window):
    """Igual que _rolling_features, pero para el RIVAL: sus ultimos `window`
    juegos con fecha ANTERIOR a before_date (el rival puede estar en un
    punto distinto de su propia temporada que el equipo que se esta
    procesando)."""
    prior = [g for g in opp_series if g["date"] and g["date"] < before_date][-window:]
    if not prior:
        return None, None
    feats = {k: r._avg([g["advanced"][k] for g in prior]) for k in ADV_KEYS}
    return feats, prior[-1]


def _baseline_projection(home_feats, away_feats, home_b2b, away_b2b):
    """Misma formula transparente de nba_report.py (project_points), aplicada
    a las features ya calculadas de esta fila - sirve para comparar en
    nba_train.py si el modelo entrenado le gana a la formula simple."""
    pace = r._avg([home_feats.get("pace"), away_feats.get("pace")]) or 99.0
    home_rtg = r._avg([home_feats.get("off_rtg"), away_feats.get("def_rtg")])
    away_rtg = r._avg([away_feats.get("off_rtg"), home_feats.get("def_rtg")])
    if home_rtg is None or away_rtg is None:
        return None
    home_pts = home_rtg * (pace / 100)
    away_pts = away_rtg * (pace / 100)
    if home_b2b:
        home_pts *= 0.97
    if away_b2b:
        away_pts *= 0.97
    return home_pts, away_pts


def build_rows_for_team(team, season, all_series, window, min_prior):
    """Filas de entrenamiento donde `team` jugo de LOCAL - asi cada juego
    real produce exactamente una fila (no dos), sin importar en el orden en
    que se recorran los 30 equipos."""
    series = all_series[str(team["espn_id"])]
    rows = []
    for i, g in enumerate(series):
        if not g["is_home"]:
            continue
        if i < min_prior:
            continue
        home_feats = _rolling_features(series, i, window)
        if home_feats is None:
            continue

        opp_id = g["opponent_espn_id"]
        opp_series = all_series.get(str(opp_id))
        if not opp_series:
            continue
        away_feats, away_last_game = _opponent_rolling_asof(opp_series, g["date"], window)
        if away_feats is None:
            continue

        home_prior = series[max(0, i - window):i]
        home_last_date = home_prior[-1]["date"] if home_prior else None
        home_days_rest = None
        if home_last_date and g["date"]:
            d1 = datetime.fromisoformat(home_last_date.replace("Z", "+00:00")).date()
            d2 = datetime.fromisoformat(g["date"].replace("Z", "+00:00")).date()
            home_days_rest = (d2 - d1).days
        away_days_rest = None
        if away_last_game and away_last_game["date"] and g["date"]:
            d1 = datetime.fromisoformat(away_last_game["date"].replace("Z", "+00:00")).date()
            d2 = datetime.fromisoformat(g["date"].replace("Z", "+00:00")).date()
            away_days_rest = (d2 - d1).days
        home_b2b = home_days_rest is not None and home_days_rest <= 1
        away_b2b = away_days_rest is not None and away_days_rest <= 1

        baseline = _baseline_projection(home_feats, away_feats, home_b2b, away_b2b)
        if baseline is None:
            continue
        baseline_home_points, baseline_away_points = baseline

        row = {
            "event_id": g["event_id"], "date": g["date"], "season": season,
            "home_team": team["abbreviation"], "away_team_espn_id": opp_id,
            "home_n_prior": len(home_prior), "away_n_prior": len(
                [x for x in opp_series if x["date"] and x["date"] < g["date"]]),
            "home_days_rest": home_days_rest, "away_days_rest": away_days_rest,
            "home_b2b": int(home_b2b), "away_b2b": int(away_b2b),
        }
        for k in ADV_KEYS:
            row[f"home_{k}"] = home_feats[k]
            row[f"away_{k}"] = away_feats[k]

        row["baseline_home_points"] = baseline_home_points
        row["baseline_away_points"] = baseline_away_points
        row["baseline_total_points"] = baseline_home_points + baseline_away_points
        row["baseline_home_margin"] = baseline_home_points - baseline_away_points

        row["label_home_points"] = g["points"]
        row["label_away_points"] = g["opp_points"]
        row["label_total_points"] = g["points"] + g["opp_points"]
        row["label_home_margin"] = g["points"] - g["opp_points"]
        row["label_home_win"] = int(g["points"] > g["opp_points"])
        rows.append(row)
    return rows


def collect_league_dataset(season, team_names=None, window=10, min_prior=5, verbose=True, max_games=None):
    teams = n.TEAMS
    if team_names:
        teams = [n.resolve_team(name) for name in team_names]

    # Para no perder filas cuando el rival de un equipo de la muestra no esta
    # en la muestra (--teams de prueba), se recolecta la temporada completa
    # de CUALQUIER rival que aparezca, aunque no se haya pedido explicitamente.
    # Con --max-games (solo para pruebas rapidas) esto se limita a los
    # rivales de esos mismos juegos recientes, no a toda la liga.
    # ESPN regresa los IDs de equipo como texto (ej. "13"), mientras que
    # TEAMS los guarda como entero - se normaliza todo a str() aqui para
    # que las comparaciones/busquedas no fallen en silencio (paso esto por
    # alto la primera vez: la coleccion corria sin error pero devolvia 0
    # filas porque ningun opp_id encontraba su equipo).
    needed_ids = {str(t["espn_id"]) for t in teams}
    all_series = {}
    t0 = __import__("time").time()
    to_process = list(teams)
    seen_ids = set(needed_ids)
    i = 0
    while i < len(to_process):
        team = to_process[i]
        i += 1
        series = build_team_season_series(team, season, max_games=max_games)
        all_series[str(team["espn_id"])] = series
        if verbose:
            print(f"  [{i}] {team['full_name']}: {len(series)} juegos recolectados "
                  f"({__import__('time').time() - t0:.0f}s acumulados)", file=sys.stderr)
        for g in series:
            opp_id = str(g["opponent_espn_id"])
            if opp_id not in seen_ids:
                seen_ids.add(opp_id)
                opp_team = next((t for t in n.TEAMS if str(t["espn_id"]) == opp_id), None)
                if opp_team:
                    to_process.append(opp_team)

    all_rows = []
    for team in teams:
        rows = build_rows_for_team(team, season, all_series, window, min_prior)
        all_rows.extend(rows)
        if verbose:
            print(f"  {team['full_name']}: {len(rows)} filas de entrenamiento (como local)", file=sys.stderr)
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Recolecta dataset historico de NBA para entrenar modelos")
    parser.add_argument("--season", type=int, default=None,
                         help="Anio en que TERMINA la temporada (2025 para 2024-25). Default: temporada actual")
    parser.add_argument("--teams", default=None,
                         help="Lista separada por comas para una corrida de prueba, ej. 'Lakers,Celtics'")
    parser.add_argument("--window", type=int, default=10, help="Ventana movil de juegos previos (default 10)")
    parser.add_argument("--min-prior", type=int, default=5,
                         help="Minimo de juegos previos para confiar en el promedio movil (default 5)")
    parser.add_argument("--out", default="training_data_nba.csv")
    parser.add_argument("--max-games", type=int, default=None,
                         help="Solo para pruebas rapidas: limita a los ultimos N juegos por equipo "
                              "(sin esto se recolecta la temporada completa)")
    args = parser.parse_args()

    season = args.season or n.current_nba_season()
    team_names = [t.strip() for t in args.teams.split(",")] if args.teams else None
    try:
        df = collect_league_dataset(season, team_names=team_names, window=args.window, min_prior=args.min_prior,
                                     max_games=args.max_games)
    finally:
        n.flush_cache()

    df.to_csv(args.out, index=False)
    print(f"\nGuardado {len(df)} filas en {args.out}")
    if len(df):
        print(f"Tasa de victoria local (dataset): {df['label_home_win'].mean():.1%}")


if __name__ == "__main__":
    main()
