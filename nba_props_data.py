"""
Recolecta un dataset historico point-in-time de PROPS de jugador (puntos,
rebotes, asistencias) para entrenar modelos de over/under. Reusa
nba_train_data.py para el contexto de equipo (pace/ratings propios y del
rival, ya cacheado si corriste reentrenar_modelos.bat para esta misma
temporada) y agrega la parte nueva: minutos/puntos/rebotes/asistencias de
cada jugador, juego por juego, con ventana movil.

Cada fila = UN jugador en UN juego real ya jugado:
  - features: su propio promedio movil de los `window` juegos ANTERIORES
    (puntos/rebotes/asistencias/minutos), el contexto de equipo propio y
    del rival de ESE juego, y el descanso del EQUIPO de cara a ese juego
    (dias desde su juego anterior, y si fue back-to-back) - todo
    point-in-time, nunca el resultado del juego que se esta prediciendo.
  - labels: lo que de verdad anoto/rebot/asistio en ESE juego.

Simplificacion conocida (documentada, no oculta): solo se incluyen
jugadores que EMPEZARON (starter=True) en ese juego especifico - es un
proxy barato de "juega minutos consistentes" sin tener que descargar
estadisticas de TODO el roster (30+ jugadores por equipo) para despues
descartar a la mayoria por poco tiempo en cancha. Un suplente de rotacion
fija que no arranca (ej. un "sexto hombre") se pierde con este filtro - si
hace falta cubrirlo mas adelante, se puede aflojar el filtro a costa de
mas tiempo de recoleccion (son ~3x mas jugadores por equipo).

Uso:
    python nba_props_data.py --season 2026 --out training_data_props.csv
    python nba_props_data.py --season 2026 --teams "Lakers,Celtics" --max-games 12 --out sample.csv
"""

import argparse
import sys
import time
from datetime import datetime as dt

import pandas as pd

import nba_data as n
import nba_train_data as td


def build_player_rows(team, season, all_team_series, window, min_prior, max_games=None, verbose=True):
    team_series = all_team_series[str(team["espn_id"])]
    event_to_idx = {g["event_id"]: i for i, g in enumerate(team_series)}
    games = team_series if max_games is None else team_series[-max_games:]

    player_games = {}
    for g in games:
        try:
            roster = n.espn_game_roster(g["event_id"], team["espn_id"])
        except Exception as e:
            print(f"    [salteado] roster {g['event_id']} ({team['abbreviation']}): {e}", file=sys.stderr)
            continue
        for entry in roster:
            if entry["did_not_play"] or not entry["starter"]:
                continue
            pid = entry["player_id"]
            try:
                stats = n.espn_player_game_stats(g["event_id"], team["espn_id"], pid)
            except Exception as e:
                print(f"    [salteado] stats jugador {pid} juego {g['event_id']}: {e}", file=sys.stderr)
                continue
            if not stats or stats.get("minutes") is None:
                continue
            player_games.setdefault(pid, {"name": entry["name"], "games": []})
            player_games[pid]["games"].append({
                "event_id": g["event_id"], "date": g["date"],
                "points": stats.get("points"), "rebounds": stats.get("rebounds"),
                "assists": stats.get("assists"), "minutes": stats.get("minutes"),
            })

    rows = []
    for pid, info in player_games.items():
        pgames = info["games"]
        for i, pg in enumerate(pgames):
            if i < min_prior:
                continue
            prior = pgames[max(0, i - window):i]
            avg_points = td.r._avg([x["points"] for x in prior])
            avg_rebounds = td.r._avg([x["rebounds"] for x in prior])
            avg_assists = td.r._avg([x["assists"] for x in prior])
            avg_minutes = td.r._avg([x["minutes"] for x in prior])
            if None in (avg_points, avg_rebounds, avg_assists, avg_minutes):
                continue

            i_team = event_to_idx.get(pg["event_id"])
            if i_team is None:
                continue
            team_feats = td._rolling_features(team_series, i_team, window)
            if team_feats is None:
                continue

            opp_id = team_series[i_team]["opponent_espn_id"]
            opp_series = all_team_series.get(str(opp_id))
            opp_feats = None
            if opp_series:
                opp_feats, _ = td._opponent_rolling_asof(opp_series, pg["date"], window)
            if opp_feats is None:
                continue

            # Descanso del EQUIPO (no del jugador especificamente - ESPN no
            # trae minutos de descanso individuales por lesion/rotacion) de
            # cara a este juego: dias desde el juego anterior del equipo.
            # Point-in-time (solo mira el juego previo, nunca el actual).
            days_rest = None
            if i_team > 0:
                prev_date = team_series[i_team - 1]["date"]
                if prev_date and pg["date"]:
                    d1 = dt.fromisoformat(prev_date.replace("Z", "+00:00")).date()
                    d2 = dt.fromisoformat(pg["date"].replace("Z", "+00:00")).date()
                    days_rest = (d2 - d1).days
            if days_rest is None:
                continue
            b2b = int(days_rest <= 1)

            rows.append({
                "event_id": pg["event_id"], "date": pg["date"], "season": season,
                "player_id": pid, "player_name": info["name"], "team": team["abbreviation"],
                "avg_points": avg_points, "avg_rebounds": avg_rebounds,
                "avg_assists": avg_assists, "avg_minutes": avg_minutes,
                "team_off_rtg": team_feats["off_rtg"], "team_pace": team_feats["pace"],
                "opp_def_rtg": opp_feats["def_rtg"], "opp_pace": opp_feats["pace"],
                "days_rest": days_rest, "b2b": b2b,
                "n_prior": len(prior),
                "label_points": pg["points"], "label_rebounds": pg["rebounds"],
                "label_assists": pg["assists"], "label_minutes": pg["minutes"],
            })
    if verbose:
        print(f"  {team['full_name']}: {len(player_games)} jugadores titulares, "
              f"{len(rows)} filas de props", file=sys.stderr)
    return rows


def collect_props_dataset(season, team_names=None, window=10, min_prior=5, max_games=None, verbose=True):
    teams = n.TEAMS
    if team_names:
        teams = [n.resolve_team(name) for name in team_names]

    # Reusa el mismo motor de nba_train_data.py para el contexto de equipo -
    # si ya corriste reentrenar_modelos.bat para esta temporada, la mayoria
    # de estas llamadas de equipo ya estan en cache y esto es casi
    # instantaneo; lo nuevo y lento aqui es roster+stats por JUGADOR.
    all_team_series = {}
    needed_ids = {str(t["espn_id"]) for t in teams}
    to_process = list(teams)
    seen_ids = set(needed_ids)
    i = 0
    t0 = time.time()
    while i < len(to_process):
        team = to_process[i]
        i += 1
        series = td.build_team_season_series(team, season, max_games=max_games)
        all_team_series[str(team["espn_id"])] = series
        if verbose:
            print(f"  [equipo {i}] {team['full_name']}: {len(series)} juegos de contexto "
                  f"({time.time()-t0:.0f}s)", file=sys.stderr)
        for g in series:
            opp_id = str(g["opponent_espn_id"])
            if opp_id not in seen_ids:
                seen_ids.add(opp_id)
                opp_team = next((t for t in n.TEAMS if str(t["espn_id"]) == opp_id), None)
                if opp_team:
                    to_process.append(opp_team)

    all_rows = []
    for team in teams:
        rows = build_player_rows(team, season, all_team_series, window, min_prior, max_games=max_games)
        all_rows.extend(rows)
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description="Recolecta dataset historico de props de jugador NBA")
    parser.add_argument("--season", type=int, default=None,
                         help="Anio en que TERMINA la temporada. Default: temporada actual")
    parser.add_argument("--teams", default=None,
                         help="Lista separada por comas para una corrida de prueba, ej. 'Lakers,Celtics'")
    parser.add_argument("--window", type=int, default=10, help="Ventana movil de juegos previos (default 10)")
    parser.add_argument("--min-prior", type=int, default=5,
                         help="Minimo de juegos previos para confiar en el promedio movil (default 5)")
    parser.add_argument("--out", default="training_data_props.csv")
    parser.add_argument("--max-games", type=int, default=None,
                         help="Solo para pruebas rapidas: limita a los ultimos N juegos por equipo")
    args = parser.parse_args()

    season = args.season or n.current_nba_season()
    team_names = [t.strip() for t in args.teams.split(",")] if args.teams else None
    try:
        df = collect_props_dataset(season, team_names=team_names, window=args.window, min_prior=args.min_prior,
                                    max_games=args.max_games)
    finally:
        n.flush_cache()

    df.to_csv(args.out, index=False)
    print(f"\nGuardado {len(df)} filas en {args.out}")
    if len(df):
        print(f"Jugadores unicos: {df['player_id'].nunique()}")


if __name__ == "__main__":
    main()
