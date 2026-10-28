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

(Antes solo se incluian titulares; desde oct-2026 entran tambien los
suplentes con 12+ minutos promedio - ver collect_player_games. Nota
historica de la version anterior:) solo se incluian
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
import props_config as pc


# Jugadores con menos minutos promedio que esto no generan filas: sus
# lineas casi no existen en las casas y su produccion es puro ruido.
MIN_AVG_MINUTES = 12


def collect_player_games(team, team_series, max_games=None, bench_deadline=None):
    """Juego por juego de cada jugador del equipo que SI jugo (titulares Y
    suplentes), con su grupo de posicion y si arranco.

    Suplentes: son ~2x mas llamadas a ESPN que los titulares, asi que solo se
    piden sus stats si ya estan en cache o si queda tiempo antes de
    `bench_deadline` (time.time()). El cache se sube a GitHub con cada
    reentrenamiento, asi que en pocas corridas se completa solo sin pasarse
    del limite de tiempo de GitHub Actions.

    Regresa (player_games, starters_by_event): el segundo es la lista de
    titulares de ESTE equipo en cada juego, con su produccion - es lo que
    "permitio" el rival, para la defensa por posicion."""
    games = team_series if max_games is None else team_series[-max_games:]
    player_games, starters_by_event = {}, {}
    for g in games:
        try:
            roster = n.espn_game_roster(g["event_id"], team["espn_id"])
        except Exception as e:
            print(f"    [salteado] roster {g['event_id']} ({team['abbreviation']}): {e}", file=sys.stderr)
            continue
        starters = []
        for entry in roster:
            if entry["did_not_play"]:
                continue
            pid = entry["player_id"]
            if not entry["starter"] and bench_deadline is not None and time.time() > bench_deadline \
                    and not n.player_stats_cached(g["event_id"], team["espn_id"], pid):
                continue
            try:
                stats = n.espn_player_game_stats(g["event_id"], team["espn_id"], pid)
            except Exception as e:
                print(f"    [salteado] stats jugador {pid} juego {g['event_id']}: {e}", file=sys.stderr)
                continue
            if not stats or not stats.get("minutes"):
                continue
            group = n.position_group(entry)
            pg = {"event_id": g["event_id"], "date": g["date"], "season": g.get("season"),
                  **pc.normalize_stats(stats), "starter": bool(entry["starter"]), "group": group}
            player_games.setdefault(pid, {"name": entry["name"], "games": []})["games"].append(pg)
            if entry["starter"]:
                starters.append(pg)
        starters_by_event[(str(g["event_id"]), str(team["espn_id"]))] = starters
    return player_games, starters_by_event


# Peso de un juego de la temporada ANTERIOR frente a uno de la actual, y
# cuanto pierde de peso cada juego hacia atras (0.85 -> el juego de hace 5
# pesa ~44% del mas reciente).
PREV_SEASON_WEIGHT = 0.5
RECENCY_DECAY = 0.85


def season_features(prior, cur_season):
    """Variables que le dan prioridad a ESTA temporada, a partir de los
    juegos previos de la ventana (mas reciente al final):
      - season_games: cuantos de esos juegos son de esta temporada.
      - avg_<prop>_season: promedio SOLO con juegos de esta temporada
        (vacio si aun no juega ninguno).
      - avg_<prop>_w: promedio ponderado - recientes pesan mas, y los de la
        temporada pasada la mitad.
    El modelo aprende solo cuanto creerle a la temporada actual segun
    cuantos juegos lleva. Misma funcion al entrenar y al predecir."""
    cur = [g for g in prior if g.get("season") == cur_season]
    weights = []
    for age, g in enumerate(reversed(prior)):
        w = RECENCY_DECAY ** age
        if g.get("season") is not None and g.get("season") != cur_season:
            w *= PREV_SEASON_WEIGHT
        weights.append(w)
    weights.reverse()
    out = {"season_games": len(cur)}
    for k in list(pc.BASE_STATS) + list(pc.COMBOS):
        vals = [pc.stat_value(g, k) for g in cur]
        vals = [v for v in vals if v is not None]
        out[f"{pc.avg_feature(k)}_season"] = sum(vals) / len(vals) if vals else None
        pairs = [(pc.stat_value(g, k), w) for g, w in zip(prior, weights) if pc.stat_value(g, k) is not None]
        tot = sum(w for _, w in pairs)
        out[f"{pc.avg_feature(k)}_w"] = sum(v * w for v, w in pairs) / tot if tot else None
    return out


# Un companero "deja hueco" solo si de verdad estaba en la rotacion
# reciente: jugo al menos MIN_GAMES de los ultimos `window` juegos del
# equipo con MIN_MINUTES+ de promedio. Si lleva semanas fuera, sus
# companeros YA absorbieron su produccion en sus propios promedios - contarlo
# otra vez seria doble conteo.
VACATED_MIN_GAMES = 2
VACATED_MIN_MINUTES = 15
VACATED_STATS = ["points", "rebounds", "assists", "threes", "minutes"]


def absent_info(recent_games):
    """Resumen de un companero ausente a partir de sus juegos recientes
    (los de la ventana del equipo): None si no estaba en la rotacion."""
    if len(recent_games) < VACATED_MIN_GAMES:
        return None
    mins = sum(g["minutes"] or 0 for g in recent_games) / len(recent_games)
    if mins < VACATED_MIN_MINUTES:
        return None
    info = {"group": main_group(recent_games)}
    for k in VACATED_STATS:
        vals = [g.get(k) or 0 for g in recent_games]
        info[k] = sum(vals) / len(vals)
    return info


def vacated_features(absent, player_group):
    """Produccion que dejan libre los companeros de rotacion que NO juegan:
      - vacated_<stat>: total (puntos, rebotes, asistencias, triples, minutos)
      - vacated_<stat>_same_pos: solo la de companeros de SU misma posicion
        (lo que deja un base le cae sobre todo a otros guardias, lo de un
        pivot a otros grandes)
      - teammates_out: cuantos son."""
    absent = [a for a in absent if a]
    out = {"teammates_out": len(absent)}
    for k in VACATED_STATS:
        out[f"vacated_{k}"] = sum(a[k] for a in absent)
        out[f"vacated_{k}_same_pos"] = sum(a[k] for a in absent if player_group and a["group"] == player_group)
    return out


def main_group(pgames):
    """Grupo de posicion mas frecuente del jugador (None si ESPN nunca lo trajo)."""
    groups = [x["group"] for x in pgames if x.get("group")]
    return max(set(groups), key=groups.count) if groups else None


def allowed_per36(entries, group):
    """Puntos/rebotes/asistencias por 36 minutos que lograron los titulares de
    `group` en `entries` (lo que permitio una defensa). None si hay muy pocos
    minutos para que el numero signifique algo."""
    sel = [e for e in entries if e.get("group") == group and e.get("minutes")]
    mins = sum(e["minutes"] for e in sel)
    if mins < 60:
        return None
    return {stat: sum((e.get(stat) or 0) for e in sel) / mins * 36 for stat in pc.ALLOWED_STATS}


def opp_allowed_asof(opp_id, opp_series, before_date, window, starters_index, group):
    """Lo que la defensa del rival le permitio a titulares del grupo de
    posicion `group` en sus ultimos `window` juegos ANTES de before_date."""
    if not group or not opp_series:
        return None
    prior = [g for g in opp_series if g["date"] and g["date"] < before_date][-window:]
    entries = []
    for g in prior:
        entries.extend(starters_index.get((str(g["event_id"]), str(g["opponent_espn_id"])), []))
    return allowed_per36(entries, group)


def build_player_rows(team, season, all_team_series, player_games, starters_index, window, min_prior,
                      verbose=True):
    team_series = all_team_series[str(team["espn_id"])]
    event_to_idx = {g["event_id"]: i for i, g in enumerate(team_series)}

    rows = []
    for pid, info in player_games.items():
        pgames = info["games"]
        group = main_group(pgames)
        for i, pg in enumerate(pgames):
            if i < min_prior:
                continue
            prior = pgames[max(0, i - window):i]
            avgs = {k: td.r._avg([pc.stat_value(x, k) for x in prior]) for k in list(pc.BASE_STATS) + list(pc.COMBOS)}
            avg_minutes = avgs["minutes"]
            if any(avgs[k] is None for k in ("points", "rebounds", "assists", "minutes")):
                continue
            if avg_minutes < MIN_AVG_MINUTES:
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

            days_rest = None
            if i_team > 0:
                days_rest = td.rest_days(team_series[i_team - 1]["date"], pg["date"])
            if days_rest is None:
                continue
            b2b = int(days_rest <= 1)

            g_team = team_series[i_team]
            # Companeros de la rotacion que NO jugaron este juego y cuanta
            # produccion dejan libre (ver vacated_features).
            window_start = team_series[max(0, i_team - window)]["date"]
            absent = []
            for tid, tinfo in player_games.items():
                if str(tid) == str(pid) or str(tid) not in g_team.get("dnp_ids", set()):
                    continue
                tg = [x for x in tinfo["games"] if window_start <= x["date"] < pg["date"]]
                absent.append(absent_info(tg))
            vacated = vacated_features(absent, group)
            last3 = prior[-3:]
            proj_margin = td.r.matchup_margin(team_feats, opp_feats)
            allowed = opp_allowed_asof(opp_id, opp_series, pg["date"], window, starters_index, group) or {}

            row = {
                "event_id": pg["event_id"], "date": pg["date"],
                "season": g_team.get("season", season),
                "player_id": pid, "player_name": info["name"], "team": team["abbreviation"],
                "position_group": group,
                "start_rate": sum(x["starter"] for x in prior) / len(prior),
                "team_off_rtg": team_feats["off_rtg"], "team_pace": team_feats["pace"],
                "opp_def_rtg": opp_feats["def_rtg"], "opp_pace": opp_feats["pace"],
                "days_rest": days_rest, "b2b": b2b,
                "is_home": int(bool(g_team.get("is_home"))),
                "proj_margin": proj_margin,
                "proj_blowout": abs(proj_margin) if proj_margin is not None else None,
                "n_prior": len(prior),
            }
            for k, v in avgs.items():
                row[pc.avg_feature(k)] = v
                row[f"{pc.avg_feature(k)}_last3"] = td.r._avg([pc.stat_value(x, k) for x in last3])
            row.update(season_features(prior, g_team.get("season", season)))
            row.update(vacated)
            for k in pc.ALLOWED_STATS:
                row[pc.allow_feature(k)] = allowed.get(k)
            for k in pc.TARGETS + ["minutes"]:
                row[f"label_{k}"] = pc.stat_value(pg, k)
            rows.append(row)
    if verbose:
        n_starters = sum(1 for info in player_games.values() if any(x["starter"] for x in info["games"]))
        print(f"  {team['full_name']}: {len(player_games)} jugadores ({n_starters} titulares alguna vez), "
              f"{len(rows)} filas de props", file=sys.stderr)
    return rows


def collect_props_dataset(season, team_names=None, window=10, min_prior=5, max_games=None, verbose=True,
                          seasons=None, bench_budget_min=90):
    """seasons: lista de temporadas a unir (ej. [2026, 2027]) - ver
    nba_train_data.collect_series_for_seasons. bench_budget_min: minutos
    maximos para pedir stats NUEVAS de suplentes (ver collect_player_games);
    None = sin limite."""
    teams = n.TEAMS
    if team_names:
        teams = [n.resolve_team(name) for name in team_names]
    all_team_series = td.load_reusable_series(teams, seasons or [season], max_games=max_games)
    if all_team_series is not None:
        print("  Reusando el historial de equipos del paso anterior (no se vuelve a descargar).",
              file=sys.stderr)
    else:
        all_team_series = td.collect_series_for_seasons(teams, seasons or [season], max_games=max_games,
                                                        verbose=verbose)

    bench_deadline = time.time() + bench_budget_min * 60 if bench_budget_min is not None else None
    player_games_by_team, starters_index = {}, {}
    for team in teams:
        pg, st = collect_player_games(team, all_team_series[str(team["espn_id"])], max_games=max_games,
                                      bench_deadline=bench_deadline)
        player_games_by_team[str(team["espn_id"])] = pg
        starters_index.update(st)
    if bench_deadline is not None and time.time() > bench_deadline:
        print("  AVISO: se acabo el tiempo para suplentes nuevos - los que falten se completan en las "
              "siguientes corridas (el cache se guarda).", file=sys.stderr)

    all_rows = []
    for team in teams:
        all_rows.extend(build_player_rows(team, season, all_team_series, player_games_by_team[str(team["espn_id"])],
                                          starters_index, window, min_prior, verbose=verbose))
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
    parser.add_argument("--seasons", default=None,
                         help="Varias temporadas separadas por coma, ej. '2026,2027'. Tiene prioridad sobre --season")
    parser.add_argument("--bench-budget-min", type=float, default=90,
                         help="Minutos maximos para descargar stats NUEVAS de suplentes (default 90; "
                              "lo ya cacheado no cuenta). Usa -1 para sin limite")
    parser.add_argument("--out", default="training_data_props.csv")
    parser.add_argument("--max-games", type=int, default=None,
                         help="Solo para pruebas rapidas: limita a los ultimos N juegos por equipo")
    args = parser.parse_args()

    season = args.season or n.current_nba_season()
    team_names = [t.strip() for t in args.teams.split(",")] if args.teams else None
    try:
        df = collect_props_dataset(season, team_names=team_names, window=args.window, min_prior=args.min_prior,
                                    max_games=args.max_games, seasons=td.parse_seasons(args.seasons, season),
                                    bench_budget_min=None if args.bench_budget_min < 0 else args.bench_budget_min)
    finally:
        n.flush_cache()

    df.to_csv(args.out, index=False)
    print(f"\nGuardado {len(df)} filas en {args.out}")
    if len(df):
        print(f"Jugadores unicos: {df['player_id'].nunique()}")
        print(f"Filas por temporada: {df['season'].value_counts().sort_index().to_dict()}")
    else:
        print("ERROR: el dataset de props salio VACIO - no hay juegos suficientes para entrenar.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
