"""
Usa los modelos entrenados (nba_props_train.py) para predecir los props de
jugador (puntos, rebotes, asistencias, triples, combinados PRA/PR/PA/RA,
robos, bloqueos, perdidas - ver props_config.py) en su proximo juego real.

Dos modos:
  - UN jugador:  python nba_props_predict.py "LeBron James" "Lakers" "Celtics"
  - TODOS los jugadores de TODOS los juegos de hoy (predict_slate, lo usa la
    app de Streamlit): python nba_props_predict.py --hoy

Si le pasas la linea real de tu casa (--linea points=24.5 --linea pra=38.5)
calcula la probabilidad de over/under y la ventaja contra el momio (--momio).
"""

import argparse
import os
import statistics
import sys
from datetime import date

import joblib
import pandas as pd

import nba_data as n
import nba_props_data as pdata
import nba_props_train as ptr
import nba_props_track as pt
import nba_report as r
import props_config as pc

WINDOW = 10
APP_DIR = os.path.dirname(os.path.abspath(__file__))


# ---------------------------------------------------------------------------
# Datos recientes
# ---------------------------------------------------------------------------

def _game_entry(g, entry, stats):
    return {"date": g["date"], "season": g.get("season"), **pc.normalize_stats(stats),
            "starter": bool(entry["starter"]), "group": n.position_group(entry)}


def player_recent_games(team_espn_id, player_id, season, window=WINDOW, buffer=6):
    """Ultimos `window` juegos REALES en los que el jugador SI jugo, en orden
    cronologico. Completa con la temporada anterior al arranque (ver
    nba_data.espn_recent_games). `buffer` juegos extra por si se salto alguno."""
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
        if not stats or not stats.get("minutes"):
            continue
        out.append(_game_entry(g, entry, stats))
    out.reverse()
    return out


def team_players_recent_games(team_espn_id, season, window=WINDOW, buffer=4, on_progress=None):
    """Lo mismo que player_recent_games pero para TODO el roster a la vez:
    se pide el roster de cada juego una sola vez. Regresa
    {player_id: {"name": ..., "games": [...]}} con los ultimos `window`
    juegos de cada jugador (mas reciente al final)."""
    games = n.espn_recent_games(team_espn_id, season=season, last_n=window + buffer)
    players = {}
    for gi, g in enumerate(reversed(games)):
        if on_progress:
            on_progress(gi, len(games))
        try:
            roster = n.espn_game_roster(g["id"], team_espn_id)
        except Exception:
            continue
        for entry in roster:
            if entry["did_not_play"]:
                continue
            pid = str(entry["player_id"])
            info = players.setdefault(pid, {"name": entry["name"], "games": []})
            if len(info["games"]) >= window:
                continue
            try:
                stats = n.espn_player_game_stats(g["id"], team_espn_id, pid)
            except Exception:
                continue
            if stats and stats.get("minutes"):
                info["games"].append(_game_entry(g, entry, stats))
    for info in players.values():
        info["games"].reverse()
    # Solo quien SIGUE en el equipo: al arranque de temporada la ventana trae
    # juegos de la temporada pasada, con jugadores que ya se fueron.
    roster = n.current_roster_ids(team_espn_id)
    if roster:
        players = {pid: info for pid, info in players.items() if pid in roster}
    return players


def opp_allowed_by_position(opponent, season, group, window=WINDOW, _cache=None):
    """Version en vivo de nba_props_data.opp_allowed_asof: lo que la defensa
    de `opponent` le permitio a titulares del grupo de posicion `group` en
    sus ultimos `window` juegos (por 36 min). Pasa `_cache` (un dict) para
    reusar las entradas entre varios jugadores contra el mismo rival."""
    if not group:
        return None
    key = str(opponent["espn_id"])
    if _cache is not None and key in _cache:
        entries = _cache[key]
    else:
        entries = []
        for g in n.espn_recent_games(opponent["espn_id"], season=season, last_n=window):
            other = g["opponent_espn_id"]
            try:
                roster = n.espn_game_roster(g["id"], other)
            except Exception:
                continue
            for entry in roster:
                if not entry["starter"] or entry["did_not_play"]:
                    continue
                try:
                    stats = n.espn_player_game_stats(g["id"], other, entry["player_id"])
                except Exception:
                    continue
                if stats and stats.get("minutes"):
                    entries.append({**pc.normalize_stats(stats), "group": n.position_group(entry)})
        if _cache is not None:
            _cache[key] = entries
    return pdata.allowed_per36(entries, group)


# Estados de ESPN que cuentan como "no juega" para repartir su produccion.
# Questionable / Day-To-Day NO cuentan (la mayoria termina jugando) - se
# muestran aparte como aviso.
OUT_STATUSES = ("out", "doubtful", "suspension", "suspended")


def parse_injuries(rep_team):
    """[(nombre_minusculas, estado_minusculas)] del reporte de lesionados."""
    out = []
    for item in rep_team.get("injuries_auto") or []:
        name, _, status = item.partition(" (")
        out.append((name.strip().lower(), status.rstrip(")").strip().lower()))
    return out


def _name_match(a, b):
    return a == b or (len(a) > 4 and (a in b or b in a))


def absent_teammates(rep_team, team_players, player_id=None):
    """Version en vivo de la deteccion de ausentes del entrenamiento:
    companeros reportados Out/Doubtful HOY que estaban en la rotacion
    reciente. Regresa (lista para vacated_features, nombres fuera, nombres
    en duda)."""
    injuries = parse_injuries(rep_team)
    absent, out_names, doubtful_names = [], [], []
    for pid, info in (team_players or {}).items():
        if player_id is not None and str(pid) == str(player_id):
            continue
        name = (info.get("name") or "").lower()
        status = next((st for nm, st in injuries if _name_match(nm, name)), None)
        if status is None:
            continue
        if any(s in status for s in OUT_STATUSES):
            a = pdata.absent_info(info["games"])
            if a:
                absent.append(a)
                out_names.append(info["name"])
        else:
            doubtful_names.append(f"{info['name']} ({status})")
    return absent, out_names, doubtful_names


# ---------------------------------------------------------------------------
# Features y prediccion
# ---------------------------------------------------------------------------

def build_props_row(recent, rep_team, rep_opp, is_home=None, opp_allowed=None, season=None, absent=None):
    """Fila de features para los modelos de props - la MISMA para el CLI, la
    app y la tanda del dia, con las mismas columnas que arma
    nba_props_data.py al entrenar (las que un modelo viejo no use se ignoran)."""
    last3 = recent[-3:]
    team_adv, opp_adv = rep_team["advanced"], rep_opp["advanced"]
    proj_margin = r.matchup_margin(team_adv, opp_adv)
    row = {
        "team_off_rtg": team_adv["off_rtg"], "team_pace": team_adv["pace"],
        "opp_def_rtg": opp_adv["def_rtg"], "opp_pace": opp_adv["pace"],
        "days_rest": rep_team.get("days_rest"), "b2b": int(bool(rep_team.get("is_b2b"))),
        "is_home": None if is_home is None else int(bool(is_home)),
        "proj_margin": proj_margin,
        "proj_blowout": abs(proj_margin) if proj_margin is not None else None,
        "start_rate": r._avg([1.0 if g.get("starter") else 0.0 for g in recent]),
    }
    for k in list(pc.BASE_STATS) + list(pc.COMBOS):
        row[pc.avg_feature(k)] = r._avg([pc.stat_value(g, k) for g in recent])
        row[f"{pc.avg_feature(k)}_last3"] = r._avg([pc.stat_value(g, k) for g in last3])
    row.update(pdata.season_features(recent, season or n.current_nba_season()))
    # Companeros fuera hoy y la produccion que dejan libre (ver absent_teammates).
    row.update(pdata.vacated_features(absent or [], pdata.main_group(recent)))
    for k in pc.ALLOWED_STATS:
        row[pc.allow_feature(k)] = (opp_allowed or {}).get(k)
    return row


def load_bundles(model_dir=APP_DIR):
    """Todos los modelos de props entrenados que existan, en el orden de
    props_config.TARGETS: {llave: bundle}."""
    out = {}
    for k in pc.TARGETS:
        path = os.path.join(model_dir, f"model_props_{k}.joblib")
        if os.path.exists(path):
            out[k] = joblib.load(path)
    return out


def predict_row(bundles, row):
    """{llave: prediccion} para todos los modelos cargados."""
    preds = {}
    for k, bundle in bundles.items():
        X = r.model_input(bundle, row)
        preds[k] = float(bundle["model"].predict(X)[0])
    return preds


def default_line(pred):
    """Linea inicial sugerida: la prediccion redondeada a .5."""
    return max(0.5, (pred // 1) + 0.5)


def american_to_prob(odds):
    """Momio americano -> probabilidad implicita (con el margen de la casa)."""
    odds = float(odds)
    return 100 / (odds + 100) if odds > 0 else -odds / (-odds + 100)


def over_under(bundle, pred, line, odds=-110):
    """Probabilidad de over/under contra una linea y ventaja contra el momio.
    None si el modelo no trae curva de error o no se dio linea."""
    curve = bundle.get("sigma_curve")
    if line is None or not curve:
        return None
    p_over = ptr.prob_over(pred, line, curve)
    p_need = american_to_prob(odds)
    best = "OVER" if p_over >= 0.5 else "UNDER"
    p_best = max(p_over, 1 - p_over)
    return {"line": line, "p_over": p_over, "p_under": 1 - p_over, "pick": best,
            "p_pick": p_best, "p_needed": p_need, "edge": p_best - p_need}


# ---------------------------------------------------------------------------
# Tanda del dia: todos los jugadores de todos los juegos
# ---------------------------------------------------------------------------

def predict_slate(bundles, season=None, game_date=None, window=WINDOW, min_minutes=pdata.MIN_AVG_MINUTES,
                  on_progress=None):
    """Predice TODOS los props de TODOS los jugadores (12+ min promedio) de
    los juegos de `game_date` (hoy por default). Excluye a quien ESPN
    reporta fuera (Out). Regresa un DataFrame: una fila por jugador con
    pred_<prop> para cada prop, mas datos del juego para registrar apuestas.

    on_progress(texto, fraccion) para mostrar avance (puede tardar varios
    minutos la primera vez del dia: los juegos recientes se descargan una
    vez y quedan en cache)."""
    season = season or n.current_nba_season()
    games = n.games_on_date(game_date or date.today())
    if not games:
        return pd.DataFrame()

    allowed_cache = {}
    rows = []
    total_teams = len(games) * 2
    done = 0
    for game in games:
        home, away = n.team_by_espn_id(game["home_espn_id"]), n.team_by_espn_id(game["away_espn_id"])
        if not home or not away:
            continue
        for team, opponent, is_home in ((home, away, True), (away, home, False)):
            label = f"{team['abbreviation']} vs {opponent['abbreviation']}"

            def prog(i, tot, label=label, done=done):
                if on_progress:
                    on_progress(f"{label}: juego {i + 1}/{tot}", (done + i / max(tot, 1)) / total_teams)

            try:
                rep_team = r.team_side_report(team, opponent, season, window, as_of_date=game["date"])
                rep_opp = r.team_side_report(opponent, team, season, window, as_of_date=game["date"])
                if not rep_team["advanced"] or not rep_opp["advanced"]:
                    continue
                players = team_players_recent_games(team["espn_id"], season, window=window, on_progress=prog)
            except Exception as e:
                print(f"  [salteado] {label}: {e}", file=sys.stderr)
                continue
            finally:
                done += 1
                n.flush_cache()

            _, team_out, _ = absent_teammates(rep_team, players)
            team_out = {x.lower() for x in team_out}
            for pid, info in players.items():
                recent = info["games"]
                if len(recent) < 3 or info["name"].lower() in team_out:
                    continue
                absent, out_names, doubt_names = absent_teammates(rep_team, players, player_id=pid)
                avg_min = r._avg([g["minutes"] for g in recent])
                if avg_min is None or avg_min < min_minutes:
                    continue
                try:
                    allowed = opp_allowed_by_position(opponent, season, pdata.main_group(recent), window,
                                                      _cache=allowed_cache)
                except Exception:
                    allowed = None
                row = build_props_row(recent, rep_team, rep_opp, is_home=is_home, opp_allowed=allowed,
                                      season=season, absent=absent)
                preds = predict_row(bundles, row)
                rows.append({
                    "player_id": pid, "player": info["name"], "team": team["full_name"],
                    "team_abbr": team["abbreviation"], "team_espn_id": team["espn_id"],
                    "opponent": opponent["full_name"], "opp_abbr": opponent["abbreviation"],
                    "event_id": game["event_id"], "game_date": game["date"],
                    "min": round(avg_min, 1), "starter_rate": row["start_rate"],
                    "companeros_fuera": ", ".join(out_names), "en_duda": ", ".join(doubt_names),
                    "pts_libres_misma_pos": round(row["vacated_points_same_pos"], 1),
                    **{f"pred_{k}": v for k, v in preds.items()},
                    **{f"avg_{k}": row.get(pc.avg_feature(k)) for k in preds},
                })
    n.flush_cache()
    return pd.DataFrame(rows).sort_values(["game_date", "team", "min"], ascending=[True, True, False])


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_lines(items):
    out = {}
    for item in items or []:
        k, _, v = item.partition("=")
        if k in pc.TARGETS and v:
            out[k] = float(v)
    return out


def main():
    parser = argparse.ArgumentParser(description="Predice props de jugador NBA")
    parser.add_argument("jugador", nargs="?", help="Nombre (o parte del nombre) del jugador")
    parser.add_argument("equipo_jugador", nargs="?", help="Equipo del jugador")
    parser.add_argument("equipo_rival", nargs="?", help="Equipo rival")
    parser.add_argument("--hoy", action="store_true",
                         help="Predice a TODOS los jugadores de los juegos de hoy y guarda props_hoy.csv")
    parser.add_argument("--season", type=int, default=None,
                         help="Anio en que TERMINA la temporada. Default: temporada actual")
    parser.add_argument("--window", type=int, default=WINDOW)
    parser.add_argument("--linea", action="append", metavar="PROP=LINEA",
                         help=f"Linea de tu casa, ej. --linea points=24.5 --linea pra=38.5 "
                              f"(props: {', '.join(pc.TARGETS)})")
    parser.add_argument("--momio", type=float, default=-110,
                         help="Momio americano del over/under (default -110)")
    args = parser.parse_args()

    bundles = load_bundles()
    if not bundles:
        print("No hay modelos de props entrenados - corre nba_retrain_pipeline.py primero.")
        return
    season = args.season or n.current_nba_season()

    if args.hoy:
        df = predict_slate(bundles, season=season, window=args.window,
                           on_progress=lambda t, f: print(f"  [{f:.0%}] {t}", file=sys.stderr))
        if df.empty:
            print("No hay juegos hoy (o no se pudo descargar el calendario).")
            return
        df.to_csv("props_hoy.csv", index=False)
        cols = ["player", "team_abbr", "opp_abbr", "min"] + [f"pred_{k}" for k in bundles]
        print(df[cols].round(1).to_string(index=False))
        print(f"\n{len(df)} jugadores - guardado en props_hoy.csv")
        return

    if not (args.jugador and args.equipo_jugador and args.equipo_rival):
        parser.error("indica jugador, equipo y rival, o usa --hoy")

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
        opp_allowed = opp_allowed_by_position(opponent, season, pdata.main_group(recent), window=args.window) \
            if len(recent) >= 3 else None
        team_players = team_players_recent_games(team["espn_id"], season, window=args.window)
    finally:
        n.flush_cache()

    if len(recent) < 3:
        print(f"Muy pocos juegos recientes registrados ({len(recent)}) para predecir con confianza.")
        return
    if not rep_team["advanced"] or not rep_opp["advanced"]:
        print("No hay suficientes datos de equipo recientes para calcular el contexto del rival.")
        return

    is_home = str(matchup.get("home_espn_id")) == str(team["espn_id"]) if matchup else None
    absent, out_names, doubt_names = absent_teammates(rep_team, team_players, player_id=player_id)
    row = build_props_row(recent, rep_team, rep_opp, is_home=is_home, opp_allowed=opp_allowed, season=season,
                          absent=absent)
    if out_names:
        print(f"Companeros FUERA hoy: {', '.join(out_names)} -> dejan {row['vacated_points']:.1f} pts, "
              f"{row['vacated_assists']:.1f} ast, {row['vacated_minutes']:.0f} min libres "
              f"({row['vacated_points_same_pos']:.1f} pts de su misma posicion)")
    if doubt_names:
        print(f"En duda (no se cuentan como fuera): {', '.join(doubt_names)}")
    preds = predict_row(bundles, row)
    lines = _parse_lines(args.linea)

    print(f"Ultimos {len(recent)} juegos: " + ", ".join(
        f"{g['points']:.0f}p/{g['rebounds']:.0f}r/{g['assists']:.0f}a" for g in recent) + "\n")
    print("=" * 70)
    for k, pred in preds.items():
        bundle = bundles[k]
        avg = row[pc.avg_feature(k)]
        line = lines.get(k, default_line(pred))
        ou = over_under(bundle, pred, line, args.momio)
        txt = f"{pc.LABELS[k]:<12} {pred:5.1f}  (promedio {avg:.1f})"
        if ou:
            veredicto = "VALOR" if ou["edge"] > 0 else "sin ventaja"
            txt += (f"  | linea {line}: over {ou['p_over']:.0%} / under {ou['p_under']:.0%} -> "
                    f"{ou['pick']} ({ou['edge']:+.1%}, {veredicto})")
        print(txt)

    if matchup and all(k in preds for k in ("points", "rebounds", "assists")):
        pt.log_prediction(pt.build_log_row(
            event_id=matchup["event_id"], game_date=matchup["date"],
            player_id=player_id, player_name=player_name,
            team=team["full_name"], team_espn_id=team["espn_id"], opponent=opponent["full_name"],
            pred_points=preds["points"], model_points=bundles["points"]["model_name"],
            baseline_points=row["avg_points"],
            pred_rebounds=preds["rebounds"], model_rebounds=bundles["rebounds"]["model_name"],
            baseline_rebounds=row["avg_rebounds"],
            pred_assists=preds["assists"], model_assists=bundles["assists"]["model_name"],
            baseline_assists=row["avg_assists"],
        ))


if __name__ == "__main__":
    main()
