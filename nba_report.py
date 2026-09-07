"""
Reporte pre-apuesta para un juego de NBA entre dos equipos: ofensiva/defensiva
avanzada (calculada con formulas estandar a partir de boxscores reales de
ESPN, ventana movil de N juegos), impacto de lesionados (On/Off Net Rating
via pbpstats.com), fatiga de calendario, y proyeccion de puntos por cuarto y
marcador final.

FASE 1 (formulas transparentes, sin modelo entrenado todavia - eso es la
Fase 2, una vez confirmado que este reporte funciona bien).

Uso:
    python nba_report.py "Lakers" "Celtics"
    python nba_report.py "Lakers" "Celtics" --last-n 5 --season 2025
    python nba_report.py "Lakers" "Celtics" --lesionado-a "LeBron James"

Notas:
  - stats.nba.com esta bloqueado a nivel de red para este usuario - este
    reporte usa ESPN (boxscores) + pbpstats.com (on/off por lesionados) en
    su lugar. Ver nba_data.py para el detalle de por que y que se pierde
    (Screen Assists, Deflection Rate, Box-Outs, Loose Balls - solo existen
    en tracking de video propietario de la NBA, ninguna fuente publica las
    tiene).
  - --season es el anio en que TERMINA la temporada (2025 para 2024-25).
    Como la temporada nueva puede no haber arrancado, usa una temporada
    pasada para practicar (ej. --season 2025).
"""

import argparse
import sys

import nba_data as n

DEFAULT_LAST_N = 10


def pct(x):
    return "N/D" if x is None else f"{x * 100:.1f}%"


def num(x, nd=1):
    return "N/D" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# Formulas estandar (Dean Oliver / Four Factors) - publicas, no propietarias.
# Se calculan a partir de los conteos crudos del boxscore de ESPN porque
# ESPN no trae los ratings ya calculados (a diferencia de stats.nba.com).
# ---------------------------------------------------------------------------

def _v(stats, key, default=0.0):
    val = stats.get(key, default)
    return val if isinstance(val, (int, float)) else default


def possessions(stats):
    fga = _v(stats, "fieldGoalsAttempted")
    fta = _v(stats, "freeThrowsAttempted")
    oreb = _v(stats, "offensiveRebounds")
    tov = _v(stats, "totalTurnovers") or _v(stats, "turnovers")
    return fga - oreb + tov + 0.4 * fta


def game_advanced(my_stats, opp_stats, my_points, opp_points, n_quarters):
    """OffRtg/DefRtg/Net/Pace/eFG%/TS%/TOV%/OREB%/DREB%/AST-Ratio de UN
    juego, desde la perspectiva de 'my_stats'."""
    my_poss = possessions(my_stats)
    opp_poss = possessions(opp_stats)
    game_poss = (my_poss + opp_poss) / 2
    if game_poss <= 0:
        return None

    minutes = 48 + max(0, n_quarters - 4) * 5
    fgm = _v(my_stats, "fieldGoalsMade")
    fga = _v(my_stats, "fieldGoalsAttempted")
    fg3m = _v(my_stats, "threePointFieldGoalsMade")
    fta = _v(my_stats, "freeThrowsAttempted")
    oreb = _v(my_stats, "offensiveRebounds")
    dreb = _v(my_stats, "defensiveRebounds")
    ast = _v(my_stats, "assists")
    tov = _v(my_stats, "totalTurnovers") or _v(my_stats, "turnovers")
    opp_oreb = _v(opp_stats, "offensiveRebounds")
    opp_dreb = _v(opp_stats, "defensiveRebounds")

    return {
        "off_rtg": my_points / game_poss * 100,
        "def_rtg": opp_points / game_poss * 100,
        "net_rtg": (my_points - opp_points) / game_poss * 100,
        "pace": game_poss * (48 / minutes) if minutes else None,
        "efg_pct": (fgm + 0.5 * fg3m) / fga if fga else None,
        "ts_pct": my_points / (2 * (fga + 0.44 * fta)) if (fga or fta) else None,
        "tov_pct": tov / game_poss if game_poss else None,
        "oreb_pct": oreb / (oreb + opp_dreb) if (oreb + opp_dreb) else None,
        "dreb_pct": dreb / (dreb + opp_oreb) if (dreb + opp_oreb) else None,
        "ast_pct": ast / fgm if fgm else None,
        "fast_break_pts": _v(my_stats, "fastBreakPoints", None),
        "points_in_paint": _v(my_stats, "pointsInPaint", None),
    }


def _avg(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def team_rolling_report(team, last_n, season):
    """Recolecta los ultimos N juegos YA jugados del equipo, junta el
    boxscore propio y del rival de cada uno, y promedia las metricas
    avanzadas calculadas juego por juego (ventana movil, no temporada
    completa)."""
    try:
        games = n.espn_played_games(team["espn_id"], season=season, last_n=last_n)
    except Exception as e:
        print(f"  ERROR recolectando juegos de {team['full_name']}: {e}", file=sys.stderr)
        return []
    per_game = []
    for e in games:
        opp_espn_id = e["opponent_espn_id"]
        try:
            my_stats, quarters = n.team_game_boxscore(e["id"], team["espn_id"])
            opp_stats, opp_quarters = n.team_game_boxscore(e["id"], opp_espn_id)
        except Exception as ex:
            print(f"    [salteado] juego {e['id']}: {ex}", file=sys.stderr)
            continue
        if not my_stats or not opp_stats or not quarters or not opp_quarters:
            continue
        my_points = sum(quarters)
        opp_points = sum(opp_quarters)
        adv = game_advanced(my_stats, opp_stats, my_points, opp_points, len(quarters))
        if adv:
            per_game.append((e, adv, quarters))
    return per_game


def probable_lineup(team, season, last_n=5, injured_names=None):
    """Aproximacion de la alineacion titular probable: los 5 jugadores que
    MAS seguido empezaron de titulares en los ultimos `last_n` juegos ya
    jugados de este equipo, EXCLUYENDO a quien este reportado lesionado
    ahora mismo (injured_names) - si el titular habitual esta lesionado, se
    promueve al siguiente jugador con mas arranques recientes en su lugar,
    en vez de solo marcarlo con una advertencia y dejarlo en la lista. La
    NBA no publica una alineacion 'confirmada' con dias de anticipacion
    como si lo hace MLB con el orden al bat (los titulares se confirman
    horas antes del juego) - esto sigue siendo una inferencia a partir del
    patron reciente, marcada como tal en el reporte, no una confirmacion
    oficial."""
    try:
        games = n.espn_played_games(team["espn_id"], season=season, last_n=last_n)
    except Exception as e:
        print(f"    (no se pudo calcular alineacion probable: {e})", file=sys.stderr)
        return None

    counts = {}
    names = {}
    n_games_checked = 0
    for g in games:
        try:
            roster = n.espn_game_roster(g["id"], team["espn_id"])
        except Exception:
            continue
        n_games_checked += 1
        for entry in roster:
            if entry["starter"] and not entry["did_not_play"]:
                pid = entry["player_id"]
                counts[pid] = counts.get(pid, 0) + 1
                names[pid] = entry["name"]
    if not counts or not n_games_checked:
        return None

    injured_names = injured_names or []
    ranked = sorted(counts.items(), key=lambda x: -x[1])

    def _is_injured(name):
        name_l = name.lower()
        return any(name_l in inj or inj in name_l for inj in injured_names)

    players = []
    excluidos = []
    for pid, c in ranked:
        if _is_injured(names[pid]):
            excluidos.append({"player_id": pid, "name": names[pid], "starts": c})
            continue
        players.append({"player_id": pid, "name": names[pid], "starts": c})
        if len(players) == 5:
            break

    return {"players": players, "excluidos_por_lesion": excluidos, "n_games_checked": n_games_checked}


def team_side_report(team, opponent, season, last_n, lesionado_override=None, as_of_date=None):
    per_game = team_rolling_report(team, last_n, season)
    if not per_game:
        return {"team": team, "opponent": opponent, "advanced": None, "quarter_profile": None,
                "n_games": 0, "injury_impact": None}

    advanced = {
        "off_rtg": _avg([g[1]["off_rtg"] for g in per_game]),
        "def_rtg": _avg([g[1]["def_rtg"] for g in per_game]),
        "net_rtg": _avg([g[1]["net_rtg"] for g in per_game]),
        "pace": _avg([g[1]["pace"] for g in per_game]),
        "efg_pct": _avg([g[1]["efg_pct"] for g in per_game]),
        "ts_pct": _avg([g[1]["ts_pct"] for g in per_game]),
        "tov_pct": _avg([g[1]["tov_pct"] for g in per_game]),
        "oreb_pct": _avg([g[1]["oreb_pct"] for g in per_game]),
        "dreb_pct": _avg([g[1]["dreb_pct"] for g in per_game]),
        "ast_pct": _avg([g[1]["ast_pct"] for g in per_game]),
        "fast_break_pts": _avg([g[1]["fast_break_pts"] for g in per_game]),
        "points_in_paint": _avg([g[1]["points_in_paint"] for g in per_game]),
    }

    # Perfil de reparto por cuarto (promedio de que % de sus puntos anota en
    # cada cuarto en estos mismos N juegos)
    q_totals = [0.0, 0.0, 0.0, 0.0]
    counted = 0
    for _, _, quarters in per_game:
        if len(quarters) < 4:
            continue
        total = sum(quarters[:4])
        if total <= 0:
            continue
        for i in range(4):
            q_totals[i] += quarters[i] / total
        counted += 1
    quarter_profile = {
        "q1": q_totals[0] / counted, "q2": q_totals[1] / counted,
        "q3": q_totals[2] / counted, "q4": q_totals[3] / counted,
        "n_games": counted,
    } if counted else {"q1": 0.25, "q2": 0.25, "q3": 0.25, "q4": 0.25, "n_games": 0}

    # Fatiga: dias de descanso / back-to-back antes del PROXIMO juego real.
    # Solo tiene sentido si hay un juego programado real (as_of_date) - al
    # practicar con una temporada pasada (sin partido futuro), calcularlo
    # contra la fecha de hoy da un numero sin sentido (cientos de "dias de
    # descanso"), asi que se omite en ese caso.
    last_game_date = per_game[-1][0].get("date")
    days_rest = None
    is_b2b = False
    if last_game_date and as_of_date:
        from datetime import datetime as dt
        last_dt = dt.fromisoformat(last_game_date.replace("Z", "+00:00")).date()
        ref_date = dt.fromisoformat(as_of_date).date() if isinstance(as_of_date, str) else as_of_date
        days_rest = (ref_date - last_dt).days
        is_b2b = days_rest <= 1

    # Lesionados actuales de ESPN (roster + status) - solo informativo, para
    # saber a quien vale la pena forzar con --lesionado-a/-b (el impacto
    # numerico via On/Off solo se calcula para el jugador forzado, no para
    # toda la lista, porque cada consulta a pbpstats cuesta una llamada).
    injuries_auto = []
    try:
        for entry in n.espn_team_injuries(team["espn_id"]):
            for inj in entry.get("injuries", entry.get("athletes", [])):
                name = (inj.get("athlete") or {}).get("displayName") or inj.get("name")
                status = (inj.get("status") or inj.get("type", {}).get("description"))
                if name:
                    injuries_auto.append(f"{name} ({status or 'N/D'})")
    except Exception as e:
        print(f"    (no se pudo descargar lista de lesionados: {e})", file=sys.stderr)

    # injured_names alimenta la EXCLUSION dentro de probable_lineup (no solo
    # una advertencia despues) - si el titular habitual esta en el reporte
    # de lesionados, no debe aparecer como probable hasta que ya no este en
    # esa lista (sin importar el status exacto: Out/Questionable/Day-to-Day
    # todos cuentan como "lesionado" para efectos de la alineacion probable).
    injured_names_lower = [inj.split(" (")[0].lower() for inj in injuries_auto]
    lineup_probable = probable_lineup(team, season, injured_names=injured_names_lower)

    injury_impact = None
    if lesionado_override:
        try:
            pid, pname = n.resolve_player_id(team["nba_id"], lesionado_override, season=season)
            if pid:
                results = n.pbp_player_on_off(team["nba_id"], pid, season=season)
                net = results.get("OffPossessions") or results.get("NetRating") or results
                injury_impact = {"player": pname, "raw": net}
        except Exception as e:
            print(f"    (no se pudo calcular impacto de lesionado: {e})", file=sys.stderr)

    return {
        "team": team, "opponent": opponent, "advanced": advanced,
        "quarter_profile": quarter_profile, "n_games": len(per_game),
        "injuries_auto": injuries_auto, "lineup_probable": lineup_probable,
        "days_rest": days_rest, "is_b2b": is_b2b, "injury_impact": injury_impact,
    }


def project_points(team_rep, opp_rep):
    adv, opp_adv = team_rep["advanced"], opp_rep["advanced"]
    if not adv or not opp_adv or adv.get("off_rtg") is None or opp_adv.get("def_rtg") is None:
        return None
    pace = _avg([adv.get("pace"), opp_adv.get("pace")]) or 99.0
    expected_rtg = _avg([adv["off_rtg"], opp_adv["def_rtg"]])
    projected_points = expected_rtg * (pace / 100)
    if team_rep.get("is_b2b"):
        projected_points *= 0.97
    return {"pace": pace, "projected_points": projected_points}


def quarter_breakdown(projected_points, quarter_profile):
    qp = quarter_profile
    q = [projected_points * qp[f"q{i}"] for i in range(1, 5)]
    return {"q1": q[0], "q2": q[1], "q3": q[2], "q4": q[3],
            "mitad1": q[0] + q[1], "mitad2": q[2] + q[3], "n_games_muestra": qp["n_games"]}


def print_team_block(rep, projection, quarters):
    t = rep["team"]
    print(f"\n{'-' * 60}")
    print(f"{t['full_name']} ({t['abbreviation']}) - {rep['n_games']} juegos en la ventana")
    print(f"{'-' * 60}")

    adv = rep["advanced"]
    if not adv:
        print("  Sin datos suficientes (revisa que el equipo tenga juegos jugados en la temporada elegida).")
        return
    print(f"  OffRtg: {num(adv['off_rtg'])}  |  DefRtg: {num(adv['def_rtg'])}  |  "
          f"Net: {num(adv['net_rtg'])}  |  Pace: {num(adv['pace'])}")
    print(f"  eFG%: {pct(adv['efg_pct'])}  |  TS%: {pct(adv['ts_pct'])}  |  TOV%: {pct(adv['tov_pct'])}")
    print(f"  OREB%: {pct(adv['oreb_pct'])}  |  DREB%: {pct(adv['dreb_pct'])}  |  "
          f"AST% (de canastas asistidas): {pct(adv['ast_pct'])}")
    print(f"  Pts contraataque: {num(adv['fast_break_pts'])}  |  Pts en la pintura: {num(adv['points_in_paint'])}")

    alerta_b2b = " [BACK-TO-BACK]" if rep.get("is_b2b") else ""
    print(f"  Descanso: {rep.get('days_rest', 'N/D')} dia(s){alerta_b2b}")

    lp = rep.get("lineup_probable")
    if lp:
        print(f"  Alineacion PROBABLE (mas titular en sus ultimos {lp['n_games_checked']} juegos, "
              f"NO confirmada oficialmente, excluye lesionados):")
        for p in lp["players"]:
            print(f"    - {p['name']} (titular en {p['starts']}/{lp['n_games_checked']})")
        if lp.get("excluidos_por_lesion"):
            excl = ", ".join(f"{p['name']} ({p['starts']}/{lp['n_games_checked']})"
                              for p in lp["excluidos_por_lesion"])
            print(f"    (excluidos por lesion: {excl})")
    else:
        print("  Alineacion probable: N/D (sin juegos recientes suficientes).")

    if rep.get("injuries_auto"):
        print(f"  Lesionados (reporte de ESPN): {', '.join(rep['injuries_auto'][:8])}")
    else:
        print("  Lesionados: ninguno reportado por ESPN ahora mismo (o no se pudo consultar).")

    if rep.get("injury_impact"):
        print(f"  LESIONADO forzado: {rep['injury_impact']['player']} - impacto On/Off: "
              f"{rep['injury_impact']['raw']}")

    if projection:
        print(f"\n  >>> Puntos proyectados: {num(projection['projected_points'])} "
              f"(pace combinado: {num(projection['pace'])})")
    if quarters:
        print(f"  Desglose por cuarto (patron historico, {quarters['n_games_muestra']} juegos de muestra):")
        print(f"    Q1: {num(quarters['q1'])}  Q2: {num(quarters['q2'])}  (Mitad 1: {num(quarters['mitad1'])})")
        print(f"    Q3: {num(quarters['q3'])}  Q4: {num(quarters['q4'])}  (Mitad 2: {num(quarters['mitad2'])})")


def main():
    parser = argparse.ArgumentParser(description="Reporte pre-apuesta NBA para dos equipos")
    parser.add_argument("equipo_a")
    parser.add_argument("equipo_b")
    parser.add_argument("--season", type=int, default=None,
                         help="Anio en que TERMINA la temporada (2025 para 2024-25). Default: temporada actual")
    parser.add_argument("--last-n", type=int, default=DEFAULT_LAST_N,
                         help=f"Ventana movil de juegos (default {DEFAULT_LAST_N})")
    parser.add_argument("--lesionado-a", default=None, help="Forzar jugador lesionado del Equipo A")
    parser.add_argument("--lesionado-b", default=None, help="Forzar jugador lesionado del Equipo B")
    args = parser.parse_args()

    season = args.season or n.current_nba_season()
    team_a = n.resolve_team(args.equipo_a)
    team_b = n.resolve_team(args.equipo_b)
    if team_a["nba_id"] == team_b["nba_id"]:
        print(f"ERROR: '{args.equipo_a}' y '{args.equipo_b}' son el mismo equipo.")
        return

    print("=" * 60)
    print(f"REPORTE NBA: {team_a['full_name']} vs {team_b['full_name']}")
    print(f"Temporada: {season - 1}-{str(season)[2:]}  |  Ventana movil: ultimos {args.last_n} juegos")
    print("=" * 60)

    try:
        matchup = n.find_next_matchup(team_a, team_b)
    except Exception as e:
        print(f"\n(no se pudo buscar el proximo juego: {e})", file=sys.stderr)
        matchup = None
    if matchup:
        print(f"\nProximo juego programado: {matchup['date']} "
              f"({'Local' if matchup['home_espn_id'] == team_a['espn_id'] else 'Visitante'}: {team_a['full_name']})")
    else:
        print("\nNo encontre un juego programado entre estos dos equipos en los proximos 14 dias - "
              "se muestran datos generales de cada equipo (util para practicar con temporadas pasadas).")

    print("\nCalculando metricas avanzadas de cada equipo (esto pide varios juegos, puede tardar un minuto)...")
    try:
        as_of_date = matchup["date"] if matchup else None
        rep_a = team_side_report(team_a, team_b, season, args.last_n, lesionado_override=args.lesionado_a, as_of_date=as_of_date)
        rep_b = team_side_report(team_b, team_a, season, args.last_n, lesionado_override=args.lesionado_b, as_of_date=as_of_date)
    finally:
        # Guarda en cache los juegos ya jugados que se hayan descargado en
        # esta corrida (aunque algo mas haya fallado a medias) - la proxima
        # vez que se pida cualquiera de esos mismos juegos, ya no se le pide
        # nada a ESPN.
        n.flush_cache()

    proj_a = project_points(rep_a, rep_b)
    proj_b = project_points(rep_b, rep_a)
    q_a = quarter_breakdown(proj_a["projected_points"], rep_a["quarter_profile"]) if proj_a else None
    q_b = quarter_breakdown(proj_b["projected_points"], rep_b["quarter_profile"]) if proj_b else None

    print_team_block(rep_a, proj_a, q_a)
    print_team_block(rep_b, proj_b, q_b)

    if proj_a and proj_b:
        print(f"\n{'=' * 60}")
        print("MARCADOR FINAL PROYECTADO")
        print(f"{'=' * 60}")
        print(f"  {team_a['full_name']}: {num(proj_a['projected_points'])}")
        print(f"  {team_b['full_name']}: {num(proj_b['projected_points'])}")
        total = proj_a["projected_points"] + proj_b["projected_points"]
        favorito = team_a if proj_a["projected_points"] > proj_b["projected_points"] else team_b
        margen = abs(proj_a["projected_points"] - proj_b["projected_points"])
        print(f"  Total proyectado: {num(total)}")
        print(f"  Favorito (formula, no modelo entrenado): {favorito['full_name']} por {num(margen)}")
        print("\n  NOTA: Fase 1 (formula transparente) - sin modelos de ML ni picks con confianza "
              "calibrada todavia. Eso es la Fase 2, una vez confirmado que este reporte funciona bien.")
    else:
        print("\nNo se pudo proyectar el marcador (faltan datos suficientes de alguno de los dos equipos).")


if __name__ == "__main__":
    main()
