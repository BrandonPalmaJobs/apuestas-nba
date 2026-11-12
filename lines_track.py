"""
Registro de LINEAS REALES de la casa (Playdoit u otra), aunque no se apueste.

Hasta ahora el modelo solo se podia evaluar contra una linea "de
referencia" inventada (el promedio del jugador, o la fraccion del total en
el 1Q). Con las lineas reales se mide lo que importa: si el modelo le gana a
TU casa. Cada linea guardada se liquida sola cuando termina el juego.

Mercados:
  - prop           : prop de jugador (puntos, PRA, triples...)
  - q_total        : total de puntos de un cuarto (stat = q1 / q2)
  - q_team         : puntos de UN equipo en un cuarto (stat = q1 / q2)
"""

import os
from datetime import datetime

import numpy as np
import pandas as pd

import nba_data as n
import props_config as pc

LINES_PATH = "house_lines_log.csv"
COLUMNS = ["logged_at", "market", "event_id", "game_date", "subject_id", "subject", "team_espn_id",
           "opp_espn_id", "stat", "line", "odds", "pred", "p_over", "model",
           "actual", "result", "evaluated"]


def log_lines(rows, path=LINES_PATH):
    """Guarda varias lineas. Si ya habia una para el mismo juego, sujeto y
    mercado/stat, se reemplaza (cuenta la ultima que viste)."""
    if not rows:
        return 0
    now = datetime.now().isoformat(timespec="seconds")
    new = pd.DataFrame([{**{c: r.get(c) for c in COLUMNS}, "logged_at": now, "actual": None,
                         "result": None, "evaluated": False} for r in rows])
    if os.path.exists(path):
        old = pd.read_csv(path, dtype=str)
        key = lambda d: (d["market"].astype(str) + "|" + d["event_id"].astype(str) + "|" +
                         d["subject_id"].astype(str) + "|" + d["stat"].astype(str))
        old = old[~key(old).isin(set(key(new.astype(str))))]
        new = pd.concat([old, new.astype(str)], ignore_index=True)
    new.to_csv(path, index=False)
    return len(rows)


def _actual(row):
    eid, market, stat = str(row["event_id"]), row["market"], row["stat"]
    if market == "prop":
        stats = n.espn_player_game_stats(eid, row["team_espn_id"], row["subject_id"])
        if not stats or not stats.get("minutes"):
            return None  # no jugo
        return pc.stat_value(pc.normalize_stats(stats), stat)
    i = {"q1": 0, "q2": 1}.get(stat)
    if i is None:
        return None
    mine = n.espn_team_game_linescores(eid, row["team_espn_id"])
    if market == "q_team":
        return mine[i] if mine and len(mine) > i else None
    theirs = n.espn_team_game_linescores(eid, row["opp_espn_id"])
    if mine and theirs and len(mine) > i and len(theirs) > i:
        return mine[i] + theirs[i]
    return None


def settle(path=LINES_PATH):
    """Liquida las lineas de juegos ya terminados."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, dtype=str)
    for idx, row in df[df["evaluated"].str.lower() != "true"].iterrows():
        try:
            if not n.espn_competition_detail(row["event_id"]).get("completed"):
                continue
            act = _actual(row)
        except Exception:
            continue
        line = float(row["line"])
        if act is None:
            res = "DNP"
        elif act > line:
            res = "OVER"
        elif act < line:
            res = "UNDER"
        else:
            res = "PUSH"
        df.loc[idx, ["actual", "result", "evaluated"]] = [str(act) if act is not None else "", res, "True"]
    df.to_csv(path, index=False)
    n.flush_cache()
    return df


def _profit(win, odds):
    o = float(odds)
    return (o / 100 if o > 0 else 100 / -o) if win else -1.0


def summarize(df):
    """Por mercado: que tan seguido el lado que el modelo veia mas probable
    le gano a la linea REAL, y que habria pasado apostando 1 unidad solo
    donde el modelo veia ventaja contra el momio."""
    if df is None or df.empty:
        return pd.DataFrame()
    d = df[(df["evaluated"].str.lower() == "true") & df["result"].isin(["OVER", "UNDER"])].copy()
    if d.empty:
        return pd.DataFrame()
    d["p_over"] = d["p_over"].astype(float)
    d["odds"] = d["odds"].astype(float)
    d["pick"] = np.where(d["p_over"] >= 0.5, "OVER", "UNDER")
    # Props: solo overs (decision del usuario)
    d.loc[d["market"] == "prop", "pick"] = "OVER"
    d["p_pick"] = np.where(d["pick"] == "OVER", d["p_over"], 1 - d["p_over"])
    d["win"] = d["pick"] == d["result"]
    d["need"] = np.where(d["odds"] > 0, 100 / (d["odds"] + 100), -d["odds"] / (-d["odds"] + 100))
    d["value"] = d["p_pick"] > d["need"]
    d["units"] = [_profit(w, o) for w, o in zip(d["win"], d["odds"])]
    names = {"prop": "Props de jugador", "q_total": "Total del cuarto", "q_team": "Puntos de equipo por cuarto"}
    rows = []
    for (market, stat), g in d.groupby(["market", "stat"]):
        v = g[g["value"]]
        label = pc.LABELS.get(stat, {"q1": "1Q", "q2": "2Q"}.get(stat, stat))
        rows.append({
            "Mercado": f"{names.get(market, market)} · {label}", "Lineas": len(g),
            "Acierto del modelo": f"{g['win'].mean():.0%}",
            "Con 60%+ de confianza": (f"{g[g['p_pick'] >= 0.6]['win'].mean():.0%} "
                                      f"({(g['p_pick'] >= 0.6).sum()})") if (g["p_pick"] >= 0.6).any() else "-",
            "Si apostabas solo con ventaja": (f"{v['win'].sum()}-{(~v['win']).sum()}, "
                                              f"{v['units'].sum():+.1f} u") if len(v) else "-",
        })
    return pd.DataFrame(rows)
