"""
Modelo dedicado a los TOTALES DEL PRIMER Y SEGUNDO CUARTO (over/under de
puntos de ambos equipos en el 1Q y en el 2Q).

Por que un modelo aparte: las casas suelen sacar la linea del 1Q como una
fraccion fija del total del partido (~25-26%). Aqui se predice el cuarto
con lo que de verdad pasa en ESE cuarto:
  - puntos anotados y permitidos por cada equipo en el 1Q / 2Q (ultimos 10
    y ultimos 5 juegos) - es la mejor senal disponible de ritmo y eficiencia
    de los titulares en ese tramo (ESPN no da posesiones por cuarto);
  - ritmo, rating ofensivo/defensivo, eFG% y perdidas del partido completo;
  - back-to-back y dias de descanso, local/visita;
  - hora de inicio (juegos de tarde / matinee);
  - titulares habituales fuera (en vivo: reporte de lesionados Out/Doubtful);
  - la linea "estilo casa" (fraccion del total proyectado) como referencia.

Entrenamiento: SOLO con datos ya guardados en cache/game_summaries.json
(marcador por cuarto, estadisticas y rosters de cada juego), sin descargar
nada: tarda segundos. Se evalua contra la linea estilo casa en juegos no
vistos, se elige el mejor modelo, se corrige su sesgo reciente y se
reentrena con todo.

Uso:
    python nba_quarters.py --train
"""

import argparse
import json
import os
import statistics
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import nba_data as n
import nba_props_train as ptr
import nba_report as r

APP_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(APP_DIR, "cache", "game_summaries.json")
WINDOW = 10
SHORT = 5
MIN_PRIOR = 5
MAX_REST = 10
QUARTERS = {"q1": 0, "q2": 1}
LABELS = {"q1": "1er cuarto", "q2": "2do cuarto"}
ET = ZoneInfo("America/New_York")

SIDE_FEATS = ["q1_poss", "q1_ppp_for", "q1_ppp_against", "q2_poss", "q2_ppp_for", "q2_ppp_against",
              "q1_for", "q1_against", "q1_for_l5", "q1_against_l5",
              "q2_for", "q2_against", "q2_for_l5", "q2_against_l5",
              "pts_for", "pts_against", "pace", "off_rtg", "def_rtg", "efg_pct", "tov_pct",
              "days_rest", "b2b", "starters_out", "n_games"]
GAME_FEATS = ["start_hour_et", "matinee", "base_q1_matchup", "base_q2_matchup", "base_q1_pace", "base_q2_pace",
              "base_full_total", "base_q1_casa", "base_q2_casa"]
FEATURES = [f"home_{f}" for f in SIDE_FEATS] + [f"away_{f}" for f in SIDE_FEATS] + GAME_FEATS

# Fraccion historica del total que cae en cada cuarto (se recalcula al
# entrenar y se guarda en el modelo; esto es solo el valor por defecto).
DEFAULT_SHARE = {"q1": 0.255, "q2": 0.255}


# ---------------------------------------------------------------------------
# Historial por equipo
# ---------------------------------------------------------------------------

def _record(team_id, my_q, opp_q, my_stats, opp_stats, date, is_home, played_ids, started_ids, qposs=None):
    """Un juego de un equipo, en el formato que usan las variables."""
    adv = None
    if my_stats and opp_stats:
        try:
            adv = r.game_advanced(my_stats, opp_stats, sum(my_q), sum(opp_q), len(my_q))
        except Exception:
            adv = None
    adv = adv or {}
    return {
        "team_id": str(team_id), "date": date, "is_home": is_home,
        "q1_for": my_q[0], "q1_against": opp_q[0], "q2_for": my_q[1], "q2_against": opp_q[1],
        "pts_for": sum(my_q), "pts_against": sum(opp_q),
        "pace": adv.get("pace"), "off_rtg": adv.get("off_rtg"), "def_rtg": adv.get("def_rtg"),
        "efg_pct": adv.get("efg_pct"), "tov_pct": adv.get("tov_pct"),
        "played": played_ids, "started": started_ids,
        # Posesiones del cuarto (promedio de ambos equipos = ritmo del cuarto)
        # y puntos por posesion propios / permitidos.
        **_pace_fields(team_id, my_q, opp_q, qposs),
    }


def _pace_fields(team_id, my_q, opp_q, qposs):
    out = {}
    if not qposs:
        return {f"{q}_{k}": None for q in QUARTERS for k in ("poss", "ppp_for", "ppp_against")}
    other = next((t for t in qposs if t != str(team_id)), None)
    for q, i in QUARTERS.items():
        mine = (qposs.get(str(team_id)) or {}).get(q)
        theirs = (qposs.get(other) or {}).get(q) if other else None
        poss = _avg([mine, theirs])
        out[f"{q}_poss"] = poss
        out[f"{q}_ppp_for"] = my_q[i] / poss if poss else None
        out[f"{q}_ppp_against"] = opp_q[i] / poss if poss else None
    return out


def _roster_sets(roster):
    if not roster:
        return None, None
    played = {str(e["player_id"]) for e in roster if not e.get("did_not_play")}
    started = {str(e["player_id"]) for e in roster if e.get("starter")}
    return played, started


def load_cached_games(cache=None):
    """Todos los juegos terminados con marcador por cuarto de ambos equipos
    en el cache, en orden cronologico."""
    if cache is None:
        with open(CACHE_PATH) as f:
            cache = json.load(f)
    games = []
    for key, comp in cache.items():
        if not key.startswith("comp:") or not isinstance(comp, dict) or not comp.get("completed"):
            continue
        eid = key[5:]
        comps = comp.get("competitors") or []
        home = next((c["espn_id"] for c in comps if c.get("home_away") == "home"), None)
        away = next((c["espn_id"] for c in comps if c.get("home_away") == "away"), None)
        if not home or not away or not comp.get("date"):
            continue
        hq, aq = cache.get(f"lines:{eid}:{home}"), cache.get(f"lines:{eid}:{away}")
        if not hq or not aq or len(hq) < 4 or len(aq) < 4:
            continue
        hs, as_ = cache.get(f"stats:{eid}:{home}"), cache.get(f"stats:{eid}:{away}")
        hp, hst = _roster_sets(cache.get(f"roster:{eid}:{home}"))
        ap, ast = _roster_sets(cache.get(f"roster:{eid}:{away}"))
        games.append({
            "event_id": eid, "date": comp["date"], "home": str(home), "away": str(away),
            "home_rec": _record(home, hq, aq, hs, as_, comp["date"], True, hp, hst, cache.get(f"qposs:{eid}")),
            "away_rec": _record(away, aq, hq, as_, hs, comp["date"], False, ap, ast, cache.get(f"qposs:{eid}")),
        })
    games.sort(key=lambda g: g["date"])
    return games


# ---------------------------------------------------------------------------
# Ritmo por cuarto desde la jugada por jugada (ESPN summary)
# ---------------------------------------------------------------------------

def _play_flags(play):
    text = ((play.get("type") or {}).get("text") or "").lower()
    desc = (play.get("text") or "").lower()
    return text, desc


def quarter_possessions_from_plays(plays):
    """Posesiones por equipo en el 1Q y 2Q, con la formula estandar:
    tiros de campo intentados + 0.44 x tiros libres - rebotes ofensivos +
    perdidas, contando solo las jugadas de cada cuarto. Regresa
    {team_id: {"q1": x, "q2": y}} o {} si no se pudo."""
    acc = {}
    for play in plays or []:
        period = (play.get("period") or {}).get("number")
        if period not in (1, 2):
            continue
        team = str((play.get("team") or {}).get("id") or "")
        if not team:
            continue
        q = f"q{period}"
        a = acc.setdefault(team, {}).setdefault(q, {"fga": 0, "fta": 0, "oreb": 0, "tov": 0})
        text, desc = _play_flags(play)
        if "free throw" in text or "free throw" in desc:
            a["fta"] += 1
        elif play.get("shootingPlay"):
            a["fga"] += 1
        elif "offensive rebound" in text:
            a["oreb"] += 1
        elif "turnover" in text or "turnover" in desc or "traveling" in text or "bad pass" in text:
            a["tov"] += 1
    out = {}
    for team, qs in acc.items():
        out[team] = {q: round(v["fga"] + 0.44 * v["fta"] - v["oreb"] + v["tov"], 1) for q, v in qs.items()}
    # Sanidad: un cuarto tiene ~20-32 posesiones por equipo
    if not out or any(not (12 <= v <= 40) for qs in out.values() for v in qs.values()):
        return {}
    return out


def get_quarter_possessions(event_id):
    """Posesiones por cuarto de un juego: del cache, o se descarga una vez."""
    cache = n._load_summary_cache()
    key = f"qposs:{event_id}"
    if key in cache:
        return cache[key]
    try:
        data = n.get_json(f"{n.ESPN_BASE}/summary", params={"event": event_id})
        res = quarter_possessions_from_plays(data.get("plays"))
    except Exception:
        return {}
    cache[key] = res
    n._cache_dirty = True
    return res


def collect_quarter_pace(budget_min=40, verbose=True):
    """Descarga la jugada por jugada de los juegos que aun no la tengan y
    guarda SOLO las posesiones por cuarto (cache 'qposs:<id>'; la jugada
    completa pesa demasiado para guardarla). Con presupuesto de tiempo: lo
    que falte se completa en las siguientes noches."""
    import time
    start = time.time()
    cache = n._load_summary_cache()
    games = [k[5:] for k, v in cache.items()
             if k.startswith("comp:") and isinstance(v, dict) and v.get("completed")]
    todo = [eid for eid in games if f"qposs:{eid}" not in cache]
    todo.sort(reverse=True)  # primero los mas recientes (los que mas pesan al predecir)
    done = fails = 0
    for eid in todo:
        if time.time() - start > budget_min * 60:
            break
        try:
            data = n.get_json(f"{n.ESPN_BASE}/summary", params={"event": eid})
            res = quarter_possessions_from_plays(data.get("plays"))
        except Exception as e:
            fails += 1
            if verbose and fails <= 5:
                print(f"  [salteado] jugada por jugada {eid}: {str(e)[:100]}", file=sys.stderr)
            continue
        cache[f"qposs:{eid}"] = res  # {} = sin datos utiles: no se vuelve a pedir
        n._cache_dirty = True
        done += 1
        if done % 100 == 0:
            n.flush_cache()
            if verbose:
                print(f"  ritmo por cuarto: {done} juegos nuevos ({time.time() - start:.0f}s)", file=sys.stderr)
        time.sleep(0.3)
    n.flush_cache()
    have = sum(1 for eid in games if cache.get(f"qposs:{eid}"))
    print(f"Ritmo por cuarto: {done} juegos nuevos, {have}/{len(games)} juegos con datos "
          f"({len(todo) - done} pendientes)")
    return have, len(games)


# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

def _avg(vals):
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


def _days_between(d1, d2):
    a = datetime.fromisoformat(d1.replace("Z", "+00:00"))
    b = datetime.fromisoformat(d2.replace("Z", "+00:00"))
    return (b.astimezone(ET).date() - a.astimezone(ET).date()).days


def side_features(history, game_date, starters_out=None):
    """Variables de UN equipo a partir de sus juegos previos (mas reciente al
    final). starters_out: si es None se calcula del roster del juego (solo
    entrenamiento); en vivo viene del reporte de lesionados."""
    prior = history[-WINDOW:]
    short = history[-SHORT:]
    f = {"n_games": len(prior)}
    for k in ("q1_for", "q1_against", "q2_for", "q2_against", "pts_for", "pts_against",
              "pace", "off_rtg", "def_rtg", "efg_pct", "tov_pct",
              "q1_poss", "q1_ppp_for", "q1_ppp_against", "q2_poss", "q2_ppp_for", "q2_ppp_against"):
        f[k] = _avg([g.get(k) for g in prior])
    for k in ("q1_for", "q1_against", "q2_for", "q2_against"):
        f[f"{k}_l5"] = _avg([g[k] for g in short])
    rest = _days_between(history[-1]["date"], game_date) if history else None
    f["days_rest"] = min(rest, MAX_REST) if rest is not None else None
    f["b2b"] = int(rest == 1) if rest is not None else 0
    f["starters_out"] = starters_out
    return f


def regular_starters(history, n_last=5, min_starts=3):
    """Titulares habituales: arrancaron al menos 3 de sus ultimos 5 juegos."""
    counts = {}
    for g in history[-n_last:]:
        for pid in g.get("started") or ():
            counts[pid] = counts.get(pid, 0) + 1
    return {pid for pid, c in counts.items() if c >= min_starts}


def game_row(home_f, away_f, start_iso, share):
    row = {}
    for side, f in (("home", home_f), ("away", away_f)):
        for k in SIDE_FEATS:
            row[f"{side}_{k}"] = f.get(k)
    start = datetime.fromisoformat(start_iso.replace("Z", "+00:00")).astimezone(ET) if start_iso and "T" in start_iso else None
    row["start_hour_et"] = start.hour + start.minute / 60 if start else None
    row["matinee"] = int(start.hour < 16) if start else 0
    for q in QUARTERS:
        hf, ha = home_f.get(f"{q}_for"), home_f.get(f"{q}_against")
        af, aa = away_f.get(f"{q}_for"), away_f.get(f"{q}_against")
        row[f"base_{q}_matchup"] = (_avg([hf, aa]) + _avg([af, ha])) if None not in (hf, ha, af, aa) else None
    # Proyeccion por ritmo: posesiones esperadas del cuarto x puntos por
    # posesion (ofensiva propia vs defensa rival) de cada equipo.
    for q in QUARTERS:
        vals = [home_f.get(f"{q}_poss"), away_f.get(f"{q}_poss"), home_f.get(f"{q}_ppp_for"),
                away_f.get(f"{q}_ppp_against"), away_f.get(f"{q}_ppp_for"), home_f.get(f"{q}_ppp_against")]
        if None in vals:
            row[f"base_{q}_pace"] = None
        else:
            poss = _avg(vals[:2])
            row[f"base_{q}_pace"] = poss * (_avg(vals[2:4]) + _avg(vals[4:6]))
    hpf, hpa, apf, apa = (home_f.get("pts_for"), home_f.get("pts_against"),
                          away_f.get("pts_for"), away_f.get("pts_against"))
    full = (_avg([hpf, apa]) + _avg([apf, hpa])) if None not in (hpf, hpa, apf, apa) else None
    row["base_full_total"] = full
    for q in QUARTERS:
        row[f"base_{q}_casa"] = full * share[q] if full is not None else None
    return row


def build_dataset(games=None):
    """Una fila por juego (con al menos MIN_PRIOR juegos previos de ambos
    equipos), con variables y resultado real del 1Q y 2Q."""
    games = games if games is not None else load_cached_games()
    hist = {}
    pending = []
    for g in games:
        hh, ah = hist.setdefault(g["home"], []), hist.setdefault(g["away"], [])
        if len(hh) >= MIN_PRIOR and len(ah) >= MIN_PRIOR:
            outs = []
            for h, rec in ((hh, g["home_rec"]), (ah, g["away_rec"])):
                regs = regular_starters(h)
                outs.append(len(regs - rec["played"]) if rec["played"] is not None and regs else None)
            pending.append((g, side_features(hh, g["date"], outs[0]), side_features(ah, g["date"], outs[1])))
        hh.append(g["home_rec"])
        ah.append(g["away_rec"])
    # Fraccion real del total que cae en cada cuarto (para la linea estilo casa)
    q_share = {q: float(np.mean([(g["home_rec"][f"{q}_for"] + g["away_rec"][f"{q}_for"]) /
                                 (g["home_rec"]["pts_for"] + g["away_rec"]["pts_for"]) for g in games]))
               for q in QUARTERS} if games else DEFAULT_SHARE
    rows = []
    for g, hf, af in pending:
        row = game_row(hf, af, g["date"], q_share)
        row.update({"event_id": g["event_id"], "date": g["date"], "home": g["home"], "away": g["away"],
                    "label_q1": g["home_rec"]["q1_for"] + g["away_rec"]["q1_for"],
                    "label_q2": g["home_rec"]["q2_for"] + g["away_rec"]["q2_for"]})
        rows.append(row)
    return pd.DataFrame(rows), q_share


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------

class ColumnBaseline:
    """'Modelo' que regresa una columna tal cual (una de las lineas base)."""

    def __init__(self, col):
        self.col = col

    def fit(self, X, y):
        return self

    def predict(self, X):
        v = X[self.col].astype(float)
        if self.col.endswith("_pace"):  # juegos sin jugada por jugada -> promedio del cuarto
            v = v.fillna(X[self.col.replace("_pace", "_matchup")].astype(float))
        return v.values


def train_quarter(df, q, share, test_frac=0.2):
    label = f"label_{q}"
    d = df.dropna(subset=[label, f"base_{q}_casa", f"base_{q}_matchup"]).sort_values("date").reset_index(drop=True)
    cut = int(len(d) * (1 - test_frac))
    train, test = d.iloc[:cut], d.iloc[cut:]
    X_tr, y_tr, X_te, y_te = train[FEATURES], train[label], test[FEATURES], test[label]

    candidates = {
        "Linea estilo casa (fraccion del total)": ColumnBaseline(f"base_{q}_casa"),
        "Promedio del cuarto (anotados/permitidos)": ColumnBaseline(f"base_{q}_matchup"),
        "Ritmo del cuarto x puntos por posesion": ColumnBaseline(f"base_{q}_pace"),
        "Regresion lineal": Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler()),
                                      ("reg", LinearRegression())]),
        "Gradient Boosting": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                                            random_state=0),
    }
    results, preds = [], {}
    for name, m in candidates.items():
        m.fit(X_tr, y_tr)
        p = m.predict(X_te)
        preds[name] = p
        results.append({"modelo": name, "mae": round(mean_absolute_error(y_te, p), 3)})
    rdf = pd.DataFrame(results).sort_values("mae")
    print(f"\n--- Total del {LABELS[q]}: {len(train)} juegos de entrenamiento, {len(test)} de prueba ---")
    print(rdf.to_string(index=False))
    best = rdf.iloc[0]["modelo"]
    # El cuarto es muy ruidoso: un modelo "aprendido" solo se usa si le gana
    # CLARAMENTE (0.05+ pts) al promedio del cuarto; si no, el promedio, que
    # es mas estable (en pruebas el boosting gano en un corte y perdio en otro).
    simple = "Promedio del cuarto (anotados/permitidos)"
    mae = dict(zip(rdf["modelo"], rdf["mae"]))
    if best not in (simple, "Linea estilo casa (fraccion del total)") and mae[best] > mae[simple] - 0.05:
        best = simple
    best_pred = preds[best]
    print(f"Mejor: {best}")

    # Over/under contra la linea estilo casa (lo que hacen las casas), en la
    # segunda mitad de la prueba, con la curva de error de la primera mitad.
    line_ref = np.floor(test[f"base_{q}_casa"].values) + 0.5
    ou = ptr.evaluate_over_under(best_pred[: len(test) // 2], y_te.values[: len(test) // 2],
                                 best_pred[len(test) // 2:], y_te.values[len(test) // 2:],
                                 line_ref[len(test) // 2:])
    print(f"Over/under vs linea estilo casa: acierto {ou['acierto_over_under']:.1%}; con confianza >=60%: "
          f"{ou['acierto_si_confianza_60']} en {ou['pct_jugadas_confianza_60']:.0%} de los juegos")
    sigma_curve = ptr.fit_sigma_curve(best_pred, y_te.values)

    model = candidates[best]
    bias = 0.0
    if not isinstance(model, ColumnBaseline):
        model = clone(model).fit(d[FEATURES], d[label])
        bias = ptr.recent_bias(d, FEATURES, label, model) if len(d) > 400 else 0.0
        print(f"Modelo final con los {len(d)} juegos; correccion de sesgo reciente {bias:+.2f}")
    bundle = {"model": model, "features": FEATURES, "model_name": best, "sigma_curve": sigma_curve,
              "bias": bias, "share": share, "trained_at": datetime.now().isoformat(timespec="seconds")}
    metrics = {"modelos": results, "over_under": ou, "n_train": len(train), "n_test": len(test)}
    return bundle, metrics


def train_all(out_dir=APP_DIR):
    df, share = build_dataset()
    if df.empty:
        print("ERROR: no hay juegos con marcador por cuarto en el cache.", file=sys.stderr)
        sys.exit(1)
    print(f"Dataset de cuartos: {len(df)} juegos ({df['date'].min()[:10]} a {df['date'].max()[:10]}); "
          f"fraccion del total: 1Q {share['q1']:.1%}, 2Q {share['q2']:.1%}")
    hist_row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "n_games": len(df)}
    for q in QUARTERS:
        bundle, metrics = train_quarter(df, q, share)
        path = os.path.join(out_dir, f"model_quarter_{q}.joblib")
        joblib.dump(bundle, path)
        with open(path + ".metrics.json", "w") as f:
            json.dump(metrics, f, indent=2, default=float)
        best = min(metrics["modelos"], key=lambda x: x["mae"])
        casa = next(x for x in metrics["modelos"] if x["modelo"].startswith("Linea estilo casa"))
        hist_row.update({f"modelo_{q}": best["modelo"], f"mae_{q}": best["mae"], f"mae_{q}_casa": casa["mae"],
                         f"acierto_ou_{q}": metrics["over_under"]["acierto_over_under"]})
    hp = os.path.join(out_dir, "training_history_quarters.csv")
    new = pd.DataFrame([hist_row])
    if os.path.exists(hp):
        new = pd.concat([pd.read_csv(hp), new], ignore_index=True)
    new.to_csv(hp, index=False)


# ---------------------------------------------------------------------------
# Prediccion en vivo
# ---------------------------------------------------------------------------

def load_bundles(model_dir=APP_DIR):
    out = {}
    for q in QUARTERS:
        p = os.path.join(model_dir, f"model_quarter_{q}.joblib")
        if os.path.exists(p):
            out[q] = joblib.load(p)
    return out


def live_history(team_espn_id, season=None):
    """Ultimos juegos del equipo con marcador por cuarto, en el mismo formato
    del entrenamiento (usa el cache; solo descarga lo que falte)."""
    hist = []
    for g in n.espn_recent_games(team_espn_id, season=season, last_n=WINDOW):
        eid, opp = g["id"], g["opponent_espn_id"]
        try:
            my_stats, my_q = n.team_game_boxscore(eid, team_espn_id)
            opp_stats, opp_q = n.team_game_boxscore(eid, opp)
        except Exception:
            continue
        if not my_q or not opp_q or len(my_q) < 4 or len(opp_q) < 4:
            continue
        try:
            played, started = _roster_sets(n.espn_game_roster(eid, team_espn_id))
        except Exception:
            played, started = None, None
        rec = _record(team_espn_id, my_q, opp_q, my_stats, opp_stats, g.get("date") or "", None, played, started,
                      get_quarter_possessions(eid))
        try:
            rec["names"] = {str(e["player_id"]): e["name"] for e in n.espn_game_roster(eid, team_espn_id)}
        except Exception:
            rec["names"] = {}
        hist.append(rec)
    hist.sort(key=lambda x: x["date"])
    return hist


OUT_WORDS = ("out", "doubtful", "susp")


def starters_out_today(history, team_espn_id):
    """(cuantos, nombres) de titulares habituales reportados Out/Doubtful hoy."""
    regs = regular_starters(history)
    names = {}
    for g in history:
        names.update(g.get("names") or {})
    try:
        injuries, _ = n.espn_team_injuries(team_espn_id)
    except Exception:
        return None, []
    out = []
    for pid in regs:
        nm = (names.get(pid) or "").lower()
        if not nm:
            continue
        for inj in injuries:
            if inj["name"].lower() == nm and any(w in inj["status"].lower() for w in OUT_WORDS):
                out.append(names[pid])
    return len(out), out


def predict_game(bundles, home, away, start_iso, season=None):
    """Prediccion del total del 1Q y 2Q de un juego. home/away: dicts de
    equipo (n.TEAMS). Incluye titulares habituales fuera hoy."""
    hh = live_history(home["espn_id"], season)
    ah = live_history(away["espn_id"], season)
    if len(hh) < 3 or len(ah) < 3:
        return None
    home_out, home_out_names = starters_out_today(hh, home["espn_id"])
    away_out, away_out_names = starters_out_today(ah, away["espn_id"])
    share = next(iter(bundles.values())).get("share", DEFAULT_SHARE)
    start = start_iso if start_iso and "T" in start_iso else datetime.now(ET).isoformat()
    hf = side_features(hh, start, home_out)
    af = side_features(ah, start, away_out)
    row = game_row(hf, af, start_iso, share)
    out = {"row": row, "home_f": hf, "away_f": af,
           "home_out_names": home_out_names, "away_out_names": away_out_names}
    for q, b in bundles.items():
        X = pd.DataFrame([row]).reindex(columns=b["features"] + ([b["model"].col]
                                                                 if isinstance(b["model"], ColumnBaseline)
                                                                 and b["model"].col not in b["features"] else []))
        out[q] = float(b["model"].predict(X)[0]) + float(b.get("bias") or 0)
        out[f"{q}_casa"] = row.get(f"base_{q}_casa")
        # Reparto del total entre los dos equipos: lo que cada uno anota en
        # ese cuarto contra lo que permite el rival (promedio de ambos lados),
        # escalado para que sume la prediccion del total.
        h_exp = _avg([hf.get(f"{q}_for"), af.get(f"{q}_against")])
        a_exp = _avg([af.get(f"{q}_for"), hf.get(f"{q}_against")])
        if h_exp and a_exp:
            out[f"{q}_home"] = out[q] * h_exp / (h_exp + a_exp)
            out[f"{q}_away"] = out[q] * a_exp / (h_exp + a_exp)
    return out


def over_under(bundle, pred, line, odds=-110):
    curve = bundle.get("sigma_curve")
    if not curve or line is None:
        return None
    p_over = ptr.prob_over(pred, line, curve)
    o = float(odds)
    need = 100 / (o + 100) if o > 0 else -o / (-o + 100)
    side = "OVER" if p_over >= 0.5 else "UNDER"
    p = max(p_over, 1 - p_over)
    return {"p_over": p_over, "p_under": 1 - p_over, "pick": side, "p_pick": p, "p_needed": need,
            "edge": p - need}


def main():
    parser = argparse.ArgumentParser(description="Modelo de totales del 1er y 2do cuarto")
    parser.add_argument("--train", action="store_true", help="Entrena con los datos del cache")
    parser.add_argument("--collect-pace", action="store_true",
                         help="Descarga la jugada por jugada pendiente (ritmo por cuarto)")
    parser.add_argument("--budget-min", type=float, default=40)
    args = parser.parse_args()
    if args.collect_pace:
        collect_quarter_pace(args.budget_min)
    if args.train:
        train_all()
    if not (args.train or args.collect_pace):
        parser.print_help()
        return


if __name__ == "__main__":
    import nba_quarters
    nba_quarters.main()
