"""
Interfaz web (Streamlit) para el sistema de apuestas NBA - reemplaza a los
.bat: se corre desde cualquier navegador (incluido el del celular) una vez
publicada en Streamlit Community Cloud.

No reimplementa el analisis: llama exactamente a las mismas funciones que
ya usan nba_report.py / nba_predict.py / nba_props_predict.py /
nba_track.py. Este archivo solo se encarga de la parte visual (formularios,
tarjetas, tablas) y de que el reentrenamiento se guarde de forma permanente
en GitHub (ver git_sync.py) - mismo patron que apuestas_mlb/streamlit_app.py.

Correr localmente para probar:
    streamlit run streamlit_app.py
"""

import os
import statistics
import subprocess
import sys
from contextlib import contextmanager
from datetime import date

import joblib
import pandas as pd
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import git_sync
import nba_data as n
import nba_props_predict as pp
import nba_report as r
import nba_track as track

st.set_page_config(page_title="NBA Apuestas", page_icon="🏀", layout="wide")


# ---------------------------------------------------------------------------
# Utilidades compartidas (mismo patron que apuestas_mlb/streamlit_app.py)
# ---------------------------------------------------------------------------

class _LiveLog:
    def __init__(self, placeholder, max_lines=300):
        self.placeholder = placeholder
        self.lines = []
        self.max_lines = max_lines

    def write(self, s):
        for part in s.splitlines():
            if part.strip():
                self.lines.append(part)
        self.placeholder.code("\n".join(self.lines[-self.max_lines:]) or " ")

    def flush(self):
        pass


@contextmanager
def live_log(placeholder):
    log = _LiveLog(placeholder)
    old_out, old_err = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = log, log
    try:
        yield log
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def check_password():
    """Si se configuro APP_PASSWORD en Secrets, pide un PIN antes de mostrar
    la app. Si no se configuro, no bloquea nada (util para probar local)."""
    if "APP_PASSWORD" not in st.secrets:
        return True
    if st.session_state.get("_authed"):
        return True
    st.title("🏀 NBA Apuestas")
    with st.form("login"):
        pw = st.text_input("PIN de acceso", type="password")
        submitted = st.form_submit_button("Entrar")
    if submitted:
        if pw == st.secrets["APP_PASSWORD"]:
            st.session_state["_authed"] = True
            st.rerun()
        else:
            st.error("PIN incorrecto.")
    return False


@st.cache_resource
def _load_bundle_cached(path, mtime):
    return joblib.load(path)


def load_bundle(path):
    """Carga un modelo .joblib, cacheado por ruta + fecha de modificacion -
    si se reentrena y el archivo cambia, se vuelve a cargar solo."""
    if not os.path.exists(path):
        return None
    return _load_bundle_cached(path, os.path.getmtime(path))


def _stream_subprocess(cmd, log_placeholder):
    process = subprocess.Popen(
        cmd, cwd=APP_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1, encoding="utf-8", errors="replace",
    )
    lines = []
    for line in process.stdout:
        lines.append(line.rstrip("\n"))
        log_placeholder.code("\n".join(lines[-300:]))
    process.wait()
    return process.returncode


def pct(x):
    return "N/D" if x is None else f"{x * 100:.1f}%"


def num(x, nd=1):
    return "N/D" if x is None else f"{x:.{nd}f}"


# ---------------------------------------------------------------------------
# Reporte del juego (reemplaza correr_reporte.bat + correr_prediccion.bat)
# ---------------------------------------------------------------------------

def render_team_block(rep, projection, quarters, col):
    t = rep["team"]
    with col:
        st.markdown(f"#### {t['full_name']} ({t['abbreviation']})")
        adv = rep["advanced"]
        if not adv:
            st.warning("Sin datos suficientes para este equipo en la temporada elegida.")
            return
        st.caption(f"{rep['n_games']} juegos en la ventana")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("OffRtg", num(adv["off_rtg"]))
        c2.metric("DefRtg", num(adv["def_rtg"]))
        c3.metric("Net", num(adv["net_rtg"]))
        c4.metric("Pace", num(adv["pace"]))
        c1.metric("eFG%", pct(adv["efg_pct"]))
        c2.metric("TS%", pct(adv["ts_pct"]))
        c3.metric("TOV%", pct(adv["tov_pct"]))
        c4.metric("AST%", pct(adv["ast_pct"]))
        c1.metric("OREB%", pct(adv["oreb_pct"]))
        c2.metric("DREB%", pct(adv["dreb_pct"]))
        c3.metric("Contraataque", num(adv["fast_break_pts"]))
        c4.metric("Pintura", num(adv["points_in_paint"]))

        if rep.get("days_rest") is not None:
            alerta = " ⚠️ BACK-TO-BACK" if rep.get("is_b2b") else ""
            st.caption(f"Descanso: {rep['days_rest']} dia(s){alerta}")

        if rep.get("injuries_auto"):
            st.caption("Lesionados (ESPN): " + ", ".join(rep["injuries_auto"][:6]))
        if rep.get("injury_impact"):
            st.caption(f"LESIONADO forzado: {rep['injury_impact']['player']} - "
                       f"impacto On/Off: {rep['injury_impact']['raw']}")

        if projection:
            st.metric("Puntos proyectados (formula Fase 1)", num(projection["projected_points"]),
                      help=f"Pace combinado: {num(projection['pace'])}")
        if quarters:
            st.caption(f"Q1: {num(quarters['q1'])}  Q2: {num(quarters['q2'])}  "
                       f"Q3: {num(quarters['q3'])}  Q4: {num(quarters['q4'])}  "
                       f"(muestra: {quarters['n_games_muestra']} juegos)")


def build_full_report_nba(equipo_a, equipo_b, season, last_n, lesionado_a, lesionado_b, local_practica):
    team_a = n.resolve_team(equipo_a)
    team_b = n.resolve_team(equipo_b)
    if team_a["nba_id"] == team_b["nba_id"]:
        raise ValueError(f"'{equipo_a}' y '{equipo_b}' son el mismo equipo.")

    try:
        matchup = n.find_next_matchup(team_a, team_b)
    except Exception:
        matchup = None

    if matchup:
        home_is_a = matchup["home_espn_id"] == team_a["espn_id"]
        team_home, team_away = (team_a, team_b) if home_is_a else (team_b, team_a)
        as_of_date = matchup["date"]
    else:
        team_home, team_away = (team_a, team_b) if local_practica == "A" else (team_b, team_a)
        as_of_date = None

    try:
        rep_a = r.team_side_report(team_a, team_b, season, last_n, lesionado_override=lesionado_a or None,
                                    as_of_date=as_of_date)
        rep_b = r.team_side_report(team_b, team_a, season, last_n, lesionado_override=lesionado_b or None,
                                    as_of_date=as_of_date)
    finally:
        n.flush_cache()

    proj_a = r.project_points(rep_a, rep_b)
    proj_b = r.project_points(rep_b, rep_a)
    q_a = r.quarter_breakdown(proj_a["projected_points"], rep_a["quarter_profile"]) if proj_a else None
    q_b = r.quarter_breakdown(proj_b["projected_points"], rep_b["quarter_profile"]) if proj_b else None

    rep_home, rep_away = (rep_a, rep_b) if team_home is team_a else (rep_b, rep_a)
    ml_row = None
    if rep_home["advanced"] and rep_away["advanced"]:
        ml_row = {}
        for k in rep_home["advanced"].keys():
            ml_row[f"home_{k}"] = rep_home["advanced"].get(k)
            ml_row[f"away_{k}"] = rep_away["advanced"].get(k)
        ml_row["home_days_rest"] = rep_home.get("days_rest")
        ml_row["away_days_rest"] = rep_away.get("days_rest")
        ml_row["home_b2b"] = int(bool(rep_home.get("is_b2b")))
        ml_row["away_b2b"] = int(bool(rep_away.get("is_b2b")))

    return {
        "team_a": team_a, "team_b": team_b, "team_home": team_home, "team_away": team_away,
        "matchup": matchup, "rep_a": rep_a, "rep_b": rep_b, "proj_a": proj_a, "proj_b": proj_b,
        "q_a": q_a, "q_b": q_b, "ml_row": ml_row, "season": season,
    }


def render_nba_report(team_a, team_b, team_home, team_away, matchup, rep_a, rep_b, proj_a, proj_b,
                       q_a, q_b, ml_row, season):
    if matchup:
        st.success(f"Proximo juego real: {matchup['date']} - Local: {team_home['full_name']}")
    else:
        st.info(f"No hay un juego programado entre estos dos equipos en los proximos 14 dias - "
                f"modo practica (local asumido: {team_home['full_name']}).")

    c1, c2 = st.columns(2)
    render_team_block(rep_a, proj_a, q_a, c1)
    render_team_block(rep_b, proj_b, q_b, c2)

    if proj_a and proj_b:
        st.subheader("Marcador proyectado (formula Fase 1)")
        total = proj_a["projected_points"] + proj_b["projected_points"]
        favorito = team_a if proj_a["projected_points"] > proj_b["projected_points"] else team_b
        margen = abs(proj_a["projected_points"] - proj_b["projected_points"])
        st.write(f"**{team_a['full_name']}**: {num(proj_a['projected_points'])}  |  "
                 f"**{team_b['full_name']}**: {num(proj_b['projected_points'])}  |  "
                 f"Total: {num(total)}  |  Favorito (formula): {favorito['full_name']} por {num(margen)}")

    st.subheader("Prediccion con modelo entrenado (Fase 2)")
    bundle_total = load_bundle(os.path.join(APP_DIR, "model_nba_total.joblib"))
    bundle_margin = load_bundle(os.path.join(APP_DIR, "model_nba_margin.joblib"))
    bundle_win = load_bundle(os.path.join(APP_DIR, "model_nba_win.joblib"))
    if not (bundle_total and bundle_margin and bundle_win):
        st.warning("Todavia no hay modelos entrenados de equipo - ve a 'Reentrenar modelos'.")
        return
    if not ml_row:
        st.warning("No hay suficientes datos recientes de alguno de los dos equipos para predecir con el modelo.")
        return

    def _predict(bundle, row):
        X = pd.DataFrame([row])[bundle["features"]]
        model = bundle["model"]
        return model.predict_proba(X)[0, 1] if hasattr(model, "predict_proba") else model.predict(X)[0]

    pred_total = _predict(bundle_total, ml_row)
    pred_margin = _predict(bundle_margin, ml_row)
    prob_home_win = _predict(bundle_win, ml_row)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Puntos totales ({bundle_total['model_name']})", num(pred_total))
    c2.metric(f"Margen local, {team_home['abbreviation']}-{team_away['abbreviation']} "
              f"({bundle_margin['model_name']})", f"{pred_margin:+.1f}")
    favorito_ml = team_home if prob_home_win >= 0.5 else team_away
    prob_fav = prob_home_win if prob_home_win >= 0.5 else 1 - prob_home_win
    c3.metric(f"Money line ({bundle_win['model_name']})", f"{favorito_ml['abbreviation']} {prob_fav*100:.1f}%",
              help=f"Probabilidad de que gane el local ({team_home['full_name']}): {prob_home_win*100:.1f}%")

    if matchup and st.session_state.get("nba_log_this", True):
        track.log_prediction({
            "logged_date": date.today().isoformat(), "event_id": matchup["event_id"],
            "game_date": matchup["date"], "home_team": team_home["abbreviation"],
            "away_team": team_away["abbreviation"],
            "home_espn_id": team_home["espn_id"], "away_espn_id": team_away["espn_id"],
            "pred_total": pred_total, "model_total": bundle_total["model_name"], "baseline_total": None,
            "pred_margin": pred_margin, "model_margin": bundle_margin["model_name"], "baseline_margin": None,
            "pred_prob_home_win": prob_home_win, "model_win": bundle_win["model_name"],
            "actual_home_points": None, "actual_away_points": None, "actual_total": None,
            "actual_margin": None, "actual_home_win": None, "evaluated": False,
        })
        st.caption("Prediccion guardada en predictions_log_nba.csv para evaluar despues del juego.")


def render_reporte_tab():
    st.header("📋 Reporte del juego")
    st.caption("Metricas avanzadas de ventana movil, proyeccion por formula (Fase 1) y prediccion "
               "con modelo entrenado (Fase 2: puntos totales, spread, money line). 1-2 min.")

    team_names = sorted(t["full_name"] for t in n.TEAMS)
    with st.form("form_reporte_nba"):
        c1, c2 = st.columns(2)
        equipo_a = c1.selectbox("Equipo A", team_names, index=0)
        equipo_b = c2.selectbox("Equipo B", team_names, index=min(1, len(team_names) - 1))
        with st.expander("Opciones avanzadas"):
            season = st.number_input("Temporada (anio en que TERMINA)", value=n.current_nba_season(), step=1)
            last_n = st.number_input("Ventana movil (juegos)", value=r.DEFAULT_LAST_N, step=1, min_value=3)
            lesionado_a = st.text_input("Forzar lesionado Equipo A (impacto On/Off)")
            lesionado_b = st.text_input("Forzar lesionado Equipo B (impacto On/Off)")
            local_practica = st.radio("Si NO hay juego real programado, quien es local:", ["A", "B"],
                                       horizontal=True)
            st.session_state["nba_log_this"] = st.checkbox(
                "Guardar esta prediccion en el historial de seguimiento (solo si hay juego real)", value=True)
        submitted = st.form_submit_button("Generar reporte", type="primary", use_container_width=True)

    if submitted:
        with st.spinner("Calculando metricas avanzadas de ambos equipos..."):
            try:
                result = build_full_report_nba(equipo_a, equipo_b, int(season), int(last_n),
                                                lesionado_a, lesionado_b, local_practica)
                st.session_state["nba_report_result"] = result
                st.session_state["nba_report_error"] = None
            except Exception as e:
                st.session_state["nba_report_result"] = None
                st.session_state["nba_report_error"] = f"{type(e).__name__}: {e}"

    if st.session_state.get("nba_report_error"):
        st.error(st.session_state["nba_report_error"])
    if st.session_state.get("nba_report_result"):
        render_nba_report(**st.session_state["nba_report_result"])


# ---------------------------------------------------------------------------
# Props de jugador (reemplaza correr_prediccion_props.bat)
# ---------------------------------------------------------------------------

def render_props_tab():
    st.header("🏀 Props de jugador")
    st.caption("Puntos, rebotes y asistencias de un jugador especifico en su proximo juego real, "
               "comparado contra su propio promedio y contra la mediana de sus ultimos juegos "
               "(no se usa una linea real de casa de apuestas - ninguna fuente publica la tiene).")

    team_names = sorted(t["full_name"] for t in n.TEAMS)
    with st.form("form_props"):
        jugador = st.text_input("Jugador (nombre o parte del nombre)")
        c1, c2 = st.columns(2)
        equipo_jugador = c1.selectbox("Equipo del jugador", team_names, index=0)
        equipo_rival = c2.selectbox("Equipo rival", team_names, index=min(1, len(team_names) - 1))
        c3, c4 = st.columns(2)
        season = c3.number_input("Temporada (anio en que TERMINA)", value=n.current_nba_season(), step=1,
                                  key="props_season")
        window = c4.number_input("Ventana movil (juegos)", value=pp.WINDOW, step=1, min_value=3,
                                  key="props_window")
        submitted = st.form_submit_button("Predecir", type="primary", use_container_width=True)

    if not submitted:
        return

    team = n.resolve_team(equipo_jugador)
    opponent = n.resolve_team(equipo_rival)

    player_id, player_name = n.resolve_player_espn(team["espn_id"], jugador, season=int(season))
    if not player_id:
        st.error(f"No encontre a '{jugador}' en el roster de {team['full_name']} (temporada {int(season)}).")
        return

    bundle_points = load_bundle(os.path.join(APP_DIR, "model_props_points.joblib"))
    bundle_rebounds = load_bundle(os.path.join(APP_DIR, "model_props_rebounds.joblib"))
    bundle_assists = load_bundle(os.path.join(APP_DIR, "model_props_assists.joblib"))
    if not (bundle_points and bundle_rebounds and bundle_assists):
        st.warning("Todavia no hay modelos de props entrenados - ve a 'Reentrenar modelos'.")
        return

    with st.spinner(f"Calculando historial reciente de {player_name}..."):
        try:
            recent = pp.player_recent_games(team["espn_id"], player_id, int(season), window=int(window))
            rep_team = r.team_side_report(team, opponent, int(season), int(window))
            rep_opp = r.team_side_report(opponent, team, int(season), int(window))
        finally:
            n.flush_cache()

    if len(recent) < 3:
        st.warning(f"Muy pocos juegos recientes registrados ({len(recent)}) para predecir con confianza.")
        return
    if not rep_team["advanced"] or not rep_opp["advanced"]:
        st.warning("No hay suficientes datos de equipo recientes para calcular el contexto del rival.")
        return

    row = {
        "avg_points": r._avg([g["points"] for g in recent]),
        "avg_rebounds": r._avg([g["rebounds"] for g in recent]),
        "avg_assists": r._avg([g["assists"] for g in recent]),
        "avg_minutes": r._avg([g["minutes"] for g in recent]),
        "team_off_rtg": rep_team["advanced"]["off_rtg"], "team_pace": rep_team["advanced"]["pace"],
        "opp_def_rtg": rep_opp["advanced"]["def_rtg"], "opp_pace": rep_opp["advanced"]["pace"],
    }

    st.subheader(f"{player_name} ({team['full_name']}) vs {opponent['full_name']}")
    game_summaries = [f"{g['points']:.0f}p/{g['rebounds']:.0f}r/{g['assists']:.0f}a" for g in recent]
    st.caption(f"Ultimos {len(recent)} juegos: " + ", ".join(game_summaries))

    cols = st.columns(3)
    for col, stat, bundle, label in [
        (cols[0], "points", bundle_points, "Puntos"),
        (cols[1], "rebounds", bundle_rebounds, "Rebotes"),
        (cols[2], "assists", bundle_assists, "Asistencias"),
    ]:
        X = pd.DataFrame([row])[bundle["features"]]
        predicho = bundle["model"].predict(X)[0]
        mediana = statistics.median(g[stat] for g in recent)
        over_under = "OVER" if predicho > mediana else "UNDER" if predicho < mediana else "IGUAL"
        with col:
            st.metric(f"{label} ({bundle['model_name']})", num(predicho),
                      delta=f"{predicho - row[f'avg_{stat}']:+.1f} vs. promedio propio")
            st.caption(f"Mediana ultimos {len(recent)}: {mediana:.1f} → {over_under}")

    st.caption("NOTA: sin linea real de casa de apuestas - la 'mediana reciente' es un umbral "
               "calculado de los propios datos del jugador. Ajusta con la linea real que te ofrezca "
               "tu casa de apuestas.")


# ---------------------------------------------------------------------------
# Evaluar predicciones (reemplaza evaluar_predicciones.bat)
# ---------------------------------------------------------------------------

def render_evaluar():
    st.header("📈 Evaluar predicciones vs. resultado real")
    st.caption("Compara cada prediccion de equipo guardada en el historial contra lo que realmente "
               "paso en el juego, una vez que termino.")

    log_path = os.path.join(APP_DIR, "predictions_log_nba.csv")

    if st.button("Evaluar ahora", type="primary"):
        placeholder = st.empty()
        with live_log(placeholder):
            track.evaluate_log(log_path)
        ok, msg = git_sync.commit_and_push(
            ["predictions_log_nba.csv"], f"Evalua predicciones {date.today().isoformat()}", st.secrets, APP_DIR,
        )
        (st.success if ok else st.warning)(msg)

    if os.path.exists(log_path):
        st.subheader("Historial de predicciones")
        st.dataframe(pd.read_csv(log_path).tail(30), hide_index=True, use_container_width=True)
    else:
        st.info("Todavia no hay predicciones guardadas - genera un reporte con un juego real primero.")


# ---------------------------------------------------------------------------
# Reentrenar modelos (reemplaza reentrenar_modelos.bat / reentrenar_props.bat)
# ---------------------------------------------------------------------------

def _run_retrain_nba(skip_props):
    season = n.current_nba_season()

    with st.status("Paso 1/4 · Recolectando dataset historico de equipos (liga completa)...",
                    expanded=True) as status:
        log = st.empty()
        rc = _stream_subprocess(
            [sys.executable, "-u", "nba_train_data.py", "--season", str(season), "--out", "training_data_nba.csv"],
            log)
        if rc != 0:
            status.update(label="Paso 1/4 fallo", state="error")
            st.error("La recoleccion de equipos fallo (revisa el log). Se detiene el reentrenamiento.")
            return
        status.update(label="Paso 1/4 listo", state="complete")

    with st.status("Paso 2/4 · Reentrenando modelos de equipo (total/margen/money line)...",
                    expanded=True) as status:
        log = st.empty()
        rc = _stream_subprocess([sys.executable, "-u", "nba_train.py", "--data", "training_data_nba.csv"], log)
        status.update(label="Paso 2/4 listo" if rc == 0 else "Paso 2/4 con advertencias",
                       state="complete" if rc == 0 else "error")

    if skip_props:
        st.info("Paso 3-4/4 omitidos (solo equipo).")
    else:
        with st.status("Paso 3/4 · Recolectando dataset historico de props de jugador (liga completa)...",
                        expanded=True) as status:
            log = st.empty()
            rc = _stream_subprocess(
                [sys.executable, "-u", "nba_props_data.py", "--season", str(season),
                 "--out", "training_data_props.csv"], log)
            if rc != 0:
                status.update(label="Paso 3/4 fallo", state="error")
                st.warning("La recoleccion de props fallo - se guardan los modelos de equipo igual.")
            else:
                status.update(label="Paso 3/4 listo", state="complete")
                with st.status("Paso 4/4 · Reentrenando modelos de props...", expanded=True) as status2:
                    log2 = st.empty()
                    rc2 = _stream_subprocess(
                        [sys.executable, "-u", "nba_props_train.py", "--data", "training_data_props.csv"], log2)
                    status2.update(label="Paso 4/4 listo" if rc2 == 0 else "Paso 4/4 con advertencias",
                                    state="complete" if rc2 == 0 else "error")

    st.success("Reentrenamiento terminado. Los modelos nuevos ya se usan en esta app.")

    files_to_save = [
        "model_nba_total.joblib", "model_nba_total.joblib.metrics.json",
        "model_nba_margin.joblib", "model_nba_margin.joblib.metrics.json",
        "model_nba_win.joblib", "model_nba_win.joblib.metrics.json",
        "training_data_nba.csv", "training_history_nba.csv",
        "model_props_points.joblib", "model_props_points.joblib.metrics.json",
        "model_props_rebounds.joblib", "model_props_rebounds.joblib.metrics.json",
        "model_props_assists.joblib", "model_props_assists.joblib.metrics.json",
        "training_data_props.csv", "training_history_props.csv",
        "cache/game_summaries.json",
    ]
    ok, msg = git_sync.commit_and_push(
        files_to_save, f"Reentrenamiento automatico {date.today().isoformat()}", st.secrets, APP_DIR,
    )
    (st.success if ok else st.warning)(msg)


def render_reentrenar():
    st.header("🔁 Reentrenar modelos")
    st.success(
        "El reentrenamiento ya corre solo cada 3 dias en GitHub Actions (servidor de GitHub, no "
        "depende de esta pestana ni de tu conexion) - no hace falta que uses el boton de abajo a "
        "menos que quieras forzar un reentrenamiento ahora mismo. Revisa el progreso o dispáralo a "
        "mano en tu repo de GitHub → pestaña **Actions**."
    )
    with st.expander("Reentrenar manualmente desde aqui (no recomendado - ver de arriba)"):
        st.write(
            "Vuelve a entrenar los modelos con **todos** los juegos jugados hasta hoy. La PRIMERA vez "
            "(sin cache poblado) puede tardar **horas** porque recolecta la liga completa de ESPN con "
            "pausas deliberadas para no disparar su limite de solicitudes - necesita que esta pestaña "
            "se quede conectada TODO ese tiempo sin cortes. Por eso el reentrenamiento automatico de "
            "GitHub Actions (arriba) es la forma recomendada."
        )
        if not git_sync.is_configured(st.secrets):
            st.warning(
                "GITHUB_TOKEN / GITHUB_REPO no configurados en Secrets: los modelos reentrenados solo "
                "van a durar hasta que la app se reinicie o se duerma por inactividad."
            )
        skip_props = st.checkbox("Solo equipo (omitir props de jugador - mucho mas rapido)", value=False)
        confirmado = st.checkbox("Entiendo que esto puede tardar horas y no voy a cerrar la app mientras corre.")
        if st.button("Iniciar reentrenamiento", type="primary", disabled=not confirmado):
            _run_retrain_nba(skip_props)

    hist_path = os.path.join(APP_DIR, "training_history_nba.csv")
    if os.path.exists(hist_path):
        st.subheader("Historial de reentrenamientos (equipo)")
        st.dataframe(pd.read_csv(hist_path).tail(10), hide_index=True, use_container_width=True)
    hist_props_path = os.path.join(APP_DIR, "training_history_props.csv")
    if os.path.exists(hist_props_path):
        st.subheader("Historial de reentrenamientos (props)")
        st.dataframe(pd.read_csv(hist_props_path).tail(10), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Ajustes
# ---------------------------------------------------------------------------

def render_ajustes():
    st.header("⚙️ Ajustes")
    st.write("Estado de la configuracion (Settings → Secrets en Streamlit Cloud):")
    st.write(f"- GitHub (guardar reentrenamientos): {'✅ configurado' if git_sync.is_configured(st.secrets) else '❌ falta GITHUB_TOKEN / GITHUB_REPO'}")
    st.write(f"- PIN de acceso: {'✅ configurado' if 'APP_PASSWORD' in st.secrets else 'ℹ️ no configurado (app abierta)'}")


# ---------------------------------------------------------------------------
# Navegacion
# ---------------------------------------------------------------------------

def main():
    if not check_password():
        return

    st.sidebar.title("🏀 NBA Apuestas")
    section = st.sidebar.radio("Seccion", [
        "📋 Reporte del juego",
        "🏀 Props de jugador",
        "📈 Evaluar predicciones",
        "🔁 Reentrenar modelos",
        "⚙️ Ajustes",
    ])

    if section == "📋 Reporte del juego":
        render_reporte_tab()
    elif section == "🏀 Props de jugador":
        render_props_tab()
    elif section == "📈 Evaluar predicciones":
        render_evaluar()
    elif section == "🔁 Reentrenar modelos":
        render_reentrenar()
    elif section == "⚙️ Ajustes":
        render_ajustes()


if __name__ == "__main__":
    main()
