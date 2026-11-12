"""
Entrena modelos POOLED (uno solo por estadistica, compartido entre todos
los jugadores - no un modelo por jugador, no hay suficientes juegos por
jugador para eso) que predicen puntos/rebotes/asistencias de un jugador en
su PROXIMO juego, usando el dataset de nba_props_data.py.

Compara cada modelo contra el baseline mas obvio: el propio promedio movil
del jugador (avg_points/avg_rebounds/avg_assists) SIN usar el contexto de
equipo/rival - si el modelo no le gana a "lo que el jugador viene
promediando", no vale la pena.

Uso:
    python nba_props_train.py --data training_data_props.csv
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

import nba_props_data as pdata
import props_config as pc
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Contexto comun a todos los props: promedios moviles (10 y ultimos 3 juegos)
# de cada estadistica, rol (que tanto arranca), equipo/rival, descanso,
# local/visita, companeros ausentes, margen proyectado (palizas) y lo que la
# defensa rival permite por 36 min a titulares de su posicion.
FEATURES = (
    [pc.avg_feature(k) for k in list(pc.BASE_STATS) + list(pc.COMBOS)]
    + [f"{pc.avg_feature(k)}_last3" for k in list(pc.BASE_STATS) + list(pc.COMBOS)]
    # Prioridad a ESTA temporada (ver nba_props_data.season_features)
    + [f"{pc.avg_feature(k)}_season" for k in list(pc.BASE_STATS) + list(pc.COMBOS)]
    + [f"{pc.avg_feature(k)}_w" for k in list(pc.BASE_STATS) + list(pc.COMBOS)]
    + ["season_games", "start_rate", "team_off_rtg", "team_pace", "opp_def_rtg", "opp_pace",
       "days_rest", "b2b", "is_home", "teammates_out", "proj_margin", "proj_blowout"]
    # Produccion que dejan libre los companeros ausentes (total y de su posicion)
    + [f"vacated_{k}" for k in pdata.VACATED_STATS]
    + [f"vacated_{k}_same_pos" for k in pdata.VACATED_STATS]
    + [pc.allow_feature(k) for k in pc.ALLOWED_STATS]
    # Lado RIVAL: titulares/rotacion del rival que no juegan y su produccion
    # (total y de la misma posicion del jugador = su defensor natural)
    + ["opp_teammates_out"] + [f"opp_vacated_{k}" for k in pdata.VACATED_STATS]
    + [f"opp_vacated_{k}_same_pos" for k in pdata.VACATED_STATS]
    # Fatiga fina del equipo del jugador (nba_report.fatigue_features)
    + ["fat_three_in_four", "fat_games_last_7", "fat_ot_prev", "fat_tz_shift"]
)
BASE_FEATURES = list(FEATURES)

# MODELO DE MINUTOS (2 etapas): primero se predice cuantos minutos jugara y
# luego su produccion por minuto x esos minutos. Los minutos son lo que mas
# explica un prop (lesiones de companeros, back-to-back, riesgo de paliza).
MINUTES_MODEL_PATH = "model_props_minutes_proj.joblib"
MIN_DERIVED = ["proj_minutes"] + [f"proj_{k}_by_min" for k in pc.TARGETS]
FEATURES = FEATURES + MIN_DERIVED


def add_minutes_derived(df, proj_minutes):
    """Agrega proj_minutes y, por cada prop, su produccion por minuto
    reciente x los minutos proyectados. df puede ser de una sola fila."""
    df = df.copy()
    df["proj_minutes"] = proj_minutes
    mins = df[pc.avg_feature("minutes")].astype(float).replace(0, np.nan)
    for k in pc.TARGETS:
        col = pc.avg_feature(k)
        if col in df.columns:
            df[f"proj_{k}_by_min"] = df[col].astype(float) / mins * df["proj_minutes"]
    return df


def train_minutes_model(df):
    """Entrena el modelo de minutos y agrega las variables derivadas al
    dataset. Para entrenar los props se usan predicciones FUERA DE MUESTRA
    (validacion cruzada por bloques de tiempo): si no, los props aprenderian
    de minutos "perfectos" que en vivo no existen."""
    from sklearn.model_selection import KFold, cross_val_predict
    if "label_minutes" not in df.columns:
        return df
    feats = [f for f in BASE_FEATURES if f in df.columns and df[f].notna().any()]
    d = df.dropna(subset=["label_minutes"]).sort_values("date")
    model = HistGradientBoostingRegressor(max_depth=4, learning_rate=0.05, max_iter=300, random_state=0)
    oof = cross_val_predict(model, d[feats], d["label_minutes"], cv=KFold(5, shuffle=False))
    base_mae = mean_absolute_error(d["label_minutes"], d[pc.avg_feature("minutes")].fillna(d["label_minutes"].mean()))
    print(f"\n--- Modelo de minutos: error {mean_absolute_error(d['label_minutes'], oof):.2f} min "
          f"(promedio simple: {base_mae:.2f} min) ---")
    final = clone(model).fit(d[feats], d["label_minutes"])
    joblib.dump({"model": final, "features": feats}, MINUTES_MODEL_PATH)
    out = df.copy()
    pm = pd.Series(np.nan, index=out.index)
    pm.loc[d.index] = oof
    return add_minutes_derived(out, pm.values)

# (llave, columna de label, baseline = su promedio movil, titulo)
STATS = [(k, f"label_{k}", pc.avg_feature(k), pc.LABELS[k]) for k in pc.TARGETS]


def over_threshold(line):
    """Las stats son enteras: una linea de 24.5 se pasa con 25+, una de 24
    (push posible) tambien se cuenta desde 25 - el umbral siempre es N.5."""
    return np.floor(line) + 0.5


def fit_sigma_curve(preds, y, n_bins=5):
    """Que tanto se equivoca el modelo SEGUN el tamano de la prediccion: un
    jugador proyectado a 28 puntos varia mucho mas que uno proyectado a 6.
    Se agrupan las predicciones del set de prueba en bins y se guarda la
    desviacion estandar del error en cada uno - con eso se convierte la
    prediccion en una probabilidad de over/under (distribucion normal)."""
    df = pd.DataFrame({"pred": preds, "resid": np.asarray(y) - np.asarray(preds)})
    df["bin"] = pd.qcut(df["pred"], q=min(n_bins, df["pred"].nunique()), duplicates="drop")
    g = df.groupby("bin", observed=True).agg(center=("pred", "median"), sigma=("resid", "std")).dropna()
    return {"centers": [float(x) for x in g["center"]],
            "sigmas": [max(float(x), 0.5) for x in g["sigma"]]}


def prob_over(pred, line, sigma_curve, raw=False):
    """P(stat > linea) segun la prediccion y la curva de error del modelo.
    Si la curva trae calibracion ("calib"), se corrige: asi un 70% de la app
    corresponde a ~70% de aciertos reales en juegos pasados."""
    sigma = float(np.interp(pred, sigma_curve["centers"], sigma_curve["sigmas"]))
    p = float(1 - norm.cdf((over_threshold(line) - pred) / sigma))
    cal = sigma_curve.get("calib")
    if cal and not raw:
        p = float(np.interp(p, cal["x"], cal["y"]))
    return min(max(p, 0.01), 0.99)


def fit_calibration(p_raw, actual):
    """Calibracion isotonica: mapea la probabilidad 'cruda' a la frecuencia
    real de over observada (monotona). None si hay muy pocos datos."""
    from sklearn.isotonic import IsotonicRegression
    p_raw, actual = np.asarray(p_raw, float), np.asarray(actual, float)
    if len(p_raw) < 200:
        return None
    iso = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip").fit(p_raw, actual)
    xs = np.linspace(0.01, 0.99, 50)
    return {"x": [float(x) for x in xs], "y": [float(y) for y in iso.predict(xs)]}


def calibration_table(p, actual, bins=(0, .3, .4, .5, .6, .7, 1.01)):
    """Que tan bien calibradas estan las probabilidades: por rango, % predicho
    vs % real de over."""
    p, actual = np.asarray(p), np.asarray(actual)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi)
        if m.sum() >= 20:
            rows.append({"rango": f"{lo:.0%}-{min(hi, 1):.0%}", "n": int(m.sum()),
                         "predicho": round(float(p[m].mean()), 3), "real": round(float(actual[m].mean()), 3)})
    return rows


def evaluate_over_under(pred_fit, y_fit, pred_eval, y_eval, line_eval, line_fit=None):
    """Prueba honesta de las probabilidades: la curva de error se ajusta con
    la PRIMERA mitad del set de prueba y se evalua en la SEGUNDA. Como no hay
    lineas reales de casas guardadas, la linea de referencia es el promedio
    movil del jugador redondeado a .5 (parecido a como la ponen las casas)."""
    curve = fit_sigma_curve(pred_fit, y_fit)
    if line_fit is not None:  # calibracion aprendida SOLO con la primera mitad
        p_fit = [prob_over(pr, ln, curve, raw=True) for pr, ln in zip(pred_fit, line_fit)]
        curve["calib"] = fit_calibration(p_fit, np.asarray(y_fit) > over_threshold(np.asarray(line_fit)))
    p = np.array([prob_over(pr, ln, curve) for pr, ln in zip(pred_eval, line_eval)])
    actual = (np.asarray(y_eval) > over_threshold(np.asarray(line_eval))).astype(int)
    confident = np.abs(p - 0.5) >= 0.1
    return {
        "brier": round(float(np.mean((p - actual) ** 2)), 4),
        "brier_moneda": round(float(np.mean((0.5 - actual) ** 2)), 4),
        "acierto_over_under": round(float(np.mean((p > 0.5) == actual)), 4),
        "acierto_si_confianza_60": round(float(np.mean((p[confident] > 0.5) == actual[confident])), 4)
            if confident.any() else None,
        "pct_jugadas_confianza_60": round(float(confident.mean()), 4),
        "calibracion": calibration_table(p, actual),
    }


class NaiveBaselineRegressor:
    def __init__(self, col):
        self.col = col

    def predict(self, X):
        return X[self.col].values


NaiveBaselineRegressor.__module__ = "nba_props_train"


def time_split(df, test_frac=0.2):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate_regression(name, y_true, y_pred):
    return {
        "modelo": name,
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(mean_squared_error(y_true, y_pred) ** 0.5, 4),
    }


def print_feature_importance(model, X_test, y_test, features, title, n_repeats=3, max_rows=3000):
    if isinstance(model, NaiveBaselineRegressor):
        print(f"\n{title}: se omite (el modelo ganador fue el baseline, no usa las demas features).")
        return
    # Es solo informativo (no cambia el modelo): con 11 props, ~70 variables y
    # miles de filas, la version completa tardaba horas. Una muestra basta.
    if len(X_test) > max_rows:
        X_test = X_test.sample(max_rows, random_state=0)
        y_test = y_test.loc[X_test.index]
    try:
        result = permutation_importance(model, X_test, y_test, scoring="neg_mean_absolute_error",
                                         n_repeats=n_repeats, random_state=0)
    except Exception as e:
        print(f"\n{title}: no se pudo calcular ({e}).")
        return
    order = result.importances_mean.argsort()[::-1]
    print(f"\n{title}:")
    for i in order:
        print(f"  {features[i]:<16} {result.importances_mean[i]:+.4f}  (+/- {result.importances_std[i]:.4f})")


def train_stat(df, test_frac, model_out, label, naive_col, title):
    # Datasets viejos (antes de agregar las variables de contexto) siguen
    # funcionando: se usan solo las columnas que existan.
    features = [f for f in FEATURES if f in df.columns and df[f].notna().any()]
    train_df, test_df = time_split(df, test_frac)
    X_train, y_train = train_df[features], train_df[label]
    X_test, y_test = test_df[features], test_df[label]

    linreg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("reg", LinearRegression()),
    ])
    linreg.fit(X_train, y_train)
    pred_linreg = linreg.predict(X_test)

    hgb = HistGradientBoostingRegressor(max_depth=3, random_state=0)
    hgb.fit(X_train, y_train)
    pred_hgb = hgb.predict(X_test)

    pred_naive = test_df[naive_col].values

    results = [
        evaluate_regression(f"Baseline (promedio movil propio)", y_test, pred_naive),
        evaluate_regression("Regresion lineal", y_test, pred_linreg),
        evaluate_regression("Gradient Boosting", y_test, pred_hgb),
    ]
    results_df = pd.DataFrame(results)
    print(f"\n--- {title}: comparacion en el set de prueba ---")
    print(results_df.to_string(index=False))

    best_name = results_df.sort_values("mae").iloc[0]["modelo"]
    print(f"Mejor modelo ({title}) por MAE: {best_name}")

    if best_name == "Gradient Boosting":
        model_to_save, best_pred = hgb, pred_hgb
    elif best_name.startswith("Baseline"):
        model_to_save, best_pred = NaiveBaselineRegressor(naive_col), pred_naive
    else:
        model_to_save, best_pred = linreg, pred_linreg

    # Probabilidades over/under: evaluacion honesta (mitad / mitad) y curva
    # final ajustada con todo el set de prueba para usarla al predecir.
    half = len(test_df) // 2
    line_ref = np.floor(test_df[naive_col].values) + 0.5
    ou = evaluate_over_under(best_pred[:half], y_test.values[:half],
                             best_pred[half:], y_test.values[half:], line_ref[half:], line_fit=line_ref[:half])
    print(f"Over/under vs. linea de referencia (promedio movil a .5): acierto {ou['acierto_over_under']:.1%}, "
          f"Brier {ou['brier']} (moneda al aire: {ou['brier_moneda']}); "
          f"con confianza >=60%: {ou['acierto_si_confianza_60']} en {ou['pct_jugadas_confianza_60']:.0%} de los casos")
    sigma_curve = fit_sigma_curve(best_pred, y_test.values)
    p_all = [prob_over(pr, ln, sigma_curve, raw=True) for pr, ln in zip(best_pred, line_ref)]
    sigma_curve["calib"] = fit_calibration(p_all, y_test.values > over_threshold(line_ref))
    print("Calibracion (prob. predicha vs % real de over): " + ", ".join(
        f"{c['rango']}: {c['predicho']:.0%}->{c['real']:.0%}" for c in ou["calibracion"]))
    results.append({"modelo": "over_under (" + best_name + ")", **ou})

    print_feature_importance(model_to_save, X_test, y_test, features, f"Importancia de variables ({title})")

    # MODELO FINAL: el elegido se vuelve a entrenar con TODOS los datos. Antes
    # se guardaba el entrenado solo con el 80% mas viejo, asi que nunca
    # aprendia de los juegos mas recientes (toda la temporada actual).
    bias = 0.0
    if not best_name.startswith("Baseline"):
        model_to_save = clone(model_to_save).fit(df[features], df[label])
        bias = recent_bias(df, features, label, model_to_save)
        print(f"Modelo final reentrenado con los {len(df)} juegos. Correccion de sesgo reciente: {bias:+.2f}")

    joblib.dump({"model": model_to_save, "features": features, "model_name": best_name,
                 "sigma_curve": sigma_curve, "bias": bias}, model_out)
    print(f"Modelo guardado en {model_out}")
    return results


BIAS_DAYS = 21
BIAS_MIN_ROWS = 200


def recent_bias(df, features, label, final_model):
    """Cuanto se ha equivocado el modelo EN PROMEDIO en los ultimos 21 dias
    (sesgo), medido honestamente: se entrena una copia SIN esos dias y se
    compara contra lo que paso. Al arranque de temporada el modelo tendia a
    sobreestimar (~1 punto): con props de solo overs eso tumba picks. La
    prediccion final se corrige sumando este sesgo (backtest: picks de 65% a
    75% de acierto esta temporada). Acotado para que un mes raro no lo mueva
    demasiado."""
    dates = pd.to_datetime(df["date"], utc=True)
    start = dates.max() - pd.Timedelta(days=BIAS_DAYS)
    recent = dates >= start
    if recent.sum() < BIAS_MIN_ROWS or (~recent).sum() < BIAS_MIN_ROWS:
        return 0.0
    m = clone(final_model).fit(df.loc[~recent, features], df.loc[~recent, label])
    resid = df.loc[recent, label].values - m.predict(df.loc[recent, features])
    bias = float(np.mean(resid))
    cap = max(0.5, 0.15 * float(df.loc[recent, label].mean()))
    return float(np.clip(bias, -cap, cap))


def log_training_history(n_rows, n_players, results_by_stat, history_path="training_history_props.csv"):
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "n_rows": n_rows, "n_players": n_players}
    for stat_key, results in results_by_stat.items():
        rdf = pd.DataFrame([x for x in results if "mae" in x])
        best = rdf.sort_values("mae").iloc[0]
        naive = rdf[rdf["modelo"].str.startswith("Baseline")].iloc[0]
        row[f"modelo_{stat_key}"] = best["modelo"]
        row[f"mae_{stat_key}"] = best["mae"]
        row[f"mae_{stat_key}_baseline"] = naive["mae"]
    # Al agregar props nuevos cambian las columnas: se reescribe el archivo
    # con la union de columnas en vez de anexar una fila desalineada.
    new = pd.DataFrame([row])
    if os.path.exists(history_path):
        new = pd.concat([pd.read_csv(history_path), new], ignore_index=True)
    new.to_csv(history_path, index=False)
    return row


def main():
    parser = argparse.ArgumentParser(description="Entrena los modelos de props de jugador NBA")
    parser.add_argument("--data", default="training_data_props.csv")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Dataset: {len(df)} filas, {df['player_id'].nunique()} jugadores unicos")
    df = train_minutes_model(df)

    results_by_stat = {}
    for stat_key, label, naive_col, title in STATS:
        if label not in df.columns or naive_col not in df.columns:
            print(f"\n{title}: se omite (el dataset no trae {label} - recolecta de nuevo con nba_props_data.py)")
            continue
        out_path = f"model_props_{stat_key}.joblib"
        results = train_stat(df.dropna(subset=[label]), args.test_frac, out_path, label, naive_col, title)
        results_by_stat[stat_key] = results
        with open(out_path + ".metrics.json", "w") as f:
            json.dump(results, f, indent=2)

    history_row = log_training_history(len(df), df["player_id"].nunique(), results_by_stat)
    print("\n--- Guardado en training_history_props.csv ---")
    print(history_row)


if __name__ == "__main__":
    # Se corre main() desde el modulo IMPORTADO (no desde __main__) para que
    # las clases de baseline que se guardan con joblib sean las mismas que
    # se cargan despues - si no, al ganar el baseline truena el guardado
    # ("Can't pickle ... not the same object").
    import nba_props_train
    nba_props_train.main()
