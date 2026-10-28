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
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["avg_points", "avg_rebounds", "avg_assists", "avg_minutes",
            "avg_points_last3", "avg_rebounds_last3", "avg_assists_last3", "avg_minutes_last3",
            "team_off_rtg", "team_pace", "opp_def_rtg", "opp_pace",
            "days_rest", "b2b",
            # Contexto de minutos/uso: local o visita, companeros habituales
            # ausentes, y margen proyectado (las palizas recortan minutos).
            "is_home", "teammates_out", "proj_margin", "proj_blowout"]


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


def prob_over(pred, line, sigma_curve):
    """P(stat > linea) segun la prediccion y la curva de error del modelo."""
    sigma = float(np.interp(pred, sigma_curve["centers"], sigma_curve["sigmas"]))
    return float(1 - norm.cdf((over_threshold(line) - pred) / sigma))


def evaluate_over_under(pred_fit, y_fit, pred_eval, y_eval, line_eval):
    """Prueba honesta de las probabilidades: la curva de error se ajusta con
    la PRIMERA mitad del set de prueba y se evalua en la SEGUNDA. Como no hay
    lineas reales de casas guardadas, la linea de referencia es el promedio
    movil del jugador redondeado a .5 (parecido a como la ponen las casas)."""
    curve = fit_sigma_curve(pred_fit, y_fit)
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
    }

STATS = [
    ("points", "label_points", "avg_points", "Puntos"),
    ("rebounds", "label_rebounds", "avg_rebounds", "Rebotes"),
    ("assists", "label_assists", "avg_assists", "Asistencias"),
]


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


def print_feature_importance(model, X_test, y_test, features, title, n_repeats=10):
    if isinstance(model, NaiveBaselineRegressor):
        print(f"\n{title}: se omite (el modelo ganador fue el baseline, no usa las demas features).")
        return
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
    features = [f for f in FEATURES if f in df.columns]
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
                             best_pred[half:], y_test.values[half:], line_ref[half:])
    print(f"Over/under vs. linea de referencia (promedio movil a .5): acierto {ou['acierto_over_under']:.1%}, "
          f"Brier {ou['brier']} (moneda al aire: {ou['brier_moneda']}); "
          f"con confianza >=60%: {ou['acierto_si_confianza_60']} en {ou['pct_jugadas_confianza_60']:.0%} de los casos")
    sigma_curve = fit_sigma_curve(best_pred, y_test.values)
    results.append({"modelo": "over_under (" + best_name + ")", **ou})

    joblib.dump({"model": model_to_save, "features": features, "model_name": best_name,
                 "sigma_curve": sigma_curve}, model_out)
    print(f"Modelo guardado en {model_out}")
    print_feature_importance(model_to_save, X_test, y_test, features, f"Importancia de variables ({title})")
    return results


def log_training_history(n_rows, n_players, results_by_stat, history_path="training_history_props.csv"):
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "n_rows": n_rows, "n_players": n_players}
    for stat_key, results in results_by_stat.items():
        rdf = pd.DataFrame([x for x in results if "mae" in x])
        best = rdf.sort_values("mae").iloc[0]
        naive = rdf[rdf["modelo"].str.startswith("Baseline")].iloc[0]
        row[f"modelo_{stat_key}"] = best["modelo"]
        row[f"mae_{stat_key}"] = best["mae"]
        row[f"mae_{stat_key}_baseline"] = naive["mae"]
    file_exists = os.path.exists(history_path)
    pd.DataFrame([row]).to_csv(history_path, mode="a", header=not file_exists, index=False)
    return row


def main():
    parser = argparse.ArgumentParser(description="Entrena los modelos de props de jugador NBA")
    parser.add_argument("--data", default="training_data_props.csv")
    parser.add_argument("--model-points-out", default="model_props_points.joblib")
    parser.add_argument("--model-rebounds-out", default="model_props_rebounds.joblib")
    parser.add_argument("--model-assists-out", default="model_props_assists.joblib")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Dataset: {len(df)} filas, {df['player_id'].nunique()} jugadores unicos")

    outs = {"points": args.model_points_out, "rebounds": args.model_rebounds_out, "assists": args.model_assists_out}
    results_by_stat = {}
    for stat_key, label, naive_col, title in STATS:
        results = train_stat(df, args.test_frac, outs[stat_key], label, naive_col, title)
        results_by_stat[stat_key] = results
        with open(outs[stat_key] + ".metrics.json", "w") as f:
            json.dump(results, f, indent=2)

    history_row = log_training_history(len(df), df["player_id"].nunique(), results_by_stat)
    print("\n--- Guardado en training_history_props.csv ---")
    print(history_row)


if __name__ == "__main__":
    main()
