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
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["avg_points", "avg_rebounds", "avg_assists", "avg_minutes",
            "team_off_rtg", "team_pace", "opp_def_rtg", "opp_pace"]

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
    train_df, test_df = time_split(df, test_frac)
    X_train, y_train = train_df[FEATURES], train_df[label]
    X_test, y_test = test_df[FEATURES], test_df[label]

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
        model_to_save = hgb
    elif best_name.startswith("Baseline"):
        model_to_save = NaiveBaselineRegressor(naive_col)
    else:
        model_to_save = linreg
    joblib.dump({"model": model_to_save, "features": FEATURES, "model_name": best_name}, model_out)
    print(f"Modelo guardado en {model_out}")
    print_feature_importance(model_to_save, X_test, y_test, FEATURES, f"Importancia de variables ({title})")
    return results


def log_training_history(n_rows, n_players, results_by_stat, history_path="training_history_props.csv"):
    row = {"timestamp": datetime.now().isoformat(timespec="seconds"), "n_rows": n_rows, "n_players": n_players}
    for stat_key, results in results_by_stat.items():
        rdf = pd.DataFrame(results)
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
