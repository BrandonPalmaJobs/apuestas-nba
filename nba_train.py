"""
Entrena y evalua modelos de NBA usando el dataset de nba_train_data.py:
  - puntos totales del juego (regresion)
  - margen del equipo local, home - away (regresion, para el spread)
  - home_win: quien gana el juego (clasificacion, para el money line)

Compara cada modelo contra la formula transparente de nba_report.py (columnas
baseline_*, ya incluidas en el dataset) - si el modelo entrenado no le gana a
la formula simple, no vale la pena usarlo. Mismo criterio que
apuestas_mlb/ml_train.py.

Uso:
    python nba_train.py --data training_data_nba.csv
"""

import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import (brier_score_loss, roc_auc_score, accuracy_score,
                              mean_absolute_error, mean_squared_error)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "home_off_rtg", "home_def_rtg", "home_net_rtg", "home_pace", "home_efg_pct", "home_ts_pct",
    "home_tov_pct", "home_oreb_pct", "home_dreb_pct", "home_ast_pct", "home_fast_break_pts", "home_points_in_paint",
    "away_off_rtg", "away_def_rtg", "away_net_rtg", "away_pace", "away_efg_pct", "away_ts_pct",
    "away_tov_pct", "away_oreb_pct", "away_dreb_pct", "away_ast_pct", "away_fast_break_pts", "away_points_in_paint",
    "home_days_rest", "away_days_rest", "home_b2b", "away_b2b",
    "home_missing_regulars", "away_missing_regulars",
]

LABEL_TOTAL = "label_total_points"
LABEL_MARGIN = "label_home_margin"
LABEL_WIN = "label_home_win"


class NaiveBaselineRegressor:
    """Envoltorio para 'guardar' la formula de nba_report.py como si fuera un
    modelo, cuando de verdad es la que gana - evita guardar por error un
    modelo real cuando el ganador fue la formula simple."""

    def __init__(self, col):
        self.col = col

    def predict(self, X):
        return X[self.col].values


class NaiveBaselineClassifier:
    """Convierte el margen proyectado por la formula (baseline_home_margin)
    en una probabilidad de victoria local via una curva logistica - regla
    empirica comun para pasar de spread a probabilidad en NBA (una escala de
    ~11-12 puntos equivale aprox. a un cambio de 10x en las probabilidades).
    Es una aproximacion documentada, no un numero oficial de ninguna casa de
    apuestas."""

    def __init__(self, col, scale=11.5):
        self.col = col
        self.scale = scale

    def predict_proba(self, X):
        margin = X[self.col].values
        p = 1 / (1 + 10 ** (-margin / self.scale))
        return np.column_stack([1 - p, p])


NaiveBaselineRegressor.__module__ = "nba_train"
NaiveBaselineClassifier.__module__ = "nba_train"


def time_split(df, test_frac=0.2):
    df = df.sort_values("date").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_frac))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def evaluate(name, y_true, y_prob):
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "modelo": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "auc": round(roc_auc_score(y_true, y_prob), 4) if len(set(y_true)) > 1 else None,
        "brier": round(brier_score_loss(y_true, y_prob), 4),
    }


def evaluate_regression(name, y_true, y_pred):
    return {
        "modelo": name,
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(mean_squared_error(y_true, y_pred) ** 0.5, 4),
    }


def print_feature_importance(model, X_test, y_test, features, scoring, title, n_repeats=10):
    if isinstance(model, (NaiveBaselineClassifier, NaiveBaselineRegressor)):
        print(f"\n{title}: se omite (el modelo ganador fue la formula/baseline, no usa las demas features).")
        return
    try:
        result = permutation_importance(model, X_test, y_test, scoring=scoring,
                                         n_repeats=n_repeats, random_state=0)
    except Exception as e:
        print(f"\n{title}: no se pudo calcular ({e}).")
        return
    order = result.importances_mean.argsort()[::-1]
    print(f"\n{title} (caida en '{scoring}' al revolver cada columna; mas alto = pesa mas):")
    for i in order:
        print(f"  {features[i]:<24} {result.importances_mean[i]:+.4f}  (+/- {result.importances_std[i]:.4f})")


def print_calibration_table(y_true, y_prob, title, n_bins=5):
    df = pd.DataFrame({"y": y_true.values if hasattr(y_true, "values") else y_true, "p": y_prob})
    try:
        df["bin"] = pd.qcut(df["p"], q=min(n_bins, df["p"].nunique()), duplicates="drop")
    except ValueError:
        print(f"\n{title}: muy pocos valores distintos para armar bins de calibracion.")
        return
    grouped = df.groupby("bin", observed=True).agg(n=("y", "size"), predicho=("p", "mean"), real=("y", "mean"))
    print(f"\n{title} (predicho vs. real por quintil de probabilidad, set de prueba):")
    print(grouped.round(4).to_string())


def print_regression_calibration_table(y_true, y_pred, title, n_bins=5):
    df = pd.DataFrame({"y": y_true.values if hasattr(y_true, "values") else y_true, "p": y_pred})
    try:
        df["bin"] = pd.qcut(df["p"], q=min(n_bins, df["p"].nunique()), duplicates="drop")
    except ValueError:
        print(f"\n{title}: muy pocos valores distintos para armar bins de calibracion.")
        return
    grouped = df.groupby("bin", observed=True).agg(n=("y", "size"), predicho=("p", "mean"), real=("y", "mean"))
    print(f"\n{title} (predicho vs. real por quintil, set de prueba):")
    print(grouped.round(3).to_string())


def train_regression(df, test_frac, model_out, label, naive_col, title):
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
        evaluate_regression(f"Baseline (formula {title})", y_test, pred_naive),
        evaluate_regression("Regresion lineal", y_test, pred_linreg),
        evaluate_regression("Gradient Boosting", y_test, pred_hgb),
    ]
    results_df = pd.DataFrame(results)
    print(f"\n--- {title}: comparacion en el set de prueba ---")
    print(results_df.to_string(index=False))
    print("Nota: 'mae'/'rmse' mas bajo = mejor (error promedio, 0 = perfecto).")

    best_name = results_df.sort_values("mae").iloc[0]["modelo"]
    print(f"\nMejor modelo ({title}) por MAE: {best_name}")

    if best_name == "Gradient Boosting":
        model_to_save, pred_to_save = hgb, pred_hgb
    elif best_name.startswith("Baseline"):
        model_to_save, pred_to_save = NaiveBaselineRegressor(naive_col), pred_naive
    else:
        model_to_save, pred_to_save = linreg, pred_linreg
    joblib.dump({"model": model_to_save, "features": FEATURES, "model_name": best_name}, model_out)
    print(f"Modelo guardado en {model_out}")

    print_feature_importance(model_to_save, X_test, y_test, FEATURES, scoring="neg_mean_absolute_error",
                              title=f"Importancia de variables ({title})")
    print_regression_calibration_table(y_test, pred_to_save, f"Calibracion ({title})")
    return results


def train_win_classifier(df, test_frac, model_out):
    train_df, test_df = time_split(df, test_frac)
    X_train, y_train = train_df[FEATURES], train_df[LABEL_WIN]
    X_test, y_test = test_df[FEATURES], test_df[LABEL_WIN]

    logreg = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000)),
    ])
    logreg.fit(X_train, y_train)
    prob_logreg = logreg.predict_proba(X_test)[:, 1]

    hgb = HistGradientBoostingClassifier(max_depth=3, random_state=0)
    hgb.fit(X_train, y_train)
    prob_hgb = hgb.predict_proba(X_test)[:, 1]

    naive = NaiveBaselineClassifier("baseline_home_margin")
    prob_naive = naive.predict_proba(test_df)[:, 1]

    results = [
        evaluate("Baseline (formula -> logistica)", y_test, prob_naive),
        evaluate("Regresion logistica", y_test, prob_logreg),
        evaluate("Gradient Boosting", y_test, prob_hgb),
    ]
    results_df = pd.DataFrame(results)
    print("\n--- Money line (home_win): comparacion en el set de prueba ---")
    print(results_df.to_string(index=False))
    print("Nota: 'brier' mas bajo = mejor calibracion (0 = perfecto). 'auc' mas alto = mejor separando.")

    best_name = results_df.sort_values("brier").iloc[0]["modelo"]
    print(f"\nMejor modelo (money line) por Brier score: {best_name}")

    if best_name == "Gradient Boosting":
        model_to_save, prob_to_save = hgb, prob_hgb
    elif best_name.startswith("Baseline"):
        model_to_save, prob_to_save = naive, prob_naive
    else:
        model_to_save, prob_to_save = logreg, prob_logreg
    joblib.dump({"model": model_to_save, "features": FEATURES, "model_name": best_name}, model_out)
    print(f"Modelo guardado en {model_out}")

    print_feature_importance(model_to_save, X_test, y_test, FEATURES, scoring="neg_brier_score",
                              title="Importancia de variables (money line)")
    print_calibration_table(y_test, prob_to_save, "Calibracion (money line)")
    return results


def log_training_history(n_rows, results_total, results_margin, results_win,
                          history_path="training_history_nba.csv"):
    def best_row(results, metric, minimize=True):
        rdf = pd.DataFrame(results)
        best = rdf.sort_values(metric, ascending=minimize).iloc[0]
        naive = rdf[rdf["modelo"].str.startswith("Baseline")].iloc[0]
        return best["modelo"], best[metric], naive[metric]

    best_total, mae_total, mae_total_naive = best_row(results_total, "mae")
    best_margin, mae_margin, mae_margin_naive = best_row(results_margin, "mae")
    best_win, brier_win, brier_win_naive = best_row(results_win, "brier")

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "n_rows": n_rows,
        "modelo_total": best_total, "mae_total": mae_total, "mae_total_baseline": mae_total_naive,
        "modelo_margin": best_margin, "mae_margin": mae_margin, "mae_margin_baseline": mae_margin_naive,
        "modelo_win": best_win, "brier_win": brier_win, "brier_win_baseline": brier_win_naive,
    }
    file_exists = os.path.exists(history_path)
    pd.DataFrame([row]).to_csv(history_path, mode="a", header=not file_exists, index=False)
    return row


def main():
    parser = argparse.ArgumentParser(description="Entrena los modelos de NBA (total, spread, money line)")
    parser.add_argument("--data", default="training_data_nba.csv")
    parser.add_argument("--model-total-out", default="model_nba_total.joblib")
    parser.add_argument("--model-margin-out", default="model_nba_margin.joblib")
    parser.add_argument("--model-win-out", default="model_nba_win.joblib")
    parser.add_argument("--test-frac", type=float, default=0.2)
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    print(f"Dataset: {len(df)} filas, {df[LABEL_WIN].mean():.1%} tasa de victoria local")

    train_df, test_df = time_split(df, args.test_frac)
    print(f"Train: {len(train_df)} filas (hasta {train_df['date'].max()}) | "
          f"Test: {len(test_df)} filas (desde {test_df['date'].min()})")

    results_total = train_regression(df, args.test_frac, args.model_total_out,
                                      LABEL_TOTAL, "baseline_total_points", "puntos totales")
    with open(args.model_total_out + ".metrics.json", "w") as f:
        json.dump(results_total, f, indent=2)

    results_margin = train_regression(df, args.test_frac, args.model_margin_out,
                                       LABEL_MARGIN, "baseline_home_margin", "margen local (spread)")
    with open(args.model_margin_out + ".metrics.json", "w") as f:
        json.dump(results_margin, f, indent=2)

    results_win = train_win_classifier(df, args.test_frac, args.model_win_out)
    with open(args.model_win_out + ".metrics.json", "w") as f:
        json.dump(results_win, f, indent=2)

    history_row = log_training_history(len(df), results_total, results_margin, results_win)
    print("\n--- Guardado en training_history_nba.csv (para ver progreso entre reentrenamientos) ---")
    print(history_row)


if __name__ == "__main__":
    import nba_train
    nba_train.main()
