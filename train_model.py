"""
Train and save all artifacts for the WWTP Phosphate Soft Sensor web app.
Run this script ONCE before deploying or running the Streamlit app.

Usage: python train_model.py
Input:  IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv (in same directory)
Output: artifacts/ directory with model, data, and metrics
"""

import json
import warnings
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor,
    GradientBoostingRegressor, HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
CSV = ROOT / "IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv"
ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

BASE_COLS = [
    "IN_METAL_Q", "T1_O2", "METAL_Q", "TEMPERATURE", "IN_Q", "MAX_CF",
    "PROCESSPHASE_INLET", "PROCESSPHASE_OUTLET",
]
ENHANCED_COLS = BASE_COLS + ["T1_NH4"]
TIME_COLS = ["hour_sin", "hour_cos", "month_sin", "month_cos"]


def load_and_engineer():
    raw = pd.read_csv(CSV, parse_dates=["date"])
    hourly = raw.set_index("date").resample("1h").mean().dropna()
    feat = hourly.copy()
    feat["hour_sin"] = np.sin(2 * np.pi * feat.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat.index.hour / 24)
    feat["month_sin"] = np.sin(2 * np.pi * feat.index.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat.index.month / 12)
    for col in ENHANCED_COLS:
        for lag in [1, 2, 3]:
            feat[f"{col}_lag{lag}h"] = feat[col].shift(lag)
        feat[f"{col}_roll3h_mean"] = feat[col].rolling(3).mean().shift(1)
        feat[f"{col}_roll3h_std"] = feat[col].rolling(3).std().shift(1)
    feat = feat.dropna()
    lag_cols = [c for c in feat.columns if "_lag" in c]
    roll_cols = [c for c in feat.columns if "_roll" in c]
    feature_cols = ENHANCED_COLS + TIME_COLS + lag_cols + roll_cols
    return raw, hourly, feat, feature_cols


def model_zoo():
    return {
        "Linear": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=20.0)),
        "Lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.001, max_iter=20000)),
        "ElasticNet": make_pipeline(StandardScaler(), ElasticNet(alpha=0.001, l1_ratio=0.25, max_iter=20000)),
        "KNN": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=25, weights="distance")),
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=180, learning_rate=0.035, max_depth=3, min_samples_leaf=35, random_state=42),
        "HistGB": HistGradientBoostingRegressor(
            max_iter=220, learning_rate=0.035, max_leaf_nodes=18, l2_regularization=0.2, random_state=42),
        "RandomForest": RandomForestRegressor(
            n_estimators=260, max_depth=8, min_samples_leaf=25, max_features=0.65, n_jobs=-1, random_state=42),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=260, max_depth=8, min_samples_leaf=18, max_features=0.7, n_jobs=-1, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=260, max_depth=3, learning_rate=0.035, subsample=0.82, colsample_bytree=0.82,
            reg_lambda=9.0, reg_alpha=0.2, min_child_weight=12, objective="reg:squarederror",
            random_state=42, n_jobs=2),
        "LightGBM": LGBMRegressor(
            n_estimators=300, max_depth=4, num_leaves=18, learning_rate=0.032, min_child_samples=45,
            subsample=0.82, colsample_bytree=0.82, reg_lambda=10.0, reg_alpha=0.1,
            random_state=42, verbose=-1, n_jobs=2),
        "MLP": make_pipeline(
            StandardScaler(),
            MLPRegressor(hidden_layer_sizes=(48, 16), alpha=0.01, learning_rate_init=0.001,
                         early_stopping=True, max_iter=450, random_state=42)),
    }


def entropy_topsis(metrics_df):
    data = metrics_df[["r2", "rmse", "mae", "gap"]].copy()
    data["rmse"] = data["rmse"].max() - data["rmse"]
    data["mae"] = data["mae"].max() - data["mae"]
    data["gap"] = data["gap"].max() - data["gap"]
    arr = data.to_numpy(float)
    arr = arr - arr.min(axis=0) + 1e-12
    p = arr / arr.sum(axis=0, keepdims=True)
    e = -(p * np.log(p + 1e-12)).sum(axis=0) / np.log(len(arr))
    w = (1 - e) / (1 - e).sum()
    norm = arr / np.sqrt((arr ** 2).sum(axis=0, keepdims=True))
    weighted = norm * w
    ideal = weighted.max(axis=0)
    nadir = weighted.min(axis=0)
    d_pos = np.sqrt(((weighted - ideal) ** 2).sum(axis=1))
    d_neg = np.sqrt(((weighted - nadir) ** 2).sum(axis=1))
    score = d_neg / (d_pos + d_neg + 1e-12)
    out = metrics_df[["model"]].copy()
    out["topsis"] = score
    out["rank"] = out["topsis"].rank(ascending=False, method="min").astype(int)
    weights = pd.DataFrame({"criterion": ["R2", "RMSE", "MAE", "Gap"], "weight": w})
    return out.sort_values("topsis", ascending=False), weights


def train_all():
    print("Loading and engineering features...")
    raw, hourly, feat, feature_cols = load_and_engineer()

    X = feat[feature_cols]
    y = feat["T1_PO4"].to_numpy()
    cut = int(len(feat) * 0.8)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y[:cut], y[cut:]

    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}, Features: {len(feature_cols)}")

    rows, preds, fitted = [], {}, {}
    for name, model in model_zoo().items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        p_train = model.predict(X_train)
        p_test = model.predict(X_test)
        rows.append({
            "model": name,
            "train_r2": r2_score(y_train, p_train),
            "r2": r2_score(y_test, p_test),
            "gap": r2_score(y_train, p_train) - r2_score(y_test, p_test),
            "rmse": np.sqrt(mean_squared_error(y_test, p_test)),
            "mae": mean_absolute_error(y_test, p_test),
        })
        preds[name] = {"train": p_train, "test": p_test}
        fitted[name] = model

    met = pd.DataFrame(rows).sort_values("r2", ascending=False)
    topsis, weights = entropy_topsis(met)
    met = met.merge(topsis[["model", "topsis", "rank"]], on="model").sort_values("r2", ascending=False)

    # SHAP for LightGBM
    print("Computing SHAP values...")
    try:
        import shap
        explainer = shap.TreeExplainer(fitted["LightGBM"])
        sample_idx = np.random.default_rng(42).choice(len(X_test), min(1500, len(X_test)), replace=False)
        shap_values = explainer.shap_values(X_test.iloc[sample_idx])
        shap_data = {
            "values": shap_values,
            "features": X_test.iloc[sample_idx].values,
            "feature_names": feature_cols,
            "sample_idx": sample_idx,
        }
    except Exception as e:
        print(f"  SHAP failed: {e}, using permutation importance")
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(fitted["LightGBM"], X_test.iloc[:1800], y_test[:1800],
                                       n_repeats=8, random_state=42, n_jobs=2)
        shap_data = {
            "values": None,
            "importance": perm.importances_mean,
            "feature_names": feature_cols,
        }

    # Save artifacts
    print("Saving artifacts...")
    with open(ARTIFACTS / "models.pkl", "wb") as f:
        pickle.dump(fitted, f)
    with open(ARTIFACTS / "results.pkl", "wb") as f:
        pickle.dump({
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "metrics": met, "preds": preds,
            "topsis_weights": weights,
            "feature_cols": feature_cols,
            "test_index": feat.index[cut:],
        }, f)
    with open(ARTIFACTS / "shap_data.pkl", "wb") as f:
        pickle.dump(shap_data, f)
    met.to_csv(ARTIFACTS / "metrics.csv", index=False)

    # Save hourly data summary for the app
    hourly_desc = hourly.describe().to_dict()
    with open(ARTIFACTS / "hourly_stats.json", "w") as f:
        json.dump(hourly_desc, f, indent=2, default=str)

    # Save feature columns info
    feature_info = {
        "base_cols": BASE_COLS,
        "enhanced_cols": ENHANCED_COLS,
        "time_cols": TIME_COLS,
        "all_feature_cols": feature_cols,
    }
    with open(ARTIFACTS / "feature_info.json", "w") as f:
        json.dump(feature_info, f, indent=2)

    print(f"\nDone! Best model: {met.iloc[0]['model']} (R2={met.iloc[0]['r2']:.4f})")
    print(f"Artifacts saved to: {ARTIFACTS}")
    return met


if __name__ == "__main__":
    train_all()
