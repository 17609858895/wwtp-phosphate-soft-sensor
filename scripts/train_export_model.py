from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


RAW_COLUMNS = [
    "IN_METAL_Q",
    "T1_O2",
    "METAL_Q",
    "TEMPERATURE",
    "IN_Q",
    "MAX_CF",
    "PROCESSPHASE_INLET",
    "PROCESSPHASE_OUTLET",
    "T1_NH4",
]

BASE_COLUMNS = RAW_COLUMNS[:-1]
TIME_COLUMNS = ["hour_sin", "hour_cos", "month_sin", "month_cos"]
TARGET = "T1_PO4"


def engineer_hourly_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    data = raw.copy()
    if "date" not in data.columns:
        raise ValueError("The raw dataset must contain a 'date' column.")
    data["date"] = pd.to_datetime(data["date"])
    hourly = data.set_index("date")[RAW_COLUMNS + [TARGET]].resample("1h").mean().dropna()

    feat = hourly.copy()
    feat["hour_sin"] = np.sin(2 * np.pi * feat.index.hour / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat.index.hour / 24)
    feat["month_sin"] = np.sin(2 * np.pi * feat.index.month / 12)
    feat["month_cos"] = np.cos(2 * np.pi * feat.index.month / 12)
    for col in RAW_COLUMNS:
        for lag in [1, 2, 3]:
            feat[f"{col}_lag{lag}h"] = feat[col].shift(lag)
        feat[f"{col}_roll3h_mean"] = feat[col].rolling(3).mean().shift(1)
        feat[f"{col}_roll3h_std"] = feat[col].rolling(3).std().shift(1)
    feat = feat.dropna()
    lag_cols = [c for c in feat.columns if "_lag" in c]
    roll_cols = [c for c in feat.columns if "_roll" in c]
    feature_cols = RAW_COLUMNS + TIME_COLUMNS + lag_cols + roll_cols
    return feat, feature_cols, hourly


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def train_lightgbm(feat: pd.DataFrame, feature_cols: list[str]) -> tuple[LGBMRegressor, dict[str, float], dict[str, float], int]:
    x = feat[feature_cols]
    y = feat[TARGET].to_numpy()
    cut = int(len(feat) * 0.8)
    model = LGBMRegressor(
        n_estimators=300,
        max_depth=4,
        num_leaves=18,
        learning_rate=0.032,
        min_child_samples=45,
        subsample=0.82,
        colsample_bytree=0.82,
        reg_lambda=10.0,
        reg_alpha=0.1,
        random_state=42,
        verbose=-1,
        n_jobs=2,
    )
    model.fit(x.iloc[:cut], y[:cut])
    train_metrics = metrics(y[:cut], model.predict(x.iloc[:cut]))
    test_metrics = metrics(y[cut:], model.predict(x.iloc[cut:]))
    return model, train_metrics, test_metrics, cut


def summary_stats(hourly: pd.DataFrame) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    for col in RAW_COLUMNS:
        s = hourly[col].dropna()
        stats[col] = {
            "min": float(s.min()),
            "p01": float(s.quantile(0.01)),
            "p05": float(s.quantile(0.05)),
            "median": float(s.median()),
            "p95": float(s.quantile(0.95)),
            "p99": float(s.quantile(0.99)),
            "max": float(s.max()),
        }
    return stats


def write_examples(feat: pd.DataFrame, hourly: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # A realistic hourly sequence. Users can upload the same format for batch prediction.
    example_sequence = hourly[RAW_COLUMNS + [TARGET]].iloc[-48:].reset_index()
    example_sequence.to_csv(out_dir / "example_hourly_sequence.csv", index=False, encoding="utf-8-sig")

    # Four rows are enough for one current-hour prediction with lag1/lag2/lag3 and rolling statistics.
    single = hourly[RAW_COLUMNS + [TARGET]].iloc[-4:].reset_index()
    single.to_csv(out_dir / "single_prediction_template.csv", index=False, encoding="utf-8-sig")

    # Feature-ready example for advanced users.
    feature_ready = feat.drop(columns=[TARGET]).iloc[-20:].reset_index(names="date")
    feature_ready.to_csv(out_dir / "example_feature_ready.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("../IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv"))
    parser.add_argument("--out", type=Path, default=Path("."))
    args = parser.parse_args()

    root = args.out.resolve()
    model_dir = root / "model"
    data_dir = root / "data"
    model_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.csv, parse_dates=["date"])
    feat, feature_cols, hourly = engineer_hourly_features(raw)
    model, train_metrics, test_metrics, cut = train_lightgbm(feat, feature_cols)

    x_train = feat[feature_cols].iloc[:cut]
    train_quantiles = x_train.quantile([0.01, 0.99]).T
    train_quantiles.columns = ["p01", "p99"]

    bundle = {
        "model": model,
        "model_name": "LightGBM",
        "target": TARGET,
        "target_label": "Reactor phosphate, PO4-P",
        "target_unit": "mg L-1",
        "raw_columns": RAW_COLUMNS,
        "feature_columns": feature_cols,
        "time_columns": TIME_COLUMNS,
        "train_feature_quantiles": train_quantiles.to_dict(orient="index"),
        "input_summary": summary_stats(hourly),
        "training_window": {
            "start": str(feat.index.min()),
            "end": str(feat.index[cut - 1]),
            "n_rows": int(cut),
        },
        "test_window": {
            "start": str(feat.index[cut]),
            "end": str(feat.index.max()),
            "n_rows": int(len(feat) - cut),
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "caveats": [
            "This model predicts in-reactor T1_PO4, not final-effluent total phosphorus.",
            "Metal dosing is treated as an operator-response signal, not as a causal dosing intervention.",
            "Reliable use requires hourly inputs and recent 1-3 h operating history.",
            "Predictions outside the training feature range should be treated as extrapolations.",
        ],
    }
    joblib.dump(bundle, model_dir / "model_bundle.joblib")

    metadata = {k: v for k, v in bundle.items() if k != "model"}
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    write_examples(feat, hourly, data_dir)
    print(json.dumps({"test_metrics": test_metrics, "features": len(feature_cols)}, indent=2))


if __name__ == "__main__":
    main()
