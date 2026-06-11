from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "model" / "model_bundle.joblib"


@st.cache_resource
def load_bundle() -> dict:
    return joblib.load(BUNDLE_PATH)


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    out["hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * idx.month / 12)
    return out


def engineer_sequence(df: pd.DataFrame, bundle: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_cols = bundle["raw_columns"]
    missing = [c for c in raw_cols if c not in df.columns]
    if missing:
        raise ValueError("Missing required raw columns: " + ", ".join(missing))

    data = df.copy()
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).sort_values("date")
        hourly = data.set_index("date")[raw_cols].resample("1h").mean().dropna()
    else:
        data = data[raw_cols].copy()
        data.index = pd.date_range(end=pd.Timestamp.now().floor("h"), periods=len(data), freq="h")
        hourly = data

    for col in raw_cols:
        hourly[col] = pd.to_numeric(hourly[col], errors="coerce")
    hourly = hourly.dropna(subset=raw_cols)
    if len(hourly) < 4:
        raise ValueError("At least 4 hourly rows are required to build lag and rolling features.")

    feat = add_time_features(hourly)
    for col in raw_cols:
        for lag in [1, 2, 3]:
            feat[f"{col}_lag{lag}h"] = feat[col].shift(lag)
        feat[f"{col}_roll3h_mean"] = feat[col].rolling(3).mean().shift(1)
        feat[f"{col}_roll3h_std"] = feat[col].rolling(3).std().shift(1)
    feat = feat.dropna()
    x = feat[bundle["feature_columns"]]
    return x, hourly.loc[feat.index]


def applicability_report(x: pd.DataFrame, bundle: dict) -> tuple[int, pd.DataFrame]:
    q = bundle.get("train_feature_quantiles", {})
    rows = []
    for col in x.columns:
        if col not in q:
            continue
        lo, hi = q[col]["p01"], q[col]["p99"]
        below = x[col] < lo
        above = x[col] > hi
        n = int((below | above).sum())
        if n:
            rows.append({"feature": col, "outside_rows": n, "training_p01": lo, "training_p99": hi})
    columns = ["feature", "outside_rows", "training_p01", "training_p99"]
    if not rows:
        return 0, pd.DataFrame(columns=columns)
    return sum(r["outside_rows"] for r in rows), pd.DataFrame(rows, columns=columns).sort_values("outside_rows", ascending=False)


def predict_frame(x: pd.DataFrame, bundle: dict) -> np.ndarray:
    return bundle["model"].predict(x[bundle["feature_columns"]])


def read_uploaded_table(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    raise ValueError("Please upload a CSV or Excel file.")


def to_excel_bytes(df: pd.DataFrame) -> bytes:
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="predictions")
    return bio.getvalue()


def css() -> None:
    st.markdown(
        """
        <style>
        .block-container {max-width: 1220px; padding-top: 1.4rem; padding-bottom: 2rem;}
        .hero {
            border: 1px solid #d8e7ee; border-radius: 14px; padding: 26px 30px;
            background: linear-gradient(135deg, #f8fcfd 0%, #edf7f8 55%, #f9fbff 100%);
            margin-bottom: 18px;
        }
        .hero h1 {font-size: 2.15rem; margin: 0 0 0.35rem 0; color: #203040;}
        .hero p {font-size: 1.02rem; color: #536579; margin: 0; line-height: 1.55;}
        .metric-card {
            border: 1px solid #dfeaf1; border-radius: 12px; padding: 16px 18px; background: #ffffff;
            box-shadow: 0 4px 18px rgba(32, 48, 64, 0.055); min-height: 112px;
        }
        .metric-card .label {font-size: 0.82rem; color: #718096; font-weight: 700; text-transform: uppercase;}
        .metric-card .value {font-size: 1.7rem; color: #203040; font-weight: 800; margin-top: 4px;}
        .result-card {
            border-radius: 16px; padding: 24px 28px; background: #f4fbfc; border: 1px solid #cce8ee;
            margin-top: 12px;
        }
        .result-card .small {color: #536579; font-weight: 700;}
        .result-card .big {font-size: 3rem; color: #1e5d70; font-weight: 850; line-height: 1.05;}
        .scope {
            border-left: 5px solid #83cdbe; background: #f7fbfa; padding: 13px 16px;
            border-radius: 10px; color: #2d4558; margin: 10px 0 18px 0;
        }
        div.stButton > button {
            width: 100%; border-radius: 10px; border: 0; background: #5b8db8; color: white;
            font-weight: 800; padding: 0.75rem 1rem;
        }
        div.stDownloadButton > button {width: 100%; border-radius: 10px; font-weight: 750;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def slider_value(label: str, col: str, stats: dict) -> float:
    s = stats[col]
    lo = float(s["p01"])
    hi = float(s["p99"])
    if lo == hi:
        hi = lo + 1.0
    default = float(s["median"])
    return float(st.slider(label, min_value=lo, max_value=hi, value=min(max(default, lo), hi), step=(hi - lo) / 100))


def current_input_form(bundle: dict) -> dict:
    stats = bundle["input_summary"]
    values: dict[str, float] = {}

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Hydraulics and aeration**")
        values["IN_Q"] = slider_value("Influent flow", "IN_Q", stats)
        values["T1_O2"] = slider_value("Dissolved oxygen (DO)", "T1_O2", stats)
        values["TEMPERATURE"] = slider_value("Water temperature", "TEMPERATURE", stats)
    with c2:
        st.markdown("**Metal dosing and control**")
        values["IN_METAL_Q"] = slider_value("Inlet metal dose", "IN_METAL_Q", stats)
        values["METAL_Q"] = slider_value("Process metal dose", "METAL_Q", stats)
        values["MAX_CF"] = slider_value("Maximum control factor", "MAX_CF", stats)
    with c3:
        st.markdown("**Nutrient and process phase**")
        values["T1_NH4"] = slider_value("Reactor NH4-N", "T1_NH4", stats)
        values["PROCESSPHASE_INLET"] = float(st.selectbox("Inlet process phase", [1, 2], index=0))
        values["PROCESSPHASE_OUTLET"] = float(st.selectbox("Outlet process phase", [1, 2], index=0))
    return values


def build_single_sequence(values: dict, bundle: dict) -> pd.DataFrame:
    ts = pd.Timestamp.combine(st.session_state["pred_date"], st.session_state["pred_time"]).floor("h")
    raw_cols = bundle["raw_columns"]
    rows = []
    for offset in [-3, -2, -1, 0]:
        row = {"date": ts + pd.Timedelta(hours=offset)}
        row.update(values)
        rows.append(row)
    return pd.DataFrame(rows)[["date"] + raw_cols]


def main() -> None:
    st.set_page_config(page_title="WWTP PO4-P Soft Sensor", layout="wide")
    css()
    bundle = load_bundle()
    test = bundle["test_metrics"]

    st.markdown(
        """
        <div class="hero">
          <h1>WWTP reactor phosphate soft sensor</h1>
          <p>Leakage-aware LightGBM model for estimating in-reactor PO4-P from hourly SCADA operating signals and recent process history.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    cards = [
        ("Model", bundle["model_name"]),
        ("Temporal test R²", f"{test['r2']:.3f}"),
        ("RMSE", f"{test['rmse']:.3f} mg L⁻¹"),
        ("Features", str(len(bundle["feature_columns"]))),
    ]
    for col, (label, value) in zip([m1, m2, m3, m4], cards):
        col.markdown(f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{value}</div></div>", unsafe_allow_html=True)

    st.markdown(
        "<div class='scope'><b>Scope:</b> This app predicts reactor T1_PO4 only. It does not predict final-effluent phosphorus, greenhouse-gas emissions, or causal effects of changing metal dose.</div>",
        unsafe_allow_html=True,
    )

    tab_single, tab_batch, tab_info = st.tabs(["Single prediction", "Batch prediction", "Model card"])

    with tab_single:
        left, right = st.columns([0.72, 0.28])
        with left:
            dcol, tcol = st.columns(2)
            with dcol:
                st.session_state["pred_date"] = st.date_input("Prediction date", value=pd.Timestamp.now().date())
            with tcol:
                st.session_state["pred_time"] = st.time_input("Prediction hour", value=pd.Timestamp.now().floor("h").time())
            values = current_input_form(bundle)
            use_same_history = st.toggle("Use current values for the previous 1-3 hours", value=True)
            seq = build_single_sequence(values, bundle)
            if not use_same_history:
                st.caption("Edit the four hourly rows below. The last row is the current prediction hour; the first three rows define lag and rolling features.")
                seq = st.data_editor(seq, use_container_width=True, hide_index=True, num_rows="fixed")
        with right:
            st.markdown("#### Input checklist")
            st.write("Provide current operating signals and recent history.")
            st.write("For best fidelity, use measured hourly values for the previous 3 h.")
            st.write("Metal dose is interpreted as a control-response indicator.")

        if st.button("Predict reactor PO4-P"):
            try:
                x, hourly_used = engineer_sequence(seq, bundle)
                pred = float(predict_frame(x.tail(1), bundle)[0])
                outside_n, outside = applicability_report(x.tail(1), bundle)
                st.markdown(
                    f"""
                    <div class="result-card">
                      <div class="small">Predicted reactor PO4-P</div>
                      <div class="big">{pred:.3f} mg L⁻¹</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if outside_n:
                    st.warning(f"{outside_n} engineered feature value(s) are outside the 1-99% training range. Treat this prediction as extrapolative.")
                    st.dataframe(outside.head(10), use_container_width=True)
                with st.expander("Hourly rows used for this prediction"):
                    st.dataframe(hourly_used.tail(4).reset_index(names="date"), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))

    with tab_batch:
        st.markdown("Upload either an hourly raw sequence with `date` and the nine SCADA input columns, or a feature-ready table containing all model features.")
        c1, c2, c3 = st.columns(3)
        c1.download_button(
            "Download hourly sequence template",
            data=(APP_DIR / "data" / "example_hourly_sequence.csv").read_bytes(),
            file_name="example_hourly_sequence.csv",
            mime="text/csv",
        )
        c2.download_button(
            "Download single prediction template",
            data=(APP_DIR / "data" / "single_prediction_template.csv").read_bytes(),
            file_name="single_prediction_template.csv",
            mime="text/csv",
        )
        c3.download_button(
            "Download feature-ready example",
            data=(APP_DIR / "data" / "example_feature_ready.csv").read_bytes(),
            file_name="example_feature_ready.csv",
            mime="text/csv",
        )

        uploaded = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx", "xls"])
        if uploaded:
            try:
                raw_upload = read_uploaded_table(uploaded)
                st.dataframe(raw_upload.head(20), use_container_width=True)
                feature_cols = bundle["feature_columns"]
                if all(c in raw_upload.columns for c in feature_cols):
                    x = raw_upload[feature_cols].copy()
                    output = raw_upload.copy()
                else:
                    x, aligned = engineer_sequence(raw_upload, bundle)
                    output = aligned.reset_index(names="date")
                output["Predicted_T1_PO4_mg_L"] = predict_frame(x, bundle)
                outside_n, outside = applicability_report(x, bundle)
                st.success(f"Generated {len(output)} prediction(s).")
                if outside_n:
                    st.warning(f"{outside_n} feature value(s) are outside the 1-99% training range across the uploaded data.")
                    st.dataframe(outside.head(20), use_container_width=True)
                st.dataframe(output.head(100), use_container_width=True)
                d1, d2 = st.columns(2)
                d1.download_button("Download predictions as CSV", output.to_csv(index=False).encode("utf-8-sig"), "wwtp_po4_predictions.csv", "text/csv")
                d2.download_button("Download predictions as Excel", to_excel_bytes(output), "wwtp_po4_predictions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as exc:
                st.error(str(exc))

    with tab_info:
        st.markdown("### Model card")
        st.json(
            {
                "target": bundle["target_label"],
                "unit": bundle["target_unit"],
                "model": bundle["model_name"],
                "training_window": bundle["training_window"],
                "test_window": bundle["test_window"],
                "test_metrics": bundle["test_metrics"],
                "caveats": bundle["caveats"],
            }
        )
        st.markdown("### Required raw inputs")
        st.dataframe(pd.DataFrame({"input_column": bundle["raw_columns"]}), use_container_width=True)


if __name__ == "__main__":
    main()
