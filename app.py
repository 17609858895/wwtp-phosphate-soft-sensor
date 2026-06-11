from __future__ import annotations

from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = APP_DIR / "model" / "model_bundle.joblib"
TARGET_TEXT = "PO₄-P"
TARGET_HTML = "PO₄-P"
NH4_TEXT = "NH₄-N"
UNIT_TEXT = "mg L⁻¹"
UNIT_HTML = "mg L<span class='unit-sup'>−1</span>"
R2_HTML = "R<span class='unit-sup'>2</span>"


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
        html, body, [class*="css"] {font-size: 16px;}
        .block-container {max-width: 1240px; padding-top: 1.25rem; padding-bottom: 2.25rem;}
        h2, h3, h4 {letter-spacing: 0;}
        sup, sub {font-size: 72%; line-height: 0; position: relative; vertical-align: baseline;}
        sup {top: -0.45em;}
        sub {bottom: -0.20em;}
        .unit-sup {
            display: inline-block; font-size: 0.62em; line-height: 0;
            transform: translateY(-0.46em); margin-left: 0.04em;
        }
        .hero {
            border: 1px solid #d6e8ec; border-radius: 16px; padding: 24px 30px 25px 30px;
            background:
                linear-gradient(135deg, rgba(248, 252, 253, 0.98) 0%, rgba(236, 247, 248, 0.98) 56%, rgba(250, 252, 255, 0.98) 100%);
            margin-bottom: 17px; box-shadow: 0 12px 34px rgba(32, 48, 64, 0.06);
        }
        .hero .eyebrow {
            color: #2d8c88; font-size: 0.82rem; font-weight: 800; letter-spacing: 0.07em;
            text-transform: uppercase; margin-bottom: 0.42rem;
        }
        .hero h1 {font-size: 2.34rem; margin: 0 0 0.42rem 0; color: #203040; font-weight: 850;}
        .hero p {font-size: 1.08rem; color: #516678; margin: 0; line-height: 1.58; max-width: 920px;}
        .metric-card {
            border: 1px solid #dfeaf1; border-radius: 14px; padding: 17px 19px; background: #ffffff;
            box-shadow: 0 8px 24px rgba(32, 48, 64, 0.052); min-height: 112px;
        }
        .metric-card .label {font-size: 0.86rem; color: #66798a; font-weight: 760; text-transform: uppercase;}
        .metric-card .value {font-size: 1.76rem; color: #203040; font-weight: 850; margin-top: 5px; white-space: nowrap;}
        .result-card {
            border-radius: 18px; padding: 25px 30px; background: #f3fbfc; border: 1px solid #c6e5eb;
            margin-top: 14px; box-shadow: 0 10px 26px rgba(30, 93, 112, 0.08);
        }
        .result-card .small {color: #506577; font-weight: 760; font-size: 1.03rem;}
        .result-card .big {font-size: 3.12rem; color: #1e5d70; font-weight: 880; line-height: 1.04; margin-top: 0.32rem;}
        .scope {
            border-left: 5px solid #83cdbe; background: #f7fbfa; padding: 14px 17px;
            border-radius: 12px; color: #2d4558; margin: 11px 0 18px 0;
            font-size: 1.01rem; line-height: 1.55;
        }
        .input-group-title {
            color: #203040; font-size: 1.04rem; font-weight: 850; margin: 1.0rem 0 0.35rem 0;
        }
        .side-card {
            border: 1px solid #dfeaf1; border-radius: 14px; padding: 18px 18px 16px 18px;
            background: #ffffff; box-shadow: 0 8px 24px rgba(32, 48, 64, 0.052);
            margin-top: 0.35rem;
        }
        .side-card h3 {font-size: 1.14rem; margin: 0 0 0.65rem 0; color: #203040;}
        .side-card p {font-size: 0.99rem; margin: 0.55rem 0; line-height: 1.48; color: #516678;}
        .side-card b {color: #244b5d;}
        div[data-testid="stTabs"] button {font-size: 1.02rem; font-weight: 760;}
        div[data-testid="stMarkdownContainer"] p {font-size: 1.01rem; line-height: 1.55;}
        div[data-testid="stWidgetLabel"] label, div[data-testid="stWidgetLabel"] p {
            font-size: 1.0rem; color: #203040; font-weight: 720;
        }
        .stSlider [data-baseweb="slider"] {padding-top: 0.28rem;}
        .stDataFrame, .stDataEditor {font-size: 0.98rem;}
        div[data-testid="stAlert"] {font-size: 1.0rem;}
        section[data-testid="stSidebar"] {font-size: 1rem;}
        hr {margin: 1.05rem 0;}
        .model-card-table {
            border-collapse: collapse; width: 100%; overflow: hidden; border-radius: 12px;
            border: 1px solid #dfeaf1; background: #ffffff;
        }
        .model-card-table td {
            padding: 0.75rem 0.9rem; border-bottom: 1px solid #edf3f6; font-size: 1.0rem;
        }
        .model-card-table td:first-child {width: 32%; color: #66798a; font-weight: 800;}
        .model-card-table tr:last-child td {border-bottom: 0;}
        .small-note {color: #516678; font-size: 0.98rem; line-height: 1.55;}
        .download-note {margin-top: -0.2rem; margin-bottom: 0.65rem; color: #516678;}
        .caption-strong {font-weight: 800; color: #203040;}
        div.stButton > button:hover {background: #4f7fa6; color: white;}
        div.stDownloadButton > button:hover {border-color: #5b8db8; color: #244b5d;}
        div[data-testid="stHorizontalBlock"] {align-items: stretch;}
        div[data-testid="column"] {min-width: 0;}
        @media (max-width: 900px) {
            .block-container {padding-left: 1rem; padding-right: 1rem;}
            .hero h1 {font-size: 1.82rem;}
            .hero p {font-size: 1.0rem;}
            div[data-testid="stHorizontalBlock"] {
                flex-wrap: wrap; gap: 0.72rem;
            }
            div[data-testid="column"] {
                width: 100% !important; flex: 1 1 100% !important; min-width: 100% !important;
            }
            .metric-card {min-height: auto; padding: 14px 16px;}
            .metric-card .value {font-size: 1.48rem; white-space: normal; overflow-wrap: anywhere;}
            .result-card .big {font-size: 2.42rem;}
            .side-card {margin-top: 1rem;}
            div[data-testid="stTabs"] button {font-size: 0.98rem;}
        }
        div.stButton > button {
            width: 100%; border-radius: 11px; border: 0; background: #5b8db8; color: white;
            font-weight: 850; padding: 0.82rem 1rem; font-size: 1.03rem;
        }
        div.stDownloadButton > button {width: 100%; border-radius: 11px; font-weight: 780; font-size: 0.99rem;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def html_metric_card(label: str, value: str) -> str:
    return f"<div class='metric-card'><div class='label'>{label}</div><div class='value'>{value}</div></div>"


def format_window(window: dict) -> str:
    start = pd.to_datetime(window["start"]).strftime("%Y-%m-%d %H:%M")
    end = pd.to_datetime(window["end"]).strftime("%Y-%m-%d %H:%M")
    return f"{start} to {end}; n = {int(window['n_rows']):,}"


def input_group_title(title: str) -> None:
    st.markdown(f"<div class='input-group-title'>{title}</div>", unsafe_allow_html=True)


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
        input_group_title("Hydraulics and aeration")
        values["IN_Q"] = slider_value("Influent flow", "IN_Q", stats)
        values["T1_O2"] = slider_value("Dissolved oxygen (O₂)", "T1_O2", stats)
        values["TEMPERATURE"] = slider_value("Water temperature", "TEMPERATURE", stats)
    with c2:
        input_group_title("Metal dosing and control")
        values["IN_METAL_Q"] = slider_value("Inlet metal dose", "IN_METAL_Q", stats)
        values["METAL_Q"] = slider_value("Process metal dose", "METAL_Q", stats)
        values["MAX_CF"] = slider_value("Maximum control factor", "MAX_CF", stats)
    with c3:
        input_group_title("Nutrient and process phase")
        values["T1_NH4"] = slider_value(f"Reactor {NH4_TEXT}", "T1_NH4", stats)
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
    st.set_page_config(page_title=f"WWTP {TARGET_TEXT} Soft Sensor", layout="wide")
    css()
    bundle = load_bundle()
    test = bundle["test_metrics"]

    st.markdown(
        """
        <div class="hero">
          <div class="eyebrow">Full-scale wastewater treatment plant</div>
          <h1>Reactor {target} soft sensor</h1>
          <p>Leakage-aware LightGBM model for estimating in-reactor {target} from hourly SCADA operating signals and recent process history.</p>
        </div>
        """.format(target=TARGET_HTML),
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    cards = [
        ("Model", bundle["model_name"]),
        (f"Temporal test {R2_HTML}", f"{test['r2']:.3f}"),
        ("RMSE", f"{test['rmse']:.3f} {UNIT_HTML}"),
        ("Features", str(len(bundle["feature_columns"]))),
    ]
    for col, (label, value) in zip([m1, m2, m3, m4], cards):
        col.markdown(html_metric_card(label, value), unsafe_allow_html=True)

    st.markdown(
        f"<div class='scope'><b>Scope:</b> This app predicts reactor T1 {TARGET_HTML} only. It does not predict final-effluent phosphorus, greenhouse-gas emissions, or causal effects of changing metal dose.</div>",
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
            use_same_history = st.toggle("Use current values for the previous 1-3 h", value=True)
            seq = build_single_sequence(values, bundle)
            if not use_same_history:
                st.caption("Edit the four hourly rows below. The last row is the current prediction hour; the first three rows define lag and rolling features.")
                seq = st.data_editor(seq, use_container_width=True, hide_index=True, num_rows="fixed")
        with right:
            st.markdown(
                """
                <div class="side-card">
                  <h3>Input checklist</h3>
                  <p><b>Current status:</b> provide the latest operating signals for the prediction hour.</p>
                  <p><b>Recent history:</b> measured hourly values from the previous 3 h give the most faithful lag and rolling features.</p>
                  <p><b>Interpretation:</b> metal dose is treated as a control-response indicator, not as a causal intervention.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if st.button(f"Predict reactor {TARGET_TEXT}"):
            try:
                x, hourly_used = engineer_sequence(seq, bundle)
                pred = float(predict_frame(x.tail(1), bundle)[0])
                outside_n, outside = applicability_report(x.tail(1), bundle)
                st.markdown(
                    f"""
                    <div class="result-card">
                      <div class="small">Predicted reactor {TARGET_HTML}</div>
                      <div class="big">{pred:.3f} {UNIT_HTML}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if outside_n:
                    st.warning(f"{outside_n} engineered feature value(s) are outside the 1st-99th percentile training range. Treat this prediction as extrapolative.")
                    st.dataframe(outside.head(10), use_container_width=True)
                with st.expander("Hourly rows used for this prediction"):
                    st.dataframe(hourly_used.tail(4).reset_index(names="date"), use_container_width=True)
            except Exception as exc:
                st.error(str(exc))

    with tab_batch:
        st.markdown(
            f"<p class='download-note'>Upload either an hourly raw sequence with <span class='caption-strong'>date</span> and the nine SCADA input columns, or a feature-ready table containing all model features. The output target is reactor {TARGET_HTML} ({UNIT_HTML}).</p>",
            unsafe_allow_html=True,
        )
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
                output[f"Predicted_T1_{TARGET_TEXT}_({UNIT_TEXT})"] = predict_frame(x, bundle)
                outside_n, outside = applicability_report(x, bundle)
                st.success(f"Generated {len(output)} prediction(s).")
                if outside_n:
                    st.warning(f"{outside_n} feature value(s) are outside the 1st-99th percentile training range across the uploaded data.")
                    st.dataframe(outside.head(20), use_container_width=True)
                st.dataframe(output.head(100), use_container_width=True)
                d1, d2 = st.columns(2)
                d1.download_button("Download predictions as CSV", output.to_csv(index=False).encode("utf-8-sig"), "wwtp_po4_predictions.csv", "text/csv")
                d2.download_button("Download predictions as Excel", to_excel_bytes(output), "wwtp_po4_predictions.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
            except Exception as exc:
                st.error(str(exc))

    with tab_info:
        st.markdown("### Model card")
        st.markdown(
            f"""
            <table class="model-card-table">
              <tr><td>Target</td><td>Reactor T1 {TARGET_HTML}</td></tr>
              <tr><td>Unit</td><td>{UNIT_HTML}</td></tr>
              <tr><td>Model</td><td>{bundle["model_name"]}</td></tr>
              <tr><td>Training window</td><td>{format_window(bundle["training_window"])}</td></tr>
              <tr><td>Temporal test window</td><td>{format_window(bundle["test_window"])}</td></tr>
              <tr><td>Temporal test {R2_HTML}</td><td>{test["r2"]:.3f}</td></tr>
              <tr><td>RMSE</td><td>{test["rmse"]:.3f} {UNIT_HTML}</td></tr>
              <tr><td>MAE</td><td>{test["mae"]:.3f} {UNIT_HTML}</td></tr>
            </table>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p class='small-note'>Caveats: the app estimates an in-reactor phosphate signal only; it should not be used to infer final-effluent compliance, greenhouse-gas emissions, or causal dosing effects.</p>",
            unsafe_allow_html=True,
        )
        st.markdown("### Required raw inputs")
        st.dataframe(pd.DataFrame({"input_column": bundle["raw_columns"]}), use_container_width=True)


if __name__ == "__main__":
    main()
