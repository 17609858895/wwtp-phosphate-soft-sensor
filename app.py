"""
WWTP Phosphate Soft Sensor — Streamlit Web Application
========================================================
An interpretable, leakage-robust machine learning soft sensor for
reactor phosphate (PO4-P) prediction in wastewater treatment plants.

Data: Agtrup (BlueKolding) WWTP, Denmark — 2 years of 2-min SCADA records
Target: T1_PO4 (reactor phosphate, mg/L)
Best model: LightGBM (temporal test R² = 0.698)
"""

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ──────────────────────────── Config ────────────────────────────
st.set_page_config(
    page_title="WWTP Phosphate Soft Sensor",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"

# ──────────────────────────── Color Palette ────────────────────────────
PALETTE = {
    "ink": "#1B1F23",
    "slate": "#6B7280",
    "grid": "#E7EEF5",
    "blue": "#3C5488",
    "cyan": "#4DBBD5",
    "teal": "#00A087",
    "green": "#7EAA55",
    "orange": "#F39B7F",
    "red": "#E64B35",
    "purple": "#8491B4",
}

FAMILY_COLORS = {
    "Linear": PALETTE["blue"],
    "Tree": PALETTE["teal"],
    "Neural": PALETTE["orange"],
    "Instance": PALETTE["purple"],
}

GROUP_COLORS = {
    "Operation": PALETTE["blue"],
    "Nutrient": PALETTE["teal"],
    "Time": PALETTE["purple"],
    "Lag": PALETTE["orange"],
    "Rolling": PALETTE["red"],
}

LABELS = {
    "IN_METAL_Q": "Inlet Metal Dose",
    "T1_O2": "Dissolved Oxygen (DO)",
    "METAL_Q": "Process Metal Dose",
    "TEMPERATURE": "Temperature",
    "IN_Q": "Influent Flow",
    "MAX_CF": "Max Control Factor",
    "PROCESSPHASE_INLET": "Inlet Phase",
    "PROCESSPHASE_OUTLET": "Outlet Phase",
    "T1_NH4": "Ammonium (NH4-N)",
    "T1_PO4": "Phosphate (PO4-P)",
    "hour_sin": "Hour (sin)",
    "hour_cos": "Hour (cos)",
    "month_sin": "Month (sin)",
    "month_cos": "Month (cos)",
}


def clean_name(name):
    out = LABELS.get(name, name)
    for key, value in LABELS.items():
        out = out.replace(key, value)
    out = out.replace("_lag", " lag ").replace("_roll3h_mean", " 3h mean")
    out = out.replace("_roll3h_std", " 3h std").replace("_", " ")
    return out


def feature_group(name):
    BASE_COLS = [
        "IN_METAL_Q", "T1_O2", "METAL_Q", "TEMPERATURE", "IN_Q", "MAX_CF",
        "PROCESSPHASE_INLET", "PROCESSPHASE_OUTLET",
    ]
    if name in BASE_COLS:
        return "Operation"
    if "NH4" in name:
        return "Nutrient"
    if "hour" in name or "month" in name:
        return "Time"
    if "_lag" in name:
        return "Lag"
    if "_roll" in name:
        return "Rolling"
    return "Other"


# ──────────────────────────── Custom CSS ────────────────────────────
st.markdown("""
<style>
    /* Main container */
    .main .block-container { padding-top: 1.5rem; max-width: 1400px; }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        height: 100%;
    }
    .metric-card h3 { color: #64748b; font-size: 0.85rem; margin: 0 0 0.3rem 0; font-weight: 500; }
    .metric-card .value { font-size: 2rem; font-weight: 700; margin: 0; }
    .metric-card .sub { color: #94a3b8; font-size: 0.75rem; margin-top: 0.2rem; }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #1e293b 0%, #334155 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        margin: 1.5rem 0 1rem 0;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    [data-testid="stSidebar"] .stMarkdown { color: #e2e8f0; }
    [data-testid="stSidebar"] .stRadio label { color: #cbd5e1 !important; }
    
    /* Hide Streamlit branding */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #f1f5f9;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────── Load Artifacts ────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load all pre-computed artifacts."""
    artifacts = {}
    
    with open(ARTIFACTS / "models.pkl", "rb") as f:
        artifacts["models"] = pickle.load(f)
    
    with open(ARTIFACTS / "results.pkl", "rb") as f:
        artifacts["results"] = pickle.load(f)
    
    with open(ARTIFACTS / "shap_data.pkl", "rb") as f:
        artifacts["shap"] = pickle.load(f)
    
    artifacts["metrics"] = pd.read_csv(ARTIFACTS / "metrics.csv")
    
    with open(ARTIFACTS / "feature_info.json", "r") as f:
        artifacts["feature_info"] = json.load(f)
    
    return artifacts


@st.cache_data
def load_raw_data():
    """Load and cache the raw CSV data."""
    csv_path = ROOT / "IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv"
    raw = pd.read_csv(csv_path, parse_dates=["date"])
    hourly = raw.set_index("date").resample("1h").mean().dropna()
    return raw, hourly


def load_or_train():
    """Load artifacts if available, otherwise train."""
    if not (ARTIFACTS / "models.pkl").exists():
        st.info("⏳ First run detected — training models... This takes ~2 minutes.")
        with st.spinner("Training 12 models..."):
            from train_model import train_all
            train_all()
        st.success("✅ Models trained successfully!")
        st.rerun()
    return load_artifacts()


# ──────────────────────────── Sidebar ────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧪 WWTP Phosphate Soft Sensor")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["📊 Dashboard", "🔬 Data Explorer", "🏆 Model Benchmark",
             "🧠 Interpretability", "🔮 Prediction", "ℹ️ About"],
            index=0,
        )
        
        st.markdown("---")
        st.markdown("""
        **Dataset**: Agtrup WWTP, Denmark  
        **Records**: 525,600 (2-min) → 17,520 (hourly)  
        **Period**: Aug 2021 – Jul 2023  
        **Target**: Reactor PO4-P  
        **Validation**: Chronological 80/20
        """)
        
        st.markdown("---")
        st.markdown("""
        <div style="text-align:center; color:#94a3b8; font-size:0.75rem;">
        Built with Streamlit · LightGBM · SHAP<br>
        Leakage-robust ML for wastewater treatment
        </div>
        """, unsafe_allow_html=True)
    
    return page


# ──────────────────────────── Dashboard Page ────────────────────────────
def page_dashboard(artifacts, hourly):
    st.markdown("# 🧪 WWTP Phosphate Soft Sensor Dashboard")
    st.markdown("**Interpretable leakage-robust ML for sustainable wastewater treatment**")
    
    met = artifacts["metrics"]
    best = met.iloc[0]
    
    # Key metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🏆 Best Model</h3>
            <p class="value" style="color:{PALETTE['teal']};">{best['model']}</p>
            <p class="sub">TOPSIS Rank #1</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📈 Test R²</h3>
            <p class="value" style="color:{PALETTE['blue']};">{best['r2']:.3f}</p>
            <p class="sub">Chronological split</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📉 RMSE</h3>
            <p class="value" style="color:{PALETTE['orange']};">{best['rmse']:.3f}</p>
            <p class="sub">mg/L</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3>📊 Train-Test Gap</h3>
            <p class="value" style="color:{PALETTE['purple']};">{best['gap']:.3f}</p>
            <p class="sub">Generalization metric</p>
        </div>
        """, unsafe_allow_html=True)
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔢 Features</h3>
            <p class="value" style="color:{PALETTE['red']};">58</p>
            <p class="sub">9 base + engineered</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 2: Time series + distribution
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📈 Phosphate Time Series (Daily Mean)")
        daily = hourly[["T1_PO4", "T1_NH4"]].resample("1D").mean()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily.index, y=daily["T1_PO4"],
            mode="lines", name="PO4-P",
            line=dict(color=PALETTE["blue"], width=1.5),
            fill="tozeroy", fillcolor="rgba(60,84,136,0.08)",
        ))
        fig.add_trace(go.Scatter(
            x=daily.index, y=daily["T1_NH4"] / 10,
            mode="lines", name="NH4-N / 10",
            line=dict(color=PALETTE["teal"], width=1.2),
        ))
        fig.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"),
            xaxis_title="Date", yaxis_title="Concentration (mg/L)",
            template="plotly_white",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📊 PO4-P Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=hourly["T1_PO4"], nbinsx=60, name="PO4-P",
            marker_color=PALETTE["blue"], opacity=0.8,
        ))
        for q, c, label in zip(
            hourly["T1_PO4"].quantile([0.5, 0.9, 0.95]),
            [PALETTE["ink"], PALETTE["orange"], PALETTE["red"]],
            ["Median", "P90", "P95"]
        ):
            fig.add_vline(x=q, line=dict(color=c, width=2, dash="dash"), annotation_text=label)
        fig.update_layout(
            height=350, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="PO4-P (mg/L)", yaxis_title="Count",
            template="plotly_white", showlegend=False,
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Row 3: Model comparison radar + TOPSIS
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 Multi-Criteria Model Comparison")
        top6 = met.head(6)
        fig = go.Figure()
        categories = ["R²", "1-RMSE", "1-MAE", "1-Gap", "TOPSIS"]
        for _, row in top6.iterrows():
            vals = [
                row["r2"],
                1 - row["rmse"],
                1 - row["mae"],
                1 - max(row["gap"], 0),
                row["topsis"],
            ]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=categories, name=row["model"],
                fill="toself", opacity=0.15,
            ))
        fig.update_layout(
            height=400, margin=dict(l=50, r=50, t=30, b=30),
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            template="plotly_white",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 🏅 TOPSIS Ranking")
        sorted_met = met.sort_values("topsis", ascending=True)
        fig = go.Figure()
        colors = [PALETTE["teal"] if i >= len(sorted_met) - 3 else PALETTE["grid"] 
                  for i in range(len(sorted_met))]
        fig.add_trace(go.Bar(
            y=sorted_met["model"], x=sorted_met["topsis"],
            orientation="h", marker_color=colors,
            text=[f"{v:.3f}" for v in sorted_met["topsis"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=400, margin=dict(l=0, r=50, t=30, b=0),
            xaxis_title="TOPSIS Closeness",
            template="plotly_white",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────── Data Explorer ────────────────────────────
def page_data_explorer(artifacts, hourly, raw):
    st.markdown("# 🔬 Data Explorer")
    st.markdown("**Explore the Agtrup WWTP SCADA dataset — 2 years of continuous monitoring**")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🔗 Correlations", "⏰ Temporal Patterns", "📋 Raw Data"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Variable Statistics (Hourly)")
            stats = hourly.describe().T.round(3)
            stats.columns = ["Count", "Mean", "Std", "Min", "Q1", "Median", "Q3", "Max"]
            st.dataframe(stats, use_container_width=True)
        
        with col2:
            st.markdown("#### Variable Distributions")
            var = st.selectbox("Select variable", hourly.columns.tolist(), 
                             format_func=lambda x: clean_name(x))
            fig = make_subplots(rows=2, cols=1, row_heights=[0.3, 0.7],
                              shared_xaxes=True, vertical_spacing=0.05)
            fig.add_trace(go.Box(x=hourly[var], name="", marker_color=PALETTE["blue"],
                               showlegend=False), row=1, col=1)
            fig.add_trace(go.Histogram(x=hourly[var], nbinsx=80, marker_color=PALETTE["teal"],
                                      opacity=0.8, showlegend=False), row=2, col=1)
            fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0),
                            template="plotly_white", font=dict(family="Inter, sans-serif"))
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Spearman Correlation Matrix")
        corr_vars = ["IN_METAL_Q", "T1_O2", "METAL_Q", "TEMPERATURE", "IN_Q", "MAX_CF",
                     "PROCESSPHASE_INLET", "PROCESSPHASE_OUTLET", "T1_NH4", "T1_PO4"]
        corr = hourly[corr_vars].corr(method="spearman")
        
        fig = go.Figure(go.Heatmap(
            z=corr.values,
            x=[clean_name(c) for c in corr.columns],
            y=[clean_name(c) for c in corr.index],
            colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=10),
        ))
        fig.update_layout(
            height=550, margin=dict(l=0, r=0, t=10, b=0),
            template="plotly_white",
            font=dict(family="Inter, sans-serif"),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("#### Hourly & Monthly Patterns")
        col1, col2 = st.columns(2)
        
        with col1:
            hourly_pattern = hourly.groupby(hourly.index.hour)["T1_PO4"].agg(["mean", "std"]).reset_index()
            hourly_pattern.columns = ["hour", "mean", "std"]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hourly_pattern["hour"], y=hourly_pattern["mean"],
                mode="lines+markers", name="Mean",
                line=dict(color=PALETTE["blue"], width=2),
                marker=dict(size=6),
            ))
            fig.add_trace(go.Scatter(
                x=pd.concat([hourly_pattern["hour"], hourly_pattern["hour"][::-1]]),
                y=pd.concat([hourly_pattern["mean"] + hourly_pattern["std"],
                            (hourly_pattern["mean"] - hourly_pattern["std"])[::-1]]),
                fill="toself", fillcolor="rgba(60,84,136,0.12)",
                line=dict(width=0), showlegend=False, name="±1 SD",
            ))
            fig.update_layout(
                title="Diurnal PO4-P Pattern", height=350,
                xaxis_title="Hour of Day", yaxis_title="PO4-P (mg/L)",
                template="plotly_white", margin=dict(l=0, r=0, t=40, b=0),
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            monthly = hourly.groupby(hourly.index.month)["T1_PO4"].agg(["mean", "std"]).reset_index()
            monthly.columns = ["month", "mean", "std"]
            months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                      "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=[months[m-1] for m in monthly["month"]], y=monthly["mean"],
                marker_color=PALETTE["teal"], opacity=0.85,
                error_y=dict(type="data", array=monthly["std"], visible=True,
                            color=PALETTE["ink"], thickness=1.5),
            ))
            fig.update_layout(
                title="Monthly PO4-P Pattern", height=350,
                xaxis_title="Month", yaxis_title="PO4-P (mg/L)",
                template="plotly_white", margin=dict(l=0, r=0, t=40, b=0),
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("#### Raw Data Sample")
        st.dataframe(hourly.head(200).style.format("{:.3f}"), use_container_width=True, height=400)
        st.caption(f"Total hourly records: {len(hourly):,}")


# ──────────────────────────── Model Benchmark ────────────────────────────
def page_model_benchmark(artifacts):
    st.markdown("# 🏆 Model Benchmark")
    st.markdown("**12-model comparison under leakage-robust chronological validation**")
    
    met = artifacts["metrics"]
    results = artifacts["results"]
    y_test = results["y_test"]
    
    tab1, tab2, tab3 = st.tabs(["📊 Performance Overview", "📈 Prediction Quality", "⚖️ Train vs Test"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Test R² Ranking")
            sorted_met = met.sort_values("r2", ascending=True)
            family_map = {
                "Linear": "Linear", "Ridge": "Linear", "Lasso": "Linear", "ElasticNet": "Linear",
                "RandomForest": "Tree", "ExtraTrees": "Tree", "GradientBoosting": "Tree",
                "HistGB": "Tree", "XGBoost": "Tree", "LightGBM": "Tree",
                "MLP": "Neural", "KNN": "Instance",
            }
            colors = [FAMILY_COLORS.get(family_map.get(m, "Other"), PALETTE["slate"]) 
                      for m in sorted_met["model"]]
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                y=sorted_met["model"], x=sorted_met["r2"],
                orientation="h", marker_color=colors,
                text=[f"{v:.3f}" for v in sorted_met["r2"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=450, margin=dict(l=0, r=60, t=10, b=0),
                xaxis_title="Temporal Test R²", xaxis_range=[0, 0.82],
                template="plotly_white",
                font=dict(family="Inter, sans-serif"),
            )
            # Add family legend
            for fam, color in FAMILY_COLORS.items():
                fig.add_trace(go.Bar(y=[None], x=[None], marker_color=color, name=fam))
            fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### RMSE vs MAE (bubble = TOPSIS)")
            fig = go.Figure()
            for _, row in met.iterrows():
                fam = family_map.get(row["model"], "Other")
                fig.add_trace(go.Scatter(
                    x=[row["rmse"]], y=[row["mae"]],
                    mode="markers+text",
                    marker=dict(size=row["topsis"] * 60 + 10, 
                              color=FAMILY_COLORS.get(fam, PALETTE["slate"]),
                              line=dict(width=1, color="white")),
                    text=row["model"] if row["model"] in {"LightGBM", "XGBoost", "Lasso", "MLP", "KNN", "Linear"} else "",
                    textposition="top center", textfont=dict(size=10, color=PALETTE["ink"]),
                    name=row["model"], showlegend=False,
                ))
            fig.update_layout(
                height=450, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="RMSE (mg/L)", yaxis_title="MAE (mg/L)",
                template="plotly_white",
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("#### Observed vs Predicted (Test Set)")
        model_name = st.selectbox("Select model", met["model"].tolist(), 
                                  index=met["model"].tolist().index("LightGBM"))
        
        pred = results["preds"][model_name]["test"]
        model_row = met[met["model"] == model_name].iloc[0]
        
        sample_idx = np.random.default_rng(42).choice(len(y_test), min(2000, len(y_test)), replace=False)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test[sample_idx], y=pred[sample_idx],
                mode="markers", name="Predictions",
                marker=dict(size=4, color=PALETTE["blue"], opacity=0.4),
            ))
            lim = [0, np.quantile(y_test, 0.995) * 1.05]
            fig.add_trace(go.Scatter(
                x=lim, y=lim, mode="lines", name="1:1 Line",
                line=dict(color=PALETTE["ink"], width=2, dash="dash"),
            ))
            fig.update_layout(
                height=450, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Observed PO4-P (mg/L)", yaxis_title="Predicted PO4-P (mg/L)",
                template="plotly_white",
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown(f"**{model_name}** Metrics")
            st.metric("Test R²", f"{model_row['r2']:.4f}")
            st.metric("RMSE", f"{model_row['rmse']:.4f} mg/L")
            st.metric("MAE", f"{model_row['mae']:.4f} mg/L")
            st.metric("Train-Test Gap", f"{model_row['gap']:.4f}")
            st.metric("TOPSIS Score", f"{model_row['topsis']:.4f}")
    
    with tab3:
        st.markdown("#### Train vs Test R² (Generalization Diagnostic)")
        fig = go.Figure()
        for _, row in met.iterrows():
            fam = family_map.get(row["model"], "Other")
            color = FAMILY_COLORS.get(fam, PALETTE["slate"])
            fig.add_trace(go.Scatter(
                x=[row["r2"], row["train_r2"]], y=[row["model"], row["model"]],
                mode="lines+markers",
                line=dict(color=PALETTE["grid"], width=3),
                marker=dict(size=10, color=[PALETTE["blue"], PALETTE["red"]]),
                showlegend=False,
            ))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                marker=dict(size=10, color=PALETTE["blue"]), name="Test"))
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                marker=dict(size=10, color=PALETTE["red"]), name="Train"))
        fig.update_layout(
            height=450, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="R²", template="plotly_white",
            font=dict(family="Inter, sans-serif"),
            legend=dict(orientation="h", y=1.05, x=0.5, xanchor="center"),
        )
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────── Interpretability ────────────────────────────
def page_interpretability(artifacts):
    st.markdown("# 🧠 Model Interpretability")
    st.markdown("**Understanding what drives phosphate predictions — SHAP, feature importance, and operating windows**")
    
    results = artifacts["results"]
    shap_data = artifacts["shap"]
    models = artifacts["models"]
    feature_cols = results["feature_cols"]
    
    tab1, tab2, tab3 = st.tabs(["🎯 Feature Importance", "📊 SHAP Analysis", "📈 Operating Windows"])
    
    with tab1:
        if shap_data["values"] is not None:
            importance = np.abs(shap_data["values"]).mean(axis=0)
        else:
            importance = shap_data["importance"]
        
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": importance,
            "group": [feature_group(f) for f in feature_cols],
        }).sort_values("importance", ascending=True).tail(20)
        
        colors = [GROUP_COLORS.get(g, PALETTE["slate"]) for g in imp_df["group"]]
        
        fig = go.Figure(go.Bar(
            y=[clean_name(f) for f in imp_df["feature"]],
            x=imp_df["importance"],
            orientation="h",
            marker_color=colors,
        ))
        fig.update_layout(
            height=600, margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Mean |SHAP Value|",
            template="plotly_white",
            font=dict(family="Inter, sans-serif"),
        )
        
        # Group legend
        for g, c in GROUP_COLORS.items():
            fig.add_trace(go.Bar(y=[None], x=[None], marker_color=c, name=g))
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.02, x=0.5, xanchor="center"))
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Group contribution pie
        col1, col2 = st.columns(2)
        with col1:
            group_sum = imp_df.groupby("group")["importance"].sum()
            fig = go.Figure(go.Pie(
                labels=group_sum.index, values=group_sum.values,
                marker=dict(colors=[GROUP_COLORS[g] for g in group_sum.index]),
                hole=0.4, textinfo="label+percent",
            ))
            fig.update_layout(
                title="Feature Group Contributions",
                height=350, margin=dict(l=0, r=0, t=40, b=0),
                template="plotly_white",
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("""
            **Feature Group Descriptions:**
            - 🔵 **Operation**: Base process variables (flow, DO, temp, phases, metal dose)
            - 🟢 **Nutrient**: Online ammonium sensor (NH4-N)
            - 🟣 **Time**: Cyclical hour/month encodings
            - 🟠 **Lag**: 1-3 hour lagged values of base features
            - 🔴 **Rolling**: 3-hour rolling mean/std of base features
            """)
    
    with tab2:
        if shap_data["values"] is not None:
            st.markdown("#### SHAP Beeswarm Plot (Top 15 Features)")
            
            shap_vals = shap_data["values"]
            features = shap_data["features"]
            names = shap_data["feature_names"]
            
            mean_abs = np.abs(shap_vals).mean(axis=0)
            top_idx = np.argsort(mean_abs)[-15:][::-1]
            
            fig = go.Figure()
            rng = np.random.default_rng(42)
            
            for rank, idx in enumerate(top_idx):
                vals = shap_vals[:, idx]
                feat_vals = features[:, idx]
                # Normalize feature values for color
                fmin, fmax = np.percentile(feat_vals, [5, 95])
                norm = (feat_vals - fmin) / (fmax - fmin + 1e-12)
                norm = np.clip(norm, 0, 1)
                
                y_jitter = np.full(len(vals), rank) + rng.normal(0, 0.12, len(vals))
                
                fig.add_trace(go.Scatter(
                    x=vals, y=y_jitter, mode="markers",
                    marker=dict(size=3, color=norm, colorscale="RdBu_r", 
                              opacity=0.5, showscale=(rank == 0),
                              colorbar=dict(title="Feature<br>Value")),
                    name=clean_name(names[idx]),
                    showlegend=False,
                ))
            
            fig.update_layout(
                height=600, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="SHAP Value (impact on prediction)",
                yaxis=dict(
                    tickvals=list(range(15)),
                    ticktext=[clean_name(names[i]) for i in top_idx],
                ),
                template="plotly_white",
                font=dict(family="Inter, sans-serif"),
            )
            fig.add_vline(x=0, line=dict(color=PALETTE["ink"], width=1))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SHAP values not available. Showing permutation importance instead.")
            imp = shap_data["importance"]
            imp_df = pd.DataFrame({"feature": feature_cols, "importance": imp})
            imp_df = imp_df.sort_values("importance", ascending=False).head(15)
            st.dataframe(imp_df.round(4), use_container_width=True)
    
    with tab3:
        st.markdown("#### Partial Dependence — Operating Windows")
        st.markdown("*How predicted PO4-P changes with each variable, holding others at their observed values*")
        
        model = models["LightGBM"]
        X_test = results["X_test"]
        
        pdp_features = ["T1_NH4", "T1_O2", "IN_METAL_Q", "METAL_Q", "IN_Q", "TEMPERATURE"]
        
        fig = make_subplots(rows=2, cols=3, 
                           subplot_titles=[clean_name(f) for f in pdp_features],
                           vertical_spacing=0.12, horizontal_spacing=0.08)
        
        for i, feat_name in enumerate(pdp_features):
            row = i // 3 + 1
            col = i % 3 + 1
            
            lo, hi = np.nanpercentile(X_test[feat_name], [2, 98])
            grid = np.linspace(lo, hi, 40)
            
            X_sample = X_test.sample(n=min(1000, len(X_test)), random_state=42)
            preds = []
            for g in grid:
                tmp = X_sample.copy()
                tmp[feat_name] = g
                preds.append(model.predict(tmp).mean())
            
            color = [PALETTE["teal"], PALETTE["blue"], PALETTE["orange"], 
                    PALETTE["red"], PALETTE["purple"], PALETTE["cyan"]][i]
            
            fig.add_trace(go.Scatter(
                x=grid, y=preds, mode="lines",
                line=dict(color=color, width=2.5),
                fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
                showlegend=False,
            ), row=row, col=col)
        
        fig.update_layout(
            height=550, margin=dict(l=0, r=0, t=30, b=0),
            template="plotly_white",
            font=dict(family="Inter, sans-serif", size=11),
        )
        for i in range(1, 7):
            fig.update_xaxes(title_text=clean_name(pdp_features[i-1]), row=(i-1)//3+1, col=(i-1)%3+1)
        fig.update_yaxes(title_text="Predicted PO4-P", row=1, col=1)
        fig.update_yaxes(title_text="Predicted PO4-P", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        > ⚠️ **Interpretation Note**: Metal dosing (IN_METAL_Q, METAL_Q) shows a positive association 
        > with phosphate. This reflects **operator control response** — operators increase dosing when 
        > phosphate is high — not a causal dose-response relationship. All interpretability results 
        > should be treated as **associational**.
        """)


# ──────────────────────────── Prediction Page ────────────────────────────
def page_prediction(artifacts):
    st.markdown("# 🔮 Phosphate Prediction")
    st.markdown("**Enter current process conditions to predict reactor PO4-P**")
    
    models = artifacts["models"]
    results = artifacts["results"]
    feature_cols = results["feature_cols"]
    
    # Model selection
    model_name = st.selectbox("Select Model", 
                              ["LightGBM", "XGBoost", "Lasso", "Ridge", "ElasticNet", "HistGB",
                               "GradientBoosting", "RandomForest", "ExtraTrees", "MLP", "KNN", "Linear"],
                              index=0)
    model = models[model_name]
    
    st.markdown("### 📝 Input Process Variables")
    
    # Get typical ranges from test data
    X_test = results["X_test"]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**🔧 Operation Variables**")
        in_metal_q = st.slider("Inlet Metal Dose (m³/h)", 
                               float(X_test["IN_METAL_Q"].min()), float(X_test["IN_METAL_Q"].max()),
                               float(X_test["IN_METAL_Q"].median()), step=1.0)
        t1_o2 = st.slider("Dissolved Oxygen (mg/L)", 
                          float(X_test["T1_O2"].min()), float(X_test["T1_O2"].max()),
                          float(X_test["T1_O2"].median()), step=0.01)
        metal_q = st.slider("Process Metal Dose (m³/h)", 
                           float(X_test["METAL_Q"].min()), float(X_test["METAL_Q"].max()),
                           float(X_test["METAL_Q"].median()), step=0.001)
        temperature = st.slider("Temperature (°C)", 
                               float(X_test["TEMPERATURE"].min()), float(X_test["TEMPERATURE"].max()),
                               float(X_test["TEMPERATURE"].median()), step=0.1)
    
    with col2:
        st.markdown("**🌊 Flow & Control**")
        in_q = st.slider("Influent Flow (m³/h)", 
                        float(X_test["IN_Q"].min()), float(X_test["IN_Q"].max()),
                        float(X_test["IN_Q"].median()), step=10.0)
        max_cf = st.slider("Max Control Factor (%)", 
                          float(X_test["MAX_CF"].min()), float(X_test["MAX_CF"].max()),
                          float(X_test["MAX_CF"].median()), step=1.0)
        processphase_inlet = st.selectbox("Inlet Phase", [1.0, 2.0], index=0)
        processphase_outlet = st.selectbox("Outlet Phase", [1.0, 2.0], index=1)
    
    with col3:
        st.markdown("**🧪 Nutrient & Time**")
        t1_nh4 = st.slider("Ammonium NH4-N (mg/L)", 
                          float(X_test["T1_NH4"].min()), float(X_test["T1_NH4"].max()),
                          float(X_test["T1_NH4"].median()), step=0.01)
        hour = st.slider("Hour of Day", 0, 23, 12)
        month = st.slider("Month", 1, 12, 6)
        
        # Show computed time features
        st.caption(f"Time features: sin/cos encoded")
    
    # Build feature vector
    if st.button("🔮 Predict PO4-P", type="primary", use_container_width=True):
        # Get median values from training data for lag/rolling features
        X_medians = X_test.median()
        
        # Create feature dict
        feat = {}
        base_vals = {
            "IN_METAL_Q": in_metal_q, "T1_O2": t1_o2, "METAL_Q": metal_q,
            "TEMPERATURE": temperature, "IN_Q": in_q, "MAX_CF": max_cf,
            "PROCESSPHASE_INLET": processphase_inlet, "PROCESSPHASE_OUTLET": processphase_outlet,
            "T1_NH4": t1_nh4,
        }
        
        # Base features
        for k, v in base_vals.items():
            feat[k] = v
        
        # Time features
        feat["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        feat["hour_cos"] = np.cos(2 * np.pi * hour / 24)
        feat["month_sin"] = np.sin(2 * np.pi * month / 12)
        feat["month_cos"] = np.cos(2 * np.pi * month / 12)
        
        # Lag features (use current value as proxy for recent history)
        enhanced = list(base_vals.keys())
        for col in enhanced:
            for lag in [1, 2, 3]:
                feat[f"{col}_lag{lag}h"] = base_vals[col]  # approximate
            feat[f"{col}_roll3h_mean"] = base_vals[col]
            feat[f"{col}_roll3h_std"] = 0.1  # small default
        
        # Build DataFrame in correct order
        input_df = pd.DataFrame([feat])[feature_cols]
        
        # Predict
        pred = model.predict(input_df)[0]
        
        # Display result
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if pred < 0.5:
                color = PALETTE["teal"]
                level = "Low — Normal operation"
                emoji = "✅"
            elif pred < 1.5:
                color = PALETTE["orange"]
                level = "Moderate — Monitor closely"
                emoji = "⚠️"
            else:
                color = PALETTE["red"]
                level = "High — Consider increasing dosing"
                emoji = "🔴"
            
            st.markdown(f"""
            <div style="text-align:center; padding:2rem; background:linear-gradient(135deg, #f8fafc, #e2e8f0); 
                        border-radius:16px; border-left:6px solid {color};">
                <h2 style="color:{color}; margin:0;">{emoji} Predicted PO4-P</h2>
                <p style="font-size:3rem; font-weight:800; color:{color}; margin:0.5rem 0;">{pred:.3f} mg/L</p>
                <p style="color:#64748b; font-size:1.1rem;">{level}</p>
                <p style="color:#94a3b8; font-size:0.85rem;">Model: {model_name}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Interpretation
        st.markdown("### 📊 Prediction Context")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"""
            **Your input vs. training data:**
            - Inlet Metal: {in_metal_q:.1f} m³/h (train median: {X_test['IN_METAL_Q'].median():.1f})
            - DO: {t1_o2:.3f} mg/L (train median: {X_test['T1_O2'].median():.3f})
            - NH4-N: {t1_nh4:.3f} mg/L (train median: {X_test['T1_NH4'].median():.3f})
            - Temperature: {temperature:.1f}°C (train median: {X_test['TEMPERATURE'].median():.1f})
            """)
        
        with col2:
            # Show where prediction falls in distribution
            test_preds = results["preds"][model_name]["test"]
            percentile = (test_preds < pred).mean() * 100
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=test_preds, nbinsx=50, marker_color=PALETTE["blue"], opacity=0.6))
            fig.add_vline(x=pred, line=dict(color=PALETTE["red"], width=3), 
                         annotation_text=f"Your prediction\n(P{percentile:.0f})")
            fig.update_layout(
                height=250, margin=dict(l=0, r=0, t=10, b=0),
                xaxis_title="Predicted PO4-P (mg/L)", yaxis_title="Count",
                template="plotly_white", showlegend=False,
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("""
        > ⚠️ **Note**: Lag and rolling features are approximated using current input values. 
        > For production deployment, these should be computed from real-time SCADA history.
        """)


# ──────────────────────────── About Page ────────────────────────────
def page_about():
    st.markdown("# ℹ️ About This Application")
    
    st.markdown("""
    ## 🧪 WWTP Phosphate Soft Sensor
    
    An **interpretable, leakage-robust machine learning soft sensor** for predicting reactor 
    phosphate (PO4-P) in wastewater treatment plants.
    
    ### 📖 Research Context
    
    This application is based on research using **2 years of 2-minute SCADA data** from the 
    **Agtrup (BlueKolding) water resource recovery facility** in Denmark, serving ~125,000 
    population equivalents.
    
    ### 🔬 Key Methodology
    
    | Aspect | Detail |
    |--------|--------|
    | **Dataset** | 525,600 records (2-min) → 17,520 (hourly) |
    | **Target** | Reactor PO4-P (T1_PO4) |
    | **Validation** | Strictly chronological 80/20 split |
    | **Models** | 12 regressors across 4 families |
    | **Best Model** | LightGBM (R² = 0.698) |
    | **Interpretability** | SHAP, PDP/ALE, permutation importance |
    
    ### 🛡️ Leakage Prevention
    
    A critical contribution is the **leakage-aware validation protocol**:
    - **No target lag features**: PO4-P history is never used as input
    - **Chronological split**: Training on past, testing on future
    - **Random split contrast**: Shows ~0.10-0.16 inflation in tree models
    - **Persistence demo**: Target lag gives R²≈0.98 by copying — exposed and excluded
    
    ### ⚠️ Interpretation Caveats
    
    - Metal dosing is treated as **operator control response**, not causal driver
    - All SHAP/PDP results are **associational**, not causal
    - The soft sensor predicts in-process PO4-P, **not** effluent compliance
    
    ### 📚 Data Source
    
    Mendeley Data: *Wastewater Treatment Plant Data for Nutrient Removal System*, Version 2  
    DOI: 10.17632/34rpmsxc4z.2
    
    ### 🛠️ Tech Stack
    
    - **Python**: scikit-learn, XGBoost, LightGBM, SHAP
    - **Web**: Streamlit, Plotly
    - **Deployment**: GitHub → Streamlit Cloud
    
    ---
    
    *Built for sustainable wastewater treatment through transparent, reproducible ML.*
    """)


# ──────────────────────────── Main ────────────────────────────
def main():
    artifacts = load_or_train()
    
    try:
        raw, hourly = load_raw_data()
    except FileNotFoundError:
        raw, hourly = None, None
    
    page = render_sidebar()
    
    if page == "📊 Dashboard":
        if hourly is not None:
            page_dashboard(artifacts, hourly)
        else:
            st.error("CSV data file not found. Please ensure `IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv` is in the project directory.")
    
    elif page == "🔬 Data Explorer":
        if hourly is not None and raw is not None:
            page_data_explorer(artifacts, hourly, raw)
        else:
            st.error("CSV data file not found.")
    
    elif page == "🏆 Model Benchmark":
        page_model_benchmark(artifacts)
    
    elif page == "🧠 Interpretability":
        page_interpretability(artifacts)
    
    elif page == "🔮 Prediction":
        page_prediction(artifacts)
    
    elif page == "ℹ️ About":
        page_about()


if __name__ == "__main__":
    main()
