# 🧪 WWTP Phosphate Soft Sensor

**Interpretable Leakage-Robust Machine Learning for Sustainable Wastewater Treatment**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)]()

## Overview

An interactive web application for predicting reactor phosphate (PO4-P) in wastewater treatment plants using 12 machine learning models with leakage-robust validation.

### Key Features

- 📊 **Dashboard** — Key metrics, time series, model comparison
- 🔬 **Data Explorer** — Interactive data exploration with correlations and temporal patterns
- 🏆 **Model Benchmark** — 12-model comparison under chronological validation
- 🧠 **Interpretability** — SHAP analysis, feature importance, partial dependence plots
- 🔮 **Prediction** — Real-time PO4-P prediction from process inputs

### Dataset

- **Source**: Agtrup (BlueKolding) WWTP, Denmark
- **Records**: 525,600 (2-min SCADA) → 17,520 (hourly)
- **Period**: August 2021 – July 2023
- **Target**: Reactor phosphate (T1_PO4)
- **DOI**: [10.17632/34rpmsxc4z.2](https://doi.org/10.17632/34rpmsxc4z.2)

### Models

| Family | Models | Best R² |
|--------|--------|---------|
| Linear | Linear, Ridge, Lasso, ElasticNet | 0.691 (Lasso) |
| Tree | RF, ExtraTrees, GB, HistGB, XGBoost, LightGBM | 0.698 (LightGBM) |
| Neural | MLP | 0.626 |
| Instance | KNN | 0.578 |

### Leakage Prevention

- ✅ No target lag features (PO4-P history never used as input)
- ✅ Strictly chronological 80/20 split (past → future)
- ✅ Random split contrast shows ~0.10-0.16 inflation
- ✅ Persistence artifact (target lag → R²≈0.98) exposed and excluded

## Deployment

### Local

```bash
pip install -r requirements.txt
python train_model.py    # Train models (~2 min)
streamlit run app.py     # Launch web app
```

### Streamlit Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set main file: `app.py`
5. Add the CSV file to the repo or use the training script

## Project Structure

```
wwtp-phosphate-soft-sensor/
├── app.py                  # Streamlit web application
├── train_model.py          # Model training script
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── config.toml         # Streamlit theme config
├── artifacts/              # Trained models & data (generated)
│   ├── models.pkl
│   ├── results.pkl
│   ├── shap_data.pkl
│   ├── metrics.csv
│   └── feature_info.json
└── IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv  # Source data
```

## Interpretability Notes

⚠️ Metal dosing variables (IN_METAL_Q, METAL_Q) show positive association with phosphate. This reflects **operator control response** — operators increase dosing when phosphate is high — **not** a causal dose-response relationship. All interpretability results are **associational**.

## Citation

If you use this application or dataset, please cite:

> Hansen, K.B., et al. (2023). Wastewater Treatment Plant Data for Nutrient Removal System, Version 2. Mendeley Data. DOI: 10.17632/34rpmsxc4z.2

## License

For research and educational purposes.
