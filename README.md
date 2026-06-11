# WWTP Reactor Phosphate Soft Sensor

This Streamlit app serves the leakage-aware LightGBM soft sensor developed for reactor phosphate prediction in a full-scale nutrient-removal wastewater treatment plant.

## What the app predicts

- Target: reactor `T1_PO4`
- Unit: `mg L-1`
- Model: LightGBM
- Inputs: hourly SCADA operating signals plus recent 1-3 h history

The app does **not** predict final-effluent total phosphorus, compliance status, greenhouse-gas emissions, or causal effects of changing metal dose.

## Model performance

The bundled model follows the paper workflow: hourly aggregation, chronological 80/20 validation, and no short target-lag feature.

Expected temporal test performance:

- R2: approximately 0.698
- RMSE: approximately 0.268 mg L-1
- MAE: approximately 0.168 mg L-1

## Files

```text
app.py
model/model_bundle.joblib
model/model_metadata.json
data/example_hourly_sequence.csv
data/single_prediction_template.csv
data/example_feature_ready.csv
scripts/train_export_model.py
requirements.txt
runtime.txt
.streamlit/config.toml
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Re-training the model

Place the original Agtrup CSV in the parent project folder or pass it explicitly:

```bash
python scripts/train_export_model.py --csv ../IOPTQCfFiFoNPo_2min_Agtrup_Aug_2023.csv --out .
```

## Streamlit Community Cloud deployment

Use these settings:

- Repository: this GitHub repository
- Branch: `main`
- Main file path: `app.py`
- Python version: `3.12`

If Streamlit Cloud uses a newer Python version, set Python 3.12 manually in **Settings -> Advanced settings** and redeploy.
