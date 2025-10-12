# GreenWatt Energy Solutions — Wind Turbine Power Forecasting

Predicting per-turbine power output from operational and environmental signals to support grid planning, performance monitoring, and operations.

## Overview
This project builds machine learning models to forecast wind turbine power output using 909,604 rows and 16 features of historical turbine and environmental data. It includes EDA, feature engineering, model training (Linear Regression, Random Forest, XGBoost), and interpretation via feature importance and partial dependence. The Random Forest model performs best and is the recommended baseline, achieving over 99.7% of predictions within ±10% of the actual power output, demonstrating its reliability and practical utility for operational forecasting, making it well-suited for real-world turbine power forecasting applications.

## Key results
- Best model: Random Forest Regressor
- Test metrics: MAE ≈ 0.433, RMSE ≈ 0.761, R² ≈ 0.915
- Top drivers: ambient/nacelle temperatures, seasonal/time features, generator winding temperature, wind direction
- Interpretability: global feature importance + PDPs for key variables

## Files
- `notebooks/major_project-GreenWatt-Energy-Solutions.ipynb` — full workflow and analysis
- `reports/Problem_Statement/Data-Science_Major-Project.pdf` — problem statement/brief
- `reports/figures/` — exported plots (feature importance, PDPs)
- `data/` — use `train.csv` at project root if available.

## How to reproduce
1. Python 3.10+; install dependencies:  
   pip install -r requirements.txt

2. Place `train.csv` at the project root or in `data/`.
3. Open the notebook in `notebooks/` and run all cells.

## Methodology (short)
- EDA: distributions, correlations, outliers, multicollinearity (VIF).
- Features: time-derived (hour, day_of_week, month, season), scaling, outlier handling.
- Models: Linear Regression, Random Forest, XGBoost; tuned with sensible defaults.
- Evaluation: train/test split; MAE, RMSE, R²; tolerance-based accuracy curves.

## Notes and limitations
- Results are from a standard holdout split; future work includes deployment readiness.
-  Production features should be causally available at prediction time.


