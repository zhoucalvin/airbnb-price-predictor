# Airbnb Price Predictor

Predicts nightly Airbnb listing prices across New York City, Los Angeles, and Chicago using ML and big data practices.

## What it does

Given a listing's characteristics (room type, neighborhood, size, amenities, walkability, host tenure, reviews), the dashboard predicts an expected nightly price and shows comparable nearby listings on a map.

Three models are trained and compared: Ridge regression (baseline), Random Forest (grid search CV), and XGBoost (Bayesian optimization + SHAP explainability).

## Data sources

- **Inside Airbnb**: publicly available Airbnb listing snapshots for NYC, LA, and Chicago (~62,500 listings after cleaning)
- **Walk Score API**: walkability, transit, and bike scores fetched by coordinate, joined to listings via geographic snapping (0.001° grid ≈ 110m resolution)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Python 3.9+ required.

## Reproducing the pipeline

Run in order:

```bash
# 1. Collect raw listing data
python collect_data.py

# 2. Clean and merge into a single parquet
python clean_data.py

# 3. Feature engineering (amenity parsing, interaction terms, PCA)
jupyter nbconvert --to notebook --execute feature_engineering.ipynb

# 4. Train models (can run independently after step 3)
jupyter nbconvert --to notebook --execute eda.ipynb
jupyter nbconvert --to notebook --execute src/ridge_model.ipynb
jupyter nbconvert --to notebook --execute src/random_forest.ipynb
jupyter nbconvert --to notebook --execute src/xgboost_model.ipynb

# 5. Launch dashboard
python src/dashboard.py
```

Dashboard runs at `http://localhost:8050`.

Note: `models/rf_model.pkl` is ~800MB and is excluded from the repo. Re-generate it by running `src/random_forest.ipynb`.

## Project structure

```
├── collect_data.py          # Fetches raw listing data from Inside Airbnb
├── clean_data.py            # Cleans, merges, and outputs airbnb_cleaned.parquet
├── eda.ipynb                # Exploratory data analysis + hypothesis testing
├── feature_engineering.ipynb # Interaction terms, PCA on amenity matrix, walk score join
├── src/
│   ├── ridge_model.ipynb    # Ridge regression baseline (RidgeCV)
│   ├── random_forest.ipynb  # Random Forest + Grid Search CV
│   ├── xgboost_model.ipynb  # XGBoost + Bayesian Optimization + SHAP
│   └── dashboard.py         # Plotly Dash interactive dashboard
├── data/                    # Parquet files (raw CSVs excluded from repo)
├── models/                  # Saved model pkl files
└── outputs/eda/             # Charts saved during modeling
```

## Results

| Model | Test RMSE (log) | R² |
|---|---|---|
| Ridge (RidgeCV) | 0.363 | 0.733 |
| Random Forest | 0.320 | 0.792 |
| XGBoost (BayesOpt) | 0.294 | 0.825 |

XGBoost's mean absolute error on the test set translates to roughly ±$42/night in actual price.

## Authors

Calvin Zhou & Sneha Sharma
