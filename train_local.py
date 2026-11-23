"""
PropValue AI - regression model comparison on the UCI Real Estate Valuation dataset.

Usage:
    python train_local.py --data data/real_estate_valuation.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

FEATURES = ["transaction_date", "house_age", "distance_to_mrt",
            "convenience_stores", "latitude", "longitude"]
TARGET = "house_price_per_unit_area"
SEED = 42


def load_data(path):
    """Load CSV, drop the row-ID column, use clean column names (renamed by position)."""
    df = pd.read_csv(path)
    if df.shape[1] == 8:  # first column is the row ID ("No")
        df = df.iloc[:, 1:]
    if df.shape[1] != 7:
        raise ValueError(f"Expected 7 data columns after dropping ID, got {df.shape[1]}")
    df.columns = FEATURES + [TARGET]
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/real_estate_valuation.csv")
    parser.add_argument("--out", default="results")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load_data(args.data)
    print(f"Loaded {len(df)} rows, {len(FEATURES)} features")
    print(df.describe().round(2).to_string())

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    models = {
        "Linear Regression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge Regression": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=SEED),
        "Gradient Boosting": GradientBoostingRegressor(random_state=SEED),
    }

    kfold = KFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, preds = [], {}
    for name, model in models.items():
        cv_rmse = -cross_val_score(model, X_train, y_train, cv=kfold,
                                   scoring="neg_root_mean_squared_error")
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        preds[name] = p
        rows.append({
            "model": name,
            "cv_rmse_mean": cv_rmse.mean(),
            "test_rmse": np.sqrt(mean_squared_error(y_test, p)),
            "test_mae": mean_absolute_error(y_test, p),
            "test_r2": r2_score(y_test, p),
        })

    results = pd.DataFrame(rows).sort_values("cv_rmse_mean").reset_index(drop=True)
    results.round(3).to_csv(os.path.join(args.out, "model_comparison.csv"), index=False)
    print("\nModel comparison (sorted by 5-fold CV RMSE on the training split):")
    print(results.round(3).to_string(index=False))

    best_name = results.loc[0, "model"]
    best_model, best_pred = models[best_name], preds[best_name]
    print(f"\nBest model by CV RMSE: {best_name}")

    # Predicted vs actual plot
    plt.figure(figsize=(5, 5))
    plt.scatter(y_test, best_pred, alpha=0.7)
    lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
    plt.plot(lims, lims, "r--")
    plt.xlabel("Actual price per unit area")
    plt.ylabel("Predicted price per unit area")
    plt.title(f"Predicted vs Actual ({best_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "predicted_vs_actual.png"), dpi=150)
    plt.close()

    # Residuals plot
    plt.figure(figsize=(6, 4))
    plt.scatter(best_pred, y_test - best_pred, alpha=0.7)
    plt.axhline(0, color="r", linestyle="--")
    plt.xlabel("Predicted price per unit area")
    plt.ylabel("Residual (actual - predicted)")
    plt.title(f"Residuals ({best_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "residuals.png"), dpi=150)
    plt.close()

    # Permutation feature importance on the test set
    imp = permutation_importance(best_model, X_test, y_test, n_repeats=30,
                                 random_state=SEED)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False).reset_index(drop=True))
    importance.round(4).to_csv(os.path.join(args.out, "feature_importance.csv"), index=False)
    print("\nFeature importance (permutation, test set):")
    print(importance.round(4).to_string(index=False))

    plt.figure(figsize=(6, 4))
    plt.barh(importance["feature"][::-1], importance["importance"][::-1])
    plt.xlabel("Mean decrease in R2 when shuffled")
    plt.title(f"Feature importance ({best_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, "feature_importance.png"), dpi=150)
    plt.close()

    # Sample predictions on unseen test rows
    sample = X_test.head(5).copy()
    sample["actual_price"] = y_test.head(5).values
    sample["predicted_price"] = best_pred[:5]
    sample["error"] = sample["actual_price"] - sample["predicted_price"]
    sample.round(2).to_csv(os.path.join(args.out, "sample_predictions.csv"), index=False)
    print("\nSample predictions on unseen test rows:")
    print(sample.round(2).to_string(index=False))

    print(f"\nSaved all outputs to '{args.out}/'")


if __name__ == "__main__":
    main()
