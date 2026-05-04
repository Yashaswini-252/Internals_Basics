import pandas as pd
import numpy as np
import os
import json
import joblib
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    df = pd.read_csv('data/training_data.csv')
    X = df.drop('delivery_time_min', axis=1)
    y = df['delivery_time_min']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("freshbasket-delivery-time-min")

    models_to_train = {
        "Ridge": Ridge(random_state=42),
        "GradientBoosting": GradientBoostingRegressor(random_state=42)
    }

    results = []
    best_mae = float('inf')
    best_model_name = None
    best_model = None

    for name, model in models_to_train.items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tag("team", "ml_engineering")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            r2 = float(r2_score(y_test, y_pred))
            
            mlflow.log_params(model.get_params())
            mlflow.log_metrics({"mae": mae, "rmse": rmse, "r2": r2})
            
            results.append({
                "name": name,
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            })
            
            if mae < best_mae:
                best_mae = mae
                best_model_name = name
                best_model = model

    joblib.dump(best_model, f'models/{best_model_name.lower()}_task1.pkl')

    output = {
        "experiment_name": "freshbasket-delivery-time-min",
        "models": results,
        "best_model": best_model_name,
        "best_metric_name": "mae",
        "best_metric_value": best_mae,
        "tags": {"team": "ml_engineering"}
    }

    with open('results/step1_s1.json', 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
