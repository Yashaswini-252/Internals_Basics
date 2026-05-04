import pandas as pd
import numpy as np
import os
import json
import joblib
import mlflow
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

def main():
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    df = pd.read_csv('data/training_data.csv')
    X = df.drop('delivery_time_min', axis=1)
    y = df['delivery_time_min']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlflow.set_experiment("freshbasket-delivery-time-min")

    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    }

    model = GradientBoostingRegressor(random_state=42)
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    rs = RandomizedSearchCV(
        model, 
        param_distributions=param_grid, 
        n_iter=10, 
        cv=kf, 
        scoring='neg_mean_absolute_error', 
        random_state=42
    )
    rs.fit(X_train, y_train)

    best_cv_mae = float(-rs.best_score_)
    best_params = {k: (int(v) if isinstance(v, (np.integer, int)) else float(v)) for k, v in rs.best_params_.items()}

    with mlflow.start_run(run_name="tuning-freshbasket") as parent_run:
        for i in range(len(rs.cv_results_['params'])):
            with mlflow.start_run(run_name=f"trial_{i}", nested=True):
                mlflow.log_params(rs.cv_results_['params'][i])
                mlflow.log_metric("cv_mae", float(-rs.cv_results_['mean_test_score'][i]))

    best_model = rs.best_estimator_
    y_pred = best_model.predict(X_test)
    test_mae = float(mean_absolute_error(y_test, y_pred))

    joblib.dump(best_model, 'models/best_tuned_model.pkl')

    output2 = {
      "search_type": "random",
      "n_folds": 3,
      "total_trials": len(rs.cv_results_['params']),
      "best_params": best_params,
      "best_mae": test_mae,
      "best_cv_mae": best_cv_mae,
      "parent_run_name": "tuning-freshbasket"
    }

    with open('results/step2_s2.json', 'w') as f:
        json.dump(output2, f, indent=2)

if __name__ == "__main__":
    main()
