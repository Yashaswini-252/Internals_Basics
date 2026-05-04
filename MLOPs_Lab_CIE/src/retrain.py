import pandas as pd
import os
import json
import joblib
import subprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

def main():
    # --- Task 3 JSON Generation (Actual Docker Run Verification) ---
    try:
        # Run docker to test the packaged model and capture the output
        result = subprocess.run([
            "docker", "run", "--rm", "freshbasket-predictor:v1",
            "--order_weight_kg", "9.0",
            "--distance_km", "4.7",
            "--is_peak_hour", "0",
            "--items_count", "6"
        ], capture_output=True, text=True, check=True)
        prediction_val = float(result.stdout.strip())
        
        output3 = {
          "image_name": "freshbasket-predictor",
          "image_tag": "v1",
          "base_image": "python:3.11-slim",
          "test_input": {"order_weight_kg": 9.0, "distance_km": 4.7, "is_peak_hour": 0, "items_count": 6},
          "prediction": prediction_val
        }
        os.makedirs('results', exist_ok=True)
        with open('results/step3_s3.json', 'w') as f:
            json.dump(output3, f, indent=2)
        print(f"Task 3: Docker run verified. Prediction: {prediction_val}")
    except Exception as e:
        print(f"Warning: Docker run could not be verified automatically. {e}")

    # --- Task 4 Retraining ---
    with open('results/step2_s2.json', 'r') as f:
        step2_results = json.load(f)
    
    champion_mae = float(step2_results['best_mae'])
    best_params = step2_results['best_params']
    
    df_train_orig = pd.read_csv('data/training_data.csv')
    df_new = pd.read_csv('data/new_data.csv')
    
    X_orig = df_train_orig.drop('delivery_time_min', axis=1)
    y_orig = df_train_orig['delivery_time_min']
    
    X_train_orig, X_test, y_train_orig, y_test = train_test_split(X_orig, y_orig, test_size=0.2, random_state=42)
    
    X_new = df_new.drop('delivery_time_min', axis=1)
    y_new = df_new['delivery_time_min']
    
    X_train_combined = pd.concat([X_train_orig, X_new], axis=0)
    y_train_combined = pd.concat([y_train_orig, y_new], axis=0)
    
    model = GradientBoostingRegressor(random_state=42, **best_params)
    model.fit(X_train_combined, y_train_combined)
    
    y_pred = model.predict(X_test)
    retrained_mae = float(mean_absolute_error(y_test, y_pred))
    
    improvement = champion_mae - retrained_mae
    threshold = 0.3
    action = "promoted" if improvement >= threshold else "kept_champion"
    
    if action == "promoted":
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/best_tuned_model.pkl')
    
    output = {
      "original_data_rows": len(df_train_orig),
      "new_data_rows": len(df_new),
      "combined_data_rows": len(df_train_orig) + len(df_new),
      "champion_mae": champion_mae,
      "retrained_mae": retrained_mae,
      "improvement": improvement,
      "min_improvement_threshold": threshold,
      "action": action,
      "comparison_metric": "mae"
    }
    
    with open('results/step4_s8.json', 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
