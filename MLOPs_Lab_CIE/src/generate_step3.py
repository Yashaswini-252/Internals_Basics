import subprocess
import json
import os

def main():
    os.makedirs('results', exist_ok=True)
    
    cmd = [
        "python", "src/predict_cli.py",
        "--order_weight_kg", "9.0",
        "--distance_km", "4.7",
        "--is_peak_hour", "0",
        "--items_count", "6"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    prediction = float(result.stdout.strip())
    
    output = {
      "image_name": "freshbasket-predictor",
      "image_tag": "v1",
      "base_image": "python:3.11-slim",
      "test_input": {"order_weight_kg": 9.0, "distance_km": 4.7, "is_peak_hour": 0, "items_count": 6},
      "prediction": prediction
    }
    
    with open('results/step3_s3.json', 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    main()
