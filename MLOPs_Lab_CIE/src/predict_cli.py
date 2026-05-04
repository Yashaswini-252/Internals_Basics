import argparse
import joblib
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Predict Delivery Time")
    parser.add_argument("--order_weight_kg", type=float, required=True)
    parser.add_argument("--distance_km", type=float, required=True)
    parser.add_argument("--is_peak_hour", type=int, required=True)
    parser.add_argument("--items_count", type=int, required=True)
    
    args = parser.parse_args()
    
    model = joblib.load('models/best_tuned_model.pkl')
    
    input_data = pd.DataFrame([{
        "order_weight_kg": args.order_weight_kg,
        "distance_km": args.distance_km,
        "is_peak_hour": args.is_peak_hour,
        "items_count": args.items_count
    }])
    
    prediction = model.predict(input_data)[0]
    print(prediction)

if __name__ == "__main__":
    main()
