import pandas as pd
import joblib

# Load test data
csv_path = "test_energy_predictions.csv"  # Update path if needed
test_data = pd.read_csv(csv_path)

# Load trained model
model = joblib.load("appliance_energy_model.pkl")

# Make predictions
predictions = model.predict(test_data)

# Add predictions to the DataFrame
test_data["Predicted_Appliances_Wh"] = predictions

# Display results
print("\nPredictions:")
print(test_data[["Predicted_Appliances_Wh"]])

# Optionally save to new CSV
output_path = "predicted_test_results.csv"
test_data.to_csv(output_path, index=False)
print(f"\n✅ Predictions saved to {output_path}")
