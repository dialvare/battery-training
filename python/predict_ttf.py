import numpy as np

# Load Trained Model
model = joblib.load("battery_ttf_model.pkl")

# New Battery Data
new_data = np.array([[45.0, 500.0, 370.0, 0.4, 98.0]])  # [Temp, Current, Voltage, SoC, SoH]

# Predict Time Before Failure
ttf_prediction = model.predict(new_data)
print(f"Predicted Time Before Failure: {ttf_prediction[0]:.2f} hours")
