import joblib
import pandas as pd

# Load Model
model = joblib.load("battery_stress_model.pkl")

# New Data
new_data = pd.DataFrame([{
    "stateOfCharge": 0.15,  # Low SoC
    "stateOfHealth": 97.00,
    "batteryCurrent": 480.00,  # High current
    "batteryVoltage": 355.00,  # Low voltage
    "kmh": 150.00,
    "distance": 50.00,
    "batteryTemp": 55.00,  # High temp
    "ambientTemp": 22.00
}])

# Make Prediction
prediction = model.predict(new_data)
print("Stress Prediction:", "STRESSED" if prediction[0] == 1 else "NORMAL")
