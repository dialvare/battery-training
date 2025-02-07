import pandas as pd
import numpy as np
import joblib
import boto3
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Simulated battery data
data = [
    {"batteryId": 1, "stateOfCharge": 0.4384, "stateOfHealth": 98.00, "batteryCurrent": 349.57, "batteryVoltage": 397.63,
     "kmh": 234.00, "distance": 79.35, "batteryTemp": 0.00, "ambientTemp": 20.40},
    {"batteryId": 2, "stateOfCharge": 0.1234, "stateOfHealth": 95.00, "batteryCurrent": 500.00, "batteryVoltage": 350.00,
     "kmh": 200.00, "distance": 100.50, "batteryTemp": 60.00, "ambientTemp": 25.40},
    {"batteryId": 3, "stateOfCharge": 0.7684, "stateOfHealth": 99.00, "batteryCurrent": 200.00, "batteryVoltage": 400.00,
     "kmh": 180.00, "distance": 120.00, "batteryTemp": 30.00, "ambientTemp": 20.00},
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Define stress condition (1 = Stress, 0 = Normal)
def detect_stress(row):
    if row["batteryCurrent"] > 450 or row["batteryTemp"] > 50 or row["stateOfCharge"] < 0.2 or row["batteryVoltage"] < 360:
        return 1  # Stress condition
    return 0  # Normal condition

# Apply stress detection
df["stressIndicator"] = df.apply(detect_stress, axis=1)

# Define Features and Target
features = ["stateOfCharge", "stateOfHealth", "batteryCurrent", "batteryVoltage", "kmh", "distance", "batteryTemp", "ambientTemp"]
X = df[features]
y = df["stressIndicator"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Save Model
joblib.dump(model, "battery_stress_model.pkl")
print("Model saved as battery_stress_model.pkl")

# Upload to S3 (Optional)
def upload_to_s3(file_name, bucket, object_name):
    s3 = boto3.client("s3")
    try:
        s3.upload_file(file_name, bucket, object_name)
        print(f"Model uploaded to S3: s3://{bucket}/{object_name}")
    except Exception as e:
        print(f"Failed to upload model: {e}")

# Example: Upload model to S3
# upload_to_s3("battery_stress_model.pkl", "your-s3-bucket-name", "models/battery_stress_model.pkl")
