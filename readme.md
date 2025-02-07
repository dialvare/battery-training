# BMS model training

## Retrieving training data

The `01_retrieve_influx_data` notebook connects to InfluxDB and exports the data of the last hour as csv file.

The `02_prepare_data` notebook pivots the data so that '_field' values become columns.

## Use Case 1: Battery stress detection 

A Python application that trains a machine learning model to detect stress indicators in the battery data. 
The model will predict whether the battery is under "stress" based on certain conditions like high current, high temperature, low state of charge, and abnormal voltage.

Packages needed:
````shellscript
pip install pandas scikit-learn joblib boto3 influxdb-client
````

The notebook `03_bms_stress_detection_training`

- Loads battery data from a CSV file.
- Extracts features like battery current, voltage, temperature, etc..
- Labels "stress conditions" based on predefined thresholds.
- Trains a model using Random Forest Classifier.
- Saves the trained model for future predictions.

### How it works.

- The detect_stress function defines what conditions indicate battery stress.
- The model is trained using RandomForestClassifier to classify stress conditions.
- The trained model is saved to a file (battery_stress_model.pkl) for later use.

After training, you can use the saved model to predict stress indicators on new data.
You can test predictions with the notebook `04_predict_stress`

## Use Case 2: Battery time to failure (ttf) detection

A Python application `05_bms_ttf_training` that trains a machine learning model with a Random Forest Regressor to predict the time before failure using battery parameters.
We assume the dataset contains historical sensor readings, with the remaining time before failure (timeBeforeFailure) decreasing over time.
We label the data with the assumption that the failure happens at the last recorded timestamp.