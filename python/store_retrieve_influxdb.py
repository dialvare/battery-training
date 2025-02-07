from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS
import pandas as pd


# InfluxDB Connection Details
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "admin_token"
INFLUXDB_ORG = "redhat"
INFLUXDB_BUCKET = "bms"

# Initialize Client
client = InfluxDBClient(url=INFLUXDB_URL, token=INFLUXDB_TOKEN, org=INFLUXDB_ORG)
write_api = client.write_api(write_options=SYNCHRONOUS)

def store_battery_data(battery_data):
    point = Point("battery_data") \
        .tag("batteryId", str(battery_data["batteryId"])) \
        .field("stateOfCharge", battery_data["stateOfCharge"]) \
        .field("stateOfHealth", battery_data["stateOfHealth"]) \
        .field("batteryCurrent", battery_data["batteryCurrent"]) \
        .field("batteryVoltage", battery_data["batteryVoltage"]) \
        .field("kmh", battery_data["kmh"]) \
        .field("distance", battery_data["distance"]) \
        .field("batteryTemp", battery_data["batteryTemp"]) \
        .field("ambientTemp", battery_data["ambientTemp"]) \
        .time(pd.Timestamp.now(), WritePrecision.NS)

    write_api.write(bucket=INFLUXDB_BUCKET, org=INFLUXDB_ORG, record=point)
    print(f"Stored data for battery ID {battery_data['batteryId']}")

def retrieve_battery_data():
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
      |> range(start: -1h)
      |> filter(fn: (r) => r["_measurement"] == "battery_data")
    '''
    query_api = client.query_api()
    tables = query_api.query(query, org=INFLUXDB_ORG)

    # Process Results
    data = []
    for table in tables:
        for record in table.records:
            data.append(record.values)

    df = pd.DataFrame(data)
    print(df)  # Print the retrieved data
    return df
    
# Example Data
battery_data_sample = {
    "batteryId": 1,
    "stateOfCharge": 0.4384,
    "stateOfHealth": 98.00,
    "batteryCurrent": 349.57,
    "batteryVoltage": 397.63,
    "kmh": 234.00,
    "distance": 79.35,
    "batteryTemp": 30.00,
    "ambientTemp": 20.40
}

#store_battery_data(battery_data_sample)

#retrieve_battery_data()

df = retrieve_battery_data()

# Drop non-numeric columns
df = df.drop(columns=["result", "table", "_start", "_stop", "_measurement", "_field", "_time"])

# Save for ML
df.to_csv("battery_data.csv", index=False)
print("Saved battery data for ML training.")

