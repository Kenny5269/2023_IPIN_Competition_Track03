import pandas as pd
import numpy as np
from geopy.distance import distance, geodesic

# wifi_df = pd.read_csv("py/T1_R1/WIFI_merged2.csv")
# imu_df = pd.read_csv("py/T1_R1/IMU_50Hz.csv").rename(columns={"AppTimestamp(s)": "timestamp"})

# for i in range(len(wifi_df)):
#     rssi_vector = wifi_df.iloc[i, 1:].to_numpy()
#     print(rssi_vector)
#     break

# for _, wifi_row in wifi_df.iterrows():
#     print(wifi_row[1:].to_numpy())
#     break

# for i in range(len(imu_df)):
#     imu_data = imu_df.drop(columns=["timestamp","SensorTimestamp(s)"]).reset_index(drop=True)
#     print(imu_data)
#     break
print(geodesic((49.46131643, 11.11097444), (49.461277, 11.11098835)).meters)