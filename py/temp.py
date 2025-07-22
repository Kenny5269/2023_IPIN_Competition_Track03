import pandas as pd
import numpy as np
from geopy.distance import distance, geodesic
from pyproj import Transformer

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
#print(geodesic((49.46131643, 11.11097444), (49.461277, 11.11098835)).meters)
# # 設定原點為你的第一個定位點
# origin_lat, origin_lon = 49.46131643, 11.11096444

# # 建立 ENU 轉換器 (經緯度 → 平面)
# transformer = Transformer.from_crs("epsg:4326", f"+proj=tmerc +lat_0={origin_lat} +lon_0={origin_lon} +units=m", always_xy=True)
# # 建立轉換器（平面 → 經緯度）
# transformer_back = Transformer.from_crs(f"+proj=tmerc +lat_0={origin_lat} +lon_0={origin_lon} +units=m","epsg:4326", always_xy=True)

# # 經緯度 → x, y（公尺）
# x, y = transformer.transform(11.11098835, 49.4612756)

# # 假設你有 EKF 輸出：
# ekf_x, ekf_y = 12.3, 45.6  # 單位：公尺

# # 轉回經緯度
# lon, lat = transformer_back.transform(ekf_x, ekf_y)
# print(x,y)

# a = [1,2,3,4,5]
# print(a[1:])
for i in range(3):
    A = {i+1}
    print(A)