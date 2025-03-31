import os
import pandas as pd
import numpy as np
import pickle
from geopy.distance import distance, geodesic
from geopy import Point
from sklearn.neighbors import KNeighborsRegressor
from scipy.signal import find_peaks

# ------------------------------
# 設定參數與路徑
# ------------------------------
index1 = '27'
index2 = '27'
INPUT_WIFI_CSV = f"py/T{index1}_R1/WIFI_merged2.csv"
INPUT_IMU_CSV = f"py/T{index1}_R1/IMU_50Hz.csv"
INPUT_GT_CSV = f"py/T{index1}_R1/POSI2.csv"
OUTPUT_DIR = f"py/aligned_trials/trial_{index2}"
IMU_WINDOW_SEC = 4.0
STEP_THRESHOLD = 1.2
FIXED_STEP_LENGTH = 0.7

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------
# 輔助函數
# ------------------------------
def smooth_acc(acc_series, alpha=0.1):
    smoothed = [acc_series[0]]
    for a in acc_series[1:]:
        smoothed.append(alpha * a + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)

def estimate_trajectory_from_imu_all(init_lat, init_lon, imu_df):
    if len(imu_df) < 2:
        return [(init_lat, init_lon)]

    acc_mag = np.sqrt(imu_df["acc_x"]**2 + imu_df["acc_y"]**2 + imu_df["acc_z"]**2)
    acc_mag = smooth_acc(acc_mag.to_numpy())

    # 使用 scipy 的 find_peaks 做步態偵測
    peaks, _ = find_peaks(acc_mag, height=STEP_THRESHOLD, distance=10)  # distance 防止過密誤判

    headings = np.cumsum(imu_df["gyro_z"].to_numpy())
    curr_pos = Point(init_lat, init_lon)
    trajectory = [(curr_pos.latitude, curr_pos.longitude)]

    for idx in peaks:
        heading_rad = headings[idx]
        heading_deg = np.degrees(heading_rad) % 360
        curr_pos = distance(meters=FIXED_STEP_LENGTH).destination(curr_pos, heading_deg)
        trajectory.append((curr_pos.latitude, curr_pos.longitude))

    print(len(trajectory))

    return trajectory

# ------------------------------
# 主流程
# ------------------------------
# 讀取資料
wifi_df = pd.read_csv(INPUT_WIFI_CSV).rename(columns={"AppTimestamp(s)": "timestamp"})
imu_df = pd.read_csv(INPUT_IMU_CSV).rename(columns={"AppTimestamp(s)": "timestamp"})
posi_df = pd.read_csv(INPUT_GT_CSV).rename(columns={"AppTimestamp(s)": "timestamp"})
posi_df = posi_df.sort_values("timestamp").reset_index(drop=True)

aligned_data = []

for i in range(len(wifi_df)):
    wifi_time = wifi_df.loc[i, "timestamp"]
    rssi_vector = wifi_df.iloc[i, 1:].to_numpy()

    closest_gt = posi_df.iloc[(posi_df['timestamp'] - wifi_time).abs().argmin()]
    gt_lat2, gt_lon2 = closest_gt["Latitude_degrees"], closest_gt["Longitude_degrees"]

    imu_window = imu_df[
        (imu_df["timestamp"] >= wifi_time) &
        (imu_df["timestamp"] < wifi_time + IMU_WINDOW_SEC)
    ].drop(columns=["SensorTimestamp(s)"]).reset_index(drop=True)

    aligned_data.append({
        "timestamp": wifi_time,
        "rssi_vector": rssi_vector,
        "imu_window": imu_window,
        "gt_lat": None,
        "gt_lon": None,
        "gt_lat2": gt_lat2,
        "gt_lon2": gt_lon2
    })

# 根據 GT 資料補上 aligned_data 對應時間點的 gt_lat/lon
used_indices = set()
for i in range(len(aligned_data)):
    if aligned_data[i]["timestamp"] < posi_df.loc[0, "timestamp"]:
        aligned_data[i]["gt_lat"] = posi_df.loc[0, "Latitude_degrees"]
        aligned_data[i]["gt_lon"] = posi_df.loc[0, "Longitude_degrees"]
        used_indices.add(i)
for _, gt_row in posi_df.iterrows():
    gt_time = gt_row["timestamp"]
    closest_idx = min(
        (i for i in range(len(aligned_data)) if i not in used_indices),
        key=lambda i: abs(aligned_data[i]["timestamp"] - gt_time),
        default=None
    )
    if closest_idx is not None:
        aligned_data[closest_idx]["gt_lat"] = gt_row["Latitude_degrees"]
        aligned_data[closest_idx]["gt_lon"] = gt_row["Longitude_degrees"]
        used_indices.add(closest_idx)
solved = 0
# 補齊其餘時間點之GT(用imu pdr)
for d in aligned_data:
    if d["gt_lat"] is not None and d["gt_lon"] is not None:
        prev_lat = d["gt_lat"]
        prev_lon = d["gt_lon"]
        prev_imu_seq = d["imu_window"]
    if d["gt_lat"] is None or d["gt_lon"] is None:
        solved += 1
        pdr_trajectory = estimate_trajectory_from_imu_all(prev_lat, prev_lon, prev_imu_seq)
        #d["pdr_trajectory"] = pdr_trajectory
        d["gt_lat"], d["gt_lon"] = pdr_trajectory[-1]  # 末端點仍保留
    
# KNN RSSI 初始定位
rssi_features = []
gt_positions = []
for d in aligned_data:
    rssi = np.nan_to_num(d["rssi_vector"], nan=-100.0)
    rssi_features.append(rssi)
    if d["gt_lat"] is not None and d["gt_lon"] is not None:
        gt_positions.append([d["gt_lat"], d["gt_lon"]])
    else:
        gt_positions.append([0.0, 0.0])  # dummy placeholder
rssi_features = np.array(rssi_features)
gt_positions = np.array(gt_positions)

knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(rssi_features, gt_positions)

for i, d in enumerate(aligned_data):
    rssi = np.nan_to_num(d["rssi_vector"], nan=-100.0)
    init_lat, init_lon = knn.predict([rssi])[0]
    d["init_lat"] = init_lat
    d["init_lon"] = init_lon

# IMU PDR 全軌跡預測 & 推估 gt_lat/lon
for d in aligned_data:
    imu_seq = d["imu_window"]
    pdr_trajectory = estimate_trajectory_from_imu_all(d["init_lat"], d["init_lon"], imu_seq)
    d["pdr_trajectory"] = pdr_trajectory
    d["pdr_lat"], d["pdr_lon"] = pdr_trajectory[-1]  # 末端點仍保留

    # 若 gt_lat/lon 為 None，則用 PDR 末端推估值補上
    # if d["gt_lat"] is None or d["gt_lon"] is None:
    #     d["gt_lat"] = d["pdr_lat"]
    #     d["gt_lon"] = d["pdr_lon"]

# 輸出所有對齊資料為 pickle
for i, d in enumerate(aligned_data):
    with open(os.path.join(OUTPUT_DIR, f"sample_{i:04d}.pkl"), "wb") as f:
        pickle.dump(d, f)

print(f"處理完成，共輸出 {len(aligned_data)} 筆對齊資料到資料夾：{OUTPUT_DIR}")
print(solved)
