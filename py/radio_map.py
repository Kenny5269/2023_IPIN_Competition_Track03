import os
import pandas as pd
import numpy as np
import pickle
import math
from geopy.distance import distance, geodesic
from geopy import Point
from sklearn.neighbors import KNeighborsRegressor
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from ahrs.filters import Madgwick, Mahony
from scipy.spatial.transform import Rotation as R
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.svm import SVR
from numpy.linalg import norm
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# 設定參數與路徑
# ------------------------------
total = [1,2,3,4,5,21,22,23,24,25,26,27]
index1 = [1]
index2 = [24,25,26,27]
INPUT_WIFI_CSV = "py/T1_R1/WIFI_merged2.csv"
INPUT_IMU_CSV = "py/T1_R1/IMU_calibrated2.csv"
INPUT_GT_CSV = "py/T1_R1/POSI2.csv"

TEMP_WIFI_CSV = []
TEMP_IMU_CSV = []
TEMP_GT_CSV = []

# for i in index1:
#     for j in range(4):
#         INPUT_WIFI_CSV.append(f"py/T{i}_R{j+1}/WIFI_merged2.csv")
#         INPUT_IMU_CSV.append(f"py/T{i}_R{j+1}/IMU_50Hz_2.csv")
#         INPUT_GT_CSV.append(f"py/T{i}_R{j+1}/POSI2.csv")

# for i in index2:
#     TEMP_WIFI_CSV.append(f"py/T{i}_R1/WIFI_merged2.csv")
#     TEMP_IMU_CSV.append(f"py/T{i}_R1/IMU_50Hz.csv")
#     TEMP_GT_CSV.append(f"py/T{i}_R1/POSI2.csv")

# TEST_WIFI_CSV = "py/TEST1/WIFI_merged2.csv"
# TEST_IMU_CSV = "py/TEST1/IMU_calibrated2.csv"
# TEST_GT_CSV = "py/TEST1/POSI2.csv"
TEST_WIFI_CSV = "py/TEST1/WIFI_merged2.csv"
TEST_IMU_CSV = "py/TEST1/IMU_calibrated2.csv"
TEST_GT_CSV = "py/TEST1/POSI2.csv"

OUTPUT_DIR = f"py/aligned_trials/T1_R1"
TEMP_OUTPUT_DIR = f"py/aligned_trials/temp_trial"
TEST_OUTPUT_DIR = f"py/aligned_trials/TEST1"

IMU_WINDOW_SEC = 4.0
STEP_THRESHOLD = 1.2
FIXED_STEP_LENGTH = 0.5
DYNAMIC_STEP_SCALE = 0.03  # 動態步長係數，越大步越長
FUSION_METHOD = "madgwick"  # IMU與地磁融合，可選: complementary, kalman, madgwick, mahony
FUSION_STRATEGY = "avg"        # WIFI與PDR融合，可選: avg, dyn, wifi_only, pdr_only, weighted_time, average_all

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

# ------------------------------

def latlon_to_xy(lat, lon, ref_lat, ref_lon):
        R = 6371000  # 地球半徑 (m)
        dlat = np.radians(lat - ref_lat)
        dlon = np.radians(lon - ref_lon)
        x = R * dlon * np.cos(np.radians(ref_lat))
        y = R * dlat
        return np.stack([x, y], axis=-1)

def xy_to_latlon(xy, ref_lat, ref_lon):
    R = 6371000
    x, y = xy[:, 0], xy[:, 1]
    dlat = y / R
    dlon = x / (R * np.cos(np.radians(ref_lat)))
    lat = ref_lat + np.degrees(dlat)
    lon = ref_lon + np.degrees(dlon)
    return np.stack([lat, lon], axis=-1)

def compute_gt_heading(lat1, lon1, lat2, lon2):
    dLon = math.radians(lon2 - lon1)
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    y = math.sin(dLon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - \
        math.sin(lat1) * math.cos(lat2) * math.cos(dLon)
    bearing = math.atan2(y, x)
    return bearing  # 回傳的是 radian（以北為 0，順時針為正）

# 實時 RSSI 緩衝區 + 濾波預測函式
from collections import deque

def predict_with_rssi_buffer(rssi_stream, model, buffer_size=3, method='sma'):
    buffer = deque(maxlen=buffer_size)
    preds = []
    for rssi in rssi_stream:
        buffer.append(rssi)
        if len(buffer) == buffer_size:
            temp_df = pd.DataFrame(buffer, columns=[f"AP_{i}" for i in range(len(rssi))])
            smoothed = filter_rssi(temp_df, method=method).iloc[-1].to_numpy()
            pred = model.predict([smoothed])[0]
            preds.append(pred)
        else:
            preds.append(None)  # 尚未足夠資料濾波
    return preds

# ------------------------------
# RSSI 濾波函數

def filter_rssi(df, method='ema', window=3, threshold=3.0):
    filtered = df.copy()
    if method == 'sma':
        filtered.iloc[:, 1:] = filtered.iloc[:, 1:].rolling(window=window, min_periods=1).mean()
    elif method == 'median':
        filtered.iloc[:, 1:] = filtered.iloc[:, 1:].rolling(window=window, min_periods=1).median()
    elif method == 'ema':
        filtered.iloc[:, 1:] = filtered.iloc[:, 1:].ewm(span=window, adjust=True).mean()
    elif method == 'zscore':
        from scipy.stats import zscore
        z = np.abs(zscore(filtered.iloc[:, 1:], nan_policy='omit'))
        filtered.iloc[:, 1:] = filtered.iloc[:, 1:].mask(z > threshold)
        filtered = filtered.fillna(method='ffill').fillna(method='bfill')
        filtered.iloc[:, 1:] = filtered.iloc[:, 1:].rolling(window=window, min_periods=1).mean()
    elif method == 'none':
        pass  # 不做處理
    else:
        raise ValueError(f"Unsupported filter method: {method}")
    return filtered

# ------------------------------
# 輔助函數
def initialize_quaternion(acc0, mag0):
    z = acc0 / np.linalg.norm(acc0)
    x = np.cross(mag0, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot_matrix = np.vstack([x, y, z]).T
    quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]
    return np.roll(quat, 1)  # 轉為 [w, x, y, z]
# ------------------------------
def smooth_acc(acc_series, alpha=0.1):
    smoothed = [acc_series[0]]
    for a in acc_series[1:]:
        smoothed.append(alpha * a + (1 - alpha) * smoothed[-1])
    return np.array(smoothed)

def estimate_heading_complementary(gyro_z, mag_headings, dt=0.02, alpha=0.98):
    fused = [mag_headings[0]]
    for i in range(1, len(gyro_z)):
        gyro_est = fused[-1] + gyro_z[i] * dt
        fused_val = alpha * gyro_est + (1 - alpha) * mag_headings[i]
        fused.append(fused_val)
    return np.unwrap(fused)

def estimate_heading_kalman(gyro_z, mag_headings, gt_heading, dt=0.02):
    Q = 0.01  # 系統雜訊協方差
    R = 0.1   # 觀測雜訊協方差
    P = 1.0   # 初始誤差協方差
    x = mag_headings[0]  # 初始估計值（使用地磁）
    headings = [x]
    for i in range(1, len(gyro_z)):
        # 預測步驟
        x_pred = x + gyro_z[i] * dt
        P_pred = P + Q

        # 更新步驟
        K = P_pred / (P_pred + R)  # Kalman 增益
        x = x_pred + K * (mag_headings[i] - x_pred)
        P = (1 - K) * P_pred

        headings.append(x)
    return np.unwrap(headings)

def estimate_heading_madgwick(imu_df, gt_heading):
    acc = imu_df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyro = imu_df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    mag = imu_df[["mag_x", "mag_y", "mag_z"]].to_numpy()
    madgwick = Madgwick()
    qs = np.zeros((len(imu_df), 4))
    qs[0] = initialize_quaternion(acc[0], mag[0])  # 根據初始 acc + mag 推估初始四元數
    for i in range(1, len(imu_df)):
        qs[i] = madgwick.updateMARG(qs[i-1], gyr=gyro[i], acc=acc[i], mag=mag[i])
    headings = np.arctan2(2*(qs[:,0]*qs[:,3] + qs[:,1]*qs[:,2]), 1 - 2*(qs[:,2]**2 + qs[:,3]**2))
    return np.unwrap(headings)

def estimate_heading_mahony(imu_df, gt_heading):
    acc = imu_df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    gyro = imu_df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    mag = imu_df[["mag_x", "mag_y", "mag_z"]].to_numpy()
    mahony = Mahony()
    qs = np.zeros((len(imu_df), 4))    
    qs[0] = initialize_quaternion(acc[0], mag[0])  # 根據初始 acc + mag 推估初始四元數
    for i in range(1, len(imu_df)):
        qs[i] = mahony.updateMARG(qs[i-1], gyr=gyro[i], acc=acc[i], mag=mag[i])
    headings = np.arctan2(2*(qs[:,0]*qs[:,3] + qs[:,1]*qs[:,2]), 1 - 2*(qs[:,2]**2 + qs[:,3]**2))
    return np.unwrap(headings)

def estimate_trajectory_from_imu_all(aligned_data, this_idx, end_idx, imu_df, heading):
    # 真實起點、終點
    true_start = np.array([aligned_data[this_idx]["gt_lat"], aligned_data[this_idx]["gt_lon"]])
    true_end = np.array([aligned_data[end_idx]["gt_lat"], aligned_data[end_idx]["gt_lon"]])

    ref_lat, ref_lon = true_start
    # print(true_start)
    # print(true_end)

    pdr_latlon = [true_start]

    if len(imu_df) < 2:
        return [(aligned_data[this_idx]["gt_lat"], aligned_data[this_idx]["gt_lon"])]

    acc_mag = np.sqrt(imu_df["acc_x"]**2 + imu_df["acc_y"]**2 + imu_df["acc_z"]**2)
    acc_mag = smooth_acc(acc_mag.to_numpy())
    #print(acc_mag)

    # 使用 scipy 的 find_peaks 做步態偵測
    peaks, _ = find_peaks(acc_mag, height=None, distance=20, prominence=0.2)  # distance 防止過密誤判
    #print(peaks)

    plt.plot(acc_mag)
    plt.plot(peaks, acc_mag[peaks], "x")
    plt.plot(np.zeros_like(acc_mag), "--", color="gray")
    plt.show()

    gyro_z = imu_df["gyro_z"].to_numpy()
    mag_x = imu_df["mag_x"].to_numpy()
    mag_y = imu_df["mag_y"].to_numpy()
    mag_headings = np.unwrap(np.arctan2(mag_y, mag_x))

    if FUSION_METHOD == "complementary":
        headings = estimate_heading_complementary(gyro_z, heading)
    elif FUSION_METHOD == "madgwick":
        headings = estimate_heading_madgwick(imu_df, heading)
    elif FUSION_METHOD == "mahony":
        headings = estimate_heading_mahony(imu_df, heading)
    elif FUSION_METHOD == "kalman":
        headings = estimate_heading_kalman(gyro_z, mag_headings, heading)
    else:
        headings = np.cumsum(gyro_z)
    
    #headings = heading
    curr_pos = Point(aligned_data[this_idx]["gt_lat"], aligned_data[this_idx]["gt_lon"])
    # trajectory = [(curr_pos.latitude, curr_pos.longitude)]

    # 固定步長
    # for idx in peaks:
    #     heading_rad = headings[idx]
    #     heading_deg = np.degrees(heading_rad) % 360
    #     curr_pos = distance(meters=FIXED_STEP_LENGTH).destination(curr_pos, heading_deg)
    #     pdr_latlon.append(np.array([curr_pos.latitude, curr_pos.longitude]))
    #     #trajectory.append((curr_pos.latitude, curr_pos.longitude))

    # 動態步長
    # for idx in peaks:
    #     win_start = max(0, idx - 15)
    #     win_end = min(len(acc_mag), idx + 15)
    #     local_rms = np.sqrt(np.mean(acc_mag[win_start:win_end]**2))
    #     step_length = DYNAMIC_STEP_SCALE * local_rms
    #     heading_deg = np.degrees(headings[idx]) % 360
    #     curr_pos = distance(meters=step_length).destination(curr_pos, heading_deg)
    #     trajectory.append((curr_pos.latitude, curr_pos.longitude))

    # 動態步長(非RMS)
    for idx in peaks:
        win_start = max(0, idx - 15)
        win_end = min(len(acc_mag), idx + 15)
        acc_segment = acc_mag[win_start:win_end]

        # 方法 A: 頻譜能量（能量越高代表震動越強 → 步長越大）
        spectrum = np.abs(np.fft.rfft(acc_segment))
        spectral_energy = np.sum(spectrum**2)
        #print(np.sqrt(spectral_energy))
        step_length_fft = DYNAMIC_STEP_SCALE * np.sqrt(spectral_energy)

        # 方法 B: 移動平均強度
        #print(np.mean(acc_segment))
        step_length_avg = DYNAMIC_STEP_SCALE * np.mean(acc_segment)

        # 方法 C: ZUPT：如果震動小於門檻，視為靜止（不推進）
        if np.max(acc_segment) - np.min(acc_segment) < 0.05:
            step_length_zupt = 0.0
        else:
            step_length_zupt = step_length_fft  # 或用 avg 也可

        # 這裡可依需求切換用哪種
        step_length = step_length_avg
        #print(step_length)

        heading_deg = np.degrees(headings[idx]) % 360
        #geo_heading = (heading_deg + 135) % (2 * np.pi)
        curr_pos = distance(meters=step_length).destination(curr_pos, heading_deg)
        # temp = np.array(curr_pos.latitude)
        # np.append(temp, curr_pos.longitude)
        # positions_dynamic.append(temp)
        pdr_latlon.append(np.array([curr_pos.latitude, curr_pos.longitude]))

        # aligned_data[this_idx+idx]["gt_lat"] = curr_pos.latitude
        # aligned_data[this_idx+idx]["gt_lon"] = curr_pos.longitude
        # aligned_data[this_idx+idx]["gt_lat_temp"] = curr_pos.latitude
        # aligned_data[this_idx+idx]["gt_lon_temp"] = curr_pos.longitude
        #trajectory.append((curr_pos.latitude, curr_pos.longitude))

    # 經緯度轉 XY 座標
    pdr_latlon = np.asarray(pdr_latlon)
    pdr_xy = latlon_to_xy(pdr_latlon[:, 0], pdr_latlon[:, 1], ref_lat, ref_lon)
    true_start_xy = latlon_to_xy(true_start[0], true_start[1], ref_lat, ref_lon)
    true_end_xy = latlon_to_xy(true_end[0], true_end[1], ref_lat, ref_lon)
    #print(true_end_xy)

    pdr_start_xy = pdr_xy[0]
    pdr_end_xy = pdr_xy[-1]

    # 計算向量
    V_true = true_end_xy - true_start_xy
    V_pdr = pdr_end_xy - pdr_start_xy

    # 縮放與旋轉參數
    scale_factor = norm(V_true) / norm(V_pdr)
    angle_true = np.arctan2(V_true[1], V_true[0])
    angle_pdr = np.arctan2(V_pdr[1], V_pdr[0])
    rotation_angle = angle_true - angle_pdr

    # 旋轉矩陣
    R_mat = np.array([
        [np.cos(rotation_angle), -np.sin(rotation_angle)],
        [np.sin(rotation_angle),  np.cos(rotation_angle)]
    ])

    # 修正：平移→旋轉→縮放→平移回真實起點（實際為原點）
    pdr_xy_centered = pdr_xy - pdr_start_xy
    pdr_xy_rotated = pdr_xy_centered @ R_mat.T
    pdr_xy_scaled = pdr_xy_rotated * scale_factor
    pdr_xy_aligned = pdr_xy_scaled + true_start_xy

    # 回轉經緯度
    aligned_latlon = xy_to_latlon(pdr_xy_aligned, ref_lat, ref_lon)

    for i in range(len(peaks)):
        aligned_data[this_idx+peaks[i]]["gt_lat"] = aligned_latlon[i][0]
        aligned_data[this_idx+peaks[i]]["gt_lon"] = aligned_latlon[i][1]
        aligned_data[this_idx+peaks[i]]["gt_lat_temp"] = aligned_latlon[i][0]
        aligned_data[this_idx+peaks[i]]["gt_lon_temp"] = aligned_latlon[i][1]

    # return trajectory

def estimate_trajectory_from_imu_all_old(aligned_data, this_idx, end_idx, imu_df, heading):
    if len(imu_df) < 2:
        return [(aligned_data[this_idx]["gt_lat"], aligned_data[this_idx]["gt_lon"])]

    acc_mag = np.sqrt(imu_df["acc_x"]**2 + imu_df["acc_y"]**2 + imu_df["acc_z"]**2)
    acc_mag = smooth_acc(acc_mag.to_numpy())
    #print(acc_mag)

    # 使用 scipy 的 find_peaks 做步態偵測
    peaks, _ = find_peaks(acc_mag, height=STEP_THRESHOLD, distance=20, prominence=0.2)  # distance 防止過密誤判
    #print(peaks)

    plt.plot(acc_mag)
    plt.plot(peaks, acc_mag[peaks], "x")
    plt.plot(np.zeros_like(acc_mag), "--", color="gray")
    plt.show()

    gyro_z = imu_df["gyro_z"].to_numpy()
    mag_x = imu_df["mag_x"].to_numpy()
    mag_y = imu_df["mag_y"].to_numpy()
    mag_headings = np.unwrap(np.arctan2(mag_y, mag_x))

    if FUSION_METHOD == "complementary":
        headings = estimate_heading_complementary(gyro_z, heading)
    elif FUSION_METHOD == "madgwick":
        headings = estimate_heading_madgwick(imu_df, heading)
    elif FUSION_METHOD == "mahony":
        headings = estimate_heading_mahony(imu_df, heading)
    elif FUSION_METHOD == "kalman":
        headings = estimate_heading_kalman(gyro_z, mag_headings, heading)
    else:
        headings = np.cumsum(gyro_z)
    
    #headings = heading
    curr_pos = Point(aligned_data[this_idx]["gt_lat"], aligned_data[this_idx]["gt_lon"])
    # trajectory = [(curr_pos.latitude, curr_pos.longitude)]

    # 固定步長
    # for idx in peaks:
    #     heading_rad = headings[idx]
    #     heading_deg = np.degrees(heading_rad) % 360
    #     curr_pos = distance(meters=FIXED_STEP_LENGTH).destination(curr_pos, heading_deg)
    #     pdr_latlon.append(np.array([curr_pos.latitude, curr_pos.longitude]))
    #     #trajectory.append((curr_pos.latitude, curr_pos.longitude))

    # 動態步長
    # for idx in peaks:
    #     win_start = max(0, idx - 15)
    #     win_end = min(len(acc_mag), idx + 15)
    #     local_rms = np.sqrt(np.mean(acc_mag[win_start:win_end]**2))
    #     step_length = DYNAMIC_STEP_SCALE * local_rms
    #     heading_deg = np.degrees(headings[idx]) % 360
    #     curr_pos = distance(meters=step_length).destination(curr_pos, heading_deg)
    #     trajectory.append((curr_pos.latitude, curr_pos.longitude))

    # 動態步長(非RMS)
    for idx in peaks:
        win_start = max(0, idx - 15)
        win_end = min(len(acc_mag), idx + 15)
        acc_segment = acc_mag[win_start:win_end]

        # 方法 A: 頻譜能量（能量越高代表震動越強 → 步長越大）
        spectrum = np.abs(np.fft.rfft(acc_segment))
        spectral_energy = np.sum(spectrum**2)
        #print(np.sqrt(spectral_energy))
        step_length_fft = DYNAMIC_STEP_SCALE * np.sqrt(spectral_energy)

        # 方法 B: 移動平均強度
        #print(np.mean(acc_segment))
        step_length_avg = DYNAMIC_STEP_SCALE * np.mean(acc_segment)

        # 方法 C: ZUPT：如果震動小於門檻，視為靜止（不推進）
        if np.max(acc_segment) - np.min(acc_segment) < 0.05:
            step_length_zupt = 0.0
        else:
            step_length_zupt = step_length_fft  # 或用 avg 也可

        # 這裡可依需求切換用哪種
        step_length = step_length_avg

        heading_deg = np.degrees(headings[idx]) % 360
        #geo_heading = (heading_deg + 135) % (2 * np.pi)
        curr_pos = distance(meters=step_length).destination(curr_pos, heading_deg)
        aligned_data[this_idx+idx]["gt_lat"] = curr_pos.latitude
        aligned_data[this_idx+idx]["gt_lon"] = curr_pos.longitude
        aligned_data[this_idx+idx]["gt_lat_temp"] = curr_pos.latitude
        aligned_data[this_idx+idx]["gt_lon_temp"] = curr_pos.longitude
        #trajectory.append((curr_pos.latitude, curr_pos.longitude))

# Wi-Fi RSSI 初始定位模型訓練與預測函式

def train_wifi_model(X, y, method="knn"):
    if method == "knn":
        model = KNeighborsRegressor(n_neighbors=3)
    elif method == "rf":
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    elif method == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=500, random_state=42)
    elif method == "ridge":
        model = Ridge(alpha=1.0)
    elif method == "svr":
        model = SVR()
    elif method == "xgb":
        model = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    elif method == "gpr":
        model = GaussianProcessRegressor()
    else:
        raise ValueError(f"Unsupported model type: {method}")
    model.fit(X, y)
    return model

def predict_wifi_position(model, rssi):
    rssi_filled = np.nan_to_num(rssi, nan=-100.0)
    return model.predict([rssi_filled])[0]

# ------------------------------
# 主流程
# 讀取資料
# ------------------------------
# 訓練軌跡
# aligned_data_all = []

aligned_data = []

wifi_df = pd.read_csv(INPUT_WIFI_CSV).rename(columns={"AppTimestamp(s)": "timestamp"})
wifi_df = filter_rssi(wifi_df)
imu_df = pd.read_csv(INPUT_IMU_CSV).rename(columns={"AppTimestamp(s)": "timestamp"})
imu_df2 = imu_df.drop(columns=["timestamp","SensorTimestamp(s)"]).reset_index(drop=True)
posi_df = pd.read_csv(INPUT_GT_CSV).rename(columns={"AppTimestamp(s)": "timestamp"})
posi_df = posi_df.sort_values("timestamp").reset_index(drop=True)

for i in range(len(imu_df)):
    imu_time = imu_df.loc[i, "timestamp"]
    # rssi_vector = wifi_df.iloc[i, 1:].to_numpy()

    # closest_gt = posi_df.iloc[(posi_df['timestamp'] - wifi_time).abs().argmin()]
    # gt_lat2, gt_lon2 = closest_gt["Latitude_degrees"], closest_gt["Longitude_degrees"]

    # imu_data = imu_df.drop(columns=["timestamp","SensorTimestamp(s)"]).reset_index(drop=True)

    aligned_data.append({
        "init_lat" : None,
        "init_lon" : None,
        "pdr_lat" : None,
        "pdr_lon" : None,
        "fused_lat" : None,
        "fused_lon" : None,
        "timestamp": imu_time,
        "rssi_vector": None,
        "gt_lat": None,
        "gt_lon": None,
        "gt_lat_ori": None,
        "gt_lon_ori": None,
        "gt_lat_temp": None,
        "gt_lon_temp": None
    })

# 根據 GT 資料補上 aligned_data 對應時間點的 gt_lat/lon

#posi_aligned_indices = []
# for i in range(len(aligned_data)):
#     if aligned_data[i]["timestamp"] < posi_df.loc[0, "timestamp"]:
#         aligned_data[i]["gt_lat"] = posi_df.loc[0, "Latitude_degrees"]
#         aligned_data[i]["gt_lon"] = posi_df.loc[0, "Longitude_degrees"]
#     elif aligned_data[i]["timestamp"] > posi_df.loc[len(posi_df)-1, "timestamp"]:
#         aligned_data[i]["gt_lat"] = posi_df.loc[len(posi_df)-1, "Latitude_degrees"]
#         aligned_data[i]["gt_lon"] = posi_df.loc[len(posi_df)-1, "Longitude_degrees"]
gt_used_indices = []
for _, gt_row in posi_df.iterrows():
    gt_time = gt_row["timestamp"]
    closest_idx = min(
        (i for i in range(len(aligned_data)) if i not in gt_used_indices),
        key=lambda i: abs(aligned_data[i]["timestamp"] - gt_time),
        default=None
    )
    if closest_idx is not None:
        aligned_data[closest_idx]["gt_lat"] = gt_row["Latitude_degrees"]
        aligned_data[closest_idx]["gt_lon"] = gt_row["Longitude_degrees"]
        aligned_data[closest_idx]["gt_lat_ori"] = gt_row["Latitude_degrees"]
        aligned_data[closest_idx]["gt_lon_ori"] = gt_row["Longitude_degrees"]
        aligned_data[closest_idx]["gt_lat_temp"] = gt_row["Latitude_degrees"]
        aligned_data[closest_idx]["gt_lon_temp"] = gt_row["Longitude_degrees"]
        gt_used_indices.append(closest_idx)
        #posi_aligned_indices.append(closest_idx)

wifi_used_indices = []
for _, wifi_row in wifi_df.iterrows():
    wifi_time = wifi_row["timestamp"]
    closest_idx = min(
        (i for i in range(len(aligned_data)) if i not in wifi_used_indices),
        key=lambda i: abs(aligned_data[i]["timestamp"] - wifi_time),
        default=None
    )
    if closest_idx is not None:
        aligned_data[closest_idx]["rssi_vector"] = wifi_row[1:].to_numpy()
        wifi_used_indices.append(closest_idx)

# 依據已知 GT 點之間做線性插值填補其餘點
# for i in range(1, len(posi_aligned_indices)):
#     start_idx = posi_aligned_indices[i - 1]
#     end_idx = posi_aligned_indices[i]
#     start_lat, start_lon = aligned_data[start_idx]["gt_lat"], aligned_data[start_idx]["gt_lon"]
#     end_lat, end_lon = aligned_data[end_idx]["gt_lat"], aligned_data[end_idx]["gt_lon"]
#     steps = end_idx - start_idx
#     for j in range(1, steps):
#         ratio = j / steps
#         interp_lat = start_lat + ratio * (end_lat - start_lat)
#         interp_lon = start_lon + ratio * (end_lon - start_lon)
#         aligned_data[start_idx + j]["gt_lat"] = interp_lat
#         aligned_data[start_idx + j]["gt_lon"] = interp_lon

# for i in aligned_data:
#     aligned_data_all.append(i)

# 臨時測試軌跡-------------------------------------------------------------------------------------------------------------
aligned_data_temp = []
for x in range(len(TEMP_WIFI_CSV)):
    aligned_data = []

    wifi_df = pd.read_csv(TEMP_WIFI_CSV[x]).rename(columns={"AppTimestamp(s)": "timestamp"})
    wifi_df = filter_rssi(wifi_df)
    imu_df = pd.read_csv(TEMP_IMU_CSV[x]).rename(columns={"AppTimestamp(s)": "timestamp"})
    posi_df = pd.read_csv(TEMP_GT_CSV[x]).rename(columns={"AppTimestamp(s)": "timestamp"})
    posi_df = posi_df.sort_values("timestamp").reset_index(drop=True)

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
            "init_lat" : None,
            "init_lon" : None,
            "pdr_trajectory" : None,
            "pdr_lat" : None,
            "pdr_lon" : None,
            "fused_lat" : None,
            "fused_lon" : None,
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
    posi_aligned_indices = []
    # for i in range(len(aligned_data)):
    #     if aligned_data[i]["timestamp"] < posi_df.loc[0, "timestamp"]:
    #         aligned_data[i]["gt_lat"] = posi_df.loc[0, "Latitude_degrees"]
    #         aligned_data[i]["gt_lon"] = posi_df.loc[0, "Longitude_degrees"]
    #     elif aligned_data[i]["timestamp"] > posi_df.loc[len(posi_df)-1, "timestamp"]:
    #         aligned_data[i]["gt_lat"] = posi_df.loc[len(posi_df)-1, "Latitude_degrees"]
    #         aligned_data[i]["gt_lon"] = posi_df.loc[len(posi_df)-1, "Longitude_degrees"]
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
            posi_aligned_indices.append(closest_idx)

    # 依據已知 GT 點之間做線性插值填補其餘點
    # for i in range(1, len(posi_aligned_indices)):
    #     start_idx = posi_aligned_indices[i - 1]
    #     end_idx = posi_aligned_indices[i]
    #     start_lat, start_lon = aligned_data[start_idx]["gt_lat"], aligned_data[start_idx]["gt_lon"]
    #     end_lat, end_lon = aligned_data[end_idx]["gt_lat"], aligned_data[end_idx]["gt_lon"]
    #     steps = end_idx - start_idx
    #     for j in range(1, steps):
    #         ratio = j / steps
    #         interp_lat = start_lat + ratio * (end_lat - start_lat)
    #         interp_lon = start_lon + ratio * (end_lon - start_lon)
    #         aligned_data[start_idx + j]["gt_lat"] = interp_lat
    #         aligned_data[start_idx + j]["gt_lon"] = interp_lon

    for i in aligned_data:
        aligned_data_temp.append(i)

# 測試軌跡 -----------------------------------------------------------------------------------------------------------------
wifi_df_test = pd.read_csv(TEST_WIFI_CSV).rename(columns={"AppTimestamp(s)": "timestamp"})
wifi_df_test = filter_rssi(wifi_df_test)
imu_df_test = pd.read_csv(TEST_IMU_CSV).rename(columns={"AppTimestamp(s)": "timestamp"})
imu_df2_test = imu_df_test.drop(columns=["timestamp","SensorTimestamp(s)"]).reset_index(drop=True)
posi_df_test = pd.read_csv(TEST_GT_CSV).rename(columns={"AppTimestamp(s)": "timestamp"})
posi_df_test = posi_df_test.sort_values("timestamp").reset_index(drop=True)

test_aligned_data = []

for i in range(0):
    imu_time = imu_df_test.loc[i, "timestamp"]
    # rssi_vector = wifi_df.iloc[i, 1:].to_numpy()

    # closest_gt = posi_df.iloc[(posi_df['timestamp'] - wifi_time).abs().argmin()]
    # gt_lat2, gt_lon2 = closest_gt["Latitude_degrees"], closest_gt["Longitude_degrees"]

    # imu_data = imu_df.drop(columns=["timestamp","SensorTimestamp(s)"]).reset_index(drop=True)

    test_aligned_data.append({
        "init_lat" : None,
        "init_lon" : None,
        "pdr_lat" : None,
        "pdr_lon" : None,
        "fused_lat" : None,
        "fused_lon" : None,
        "timestamp": imu_time,
        "rssi_vector": None,
        "gt_lat": None,
        "gt_lon": None,
        "gt_lat_ori": None,
        "gt_lon_ori": None,
        "gt_lat_temp": None,
        "gt_lon_temp": None
    })

gt_used_indices_test = []
for _, gt_row in posi_df_test.iterrows():
    gt_time = gt_row["timestamp"]
    closest_idx = min(
        (i for i in range(len(test_aligned_data)) if i not in gt_used_indices_test),
        key=lambda i: abs(test_aligned_data[i]["timestamp"] - gt_time),
        default=None
    )
    if closest_idx is not None:
        test_aligned_data[closest_idx]["gt_lat"] = gt_row["Latitude_degrees"]
        test_aligned_data[closest_idx]["gt_lon"] = gt_row["Longitude_degrees"]
        # test_aligned_data[closest_idx]["gt_lat_ori"] = gt_row["Latitude_degrees"]
        # test_aligned_data[closest_idx]["gt_lon_ori"] = gt_row["Longitude_degrees"]
        # alignedtest_aligned_data_data[closest_idx]["gt_lat_temp"] = gt_row["Latitude_degrees"]
        # test_aligned_data[closest_idx]["gt_lon_temp"] = gt_row["Longitude_degrees"]
        gt_used_indices_test.append(closest_idx)
        #posi_aligned_indices.append(closest_idx)

wifi_used_indices_test = []
for _, wifi_row in wifi_df_test.iterrows():
    wifi_time = wifi_row["timestamp"]
    closest_idx = min(
        (i for i in range(len(test_aligned_data)) if i not in wifi_used_indices_test),
        key=lambda i: abs(test_aligned_data[i]["timestamp"] - wifi_time),
        default=None
    )
    if closest_idx is not None:
        test_aligned_data[closest_idx]["rssi_vector"] = wifi_row[1:].to_numpy()
        wifi_used_indices_test.append(closest_idx)

# WIFI RSSI 初始定位
rssi_features = []
gt_positions = []
# for d in aligned_data_all:
#     if d["gt_lat"] is not None and d["gt_lon"] is not None:
#         rssi = np.nan_to_num(d["rssi_vector"], nan=-100.0)
#         rssi_features.append(rssi)
#         gt_positions.append([d["gt_lat"], d["gt_lon"]])

# rssi_features = np.array(rssi_features)
# gt_positions = np.array(gt_positions)

# wifi_model = train_wifi_model(rssi_features, gt_positions)

gt_coords_origin = [(d["gt_lat_ori"], d["gt_lon_ori"]) for d in aligned_data if d["gt_lat_ori"] is not None]
gt_lats_ori, gt_lons_ori = zip(*gt_coords_origin)
# if aligned_data[j]["gt_lat_ori"] is not None:
#     gt_coords_origin.append((aligned_data[j]["gt_lat_ori"], aligned_data[j]["gt_lon_ori"]))

for i in range(len(gt_used_indices)-1):
    this_gt_index = gt_used_indices[i]
    #print(this_gt_index)
    next_gt_index = gt_used_indices[i+1]
    #print(next_gt_index)
    imu_seq = imu_df2[this_gt_index : next_gt_index]
    # for j in range(this_gt_index+1, next_gt_index):
    #     imu_seq = pd.concat([imu_seq, aligned_data[j]["imu_data"]])
    #     print(imu_seq)
    #     break
    #print(len(imu_seq))
    print(geodesic((aligned_data[this_gt_index]["gt_lat_ori"], aligned_data[this_gt_index]["gt_lon_ori"]), (aligned_data[next_gt_index]["gt_lat_ori"], aligned_data[next_gt_index]["gt_lon_ori"])).meters)
    gt_heading = compute_gt_heading(aligned_data[this_gt_index]["gt_lat_ori"], aligned_data[this_gt_index]["gt_lon_ori"], aligned_data[next_gt_index]["gt_lat_ori"], aligned_data[next_gt_index]["gt_lon_ori"])
    #gt_heading = np.arctan2(aligned_data[next_gt_index]["gt_lat_ori"] - aligned_data[this_gt_index]["gt_lat_ori"],
    #                        aligned_data[next_gt_index]["gt_lon_ori"] - aligned_data[this_gt_index]["gt_lon_ori"])
    estimate_trajectory_from_imu_all(aligned_data, this_gt_index, next_gt_index, imu_seq, gt_heading)

    # 畫出軌跡圖
    
    gt_coords = []
    for j in range(this_gt_index, next_gt_index):
        
        if aligned_data[j]["gt_lat"] is not None:
            gt_coords.append((aligned_data[j]["gt_lat"], aligned_data[j]["gt_lon"]))

    
    gt_lats, gt_lons = zip(*gt_coords)

    plt.plot(gt_lons_ori, gt_lats_ori, label="Ground Truth origin", marker="o")
    plt.plot(gt_lons, gt_lats, label="Ground Truth", marker="o")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Wi-Fi Init vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

for i in range(1, len(gt_used_indices)):
    start_idx = gt_used_indices[i - 1]
    end_idx = gt_used_indices[i]
    start_lat, start_lon = aligned_data[start_idx]["gt_lat_temp"], aligned_data[start_idx]["gt_lon_temp"]
    end_lat, end_lon = aligned_data[end_idx]["gt_lat_temp"], aligned_data[end_idx]["gt_lon_temp"]
    steps = end_idx - start_idx
    for j in range(1, steps):
        ratio = j / steps
        interp_lat = start_lat + ratio * (end_lat - start_lat)
        interp_lon = start_lon + ratio * (end_lon - start_lon)
        aligned_data[start_idx + j]["gt_lat_temp"] = interp_lat
        aligned_data[start_idx + j]["gt_lon_temp"] = interp_lon

for d in aligned_data:
    if d["rssi_vector"] is not None and d["gt_lat_temp"] is not None:
        rssi = np.nan_to_num(d["rssi_vector"], nan=-100.0)
        rssi_features.append(rssi)
        gt_positions.append([d["gt_lat_temp"], d["gt_lon_temp"]])

rssi_features = np.array(rssi_features)
gt_positions = np.array(gt_positions)

wifi_model = train_wifi_model(rssi_features, gt_positions)

'''
for i, d in enumerate(aligned_data_all):
    rssi = np.nan_to_num(d["rssi_vector"], nan=-100.0)
    init_lat, init_lon = predict_wifi_position(wifi_model, rssi)
    imu_seq = d["imu_data"]
  
    d["init_lat"] = init_lat
    d["init_lon"] = init_lon

    if i == 0:
        d["fused_lat"] = init_lat
        d["fused_lon"] = init_lon
        pdr_trajectory = estimate_trajectory_from_imu_all(d["fused_lat"], d["fused_lon"], imu_seq)
        d["pdr_trajectory"] = pdr_trajectory
        d["pdr_lat"], d["pdr_lon"] = pdr_trajectory[-1]
        continue

    # 融合 init 與上一步 pdr 座標
    fusion_strategy = FUSION_STRATEGY

    prev = aligned_data_all[i - 1]
    prev_pdr_lat, prev_pdr_lon = prev["pdr_lat"], prev["pdr_lon"]
    
    if fusion_strategy == "avg":
        alpha = 0.7
        fused_lat = alpha * init_lat + (1 - alpha) * prev_pdr_lat
        fused_lon = alpha * init_lon + (1 - alpha) * prev_pdr_lon
    elif fusion_strategy == "dyn":
        dist_wifi = np.linalg.norm([init_lat - prev_pdr_lat, init_lon - prev_pdr_lon])
        alpha = np.clip(1 - dist_wifi / 10.0, 0.0, 1.0)
        fused_lat = alpha * init_lat + (1 - alpha) * prev_pdr_lat
        fused_lon = alpha * init_lon + (1 - alpha) * prev_pdr_lon
    elif fusion_strategy == "wifi_only":
        fused_lat, fused_lon = init_lat, init_lon
    elif fusion_strategy == "pdr_only":
        fused_lat, fused_lon = prev_pdr_lat, prev_pdr_lon
    elif fusion_strategy == "weighted_time":
        dt = d["timestamp"] - prev["timestamp"]
        alpha = np.exp(-dt / 3.0)  # 根據時間差做指數衰減
        fused_lat = alpha * init_lat + (1 - alpha) * prev_pdr_lat
        fused_lon = alpha * init_lon + (1 - alpha) * prev_pdr_lon
    elif fusion_strategy == "average_all":
        # 將 init_lat/lon、prev_pdr_lat/lon、prev["init_lat/lon"] 平均融合
        prev_init_lat, prev_init_lon = prev["init_lat"], prev["init_lon"]
        fused_lat = np.mean([init_lat, prev_init_lat, prev_pdr_lat])
        fused_lon = np.mean([init_lon, prev_init_lon, prev_pdr_lon])
    else:
        raise ValueError("Unsupported fusion strategy")

    # 寫入選擇的主融合輸出
    d["fused_lat"] = fused_lat
    d["fused_lon"] = fused_lon
    
    pdr_trajectory = estimate_trajectory_from_imu_all(d["fused_lat"], d["fused_lon"], imu_seq)
    d["pdr_trajectory"] = pdr_trajectory
    d["pdr_lat"], d["pdr_lon"] = pdr_trajectory[-1]
'''

# WIFI RSSI 初始定位(臨時測試軌跡)
first = False
for i, d in enumerate(aligned_data_temp):
    rssi = np.nan_to_num(d["rssi_vector"], nan=-100.0)
    init_lat, init_lon = predict_wifi_position(wifi_model, rssi)
    imu_seq = d["imu_data"]
  
    d["init_lat"] = init_lat
    d["init_lon"] = init_lon

    if not first and d["gt_lat"] is not None:
        first = True
        print('first')
        d["fused_lat"] = init_lat
        d["fused_lon"] = init_lon
        pdr_trajectory = estimate_trajectory_from_imu_all(d["fused_lat"], d["fused_lon"], imu_seq)
        d["pdr_trajectory"] = pdr_trajectory
        d["pdr_lat"], d["pdr_lon"] = pdr_trajectory[-1]
        continue
    elif not first:
        print('waiting')
        continue

    # 融合 init 與上一步 pdr 座標
    fusion_strategy = FUSION_STRATEGY

    prev = aligned_data_temp[i - 1]
    prev_pdr_lat, prev_pdr_lon = prev["pdr_lat"], prev["pdr_lon"]
    
    if fusion_strategy == "avg":
        alpha = 0.7
        fused_lat = alpha * init_lat + (1 - alpha) * prev_pdr_lat
        fused_lon = alpha * init_lon + (1 - alpha) * prev_pdr_lon
    elif fusion_strategy == "dyn":
        dist_wifi = np.linalg.norm([init_lat - prev_pdr_lat, init_lon - prev_pdr_lon])
        alpha = np.clip(1 - dist_wifi / 10.0, 0.0, 1.0)
        fused_lat = alpha * init_lat + (1 - alpha) * prev_pdr_lat
        fused_lon = alpha * init_lon + (1 - alpha) * prev_pdr_lon
    elif fusion_strategy == "wifi_only":
        fused_lat, fused_lon = init_lat, init_lon
    elif fusion_strategy == "pdr_only":
        fused_lat, fused_lon = prev_pdr_lat, prev_pdr_lon
    elif fusion_strategy == "weighted_time":
        dt = d["timestamp"] - prev["timestamp"]
        alpha = np.exp(-dt / 3.0)  # 根據時間差做指數衰減
        fused_lat = alpha * init_lat + (1 - alpha) * prev_pdr_lat
        fused_lon = alpha * init_lon + (1 - alpha) * prev_pdr_lon
    elif fusion_strategy == "average_all":
        # 將 init_lat/lon、prev_pdr_lat/lon、prev["init_lat/lon"] 平均融合
        prev_init_lat, prev_init_lon = prev["init_lat"], prev["init_lon"]
        fused_lat = np.mean([init_lat, prev_init_lat, prev_pdr_lat])
        fused_lon = np.mean([init_lon, prev_init_lon, prev_pdr_lon])
    else:
        raise ValueError("Unsupported fusion strategy")

    # 寫入選擇的主融合輸出
    d["fused_lat"] = fused_lat
    d["fused_lon"] = fused_lon
    
    pdr_trajectory = estimate_trajectory_from_imu_all(d["fused_lat"], d["fused_lon"], imu_seq)
    d["pdr_trajectory"] = pdr_trajectory
    d["pdr_lat"], d["pdr_lon"] = pdr_trajectory[-1]

# WIFI RSSI 初始定位(測試軌跡)
first = False
for i, d in enumerate(test_aligned_data):
    rssi = np.nan_to_num(d["rssi_vector"], nan=-100.0)
    init_lat, init_lon = predict_wifi_position(wifi_model, rssi)
    imu_seq = d["imu_data"]
  
    d["init_lat"] = init_lat
    d["init_lon"] = init_lon

    if not first and d["gt_lat"] is not None:
        first = True
        print('first')
        d["fused_lat"] = init_lat
        d["fused_lon"] = init_lon
        pdr_trajectory = estimate_trajectory_from_imu_all(d["fused_lat"], d["fused_lon"], imu_seq)
        d["pdr_trajectory"] = pdr_trajectory
        d["pdr_lat"], d["pdr_lon"] = pdr_trajectory[-1]
        continue
    elif not first:
        print('waiting')
        continue

    # 融合 init 與上一步 pdr 座標
    fusion_strategy = FUSION_STRATEGY

    prev = test_aligned_data[i - 1]
    prev_pdr_lat, prev_pdr_lon = prev["pdr_lat"], prev["pdr_lon"]
    
    if fusion_strategy == "avg":
        alpha = 0.7
        fused_lat = alpha * init_lat + (1 - alpha) * prev_pdr_lat
        fused_lon = alpha * init_lon + (1 - alpha) * prev_pdr_lon
    elif fusion_strategy == "dyn":
        dist_wifi = np.linalg.norm([init_lat - prev_pdr_lat, init_lon - prev_pdr_lon])
        alpha = np.clip(1 - dist_wifi / 10.0, 0.0, 1.0)
        fused_lat = alpha * init_lat + (1 - alpha) * prev_pdr_lat
        fused_lon = alpha * init_lon + (1 - alpha) * prev_pdr_lon
    elif fusion_strategy == "wifi_only":
        fused_lat, fused_lon = init_lat, init_lon
    elif fusion_strategy == "pdr_only":
        fused_lat, fused_lon = prev_pdr_lat, prev_pdr_lon
    elif fusion_strategy == "weighted_time":
        dt = d["timestamp"] - prev["timestamp"]
        alpha = np.exp(-dt / 3.0)  # 根據時間差做指數衰減
        fused_lat = alpha * init_lat + (1 - alpha) * prev_pdr_lat
        fused_lon = alpha * init_lon + (1 - alpha) * prev_pdr_lon
    elif fusion_strategy == "average_all":
        # 將 init_lat/lon、prev_pdr_lat/lon、prev["init_lat/lon"] 平均融合
        prev_init_lat, prev_init_lon = prev["init_lat"], prev["init_lon"]
        fused_lat = np.mean([init_lat, prev_init_lat, prev_pdr_lat])
        fused_lon = np.mean([init_lon, prev_init_lon, prev_pdr_lon])
    else:
        raise ValueError("Unsupported fusion strategy")

    # 寫入選擇的主融合輸出
    d["fused_lat"] = fused_lat
    d["fused_lon"] = fused_lon
    
    pdr_trajectory = estimate_trajectory_from_imu_all(d["fused_lat"], d["fused_lon"], imu_seq)
    d["pdr_trajectory"] = pdr_trajectory
    d["pdr_lat"], d["pdr_lon"] = pdr_trajectory[-1]

# 輸出所有對齊資料為 pickle
# for i, d in enumerate(aligned_data):
#     with open(os.path.join(OUTPUT_DIR, f"sample_{i:04d}.pkl"), "wb") as f:
#         pickle.dump(d, f)
# with open(os.path.join(OUTPUT_DIR, f"all_data.pkl"), "wb") as f:
#     pickle.dump(aligned_data, f)

# 輸出所有對齊資料為 pickle
# for i, d in enumerate(aligned_data_temp):
#     with open(os.path.join(TEMP_OUTPUT_DIR, f"sample_{i:04d}.pkl"), "wb") as f:
#         pickle.dump(d, f)

# 輸出所有對齊資料為 pickle
# for i, d in enumerate(test_aligned_data):
#     with open(os.path.join(TEST_OUTPUT_DIR, f"sample_{i:04d}.pkl"), "wb") as f:
#         pickle.dump(d, f)
# with open(os.path.join(OUTPUT_DIR, f"all_data.pkl"), "wb") as f:
#     pickle.dump(test_aligned_data, f)

# print(f"處理完成，共輸出 {len(aligned_data)} 筆對齊資料到資料夾：{OUTPUT_DIR}")
# print(f"處理完成，共輸出 {len(aligned_data_temp)} 筆對齊資料到資料夾：{TEMP_OUTPUT_DIR}")
# print(f"處理完成，共輸出 {len(test_aligned_data)} 筆對齊資料到資料夾：{TEST_OUTPUT_DIR}")
