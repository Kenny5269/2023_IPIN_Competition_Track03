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
from pyproj import Transformer
from geomag import declination
from scipy.ndimage import uniform_filter1d
import warnings
warnings.filterwarnings("ignore")

# ------------------------------
# 設定參數與路徑
# ------------------------------
total = [1,2,3,4,5,21,22,23,24,25,26,27]
index1 = [1]
index2 = [24,25,26,27]
input_file = 'T1_R1'
INPUT_WIFI_CSV = f'py/{input_file}/WIFI_merged_filtered.csv'
INPUT_IMU_CSV = f'py/{input_file}/IMU_calibrated3_temp.csv'
INPUT_GT_CSV = f'py/{input_file}/POSI2.csv'

TEMP_WIFI_CSV = []
TEMP_IMU_CSV = []
TEMP_GT_CSV = []

# for i in index1:
#     for j in range(4):
#         INPUT_WIFI_CSV.append(f"py/T{i}_R{j+1}/WIFI_merged_filtered.csv")
#         INPUT_IMU_CSV.append(f"py/T{i}_R{j+1}/IMU_50Hz_2.csv")
#         INPUT_GT_CSV.append(f"py/T{i}_R{j+1}/POSI2.csv")

# for i in index2:
#     TEMP_WIFI_CSV.append(f"py/T{i}_R1/WIFI_merged_filtered.csv")
#     TEMP_IMU_CSV.append(f"py/T{i}_R1/IMU_50Hz.csv")
#     TEMP_GT_CSV.append(f"py/T{i}_R1/POSI2.csv")

# TEST_WIFI_CSV = "py/TEST1/WIFI_merged_filtered.csv"
# TEST_IMU_CSV = "py/TEST1/IMU_calibrated2.csv"
# TEST_GT_CSV = "py/TEST1/POSI2.csv"

test_file = 'TEST1'
TEST_WIFI_CSV = f"py/{test_file}/WIFI_merged_filtered.csv"
TEST_IMU_CSV = f"py/{test_file}/IMU_calibrated3_temp.csv"
TEST_GT_CSV = f"py/{test_file}/POSI2.csv"

# TEST_WIFI_CSV = f'py/{input_file}/WIFI_merged_filtered.csv'
# TEST_IMU_CSV = f'py/{input_file}/IMU_calibrated3_temp.csv'
# TEST_GT_CSV = f'py/{input_file}/POSI2.csv'

OUTPUT_DIR = f'py/aligned_trials/{input_file}'
TEMP_OUTPUT_DIR = f'py/aligned_trials/temp_trial'
TEST_OUTPUT_DIR = f'py/aligned_trials/{test_file}'

IMU_WINDOW_SEC = 4.0
STEP_THRESHOLD = 0.7
FIXED_STEP_LENGTH = 0.5
DYNAMIC_STEP_SCALE = 0.9  # 動態步長係數，越大步越長
FUSION_METHOD = "madgwick"  # IMU與地磁融合，可選: complementary, kalman, madgwick, mahony
FUSION_STRATEGY = "avg"        # WIFI與PDR融合，可選: avg, dyn, wifi_only, pdr_only, weighted_time, average_all

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TEMP_OUTPUT_DIR, exist_ok=True)
os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

# class EKF_Localizer:
#     def __init__(self, init_pos, init_heading, process_noise_std=0.5, obs_noise_std=2.0):
#         self.x = np.array([init_pos[0], init_pos[1], init_heading], dtype=np.float64)  # 明確指定 dtype
#         self.P = np.eye(3) * 1.0
#         self.Q = np.diag([process_noise_std**2]*2 + [np.deg2rad(5)**2])
#         self.R = np.diag([obs_noise_std**2, obs_noise_std**2])

#     def predict(self, step_length, heading_delta):
#         theta = self.x[2] + heading_delta
#         dx = step_length * np.cos(theta)
#         dy = step_length * np.sin(theta)

#         self.x[0] += dx
#         self.x[1] += dy
#         self.x[2] = theta

#         F = np.array([
#             [1, 0, -step_length * np.sin(theta)],
#             [0, 1,  step_length * np.cos(theta)],
#             [0, 0, 1]
#         ])

#         self.P = F @ self.P @ F.T + self.Q

#     def update(self, z):
#         H = np.array([
#             [1, 0, 0],
#             [0, 1, 0]
#         ])
#         y = z - H @ self.x
#         S = H @ self.P @ H.T + self.R
#         K = self.P @ H.T @ np.linalg.inv(S)
#         self.x += K @ y
#         self.P = (np.eye(3) - K @ H) @ self.P

#     def get_state(self):
#         return self.x.copy()

# ------------------------------

class EKF_Localizer_xy:
    def __init__(self, init_pos, init_heading_deg, process_noise_std=0.5, obs_noise_std=2.0):
        # 初始狀態: x, y, heading (以度為單位)
        self.x = np.array([init_pos[0], init_pos[1], init_heading_deg], dtype=np.float64)
        self.P = np.eye(3) * 1.0
        self.Q = np.diag([process_noise_std**2]*2 + [5.0**2])  # 角度雜訊單位為度
        self.R = np.diag([obs_noise_std**2, obs_noise_std**2])

    def predict(self, step_length, heading_delta_deg):
        # 預測：以 degree 計算，再轉為 rad 做三角運算
        theta_deg = self.x[2] + heading_delta_deg
        theta_rad = np.deg2rad(theta_deg)

        dx = step_length * np.cos(theta_rad)
        dy = step_length * np.sin(theta_rad)

        self.x[0] += dx
        self.x[1] += dy
        self.x[2] = theta_deg % 360  # 保持 heading 在 0~360

        F = np.array([
            [1, 0, -step_length * np.sin(theta_rad)],
            [0, 1,  step_length * np.cos(theta_rad)],
            [0, 0, 1]
        ])

        self.P = F @ self.P @ F.T + self.Q

    def update(self, z):
        H = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x += K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

    def get_state(self):
        return self.x.copy()
    
class EKF_Localizer_latlon:
    def __init__(self, init_latlon, init_heading_deg, process_noise_std=0.5, obs_noise_std=2.0):
        """
        init_latlon: 初始位置 (latitude, longitude)
        init_heading_deg: 初始朝向角（以度為單位）
        """
        self.x = np.array([init_latlon[0], init_latlon[1], init_heading_deg], dtype=np.float64)
        self.P = np.eye(3) * 1.0
        self.Q = np.diag([process_noise_std**2]*2 + [2.5**2])  # 緯度、經度誤差 + heading (deg)
        self.R = np.diag([obs_noise_std**2, obs_noise_std**2])  # 觀測誤差

    def predict(self, step_length, heading_deg):
        """
        使用「絕對 heading」進行位置預測（0~360 度，以真北為 0）
        """
        # theta_deg = heading_abs_deg % 360
        # theta_rad = np.deg2rad(theta_deg)

        curr_pos = (self.x[0], self.x[1])  # (lat, lon)
        new_pos = distance(meters=step_length).destination(curr_pos, heading_deg)

        self.x[0] = new_pos.latitude
        self.x[1] = new_pos.longitude
        self.x[2] = heading_deg  # 直接設為當前絕對角度

        F = np.eye(3)  # 線性近似，不更新 Jacobian
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z_latlon):
        """
        使用觀測位置（lat, lon）來進行 EKF 更新
        """
        H = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])
        z = np.array([z_latlon[0], z_latlon[1]])
        y = z - H @ self.x
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x += K @ y
        self.P = (np.eye(3) - K @ H) @ self.P

    def get_state(self):
        return self.x.copy()  # [latitude, longitude, heading_deg]

def estimate_step_length_from_world_acc(df, step_index, window=20, method='instant'):
    """
    根據世界座標下的 acc_x, acc_y 資料估計步長。

    參數:
        df (pd.DataFrame): 包含 'acc_x', 'acc_y' 欄位的資料，應為世界座標下加速度
        step_index (int): 此次步態發生的中心索引（如 peak index）
        window (int): 使用的資料窗口大小（總長度 = 2*window + 1）
        method (str): 使用 'instant' 或 'integrate' 方法計算步長

    回傳:
        float: 預估步長（單位：任意，視原始 acc 單位與 dt 而定）
    """
    if 'acc_x' not in df.columns or 'acc_y' not in df.columns:
        raise ValueError("資料中必須包含 'acc_x' 與 'acc_y' 欄位")

    start = max(0, step_index - window)
    end = min(len(df), step_index + window + 1)
    acc_xy = df[['acc_x', 'acc_y']].iloc[start:step_index+9].to_numpy()

    if method == 'instant':
        # 使用該中心點的加速度分量
        center_idx = min(window, len(acc_xy) - 1)
        acc_vec = acc_xy[center_idx]
        step_length = np.linalg.norm(acc_vec)

    elif method == 'integrate':
        # 積分加速度 → 粗略估算速度 → 估步長
        dt = 1 / 50  # 預設 50Hz
        # print(acc_xy)
        vel_xy = np.cumsum(np.abs(acc_xy) * dt, axis=0)
        # print(vel_xy)
        pos_xy = np.cumsum(vel_xy * dt, axis=0)
        step_length = np.linalg.norm(pos_xy[-1] - pos_xy[0]) * 6

        # delta_vel = vel_xy[-1] - vel_xy[0]
        # step_length = np.linalg.norm(delta_vel) * 0.02 * window

    else:
        raise ValueError("method 必須是 'instant' 或 'integrate'")

    return step_length

def estimate_step_length_with_zupt(df, start, end, dt=1/50, gyro_thresh=1.0, turning_scale=0.3):
    """
    使用 acc_x, acc_y 進行雙重積分計算每一步步長，並根據 gyro_z 判斷是否為轉向步。
    每一步起點和終點由 peaks 指定。使用 ZUPT 將每一步的初始速度視為 0。

    Args:
        acc_x (np.ndarray): 世界座標系下 x 軸加速度
        acc_y (np.ndarray): 世界座標系下 y 軸加速度
        gyro_z (np.ndarray): 世界座標系下 z 軸角速度（判斷轉向用）
        peaks (np.ndarray): 步態波峰的 index
        dt (float): 時間間隔，預設 1/50 (50Hz)
        gyro_thresh (float): 判斷轉向的角速度閾值（單位：rad/s）
        turning_scale (float): 若為轉向步，步長縮小的倍率

    Returns:
        step_lengths (list of float): 每一步的估算步長
        turn_flags (list of bool): 每一步是否為轉向
    """
    step_lengths = []
    turn_flags = []
    if (end - start) > 50:
        start = end - 30
    # start = max(0, end - 20)
    # end = min(len(df), end + 20)

    ax = df[['acc_x']].iloc[start:end].to_numpy()
    ay = df[['acc_y']].iloc[start:end].to_numpy()
    gz = df[['gyro_z']].iloc[start:end].to_numpy()

    ax = np.abs(ax)
    ay = np.abs(ay)

    # ZUPT：假設每一步速度初始為 0
    vx = np.zeros_like(ax)
    vy = np.zeros_like(ay)

    for j in range(1, len(ax)):
        vx[j] = vx[j - 1] + (ax[j] + ax[j - 1]) / 2 * dt
        vy[j] = vy[j - 1] + (ay[j] + ay[j - 1]) / 2 * dt
    
    # print(f'vx = {vx}')
    # print(f'vy = {vy}')

    px = np.zeros_like(vx)
    py = np.zeros_like(vy)

    for j in range(1, len(vx)):
        px[j] = px[j - 1] + (vx[j] + vx[j - 1]) / 2 * dt
        py[j] = py[j - 1] + (vy[j] + vy[j - 1]) / 2 * dt

    # print(f'px = {px}')
    # print(f'py = {py}')

    # 計算總位移（向量長度）
    disp = np.linalg.norm([px[-1], py[-1]])

    # 判斷是否為轉向步
    turning = np.max(np.abs(gz)) > gyro_thresh
    if turning:
        print('turning')
        disp *= turning_scale
    else:
        disp *= 6

    # disp *= 6

    return disp

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

def estimate_heading_kalman(gyro_z, mag_headings, dt=0.02):
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

def estimate_heading_madgwick(imu_df):
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

def estimate_heading_mahony(imu_df):
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

    # 世界座標加速度
    acc_x_world = imu_df['acc_x'].values
    acc_y_world = imu_df['acc_y'].values
    acc_z_world = imu_df['acc_z'].values

    # 使用 scipy 的 find_peaks 做步態偵測
    peaks, _ = find_peaks(acc_z_world, height=STEP_THRESHOLD, distance=20, prominence=0.4)  # distance 防止過密誤判
    #print(peaks)

    # plt.plot(acc_mag)
    # plt.plot(peaks, acc_mag[peaks], "x")
    # plt.plot(np.zeros_like(acc_mag), "--", color="gray")
    # plt.show()

    gyro_z = imu_df["gyro_z"].to_numpy()

    # headings = imu_df['yaw_deg'].to_numpy()
    headings = imu_df['yaw_deg_ori'].to_numpy()
    # headings = imu_df['yaw_deg_ori_cal'].to_numpy()

    # mag_x = imu_df["mag_x"].to_numpy()
    # mag_y = imu_df["mag_y"].to_numpy()
    # mag_headings = np.unwrap(np.arctan2(mag_y, mag_x))

    # if FUSION_METHOD == "complementary":
    #     headings = estimate_heading_complementary(gyro_z, mag_headings)
    # elif FUSION_METHOD == "madgwick":
    #     headings = estimate_heading_madgwick(imu_df)
    # elif FUSION_METHOD == "mahony":
    #     headings = estimate_heading_mahony(imu_df)
    # elif FUSION_METHOD == "kalman":
    #     headings = estimate_heading_kalman(gyro_z, mag_headings)
    # else:
    #     headings = np.cumsum(gyro_z)
    
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
        # step_length = step_length_avg
        step_length = estimate_step_length_from_world_acc(imu_df, idx, method='integrate') if estimate_step_length_from_world_acc(imu_df, idx, method='integrate') <= 0.8 else 0.8
        #print(step_length)
        print(f'{aligned_data[this_idx+idx]["timestamp"]}, step_length = {step_length}')

        heading_deg = (-headings[idx] + 360) % 360

        # heading_deg = np.degrees(headings[idx]) % 360
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

    # 記錄PDR的index
    pdr_idx = [this_idx]

    for i in range(len(peaks)):
        aligned_data[this_idx+peaks[i]]["gt_lat"] = aligned_latlon[i+1][0]
        aligned_data[this_idx+peaks[i]]["gt_lon"] = aligned_latlon[i+1][1]
        aligned_data[this_idx+peaks[i]]["gt_lat_temp"] = aligned_latlon[i+1][0]
        aligned_data[this_idx+peaks[i]]["gt_lon_temp"] = aligned_latlon[i+1][1]

        pdr_idx.append(this_idx+peaks[i])

    pdr_idx.append(end_idx)

    for i in range(len(pdr_idx)-1):
        start_idx = pdr_idx[i]
        end_idx = pdr_idx[i+1]
        start_lat, start_lon = aligned_data[start_idx]["gt_lat_temp"], aligned_data[start_idx]["gt_lon_temp"]
        end_lat, end_lon = aligned_data[end_idx]["gt_lat_temp"], aligned_data[end_idx]["gt_lon_temp"]
        steps = end_idx - start_idx
        for j in range(1, steps):
            ratio = j / steps
            interp_lat = start_lat + ratio * (end_lat - start_lat)
            interp_lon = start_lon + ratio * (end_lon - start_lon)
            aligned_data[start_idx + j]["gt_lat_temp"] = interp_lat
            aligned_data[start_idx + j]["gt_lon_temp"] = interp_lon

    # return trajectory

def estimate_trajectory_from_imu_all_old(aligned_data, this_idx, end_idx, imu_df, heading):
    interval = geodesic((aligned_data[this_idx]["gt_lat_ori"], aligned_data[this_idx]["gt_lon_ori"]), (aligned_data[end_idx]["gt_lat_ori"], aligned_data[end_idx]["gt_lon_ori"])).meters
    if len(imu_df) < 2:
        return [(aligned_data[this_idx]["gt_lat"], aligned_data[this_idx]["gt_lon"])]

    acc_mag = np.sqrt(imu_df["acc_x"]**2 + imu_df["acc_y"]**2 + imu_df["acc_z"]**2)
    acc_mag = smooth_acc(acc_mag.to_numpy())
    #print(acc_mag)
    # acc = imu_df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    # acc_mag = np.linalg.norm(acc, axis=1)
    # smooth_acc1 = uniform_filter1d(acc_mag, size=5)

    # 世界座標加速度
    acc_x_world = imu_df['acc_x'].values
    acc_y_world = imu_df['acc_y'].values
    acc_z_world = imu_df['acc_z'].values

    # 使用 scipy 的 find_peaks 做步態偵測
    peaks, _ = find_peaks(acc_z_world, height=STEP_THRESHOLD, distance=20, prominence=0.4)  # distance 防止過密誤判
    #print(peaks)

    # plt.plot(acc_mag)
    # plt.plot(peaks, acc_mag[peaks], "x")
    # plt.plot(np.zeros_like(acc_mag), "--", color="gray")
    # plt.show()

    gyro_z = imu_df["gyro_z"].to_numpy()

    # 畫圖
    plt.figure(figsize=(14, 12))

    # # 第一張圖：平滑加速度
    # plt.subplot(3, 1, 1)
    # plt.plot(acc_mag)
    # plt.plot(peaks, acc_mag[peaks], "x")
    # plt.plot(np.zeros_like(acc_mag), "--", color="gray")
    # plt.title('Smooth_acc and Peaks')
    # plt.xlabel('index')
    # plt.ylabel('Acceleration (m/s²)')
    # # plt.legend()
    # plt.grid(True)

    # # 第一張圖：平滑加速度
    # plt.subplot(3, 1, 1)
    # plt.plot(acc_z_world)
    # plt.plot(peaks, acc_z_world[peaks], "x")
    # plt.plot(np.zeros_like(acc_z_world), "--", color="gray")
    # plt.title('Smooth_acc and Peaks')
    # plt.xlabel('index')
    # plt.ylabel('Acceleration (m/s²)')
    # # plt.legend()
    # plt.grid(True)

    # # 第二張圖：世界座標三軸加速度
    # plt.subplot(3, 1, 2)
    # plt.plot(acc_x_world, label='acc_x_calibrated')
    # plt.plot(acc_y_world, label='acc_y_calibrated')
    # plt.plot(acc_z_world, label='acc_z_calibrated')
    # plt.title('World coordinate Accelerations')
    # plt.xlabel('index')
    # plt.ylabel('Acceleration (m/s²)')
    # plt.legend()
    # plt.grid(True)

    # # 第三張圖：角速度
    # plt.subplot(3, 1, 3)
    # plt.plot(gyro_z)
    # plt.title('gyro_world')
    # plt.xlabel('index')
    # plt.ylabel('Angular Velocity (rad/s)')
    # # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    # headings = imu_df['yaw_deg'].to_numpy()
    headings = imu_df['yaw_deg_ori'].to_numpy()
    # headings = imu_df['yaw_deg_ori_cal'].to_numpy()

    # gyro_z = imu_df["gyro_z"].to_numpy()
    # mag_x = imu_df["mag_x"].to_numpy()
    # mag_y = imu_df["mag_y"].to_numpy()
    # mag_headings = np.unwrap(np.arctan2(mag_y, mag_x))

    # if FUSION_METHOD == "complementary":
    #     headings = estimate_heading_complementary(gyro_z, mag_headings)
    # elif FUSION_METHOD == "madgwick":
    #     headings = estimate_heading_madgwick(imu_df)
    # elif FUSION_METHOD == "mahony":
    #     headings = estimate_heading_mahony(imu_df)
    # elif FUSION_METHOD == "kalman":
    #     headings = estimate_heading_kalman(gyro_z, mag_headings)
    # else:
    #     headings = np.cumsum(gyro_z)
    
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

    # 記錄PDR的index
    pdr_idx = [this_idx]

    # 動態步長(非RMS)
    total_length = 0
    for idx in peaks:
        decl = declination(curr_pos.latitude, curr_pos.longitude)
        # print(f'{aligned_data[this_idx+idx]["timestamp"]}, {headings[idx]}, {decl}')
        win_start = max(0, idx - 10)
        win_end = min(len(acc_mag), idx + 10)
        acc_segment = acc_mag[win_start:win_end]

        # 方法 A: 頻譜能量（能量越高代表震動越強 → 步長越大）
        spectrum = np.abs(np.fft.rfft(acc_segment))
        spectral_energy = np.sum(spectrum**2)
        #print(np.sqrt(spectral_energy))
        step_length_fft = 0.03 * np.sqrt(spectral_energy)

        # 方法 B: 移動平均強度
        #print(np.mean(acc_segment))
        step_length_avg = DYNAMIC_STEP_SCALE * np.mean(acc_segment)

        # 方法 C: ZUPT：如果震動小於門檻，視為靜止（不推進）
        if np.max(acc_segment) - np.min(acc_segment) < 0.05:
            step_length_zupt = 0.0
        else:
            step_length_zupt = step_length_fft  # 或用 avg 也可

        # 這裡可依需求切換用哪種
        # step_length = step_length_fft
        # step_length = FIXED_STEP_LENGTH
        if interval == 0:
            step_length = estimate_step_length_from_world_acc(imu_df, idx, method='integrate') / 6
        else:
            step_length = estimate_step_length_from_world_acc(imu_df, idx, method='integrate') if estimate_step_length_from_world_acc(imu_df, idx, method='integrate') <= 0.8 else 0.8
            
        total_length += step_length
        print(f'{aligned_data[this_idx+idx]["timestamp"]}, step_length = {step_length}')
        # heading_deg = np.degrees(headings[idx]) % 360
        
        heading_deg = (-headings[idx] + 360) % 360
        #geo_heading = (heading_deg + 135) % (2 * np.pi)
        curr_pos = distance(meters=step_length).destination(curr_pos, heading_deg)
        aligned_data[this_idx+idx]["gt_lat"] = curr_pos.latitude
        aligned_data[this_idx+idx]["gt_lon"] = curr_pos.longitude
        aligned_data[this_idx+idx]["gt_lat_temp"] = curr_pos.latitude
        aligned_data[this_idx+idx]["gt_lon_temp"] = curr_pos.longitude
        #trajectory.append((curr_pos.latitude, curr_pos.longitude))

        pdr_idx.append(this_idx+idx)

    pdr_idx.append(end_idx)

    for i in range(len(pdr_idx)-1):
        start_idx = pdr_idx[i]
        end_idx = pdr_idx[i+1]
        start_lat, start_lon = aligned_data[start_idx]["gt_lat_temp"], aligned_data[start_idx]["gt_lon_temp"]
        end_lat, end_lon = aligned_data[end_idx]["gt_lat_temp"], aligned_data[end_idx]["gt_lon_temp"]
        steps = end_idx - start_idx
        for j in range(1, steps):
            ratio = j / steps
            interp_lat = start_lat + ratio * (end_lat - start_lat)
            interp_lon = start_lon + ratio * (end_lon - start_lon)
            aligned_data[start_idx + j]["gt_lat_temp"] = interp_lat
            aligned_data[start_idx + j]["gt_lon_temp"] = interp_lon

    print(total_length)

def estimate_trajectory_from_imu_all_test(aligned_data, this_idx, end_idx, imu_df, first_heading_deg):
    if len(imu_df) < 2:
        return [(aligned_data[this_idx]["gt_lat"], aligned_data[this_idx]["gt_lon"])]

    acc_mag = np.sqrt(imu_df["acc_x"]**2 + imu_df["acc_y"]**2 + imu_df["acc_z"]**2)
    acc_mag = smooth_acc(acc_mag.to_numpy())
    #print(acc_mag)

    acc_z_world = imu_df['acc_z'].values

    # 使用 scipy 的 find_peaks 做步態偵測
    peaks, _ = find_peaks(acc_z_world, height=STEP_THRESHOLD, distance=20, prominence=0.4)  # distance 防止過密誤判
    #print(peaks)

    # plt.plot(acc_mag)
    # plt.plot(peaks, acc_mag[peaks], "x")
    # plt.plot(np.zeros_like(acc_mag), "--", color="gray")
    # plt.show()

    # headings = imu_df['yaw_deg'].to_numpy()
    headings = imu_df['yaw_deg_ori'].to_numpy()
    # headings = imu_df['yaw_deg_ori_cal'].to_numpy()

    # gyro_z = imu_df["gyro_z"].to_numpy()
    # mag_x = imu_df["mag_x"].to_numpy()
    # mag_y = imu_df["mag_y"].to_numpy()
    # mag_headings = np.unwrap(np.arctan2(mag_y, mag_x))

    # if FUSION_METHOD == "complementary":
    #     headings = estimate_heading_complementary(gyro_z, mag_headings)
    # elif FUSION_METHOD == "madgwick":
    #     headings = estimate_heading_madgwick(imu_df)
    # elif FUSION_METHOD == "mahony":
    #     headings = estimate_heading_mahony(imu_df)
    # elif FUSION_METHOD == "kalman":
    #     headings = estimate_heading_kalman(gyro_z, mag_headings)
    # else:
    #     headings = np.cumsum(gyro_z)

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

    last_heading_deg = first_heading_deg
    last_idx = 0
    # 動態步長(非RMS)
    for idx in peaks:
        # print('yes')
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
        # step_length = step_length_avg

        # step_length = estimate_step_length_from_world_acc(imu_df, idx, method='integrate') if estimate_step_length_from_world_acc(imu_df, idx, method='integrate') <= 0.8 else 0.8

        step_length = estimate_step_length_with_zupt(imu_df, last_idx, idx)
        step_length = step_length if step_length <= 0.8 else 0.8

        heading_deg = (-headings[idx] + 360) % 360
        heading_delta_deg = (heading_deg - last_heading_deg + 180) % 360 - 180
        #print(step_length)
        # heading_rad = headings[idx]
        ekf.predict(step_length, heading_deg)
        # x, y = ekf.get_state()[0], ekf.get_state()[1]
        # lon, lat = transformer_back.transform(x, y)
        lat, lon = ekf.get_state()[0], ekf.get_state()[1]

        # heading_deg = np.degrees(headings[idx]) % 360
        

        aligned_data[this_idx+idx]["pdr_lat"] = lat
        aligned_data[this_idx+idx]["pdr_lon"] = lon
        aligned_data[this_idx+idx]["fused_lat"] = lat
        aligned_data[this_idx+idx]["fused_lon"] = lon
        # aligned_data[this_idx+idx]["gt_lat_temp"] = curr_pos.latitude
        # aligned_data[this_idx+idx]["gt_lon_temp"] = curr_pos.longitude

        # last_heading_deg = heading_deg

        last_idx = idx

    # return last_heading_deg

def estimate_trajectory_from_imu_all_test_pdr(aligned_data, this_idx, end_idx, imu_df, lat, lon):
    # interval = geodesic((aligned_data[this_idx]["gt_lat_ori"], aligned_data[this_idx]["gt_lon_ori"]), (aligned_data[end_idx]["gt_lat_ori"], aligned_data[end_idx]["gt_lon_ori"])).meters
    if len(imu_df) < 2:
        return [(aligned_data[this_idx]["gt_lat"], aligned_data[this_idx]["gt_lon"])]

    acc_mag = np.sqrt(imu_df["acc_x"]**2 + imu_df["acc_y"]**2 + imu_df["acc_z"]**2)
    acc_mag = smooth_acc(acc_mag.to_numpy())
    #print(acc_mag)
    # acc = imu_df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    # acc_mag = np.linalg.norm(acc, axis=1)
    # smooth_acc1 = uniform_filter1d(acc_mag, size=5)

    # 世界座標加速度
    acc_x_world = imu_df['acc_x'].values
    acc_y_world = imu_df['acc_y'].values
    acc_z_world = imu_df['acc_z'].values

    # 使用 scipy 的 find_peaks 做步態偵測
    peaks, _ = find_peaks(acc_z_world, height=STEP_THRESHOLD, distance=20, prominence=0.4)  # distance 防止過密誤判
    #print(peaks)

    # plt.plot(acc_mag)
    # plt.plot(peaks, acc_mag[peaks], "x")
    # plt.plot(np.zeros_like(acc_mag), "--", color="gray")
    # plt.show()

    gyro_z = imu_df["gyro_z"].to_numpy()

    # 畫圖
    # plt.figure(figsize=(14, 12))

    # # 第一張圖：平滑加速度
    # plt.subplot(3, 1, 1)
    # plt.plot(acc_mag)
    # plt.plot(peaks, acc_mag[peaks], "x")
    # plt.plot(np.zeros_like(acc_mag), "--", color="gray")
    # plt.title('Smooth_acc and Peaks')
    # plt.xlabel('index')
    # plt.ylabel('Acceleration (m/s²)')
    # # plt.legend()
    # plt.grid(True)

    # # 第一張圖：平滑加速度
    # plt.subplot(3, 1, 1)
    # plt.plot(acc_z_world)
    # plt.plot(peaks, acc_z_world[peaks], "x")
    # plt.plot(np.zeros_like(acc_z_world), "--", color="gray")
    # plt.title('Smooth_acc and Peaks')
    # plt.xlabel('index')
    # plt.ylabel('Acceleration (m/s²)')
    # # plt.legend()
    # plt.grid(True)

    # # 第二張圖：世界座標三軸加速度
    # plt.subplot(3, 1, 2)
    # plt.plot(acc_x_world, label='acc_x_calibrated')
    # plt.plot(acc_y_world, label='acc_y_calibrated')
    # plt.plot(acc_z_world, label='acc_z_calibrated')
    # plt.title('World coordinate Accelerations')
    # plt.xlabel('index')
    # plt.ylabel('Acceleration (m/s²)')
    # plt.legend()
    # plt.grid(True)

    # # 第三張圖：角速度
    # plt.subplot(3, 1, 3)
    # plt.plot(gyro_z)
    # plt.title('gyro_world')
    # plt.xlabel('index')
    # plt.ylabel('Angular Velocity (rad/s)')
    # # plt.legend()
    # plt.grid(True)

    # plt.tight_layout()
    # plt.show()

    # headings = imu_df['yaw_deg'].to_numpy()
    headings = imu_df['yaw_deg_ori'].to_numpy()
    # headings = imu_df['yaw_deg_ori_cal'].to_numpy()

    # gyro_z = imu_df["gyro_z"].to_numpy()
    # mag_x = imu_df["mag_x"].to_numpy()
    # mag_y = imu_df["mag_y"].to_numpy()
    # mag_headings = np.unwrap(np.arctan2(mag_y, mag_x))

    # if FUSION_METHOD == "complementary":
    #     headings = estimate_heading_complementary(gyro_z, mag_headings)
    # elif FUSION_METHOD == "madgwick":
    #     headings = estimate_heading_madgwick(imu_df)
    # elif FUSION_METHOD == "mahony":
    #     headings = estimate_heading_mahony(imu_df)
    # elif FUSION_METHOD == "kalman":
    #     headings = estimate_heading_kalman(gyro_z, mag_headings)
    # else:
    #     headings = np.cumsum(gyro_z)
    
    #headings = heading
    curr_pos = Point(lat, lon)
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

    # 記錄PDR的index
    # pdr_idx = [this_idx]

    # 動態步長(非RMS)
    total_length = 0
    for idx in peaks:
        '''
        decl = declination(curr_pos.latitude, curr_pos.longitude)
        # print(f'{aligned_data[this_idx+idx]["timestamp"]}, {headings[idx]}, {decl}')
        win_start = this_idx + (idx - 10)
        win_end = this_idx + (idx + 10)
        acc_segment = acc_mag[win_start:win_end]

        # 方法 A: 頻譜能量（能量越高代表震動越強 → 步長越大）
        spectrum = np.abs(np.fft.rfft(acc_segment))
        spectral_energy = np.sum(spectrum**2)
        #print(np.sqrt(spectral_energy))
        step_length_fft = 0.03 * np.sqrt(spectral_energy)

        # 方法 B: 移動平均強度
        #print(np.mean(acc_segment))
        step_length_avg = DYNAMIC_STEP_SCALE * np.mean(acc_segment)

        # 方法 C: ZUPT：如果震動小於門檻，視為靜止（不推進）
        if np.max(acc_segment) - np.min(acc_segment) < 0.05:
            step_length_zupt = 0.0
        else:
            step_length_zupt = step_length_fft  # 或用 avg 也可
        '''
        win_start = this_idx + (idx - 40)
        win_end = this_idx + (idx)
        # 這裡可依需求切換用哪種
        # step_length = step_length_fft
        # step_length = FIXED_STEP_LENGTH
        # if interval == 0:
        #     step_length = estimate_step_length_from_world_acc(imu_df, idx, method='integrate') / 6
        # else:
        #     step_length = estimate_step_length_from_world_acc(imu_df, idx, method='integrate') if estimate_step_length_from_world_acc(imu_df, idx, method='integrate') <= 0.8 else 0.8
        turn = False
        check_gyro_z = gyro_z[win_start:win_end]
        for i in check_gyro_z:
            if i > 1.0:
                turn = True
                break
        if turn:
            step_length = estimate_step_length_from_world_acc(imu_df, idx, method='integrate') / 6
        else:
            step_length = estimate_step_length_from_world_acc(imu_df, idx, method='integrate') if estimate_step_length_from_world_acc(imu_df, idx, method='integrate') <= 0.8 else 0.8
        total_length += step_length
        print(f'{aligned_data[this_idx+idx]["timestamp"]}, step_length = {step_length}, turn = {turn}')
        # heading_deg = np.degrees(headings[idx]) % 360
        
        heading_deg = (-headings[idx] + 360) % 360
        #geo_heading = (heading_deg + 135) % (2 * np.pi)
        curr_pos = distance(meters=step_length).destination(curr_pos, heading_deg)
        aligned_data[this_idx+idx]["fused_lat"] = curr_pos.latitude
        aligned_data[this_idx+idx]["fused_lon"] = curr_pos.longitude
        # aligned_data[this_idx+idx]["gt_lat_temp"] = curr_pos.latitude
        # aligned_data[this_idx+idx]["gt_lon_temp"] = curr_pos.longitude
        #trajectory.append((curr_pos.latitude, curr_pos.longitude))

        # pdr_idx.append(this_idx+idx)

    # pdr_idx.append(end_idx)

    # for i in range(len(pdr_idx)-1):
    #     start_idx = pdr_idx[i]
    #     end_idx = pdr_idx[i+1]
    #     start_lat, start_lon = aligned_data[start_idx]["gt_lat_temp"], aligned_data[start_idx]["gt_lon_temp"]
    #     end_lat, end_lon = aligned_data[end_idx]["gt_lat_temp"], aligned_data[end_idx]["gt_lon_temp"]
    #     steps = end_idx - start_idx
    #     for j in range(1, steps):
    #         ratio = j / steps
    #         interp_lat = start_lat + ratio * (end_lat - start_lat)
    #         interp_lon = start_lon + ratio * (end_lon - start_lon)
    #         aligned_data[start_idx + j]["gt_lat_temp"] = interp_lat
    #         aligned_data[start_idx + j]["gt_lon_temp"] = interp_lon

    # print(total_length)

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

# 測試軌跡 -----------------------------------------------------------------------------------------------------------------

wifi_df_test = pd.read_csv(TEST_WIFI_CSV).rename(columns={"AppTimestamp(s)": "timestamp"})
# wifi_df_test = filter_rssi(wifi_df_test)
imu_df_test = pd.read_csv(TEST_IMU_CSV).rename(columns={"AppTimestamp(s)": "timestamp"})
imu_df2_test = imu_df_test.drop(columns=["timestamp","SensorTimestamp(s)"]).reset_index(drop=True)
posi_df_test = pd.read_csv(TEST_GT_CSV).rename(columns={"AppTimestamp(s)": "timestamp"})
posi_df_test = posi_df_test.sort_values("timestamp").reset_index(drop=True)

test_aligned_data = []

for i in range(len(imu_df_test)):
    imu_time = imu_df_test.loc[i, "timestamp"]
    # rssi_vector = wifi_df.iloc[i, 1:].to_numpy()

    # closest_gt = posi_df.iloc[(posi_df['timestamp'] - wifi_time).abs().argmin()]
    # gt_lat2, gt_lon2 = closest_gt["Latitude_degrees"], closest_gt["Longitude_degrees"]

    # imu_data = imu_df.drop(columns=["timestamp","SensorTimestamp(s)"]).reset_index(drop=True)

    test_aligned_data.append({
        "knn_lat" : None,
        "knn_lon" : None,
        "knn_neighbors_distances" : None,
        "knn_neighbors_nearest_positions" : None,
        "wifi_lat" : None,
        "wifi_lon" : None,
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
        if wifi_time > 50:
            wifi_used_indices_test.append(closest_idx)

# WIFI RSSI 初始定位(測試軌跡)
repitition = 'R1234'
neighbors = 3
top = 10
with open(f'py/knn/distance/rssi_features_{repitition}_fixed_v2_neighbors{neighbors}.pkl', 'rb') as f:
    rssi_features = pickle.load(f)
with open(f'py/knn/distance/gt_positions_{repitition}_fixed_v2_neighbors{neighbors}.pkl', 'rb') as f:
    gt_positions = pickle.load(f)
with open(f'py/knn/distance/knn_model_{repitition}_fixed_v2_neighbors{neighbors}.pkl', 'rb') as f:
    wifi_model = pickle.load(f)
# with open(f'py/knn/distance/knn_model_{repitition}_fixed_v2_neighbors1.pkl', 'rb') as f:
#     wifi_model_n1 = pickle.load(f)

# 設定原點為你的第一個定位點
# test_aligned_data[wifi_used_indices_test[0]]["rssi_vector"]
origin_lat, origin_lon = wifi_model.predict([test_aligned_data[wifi_used_indices_test[0]]["rssi_vector"]])[0]
init_heading = imu_df2_test["yaw_deg_ori"][wifi_used_indices_test[0]]
init_heading = (-init_heading + 360) % 360
print(f'init heading = {init_heading}')
ekf = EKF_Localizer_latlon(init_latlon=(origin_lat, origin_lon), init_heading_deg=init_heading, process_noise_std=0.5, obs_noise_std=2.5)
test_aligned_data[wifi_used_indices_test[0]]["knn_lat"] = origin_lat
test_aligned_data[wifi_used_indices_test[0]]["knn_lon"] = origin_lon

distances, indices = wifi_model.kneighbors([test_aligned_data[wifi_used_indices_test[0]]["rssi_vector"]])
test_aligned_data[wifi_used_indices_test[0]]["knn_neighbors_distances"] = distances[0]
test_aligned_data[wifi_used_indices_test[0]]["knn_neighbors_nearest_positions"] = gt_positions[indices[0]]

# 建立 ENU 轉換器
transformer = Transformer.from_crs("epsg:4326", f"+proj=tmerc +lat_0={origin_lat} +lon_0={origin_lon} +units=m", always_xy=True)
# 建立轉換器（平面 → 經緯度）
transformer_back = Transformer.from_crs(f"+proj=tmerc +lat_0={origin_lat} +lon_0={origin_lon} +units=m","epsg:4326", always_xy=True)
first_heading = init_heading
# 經緯度 → x, y（公尺）
# x, y = transformer.transform(origin_lon, origin_lat)

# 假設你有 EKF 輸出：
# ekf_x, ekf_y = 12.3, 45.6  # 單位：公尺

# 轉回經緯度
# lon, lat = transformer_back.transform(ekf_x, ekf_y)

for i in range(len(wifi_used_indices_test)-1):
    this_wifi_index = wifi_used_indices_test[i]
    next_wifi_index = wifi_used_indices_test[i+1]
    imu_seq = imu_df2_test[this_wifi_index+1 : next_wifi_index]
    # temp_heading = estimate_trajectory_from_imu_all_test(test_aligned_data, this_wifi_index, next_wifi_index, imu_seq, first_heading)
    estimate_trajectory_from_imu_all_test(test_aligned_data, this_wifi_index, next_wifi_index, imu_seq, first_heading)
    '''
    if test_aligned_data[wifi_used_indices_test[i+1]]["timestamp"] > 268 and test_aligned_data[wifi_used_indices_test[i+1]]["timestamp"] < 300:
        lat, lon = wifi_model_n1.predict([test_aligned_data[wifi_used_indices_test[i+1]]["rssi_vector"]])[0]
        distances, indices = wifi_model_n1.kneighbors([test_aligned_data[wifi_used_indices_test[i+1]]["rssi_vector"]])
    else:
        lat, lon = wifi_model.predict([test_aligned_data[wifi_used_indices_test[i+1]]["rssi_vector"]])[0]
        distances, indices = wifi_model.kneighbors([test_aligned_data[wifi_used_indices_test[i+1]]["rssi_vector"]])
    '''
    lat, lon = wifi_model.predict([test_aligned_data[wifi_used_indices_test[i+1]]["rssi_vector"]])[0]
    distances, indices = wifi_model.kneighbors([test_aligned_data[wifi_used_indices_test[i+1]]["rssi_vector"]])
    test_aligned_data[wifi_used_indices_test[i+1]]["knn_lat"] = lat
    test_aligned_data[wifi_used_indices_test[i+1]]["knn_lon"] = lon

    # distances, indices = wifi_model.kneighbors([test_aligned_data[wifi_used_indices_test[i+1]]["rssi_vector"]])
    test_aligned_data[wifi_used_indices_test[i+1]]["knn_neighbors_distances"] = distances[0]
    test_aligned_data[wifi_used_indices_test[i+1]]["knn_neighbors_nearest_positions"] = gt_positions[indices[0]]
    # x, y = transformer.transform(lon, lat)
    # ekf.update(np.array([x, y]))
    ekf.update(np.array([lat, lon]))
    # ekf_x, ekf_y = ekf.get_state()[0], ekf.get_state()[1]
    # final_lon, final_lat = transformer_back.transform(ekf_x, ekf_y)
    final_lat, final_lon = ekf.get_state()[0], ekf.get_state()[1]
    test_aligned_data[wifi_used_indices_test[i+1]]["wifi_lat"] = final_lat
    test_aligned_data[wifi_used_indices_test[i+1]]["wifi_lon"] = final_lon
    test_aligned_data[wifi_used_indices_test[i+1]]["fused_lat"] = final_lat
    test_aligned_data[wifi_used_indices_test[i+1]]["fused_lon"] = final_lon

    # first_heading = temp_heading

wifi_last_index = wifi_used_indices_test[len(wifi_used_indices_test)-1]
gt_last_index = gt_used_indices_test[len(gt_used_indices_test)-1]

if gt_last_index > wifi_last_index:
    imu_seq = imu_df2_test[wifi_last_index : gt_last_index]
    estimate_trajectory_from_imu_all_test(test_aligned_data, wifi_last_index, gt_last_index, imu_seq, first_heading)

# imu_seq = imu_df2_test[wifi_used_indices_test[0]+1 :]
# this_wifi_index = wifi_used_indices_test[0]+1
# print(len(imu_seq))
# print(this_wifi_index)
# estimate_trajectory_from_imu_all_test(test_aligned_data, this_wifi_index, None, imu_seq, first_heading)
# estimate_trajectory_from_imu_all_test_pdr(test_aligned_data, this_wifi_index, None, imu_seq, origin_lat, origin_lon)
'''
gt_coords_origin_test = [(d["gt_lat"], d["gt_lon"]) for d in test_aligned_data if d["gt_lat"] is not None]
gt_lats_test, gt_lons_test = zip(*gt_coords_origin_test)

for i in range(len(gt_used_indices_test)-1):
    this_gt_index = gt_used_indices_test[i]
    next_gt_index = gt_used_indices_test[i+1]

    wifi_coords = []
    pdr_coords = []
    for j in range(this_gt_index, next_gt_index):
        if test_aligned_data[j]["wifi_lat"] is not None:
            wifi_coords.append((test_aligned_data[j]["wifi_lat"], test_aligned_data[j]["wifi_lon"]))
        elif test_aligned_data[j]["pdr_lat"] is not None:
            pdr_coords.append((test_aligned_data[j]["pdr_lat"], test_aligned_data[j]["pdr_lon"]))

    wifi_lats, wifi_lons = zip(*wifi_coords)
    pdr_lats, pdr_lons = zip(*pdr_coords)

    plt.plot(gt_lons_test, gt_lats_test, label="Ground Truth origin", marker="o")
    plt.plot(wifi_lons, wifi_lats, label="Wi-Fi", marker="o")
    plt.plot(pdr_lons, pdr_lats, label="PDR", marker="o")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("Wi-Fi and PDR vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
'''


# 輸出所有對齊資料為 pickle
# for i, d in enumerate(test_aligned_data):
#     with open(os.path.join(TEST_OUTPUT_DIR, f"sample_{i:04d}.pkl"), "wb") as f:
#         pickle.dump(d, f)
with open(os.path.join(TEST_OUTPUT_DIR, f"distance/final_neighbors{neighbors}.pkl"), "wb") as f:
    pickle.dump(test_aligned_data, f)

print(f"處理完成，共輸出 {len(test_aligned_data)} 筆對齊資料到資料夾：{TEST_OUTPUT_DIR}")
