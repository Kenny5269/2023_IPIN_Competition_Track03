import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from geopy.distance import distance
from scipy.signal import butter, filtfilt

def lowpass_filter(data, cutoff_freq, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        return np.stack([filtfilt(b, a, data[:, i]) for i in range(data.shape[1])], axis=1)

# 讀取IMU原始資料
index = 'T27_R4'

imu_df = pd.read_csv(f'{index}/IMU_50Hz.csv')  # 注意：應該是未經校準原始資料

# 指定靜止段時間範圍
static_start = 50  # 起點秒數
static_end = 53    # 終點秒數

# 僅保留 AppTimestamp >= 40 秒的資料，捨棄前段 calibration 動作資料
imu_df = imu_df[imu_df['AppTimestamp(s)'] >= static_start].reset_index(drop=True)
fs = 50

acc_f = lowpass_filter(imu_df[['acc_x', 'acc_y', 'acc_z']].values, 5, fs)
gyro_f = lowpass_filter(imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values, 10, fs)
mag_f = lowpass_filter(imu_df[['mag_x', 'mag_y', 'mag_z']].values, 3, fs)

timestamps = imu_df['AppTimestamp(s)'].values

# 擷取原始資料欄位
acc = imu_df[['acc_x', 'acc_y', 'acc_z']].values
gyro = imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values

# 設定初始靜止段範圍，推估初始姿態
static_mask = (imu_df['AppTimestamp(s)'] >= static_start) & (imu_df['AppTimestamp(s)'] <= static_end)
mean_acc_static = imu_df.loc[static_mask, ['acc_x', 'acc_y', 'acc_z']].mean().values

# 從重力方向反推出初始姿態對齊世界Z軸（重力朝下）
init_gravity = mean_acc_static / np.linalg.norm(mean_acc_static)
world_z = np.array([0, 0, 1])

# 建立初始旋轉矩陣 R0，將手機初始重力方向轉到世界Z軸
def rotation_matrix_from_vectors(vec1, vec2):
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))

R0 = rotation_matrix_from_vectors(init_gravity, world_z)

# 使用者選擇的模式：'full' 或 'heading'
mode = 'full'  # 或改成 'heading' 只積分 z 軸角速度

delta_t = np.diff(timestamps, prepend=timestamps[0])
orientation = [R0]  # 以初始對齊姿態為起點
heading_angle = [0]  # 初始heading

for i in range(1, len(timestamps)):
    if mode == 'full':
        wx, wy, wz = gyro_f[i]
        angle = np.linalg.norm([wx, wy, wz]) * delta_t[i]
        if angle == 0:
            R = np.eye(3)
        else:
            axis = np.array([wx, wy, wz]) / np.linalg.norm([wx, wy, wz])
            x, y, z = axis
            c = np.cos(angle)
            s = np.sin(angle)
            C = 1 - c
            R = np.array([
                [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
                [y*x*C + z*s, c + y*y*C,     y*z*C - x*s],
                [z*x*C - y*s, z*y*C + x*s, c + z*z*C    ]
            ])
        orientation.append(orientation[-1] @ R)
    elif mode == 'heading':
        dtheta = gyro_f[i][2] * delta_t[i]  # 只用Z軸角速度
        heading_angle.append(heading_angle[-1] + dtheta)
        theta = heading_angle[-1]
        R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]
        ])
        orientation.append(R0 @ R)
    else:
        raise ValueError("mode must be 'full' or 'heading'")

orientation = np.array(orientation)

# 將手機座標下的加速度資料轉換到世界座標系
acc_world = np.array([orientation[i] @ acc_f[i] for i in range(len(acc_f))])

# 扣除重力向量 [0, 0, 9.81]，得到純動態加速度
gravity = np.array([0, 0, 9.81])
dynamic_acc = acc_world - gravity

# 將 dynamic_acc 寫入 DataFrame 中
# imu_df['acc_x'] = acc_world[:, 0]
# imu_df['acc_y'] = acc_world[:, 1]
# imu_df['acc_z'] = acc_world[:, 2]

imu_df['acc_x'] = dynamic_acc[:, 0]
imu_df['acc_y'] = dynamic_acc[:, 1]
imu_df['acc_z'] = dynamic_acc[:, 2]

# 額外驗證：靜止段轉換後的加速度平均是否 ≈ [0, 0, 0]
acc_dyn_static = dynamic_acc[static_mask.values]
print("靜止段動態加速度平均值：", np.mean(acc_dyn_static, axis=0))

# 儲存校準後的資料
imu_df.to_csv(f'{index}/IMU_aligned_calibrated.csv', index=False)

print("完成：使用", mode, "模式校準加速度，並已扣除重力。")
