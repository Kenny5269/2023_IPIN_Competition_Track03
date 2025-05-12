
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm

def normalize(v):
    return v / norm(v) if norm(v) > 0 else v

def initialize_quaternion_from_acc_mag(acc0, mag0):
    acc0 = acc0 / np.linalg.norm(acc0)
    mag0 = mag0 / np.linalg.norm(mag0)

    z_axis = -acc0
    x_axis = np.cross(mag0, acc0)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R_init = np.vstack([x_axis, y_axis, z_axis]).T  # 每一欄是單位向量
    q_scipy = R.from_matrix(R_init).as_quat()  # [x, y, z, w]
    return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])  # 轉成 [w, x, y, z]

def detect_static_segment(df, window_size=100, max_time=60.0, acc_threshold=0.06, gyro_threshold=0.02):
    subset = df[df['AppTimestamp(s)'] <= max_time].reset_index(drop=True)
    for i in range(len(subset) - window_size):
        # print('fuck')
        acc_win = subset[['acc_x', 'acc_y', 'acc_z']].iloc[i:i+window_size].to_numpy()
        gyro_win = subset[['gyro_x', 'gyro_y', 'gyro_z']].iloc[i:i+window_size].to_numpy()
        acc_var = np.var(acc_win, axis=0)
        gyro_mean = np.mean(np.abs(gyro_win), axis=0)
        if acc_var.mean() < acc_threshold and gyro_mean.mean() < gyro_threshold:
            start_time = subset.loc[i, 'AppTimestamp(s)']
            end_time = subset.loc[i + window_size - 1, 'AppTimestamp(s)']
            return i, start_time, end_time
    return None, None, None


def madgwick_filter_with_mag_init(df, beta=0.1, freq=50):
    dt = 1.0 / freq
    quaternions = []

    idx, t_start, t_end = detect_static_segment(df)
    if idx is None:
        raise RuntimeError("❌ 找不到靜止段，請檢查資料或參數")
    
    # 使用該靜止段建立初始四元數
    acc0 = df[['acc_x', 'acc_y', 'acc_z']].iloc[idx:idx+100].mean().to_numpy()
    mag0 = df[['mag_x', 'mag_y', 'mag_z']].iloc[idx:idx+100].mean().to_numpy()
    q = initialize_quaternion_from_acc_mag(acc0, mag0)
    q = -q

    # 取得第一筆 acc 與 mag 資料來初始化
    # acc0 = df.loc[0, ['acc_x', 'acc_y', 'acc_z']].to_numpy()
    # mag0 = df.loc[0, ['mag_x', 'mag_y', 'mag_z']].to_numpy()
    # q = initialize_quaternion_from_acc_mag(acc0, mag0)

    # 前段填 [1, 0, 0, 0]
    for _ in range(idx):
        quaternions.append([1.0, 0.0, 0.0, 0.0])

    # 從靜止段後開始推估姿態
    for i in range(idx, len(df)):
        row = df.iloc[i]
        ax, ay, az = row[['acc_x', 'acc_y', 'acc_z']]
        gx, gy, gz = row[['gyro_x', 'gyro_y', 'gyro_z']]
        mx, my, mz = row[['mag_x', 'mag_y', 'mag_z']]

        acc = normalize([ax, ay, az])
        mag = normalize([mx, my, mz])
        q1, q2, q3, q4 = q

        f = np.array([
            2*(q2*q4 - q1*q3) - acc[0],
            2*(q1*q2 + q3*q4) - acc[1],
            2*(0.5 - q2**2 - q3**2) - acc[2]
        ])
        J = np.array([
            [-2*q3,  2*q4, -2*q1, 2*q2],
            [ 2*q2,  2*q1,  2*q4, 2*q3],
            [ 0.0 , -4*q2, -4*q3, 0.0]
        ])
        step = normalize(J.T @ f)

        q_dot = 0.5 * np.array([
            -q2*gx - q3*gy - q4*gz,
             q1*gx + q3*gz - q4*gy,
             q1*gy - q2*gz + q4*gx,
             q1*gz + q2*gy - q3*gx
        ]) - beta * step

        q += q_dot * dt
        q = normalize(q)
        quaternions.append(q.copy())

    # 寫入欄位
    q_arr = np.array(quaternions)
    df['q_w'] = q_arr[:, 0]
    df['q_x'] = q_arr[:, 1]
    df['q_y'] = q_arr[:, 2]
    df['q_z'] = q_arr[:, 3]

    print(f"✅ 偵測到的靜止段：{t_start:.2f} 秒 ～ {t_end:.2f} 秒，從該段起開始估算四元數")
    return df, t_start, t_end

# 讀取資料
index = 'T1_R1'
df = pd.read_csv(f'{index}/IMU_50Hz.csv')

# 低通濾波器定義
def lowpass_filter(data, cutoff=5, fs=50, order=2):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# 濾波 acc, gyro, mag
for sensor in ['acc', 'gyro', 'mag']:
    for axis in ['x', 'y', 'z']:
        col = f'{sensor}_{axis}'
        df[col] = lowpass_filter(df[col])

# 估計靜止區間 bias (45~50秒)
static = df[(df['AppTimestamp(s)'] >= 45) & (df['AppTimestamp(s)'] <= 50)]
gyro_bias = static[['gyro_x', 'gyro_y', 'gyro_z']].mean().values
mag_bias = static[['mag_x', 'mag_y', 'mag_z']].mean().values

# 扣除 bias
df[['gyro_x', 'gyro_y', 'gyro_z']] -= gyro_bias
df[['mag_x', 'mag_y', 'mag_z']] -= mag_bias

# Madgwick 濾波器簡易實作（不含磁力計）

# q = np.array([1.0, 0.0, 0.0, 0.0])
# beta = 0.1
# dt = 1 / 50
# quaternions = []

# for i, row in df.iterrows():
#     ax, ay, az = row[['acc_x', 'acc_y', 'acc_z']]
#     gx, gy, gz = row[['gyro_x', 'gyro_y', 'gyro_z']]
#     acc = normalize([ax, ay, az])
#     if norm(acc) == 0:
#         quaternions.append(q.copy())
#         continue
#     f = np.array([
#         2*(q[1]*q[3] - q[0]*q[2]) - ax,
#         2*(q[0]*q[1] + q[2]*q[3]) - ay,
#         2*(0.5 - q[1]**2 - q[2]**2) - az
#     ])
#     J = np.array([
#         [-2*q[2],  2*q[3], -2*q[0], 2*q[1]],
#         [ 2*q[1],  2*q[0],  2*q[3], 2*q[2]],
#         [    0.0, -4*q[1], -4*q[2],    0.0]
#     ])
#     step = normalize(J.T @ f)
#     q_dot = 0.5 * np.array([
#         -q[1]*gx - q[2]*gy - q[3]*gz,
#          q[0]*gx + q[2]*gz - q[3]*gy,
#          q[0]*gy - q[1]*gz + q[3]*gx,
#          q[0]*gz + q[1]*gy - q[2]*gx
#     ]) - beta * step
#     q += q_dot * dt
#     q = normalize(q)
#     quaternions.append(q.copy())

# # 儲存四元數
# q_arr = np.array(quaternions)
# df['q_w'], df['q_x'], df['q_y'], df['q_z'] = q_arr[:,0], q_arr[:,1], q_arr[:,2], q_arr[:,3]

# 新Madgwick濾波融合估算四元數
df, start, end = madgwick_filter_with_mag_init(df)

# 四元數比對(error_degree)
# 將四元數組裝成 array（順序為 [w, x, y, z]）
q_ref = df[['Quat_1', 'Quat_2', 'Quat_3', 'Quat_4']].to_numpy()
q_est = df[['q_w', 'q_x', 'q_y', 'q_z']].to_numpy()

# 計算內積（逐列）
dot_products = np.einsum('ij,ij->i', q_ref, q_est)
dot_products = np.clip(np.abs(dot_products), 0, 1)  # 限制在 arccos 有效範圍

# 計算角度差（弧度）
angle_diff_rad = 2 * np.arccos(dot_products)

# 若要以度表示：
angle_diff_deg = np.degrees(angle_diff_rad)

# 存入 DataFrame
df['quat_angle_error_deg'] = angle_diff_deg

# 四元數轉世界座標系 (acc, gyro, mag)(順序為 [x, y, z, w])
acc_world, gyro_world, mag_world = [], [], []
for i, row in df.iterrows():
    quat = [row['q_x'], row['q_y'], row['q_z'], row['q_w']]
    r = R.from_quat(quat)
    acc_world.append(r.apply(row[['acc_x', 'acc_y', 'acc_z']]))
    gyro_world.append(r.apply(row[['gyro_x', 'gyro_y', 'gyro_z']]))
    mag_world.append(r.apply(row[['mag_x', 'mag_y', 'mag_z']]))

acc_world = np.array(acc_world)
gyro_world = np.array(gyro_world)
mag_world = np.array(mag_world)
df['acc_wx'], df['acc_wy'], df['acc_wz'] = acc_world[:,0], acc_world[:,1], acc_world[:,2]
df['gyro_wx'], df['gyro_wy'], df['gyro_wz'] = gyro_world[:,0], gyro_world[:,1], gyro_world[:,2]
df['mag_wx'], df['mag_wy'], df['mag_wz'] = mag_world[:,0], mag_world[:,1], mag_world[:,2]

# 扣除重力與 bias
gravity = np.array([0, 0, 9.8])
acc_dynamic = acc_world - gravity
static_dyn = df[(df['AppTimestamp(s)'] >= start) & (df['AppTimestamp(s)'] <= end)][['acc_wx', 'acc_wy', 'acc_wz']].values - gravity
bias_world = static_dyn.mean(axis=0)
acc_dynamic -= bias_world
df['acc_dx'], df['acc_dy'], df['acc_dz'] = acc_dynamic[:,0], acc_dynamic[:,1], acc_dynamic[:,2]

# 驗證靜止段動態加速度平均是否接近 0
check = df[(df['AppTimestamp(s)'] >= 45) & (df['AppTimestamp(s)'] <= 50)][['acc_dx', 'acc_dy', 'acc_dz']].mean()
print("靜止段動態加速度平均（應接近 [0, 0, 0]）：\n", check.values)
final_export_df = pd.DataFrame({
    'AppTimestamp(s)': df['AppTimestamp(s)'],
    'SensorTimestamp(s)': df['SensorTimestamp(s)'],
    'acc_x': df['acc_dx'],
    'acc_y': df['acc_dy'],
    'acc_z': df['acc_dz'],
    'gyro_x': df['gyro_wx'],
    'gyro_y': df['gyro_wy'],
    'gyro_z': df['gyro_wz'],
    'mag_x': df['mag_wx'],
    'mag_y': df['mag_wy'],
    'mag_z': df['mag_wz'],
    'Quat_1': df['Quat_1'],
    'Quat_2': df['Quat_2'],
    'Quat_3': df['Quat_3'],
    'Quat_4': df['Quat_4'],
    'q_w': df['q_w'],
    'q_x': df['q_x'],
    'q_y': df['q_y'],
    'q_z': df['q_z'],
    'quat_angle_error_deg' : df['quat_angle_error_deg']
})

# 畫圖
# 抓出世界座標下的加速度分量
timestamps = final_export_df['AppTimestamp(s)'].values

# 世界座標加速度
acc_x_world = final_export_df['acc_x'].values
acc_y_world = final_export_df['acc_y'].values
acc_z_world = final_export_df['acc_z'].values

# 世界座標角速度
gyro_x_world = final_export_df['gyro_x'].values
gyro_y_world = final_export_df['gyro_y'].values
gyro_z_world = final_export_df['gyro_z'].values

# 世界座標磁力計
mag_x_world = final_export_df['mag_x'].values
mag_y_world = final_export_df['mag_y'].values
mag_z_world = final_export_df['mag_z'].values

# 畫圖
plt.figure(figsize=(14, 12))

# 第一張圖：加速度
plt.subplot(3, 1, 1)
plt.plot(timestamps, acc_x_world, label='acc_x_calibrated')
plt.plot(timestamps, acc_y_world, label='acc_y_calibrated')
plt.plot(timestamps, acc_z_world, label='acc_z_calibrated')
plt.title('Calibrated Accelerations')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)

# 第二張圖：角速度
plt.subplot(3, 1, 2)
plt.plot(timestamps, gyro_x_world, label='gyro_x_calibrated')
plt.plot(timestamps, gyro_y_world, label='gyro_y_calibrated')
plt.plot(timestamps, gyro_z_world, label='gyro_z_calibrated')
plt.title('Calibrated Angular Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

# 第三張圖：磁力計
plt.subplot(3, 1, 3)
plt.plot(timestamps, mag_x_world, label='mag_x_calibrated')
plt.plot(timestamps, mag_y_world, label='mag_y_calibrated')
plt.plot(timestamps, mag_z_world, label='mag_z_calibrated')
plt.title('Calibrated Magnetometer Readings')
plt.xlabel('Time (s)')
plt.ylabel('Magnetic Field (μT)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

final_export_df.to_csv(f'{index}/IMU_calibrated3_temp.csv', index=False)

