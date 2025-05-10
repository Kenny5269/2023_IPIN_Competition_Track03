
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm

# 讀取資料
index = 'T3_R4'
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
def normalize(v):
    return v / norm(v) if norm(v) > 0 else v

q = np.array([1.0, 0.0, 0.0, 0.0])
beta = 0.1
dt = 1 / 50
quaternions = []

for i, row in df.iterrows():
    ax, ay, az = row[['acc_x', 'acc_y', 'acc_z']]
    gx, gy, gz = row[['gyro_x', 'gyro_y', 'gyro_z']]
    acc = normalize([ax, ay, az])
    if norm(acc) == 0:
        quaternions.append(q.copy())
        continue
    f = np.array([
        2*(q[1]*q[3] - q[0]*q[2]) - ax,
        2*(q[0]*q[1] + q[2]*q[3]) - ay,
        2*(0.5 - q[1]**2 - q[2]**2) - az
    ])
    J = np.array([
        [-2*q[2],  2*q[3], -2*q[0], 2*q[1]],
        [ 2*q[1],  2*q[0],  2*q[3], 2*q[2]],
        [    0.0, -4*q[1], -4*q[2],    0.0]
    ])
    step = normalize(J.T @ f)
    q_dot = 0.5 * np.array([
        -q[1]*gx - q[2]*gy - q[3]*gz,
         q[0]*gx + q[2]*gz - q[3]*gy,
         q[0]*gy - q[1]*gz + q[3]*gx,
         q[0]*gz + q[1]*gy - q[2]*gx
    ]) - beta * step
    q += q_dot * dt
    q = normalize(q)
    quaternions.append(q.copy())

# 儲存四元數
q_arr = np.array(quaternions)
df['q_w'], df['q_x'], df['q_y'], df['q_z'] = q_arr[:,0], q_arr[:,1], q_arr[:,2], q_arr[:,3]

# 四元數轉世界座標系 (acc, gyro, mag)
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
static_dyn = df[(df['AppTimestamp(s)'] >= 45) & (df['AppTimestamp(s)'] <= 50)][['acc_wx', 'acc_wy', 'acc_wz']].values - gravity
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

final_export_df.to_csv(f'{index}/IMU_calibrated3.csv', index=False)

