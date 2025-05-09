
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

# 定義常數
g_world = np.array([0, 0, 9.81])  # 世界座標系的重力向量 (垂直向下)

def normalize(v):
    return v / np.linalg.norm(v)

def rotation_matrix_from_gyro(gyro, dt):
    """根據陀螺儀數據計算旋轉矩陣 (小角度近似)"""
    angle = np.linalg.norm(gyro) * dt  # 旋轉角度
    if angle == 0:
        return np.eye(3)
    
    axis = normalize(gyro)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    one_minus_cos = 1 - cos_a

    x, y, z = axis
    R = np.array([
        [cos_a + x * x * one_minus_cos, x * y * one_minus_cos - z * sin_a, x * z * one_minus_cos + y * sin_a],
        [y * x * one_minus_cos + z * sin_a, cos_a + y * y * one_minus_cos, y * z * one_minus_cos - x * sin_a],
        [z * x * one_minus_cos - y * sin_a, z * y * one_minus_cos + x * sin_a, cos_a + z * z * one_minus_cos]
    ])
    return R

def remove_gravity(acc_data, gyro_data, timestamps, alpha=0.98):
    """
    使用 Complementary filter 的方式融合，估計手機姿態並移除重力
    acc_data: Nx3，加速度數據 (ax, ay, az)
    gyro_data: Nx3，陀螺儀數據 (gx, gy, gz)，單位: rad/s
    timestamps: N，時間戳記，單位: 秒
    alpha: 融合係數，越接近1越依賴gyro
    """

    # 初始旋轉矩陣（假設一開始是正向）
    R = np.eye(3)
    dynamic_acc_list = []

    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i-1]
        if dt <= 0:
            dt = 1e-3  # 避免除以0

        gyro = gyro_data[i-1]
        acc = acc_data[i]

        # 根據陀螺儀推進旋轉
        R_gyro = rotation_matrix_from_gyro(gyro, dt)
        R = R @ R_gyro  # 更新姿態

        # 用加速度修正姿態（Complementary Filter）
        # 用加速度估計當前的重力方向
        acc_norm = normalize(acc)
        gravity_from_acc = acc_norm * 9.81

        gravity_est = R.T @ g_world  # 根據目前R推測的重力

        # 將估計的重力慢慢往感測到的重力調整
        gravity = alpha * gravity_est + (1 - alpha) * gravity_from_acc

        # 反推更新R
        correction_axis = np.cross(gravity_est, gravity)
        correction_angle = np.linalg.norm(correction_axis)
        if correction_angle != 0:
            correction_axis = normalize(correction_axis)
            R_correction = rotation_matrix_from_gyro(correction_axis, correction_angle)
            R = R_correction @ R

        # 轉換世界重力向量到手機座標
        g_in_device = R @ g_world

        # 扣掉重力
        dynamic_acc = acc - g_in_device
        dynamic_acc_list.append(dynamic_acc)

    return np.array(dynamic_acc_list)

# 讀取IMU資料
index = 'T27_R4'

imu_df = pd.read_csv(f'{index}/IMU_aligned_calibrated.csv')

# 指定靜止段時間範圍
static_start = 50  # 起點秒數
static_end = 53    # 終點秒數

# 選出靜止段資料（根據AppTimestamp欄位）
static_range = imu_df[(imu_df['AppTimestamp(s)'] >= static_start) & (imu_df['AppTimestamp(s)'] <= static_end)]

# --- 加速度校準 ---
mean_acc = static_range[['acc_x', 'acc_y', 'acc_z']].mean().values
print(mean_acc)
imu_df[['acc_x', 'acc_y', 'acc_z']] -= mean_acc

# --- 陀螺儀校準 ---
mean_gyro = static_range[['gyro_x', 'gyro_y', 'gyro_z']].mean().values
imu_df[['gyro_x', 'gyro_y', 'gyro_z']] -= mean_gyro

# --- 磁力計校準（簡單偏移） ---
mean_mag = static_range[['mag_x', 'mag_y', 'mag_z']].mean().values
imu_df[['mag_x', 'mag_y', 'mag_z']] -= mean_mag

# 座標對齊現實座標
# Select the static data
static_range = imu_df[(imu_df['AppTimestamp(s)'] >= static_start) & (imu_df['AppTimestamp(s)'] <= static_end)]

# Calculate mean acceleration during static period (to estimate gravity vector)
mean_acc = static_range[['acc_x', 'acc_y', 'acc_z']].mean().values
gravity_vector = mean_acc / np.linalg.norm(mean_acc)  # Normalize to unit vector

# Define world frame Z-axis pointing upward
world_z = np.array([0, 0, 1])

# Compute rotation matrix to align phone's gravity vector to world Z-axis
def rotation_matrix_from_vectors(vec1, vec2):
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)  # Already aligned
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix

R_align = rotation_matrix_from_vectors(gravity_vector, world_z)

# Rotate acceleration data
# acc_data = imu_df[['acc_x', 'acc_y', 'acc_z']].values
# acc_world = (R_align @ acc_data.T).T

# Rotate gyroscope data
gyro_data = imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values
gyro_world = (R_align @ gyro_data.T).T

# Rotate magnetometer data
mag_data = imu_df[['mag_x', 'mag_y', 'mag_z']].values
mag_world = (R_align @ mag_data.T).T

# Save the rotated data back to the DataFrame
# imu_df['acc_x'] = acc_world[:, 0]
# imu_df['acc_y'] = acc_world[:, 1]
# imu_df['acc_z'] = acc_world[:, 2]

imu_df['gyro_x'] = gyro_world[:, 0]
imu_df['gyro_y'] = gyro_world[:, 1]
imu_df['gyro_z'] = gyro_world[:, 2]

imu_df['mag_x'] = mag_world[:, 0]
imu_df['mag_y'] = mag_world[:, 1]
imu_df['mag_z'] = mag_world[:, 2]

# 畫圖
# 抓出世界座標下的加速度分量
timestamps = imu_df['AppTimestamp(s)'].values

# 世界座標加速度
acc_x_world = imu_df['acc_x'].values
acc_y_world = imu_df['acc_y'].values
acc_z_world = imu_df['acc_z'].values

# 世界座標角速度
gyro_x_world = imu_df['gyro_x'].values
gyro_y_world = imu_df['gyro_y'].values
gyro_z_world = imu_df['gyro_z'].values

# 世界座標磁力計
mag_x_world = imu_df['mag_x'].values
mag_y_world = imu_df['mag_y'].values
mag_z_world = imu_df['mag_z'].values

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

'''
# 去除重力加速度分量
timestamps = imu_df['SensorTimestamp(s)'].values        # 時間戳記
acc_data = imu_df[['acc_x', 'acc_y', 'acc_z']].values    # 加速度 (ax, ay, az)
gyro_data = imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values   # 陀螺儀 (gx, gy, gz)

dynamic_acc = remove_gravity(acc_data, gyro_data, timestamps)

# 覆蓋 ax, ay, az
imu_df.iloc[1:, [imu_df.columns.get_loc('acc_x'),
                        imu_df.columns.get_loc('acc_y'),
                        imu_df.columns.get_loc('acc_z')]] = dynamic_acc

# 去掉第一列
imu_df = imu_df.iloc[1:].reset_index(drop=True)
'''
# 儲存校準後的資料
imu_df.to_csv(f'{index}/IMU_calibrated2.csv', index=False)
