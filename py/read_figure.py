from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm
from ahrs.filters import Madgwick
from scipy.signal import find_peaks

# 讀取資料
index = 'T1_R1'
df = pd.read_csv(f'{index}/IMU_50Hz.csv')

final_export_df = pd.DataFrame({
    'AppTimestamp(s)': df['AppTimestamp(s)'],
    'SensorTimestamp(s)': df['SensorTimestamp(s)'],
    'acc_x': df['acc_x'],
    'acc_y': df['acc_y'],
    'acc_z': df['acc_z'],
    # 'acc_wx': df['acc_wx'],
    # 'acc_wy': df['acc_wy'],
    # 'acc_wz': df['acc_wz'],
    'gyro_x': df['gyro_x'],
    'gyro_y': df['gyro_y'],
    'gyro_z': df['gyro_z'],
    'mag_x': df['mag_x'],
    'mag_y': df['mag_y'],
    'mag_z': df['mag_z']
})

# 抓出世界座標下的加速度分量
timestamps = final_export_df[(final_export_df['AppTimestamp(s)'] >= 41.1)]['AppTimestamp(s)'].values

# 世界座標加速度
acc_x_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= 41.1)]['acc_x'].values
acc_y_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= 41.1)]['acc_y'].values
acc_z_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= 41.1)]['acc_z'].values

# 世界座標角速度
gyro_x_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= 41.1)]['gyro_x'].values
gyro_y_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= 41.1)]['gyro_y'].values
gyro_z_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= 41.1)]['gyro_z'].values

# 世界座標磁力計
mag_x_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= 41.1)]['mag_x'].values
mag_y_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= 41.1)]['mag_y'].values
mag_z_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= 41.1)]['mag_z'].values

parameters = {"axes.labelsize": 15, "axes.titlesize": 15, "xtick.labelsize": 15, "ytick.labelsize":15, "legend.fontsize":15}
plt.rcParams.update(parameters)

# 畫圖
plt.figure(figsize=(19.2, 10.8))
# figsize=(14, 12)
# 第一張圖：加速度
# plt.subplot(3, 1, 1)

plt.plot(timestamps, acc_x_world, label='acc_x_uncalibrated')
plt.plot(timestamps, acc_y_world, label='acc_y_uncalibrated')
plt.plot(timestamps, acc_z_world, label='acc_z_uncalibrated')
plt.title('Calibrated Accelerations')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend(loc="upper left")
plt.grid(True)

# 第二張圖：角速度
# plt.subplot(3, 1, 2)

# plt.plot(timestamps, gyro_x_world, label='gyro_x_uncalibrated')
# plt.plot(timestamps, gyro_y_world, label='gyro_y_uncalibrated')
# plt.plot(timestamps, gyro_z_world, label='gyro_z_uncalibrated')
# plt.title('Calibrated Angular Velocities')
# plt.xlabel('Time (s)')
# plt.ylabel('Angular Velocity (rad/s)')
# plt.legend(loc="upper left")
# plt.grid(True)

# 第三張圖：磁力計
# plt.subplot(3, 1, 3)

# plt.plot(timestamps, mag_x_world, label='mag_x_uncalibrated')
# plt.plot(timestamps, mag_y_world, label='mag_y_uncalibrated')
# plt.plot(timestamps, mag_z_world, label='mag_z_uncalibrated')
# plt.title('Calibrated Magnetometer Readings')
# plt.xlabel('Time (s)')
# plt.ylabel('Magnetic Field (μT)')
# plt.legend(loc="upper left")
# plt.grid(True)

plt.tight_layout()
plt.show()
