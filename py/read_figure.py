from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
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

# parameters = {"axes.labelsize": 15, "axes.titlesize": 15, "xtick.labelsize": 15, "ytick.labelsize":15, "legend.fontsize":15}
# plt.rcParams.update(parameters)

fig, ax = plt.subplots()

# 畫圖
# plt.figure(figsize=(25.6, 14.4))
# plt.rcParams['figure.figsize'] = [19, 10]
# figsize=(14, 12)
# 第一張圖：加速度
# plt.subplot(3, 1, 1)

# ax.plot(timestamps, acc_x_world, label='acc_x_uncalibrated')
# ax.plot(timestamps, acc_y_world, label='acc_y_uncalibrated')
# ax.plot(timestamps, acc_z_world, label='acc_z_uncalibrated')
# ax.set_title('Uncalibrated Accelerations', fontsize=25)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.set_xlabel('Time (s)', fontsize=25)
# ax.set_ylabel('Acceleration (m/s²)', fontsize=25)
# ax.tick_params(axis='both', labelsize=25)
# ax.legend(loc="upper left", fontsize=25)
# ax.grid(True)
# fig.set_size_inches(25.6, 14.4)
# fig.subplots_adjust(left=0.05, right=0.98, bottom=0.07, top=0.95)
# fig.savefig(f'figure/T1/imu_calibrated/R1_ACCE_uncalibrated.png')

# 第二張圖：角速度
# plt.subplot(3, 1, 2)

# ax.plot(timestamps, gyro_x_world, label='gyro_x_uncalibrated')
# ax.plot(timestamps, gyro_y_world, label='gyro_y_uncalibrated')
# ax.plot(timestamps, gyro_z_world, label='gyro_z_uncalibrated')
# ax.set_title('Uncalibrated Angular Velocities', fontsize=25)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
# ax.set_xlabel('Time (s)', fontsize=25)
# ax.set_ylabel('Angular Velocity (rad/s)', fontsize=25)
# ax.tick_params(axis='both', labelsize=25)
# ax.legend(loc="upper left", fontsize=25)
# ax.grid(True)
# fig.set_size_inches(25.6, 14.4)
# fig.subplots_adjust(left=0.05, right=0.98, bottom=0.07, top=0.95)
# fig.savefig(f'figure/T1/imu_calibrated/R1_GYRO_uncalibrated.png')

# 第三張圖：磁力計
# plt.subplot(3, 1, 3)

ax.plot(timestamps, mag_x_world, label='mag_x_uncalibrated')
ax.plot(timestamps, mag_y_world, label='mag_y_uncalibrated')
ax.plot(timestamps, mag_z_world, label='mag_z_uncalibrated')
ax.set_title('Uncalibrated Magnetometer Readings', fontsize=25)
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
ax.set_xlabel('Time (s)', fontsize=25)
ax.set_ylabel('Magnetic Field (μT)', fontsize=25)
ax.tick_params(axis='both', labelsize=25)
ax.legend(loc="upper left", fontsize=25)
ax.grid(True)
fig.set_size_inches(25.6, 14.4)
fig.subplots_adjust(left=0.07, right=0.98, bottom=0.07, top=0.95)
fig.savefig(f'figure/T1/imu_calibrated/R1_MAGN_uncalibrated.png')

# plt.tight_layout(rect=(0.1,0.1,1,1))
# plt.subplots_adjust(left=0.05, right=0.98, bottom=0.05, top=0.95)
# plt.tight_layout()
# plt.show()
