
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt

def lowpass_filter(data, cutoff_freq, fs, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff_freq / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    if data.ndim == 1:
        return filtfilt(b, a, data)
    else:
        return np.stack([filtfilt(b, a, data[:, i]) for i in range(data.shape[1])], axis=1)

class MadgwickAHRS:
    def __init__(self, sampleperiod=1/50, beta=0.1):
        self.sampleperiod = sampleperiod
        self.beta = beta
        self.q = np.array([1.0, 0.0, 0.0, 0.0])

    def update(self, gyro, acc, mag):
        q = self.q
        if np.linalg.norm(acc) == 0 or np.linalg.norm(mag) == 0:
            return
        acc /= np.linalg.norm(acc)
        mag /= np.linalg.norm(mag)
        h = self._quat_mult(q, self._quat_mult(np.hstack([0, mag]), self._quat_conj(q)))
        b = np.array([0, np.linalg.norm(h[1:3]), 0, h[3]])
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - acc[0],
            2*(q[0]*q[1] + q[2]*q[3]) - acc[1],
            2*(0.5 - q[1]**2 - q[2]**2) - acc[2],
            2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]) - mag[0],
            2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - mag[1],
            2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - mag[2]
        ])
        J = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0],
            [-2*b[3]*q[2], 2*b[3]*q[3], -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1], 2*b[1]*q[2]+2*b[3]*q[0], 2*b[1]*q[1]+2*b[3]*q[3], -2*b[1]*q[0]+2*b[3]*q[2]],
            [2*b[1]*q[2], 2*b[1]*q[3]-4*b[3]*q[1], 2*b[1]*q[0]-4*b[3]*q[2], 2*b[1]*q[1]]
        ])
        step = J.T @ f
        step /= np.linalg.norm(step)
        q_dot = 0.5 * self._quat_mult(q, np.hstack([0, gyro])) - self.beta * step
        self.q += q_dot * self.sampleperiod
        self.q /= np.linalg.norm(self.q)

    def _quat_mult(self, q, r):
        return np.array([
            q[0]*r[0] - q[1]*r[1] - q[2]*r[2] - q[3]*r[3],
            q[0]*r[1] + q[1]*r[0] + q[2]*r[3] - q[3]*r[2],
            q[0]*r[2] - q[1]*r[3] + q[2]*r[0] + q[3]*r[1],
            q[0]*r[3] + q[1]*r[2] - q[2]*r[1] + q[3]*r[0]
        ])

    def _quat_conj(self, q):
        return np.array([q[0], -q[1], -q[2], -q[3]])

# 主流程
index = 'T1_R1'

imu_df = pd.read_csv(f'{index}/IMU_50Hz.csv')
imu_df = imu_df[imu_df['AppTimestamp(s)'] >= 40].reset_index(drop=True)
fs = 50
timestamps = imu_df['AppTimestamp(s)'].values

acc_f = lowpass_filter(imu_df[['acc_x', 'acc_y', 'acc_z']].values, 5, fs)
gyro_f = lowpass_filter(imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values, 10, fs)
mag_f = lowpass_filter(imu_df[['mag_x', 'mag_y', 'mag_z']].values, 3, fs)

imu_df['acc_x'] = acc_f[:, 0]
imu_df['acc_y'] = acc_f[:, 1]
imu_df['acc_z'] = acc_f[:, 2]

static_mask = (timestamps >= 40) & (timestamps <= 50)
#acc_bias = acc_f[static_mask].mean(axis=0)
gyro_bias = gyro_f[static_mask].mean(axis=0)
mag_bias = mag_f[static_mask].mean(axis=0)

#acc_corr = acc_f - acc_bias
gyro_corr = gyro_f - gyro_bias
mag_corr = mag_f - mag_bias

madgwick = MadgwickAHRS(sampleperiod=1/fs)
R_list = []
for i in range(len(imu_df)):
    madgwick.update(gyro_corr[i], acc_f[i], mag_corr[i])
    q = madgwick.q
    R = np.array([
        [1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
        [2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[1]**2 + q[3]**2), 2*(q[2]*q[3] - q[0]*q[1])],
        [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2)]
    ])
    R_list.append(R)

acc_world = np.array([R_list[i] @ acc_f[i] for i in range(len(imu_df))])
gyro_world = np.array([R_list[i] @ gyro_corr[i] for i in range(len(imu_df))])
mag_world = np.array([R_list[i] @ mag_corr[i] for i in range(len(imu_df))])
acc_dyn = acc_world - np.array([0, 0, 9.81])
acc_bias = acc_dyn[static_mask].mean(axis=0)
acc_dyn = acc_dyn - acc_bias
#acc_dyn = acc_world

# imu_df['acc_x'] = acc_f[:, 0]
# imu_df['acc_y'] = acc_f[:, 1]
# imu_df['acc_z'] = acc_f[:, 2]
imu_df['gyro_x'] = gyro_world[:, 0]
imu_df['gyro_y'] = gyro_world[:, 1]
imu_df['gyro_z'] = gyro_world[:, 2]
imu_df['mag_x'] = mag_world[:, 0]
imu_df['mag_y'] = mag_world[:, 1]
imu_df['mag_z'] = mag_world[:, 2]

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
plt.plot(timestamps, acc_x_world, label='acc_x_world')
plt.plot(timestamps, acc_y_world, label='acc_y_world')
plt.plot(timestamps, acc_z_world, label='acc_z_world')
plt.title('World-aligned Accelerations')
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s²)')
plt.legend()
plt.grid(True)

# 第二張圖：角速度
plt.subplot(3, 1, 2)
plt.plot(timestamps, gyro_x_world, label='gyro_x_world')
plt.plot(timestamps, gyro_y_world, label='gyro_y_world')
plt.plot(timestamps, gyro_z_world, label='gyro_z_world')
plt.title('World-aligned Angular Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend()
plt.grid(True)

# 第三張圖：磁力計
plt.subplot(3, 1, 3)
plt.plot(timestamps, mag_x_world, label='mag_x_world')
plt.plot(timestamps, mag_y_world, label='mag_y_world')
plt.plot(timestamps, mag_z_world, label='mag_z_world')
plt.title('World-aligned Magnetometer Readings')
plt.xlabel('Time (s)')
plt.ylabel('Magnetic Field (μT)')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

imu_df.to_csv(f'{index}/IMU_Calibrated.csv', index=False)
