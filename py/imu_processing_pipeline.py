
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as R
from numpy.linalg import norm
from ahrs.filters import Madgwick
from scipy.signal import find_peaks

def normalize(v):
    return v / norm(v) if norm(v) > 0 else v

def angular_difference_deg(a, b):
    """
    計算兩個角度之間的最小差值（單位：度），範圍為 [0, 180]
    支援 a, b 為 scalar 或 numpy array
    """
    diff = (a - b + 180) % 360 - 180
    return np.abs(diff)

def initialize_quaternion(acc0, mag0):
    z = acc0 / np.linalg.norm(acc0)
    x = np.cross(mag0, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    rot_matrix = np.vstack([x, y, z]).T
    quat = R.from_matrix(rot_matrix).as_quat()  # [x, y, z, w]
    return np.roll(quat, 1)  # 轉為 [w, x, y, z]

def initialize_quaternion_from_acc_mag(acc0, mag0):
    acc0 = acc0 / np.linalg.norm(acc0)
    mag0 = mag0 / np.linalg.norm(mag0)

    z_axis = acc0
    x_axis = np.cross(mag0, acc0)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)

    R_init = np.vstack([x_axis, y_axis, z_axis]).T  # 每一欄是單位向量
    q_scipy = R.from_matrix(R_init).as_quat()  # [x, y, z, w]
    return np.array([q_scipy[3], q_scipy[0], q_scipy[1], q_scipy[2]])  # 轉成 [w, x, y, z]

def detect_best_static_segment(df, window_size=100, min_time=40.0, max_time=60.0, 
                                acc_threshold=0.06, gyro_threshold=0.02, mag_threshold=0.5):
    subset = df[(df['AppTimestamp(s)'] >= min_time) & (df['AppTimestamp(s)'] <= max_time)].copy()

    best_score = float('inf')
    best_idx = None
    best_start, best_end = None, None

    for i in range(len(subset) - window_size):
        acc_win = subset[['acc_x', 'acc_y', 'acc_z']].iloc[i:i+window_size].to_numpy()
        gyro_win = subset[['gyro_x', 'gyro_y', 'gyro_z']].iloc[i:i+window_size].to_numpy()
        mag_win = subset[['mag_x', 'mag_y', 'mag_z']].iloc[i:i+window_size].to_numpy()

        acc_var = np.var(acc_win, axis=0)
        gyro_mean = np.mean(np.abs(gyro_win), axis=0)
        mag_var = np.var(mag_win, axis=0)

        acc_score = acc_var.mean()
        gyro_score = gyro_mean.mean()
        mag_score = mag_var.mean()

        if acc_score < acc_threshold and gyro_score < gyro_threshold and mag_score < mag_threshold:
            # total_score = (acc_score + gyro_score + mag_score) / 3
            # if total_score < best_score:
            #     best_score = total_score
            #     best_idx = subset.index[i]  # ✅ 回傳對應到原始 df 的 index
            #     best_start = subset.iloc[i]['AppTimestamp(s)']
            #     best_end = subset.iloc[i + window_size - 1]['AppTimestamp(s)']

            best_idx = subset.index[i]  # ✅ 回傳對應到原始 df 的 index
            best_start = subset.iloc[i]['AppTimestamp(s)']
            best_end = subset.iloc[i + window_size - 1]['AppTimestamp(s)']

    if best_idx is not None:
        return best_idx, best_start, best_end
    else:
        return None, None, None

def detect_best_static_acc_segment(df, window_size=100, min_time=40.0, max_time=60.0, acc_threshold=0.06):
    subset = df[(df['AppTimestamp(s)'] >= min_time) & (df['AppTimestamp(s)'] <= max_time)].copy()

    best_score = float('inf')
    best_idx = None
    best_start, best_end = None, None

    for i in range(len(subset) - window_size):
        acc_win = subset[['acc_x', 'acc_y', 'acc_z']].iloc[i:i+window_size].to_numpy()
        acc_var = np.var(acc_win, axis=0)
        acc_score = acc_var.mean()

        # if acc_score < acc_threshold and acc_score < best_score:
        #     best_score = acc_score
        #     best_idx = subset.index[i]
        #     best_start = subset.iloc[i]['AppTimestamp(s)']
        #     best_end = subset.iloc[i + window_size - 1]['AppTimestamp(s)']

        if acc_score < acc_threshold:
            best_idx = subset.index[i]
            best_start = subset.iloc[i]['AppTimestamp(s)']
            best_end = subset.iloc[i + window_size - 1]['AppTimestamp(s)']
            return best_idx, best_start, best_end

    # if best_idx is not None:
    #     return best_idx, best_start, best_end
    # else:
    #     return None, None, None


def detect_static_gyro_segment(df, window_size=100, max_time=60.0, gyro_threshold=0.02):
    subset = df[df['AppTimestamp(s)'] <= max_time].reset_index(drop=True)
    for i in range(len(subset) - window_size):
        gyro_win = subset[['gyro_x', 'gyro_y', 'gyro_z']].iloc[i:i+window_size].to_numpy()
        gyro_mean = np.mean(np.abs(gyro_win), axis=0)
        if gyro_mean.mean() < gyro_threshold:
            start_time = subset.loc[i, 'AppTimestamp(s)']
            end_time = subset.loc[i + window_size - 1, 'AppTimestamp(s)']
            return i, start_time, end_time
    return None, None, None

def detect_best_static_gyro_segment(df, window_size=100, min_time=30.0, max_time=60.0, gyro_threshold=0.02):
    subset = df[(df['AppTimestamp(s)'] >= min_time) & (df['AppTimestamp(s)'] <= max_time)].copy()

    best_score = float('inf')
    best_idx = None
    best_start, best_end = None, None

    for i in range(len(subset) - window_size):
        gyro_win = subset[['gyro_x', 'gyro_y', 'gyro_z']].iloc[i:i+window_size].to_numpy()
        gyro_mean = np.mean(np.abs(gyro_win), axis=0)
        gyro_score = gyro_mean.mean()

        if gyro_score < gyro_threshold and gyro_score < best_score:
            best_score = gyro_score
            best_idx = subset.index[i]
            best_start = subset.iloc[i]['AppTimestamp(s)']
            best_end = subset.iloc[i + window_size - 1]['AppTimestamp(s)']

    if best_idx is not None:
        return best_idx, best_start, best_end
    else:
        return None, None, None


def detect_static_mag_segment(df, window_size=100, max_time=60.0, mag_threshold=0.5):
    subset = df[df['AppTimestamp(s)'] <= max_time].reset_index(drop=True)
    for i in range(len(subset) - window_size):
        mag_win = subset[['mag_x', 'mag_y', 'mag_z']].iloc[i:i+window_size].to_numpy()
        mag_var = np.var(mag_win, axis=0)
        if mag_var.mean() < mag_threshold:
            start_time = subset.loc[i, 'AppTimestamp(s)']
            end_time = subset.loc[i + window_size - 1, 'AppTimestamp(s)']
            return i, start_time, end_time
    return None, None, None

def detect_best_static_mag_segment(df, window_size=100, min_time=30.0, max_time=60.0, mag_threshold=0.5):
    subset = df[(df['AppTimestamp(s)'] >= min_time) & (df['AppTimestamp(s)'] <= max_time)].copy()

    best_score = float('inf')
    best_idx = None
    best_start, best_end = None, None

    for i in range(len(subset) - window_size):
        mag_win = subset[['mag_x', 'mag_y', 'mag_z']].iloc[i:i+window_size].to_numpy()
        mag_var = np.var(mag_win, axis=0)
        mag_score = mag_var.mean()

        if mag_score < mag_threshold and mag_score < best_score:
            best_score = mag_score
            best_idx = subset.index[i]
            best_start = subset.iloc[i]['AppTimestamp(s)']
            best_end = subset.iloc[i + window_size - 1]['AppTimestamp(s)']

    if best_idx is not None:
        return best_idx, best_start, best_end
    else:
        return None, None, None


def madgwick_filter_with_mag_init(df, idx, t_start, t_end, beta=0.1, freq=50):
    dt = 1.0 / freq
    quaternions = []

    # idx, t_start, t_end = detect_static_segment(df)
    if idx is None:
        raise RuntimeError("❌ 找不到靜止段，請檢查資料或參數")
    
    # # 偵測 gyro norm 的 peak（可只用 gyro_z）
    # gyro_z = df['gyro_z'].values.astype(float)
    # gyro_peaks, _ = find_peaks(np.abs(gyro_z), height=0.5)  # 可調整 threshold
    # print(len(gyro_peaks))

    gyro_moving = (
        (np.abs(df['gyro_x']) > 0.35) |
        (np.abs(df['gyro_y']) > 0.35) |
        (np.abs(df['gyro_z']) > 0.35)
    )
    
    # 使用該靜止段建立初始四元數
    acc0 = df[['acc_x', 'acc_y', 'acc_z']].iloc[idx:idx+100].mean().to_numpy()
    mag0 = df[['mag_x', 'mag_y', 'mag_z']].iloc[idx:idx+100].mean().to_numpy()
    q = initialize_quaternion_from_acc_mag(acc0, mag0)
    # q = initialize_quaternion(acc0, mag0)
    q = -q
    print(f'初始四元數 = {q}')

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

        if gyro_moving.iloc[i]:
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

        # f = np.array([
        #     2*(q2*q4 - q1*q3) - acc[0],
        #     2*(q1*q2 + q3*q4) - acc[1],
        #     2*(0.5 - q2**2 - q3**2) - acc[2]
        # ])
        # J = np.array([
        #     [-2*q3,  2*q4, -2*q1, 2*q2],
        #     [ 2*q2,  2*q1,  2*q4, 2*q3],
        #     [ 0.0 , -4*q2, -4*q3, 0.0]
        # ])
        # step = normalize(J.T @ f)

        # q_dot = 0.5 * np.array([
        #     -q2*gx - q3*gy - q4*gz,
        #     q1*gx + q3*gz - q4*gy,
        #     q1*gy - q2*gz + q4*gx,
        #     q1*gz + q2*gy - q3*gx
        # ]) - beta * step

        # q += q_dot * dt
        # q = normalize(q)

        quaternions.append(q.copy())

    # 寫入欄位
    q_arr = np.array(quaternions)
    df['q_w'] = q_arr[:, 0]
    df['q_x'] = q_arr[:, 1]
    df['q_y'] = q_arr[:, 2]
    df['q_z'] = q_arr[:, 3]

    # # 第二種計算四元數方法
    # quaternions.append(q)
    # # quaternions.append([1.0, 0.0, 0.0, 0.0])
    # acc = df[["acc_x", "acc_y", "acc_z"]].to_numpy()
    # gyro = df[["gyro_x", "gyro_y", "gyro_z"]].to_numpy()
    # mag = df[["mag_x", "mag_y", "mag_z"]].to_numpy()
    # madgwick = Madgwick()
    # print(f'前 = {quaternions[idx]}')
    # for i in range(idx+1, len(df)):
    #     quaternions.append(madgwick.updateIMU(quaternions[i-1].copy(), gyr=gyro[i], acc=acc[i]))
    #     # quaternions.append(madgwick.updateMARG(quaternions[i-1].copy(), gyr=gyro[i], acc=acc[i], mag=mag[i]))
    #     # if gyro_moving.iloc[i]:
    #     #     # quaternions.append(madgwick.updateIMU(quaternions[i-1].copy(), gyr=gyro[i], acc=acc[i]))
    #     #     quaternions.append(madgwick.updateMARG(quaternions[i-1].copy(), gyr=gyro[i], acc=acc[i], mag=mag[i]))
    #     # else:
    #     #     quaternions.append(quaternions[i-1].copy())
    # print(f'後 = {quaternions[idx]}')
    # q_arr = np.array(quaternions)
    # df['q_w'] = q_arr[:, 0]
    # df['q_x'] = q_arr[:, 1]
    # df['q_y'] = q_arr[:, 2]
    # df['q_z'] = q_arr[:, 3]

    print(f"✅ 偵測到的靜止段：{t_start:.2f} 秒 ～ {t_end:.2f} 秒，從該段起開始估算四元數")
    return df

# 讀取資料
index = 'T1_R2'
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

# id_acc, start_acc, end_acc = detect_static_acc_segment(df)
# id_gyro, start_gyro, end_gyro = detect_static_gyro_segment(df)
# id_mag, start_mag, end_mag = detect_static_mag_segment(df)

id_acc, start_acc, end_acc = detect_best_static_acc_segment(df)
id_gyro, start_gyro, end_gyro = detect_best_static_gyro_segment(df)
id_mag, start_mag, end_mag = detect_best_static_mag_segment(df)

# id_acc, start_acc, end_acc = detect_best_static_segment(df)

gyro_bias = df[(df['AppTimestamp(s)'] >= start_gyro) & (df['AppTimestamp(s)'] <= end_gyro)][['gyro_x', 'gyro_y', 'gyro_z']].mean().values
mag_bias = df[(df['AppTimestamp(s)'] >= start_mag) & (df['AppTimestamp(s)'] <= end_mag)][['mag_x', 'mag_y', 'mag_z']].mean().values

df[['gyro_x', 'gyro_y', 'gyro_z']] -= gyro_bias
# df[['mag_x', 'mag_y', 'mag_z']] -= mag_bias

print(f'start_gyro = {start_acc}, end_gyro = {end_acc}')
print(f'gyro_bias = {gyro_bias}')

print(f'start_mag = {start_acc}, end_mag = {end_acc}')
print(f'mag_bias = {mag_bias}')


# 估計靜止區間 bias (start ~ end秒)
# static = df[(df['AppTimestamp(s)'] >= start) & (df['AppTimestamp(s)'] <= end)]
# gyro_bias = df[(df['AppTimestamp(s)'] >= start) & (df['AppTimestamp(s)'] <= end)][['gyro_x', 'gyro_y', 'gyro_z']].mean().values
# mag_bias = df[(df['AppTimestamp(s)'] >= start) & (df['AppTimestamp(s)'] <= end)][['mag_x', 'mag_y', 'mag_z']].mean().values

# print(f'start_gyro = {start_gyro}, end_gyro = {end_gyro}')
# print(f'gyro_bias = {gyro_bias}')

# print(f'start_mag = {start_mag}, end_mag = {end_mag}')
# print(f'mag_bias = {mag_bias}')

# 扣除 bias
# df[['gyro_x', 'gyro_y', 'gyro_z']] -= gyro_bias
# df[['mag_x', 'mag_y', 'mag_z']] -= mag_bias

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
df = madgwick_filter_with_mag_init(df, id_acc, start_acc , end_acc)

q_xyzw = df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()
q_xyzw_ori = df[['Quat_x', 'Quat_y', 'Quat_z', 'Quat_w']].to_numpy()

r = R.from_quat(q_xyzw)
eulers = r.as_euler('zyx', degrees=True)
df['yaw_deg'] = eulers[:, 0]

r2 = R.from_quat(q_xyzw_ori)
eulers2 = r2.as_euler('zyx', degrees=True)
df['yaw_deg_ori_cal'] = eulers2[:, 0]

df['yaw_error'] = angular_difference_deg(df['YawZ'], df['yaw_deg'])

# 四元數比對(error_degree)
# 將四元數組裝成 array（順序為 [x, y, z, w]）
q_ref = df[['Quat_x', 'Quat_y', 'Quat_z', 'Quat_w']].to_numpy()
q_est = df[['q_x', 'q_y', 'q_z', 'q_w']].to_numpy()

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
    quat = [row['Quat_x'], row['Quat_y'], row['Quat_z'], row['Quat_w']]
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
# id_acc, start_acc, end_acc = detect_best_static_acc_segment(df)
# print(f'start_acc = {start_acc}, end_acc = {end_acc}')
gravity = np.array([0, 0, 9.8])
acc_dynamic = acc_world - gravity
static_dyn = df[(df['AppTimestamp(s)'] >= start_acc) & (df['AppTimestamp(s)'] <= end_acc)][['acc_wx', 'acc_wy', 'acc_wz']].values - gravity
bias_world = static_dyn.mean(axis=0)
print(bias_world)
# acc_dynamic -= bias_world
df['acc_dx'], df['acc_dy'], df['acc_dz'] = acc_dynamic[:,0], acc_dynamic[:,1], acc_dynamic[:,2]

# id_gyro, start_gyro, end_gyro = detect_best_static_gyro_segment(df)
# id_mag, start_mag, end_mag = detect_best_static_mag_segment(df)

# gyro_bias = df[(df['AppTimestamp(s)'] >= start_gyro) & (df['AppTimestamp(s)'] <= end_gyro)][['gyro_wx', 'gyro_wy', 'gyro_wz']].mean().values
# mag_bias = df[(df['AppTimestamp(s)'] >= start_mag) & (df['AppTimestamp(s)'] <= end_mag)][['mag_wx', 'mag_wy', 'mag_wz']].mean().values

# df[['gyro_wx', 'gyro_wy', 'gyro_wz']] -= gyro_bias
# df[['mag_wx', 'mag_wy', 'mag_wz']] -= mag_bias

# print(f'start_gyro = {start_gyro}, end_gyro = {end_gyro}')
# print(f'gyro_bias = {gyro_bias}')

# print(f'start_mag = {start_mag}, end_mag = {end_mag}')
# print(f'mag_bias = {mag_bias}')

# 驗證靜止段動態加速度平均是否接近 0
check = df[(df['AppTimestamp(s)'] >= start_acc) & (df['AppTimestamp(s)'] <= end_acc)][['acc_dx', 'acc_dy', 'acc_dz']].mean()
print("靜止段動態加速度平均（應接近 [0, 0, 0]）：\n", check.values)
final_export_df = pd.DataFrame({
    'AppTimestamp(s)': df['AppTimestamp(s)'],
    'SensorTimestamp(s)': df['SensorTimestamp(s)'],
    'acc_x': df['acc_dx'],
    'acc_y': df['acc_dy'],
    'acc_z': df['acc_dz'],
    # 'acc_wx': df['acc_wx'],
    # 'acc_wy': df['acc_wy'],
    # 'acc_wz': df['acc_wz'],
    'gyro_x': df['gyro_wx'],
    'gyro_y': df['gyro_wy'],
    'gyro_z': df['gyro_wz'],
    'mag_x': df['mag_wx'],
    'mag_y': df['mag_wy'],
    'mag_z': df['mag_wz'],
    'yaw_deg_ori': df['YawZ'],
    'yaw_deg_ori_cal': df['yaw_deg_ori_cal'],
    'yaw_deg': df['yaw_deg'],
    'yaw_error': df['yaw_error'],
    'Quat_w': df['Quat_w'],
    'Quat_x': df['Quat_x'],
    'Quat_y': df['Quat_y'],
    'Quat_z': df['Quat_z'],
    'q_w': df['q_w'],
    'q_x': df['q_x'],
    'q_y': df['q_y'],
    'q_z': df['q_z'],
    'quat_angle_error_deg' : df['quat_angle_error_deg']
})

# # 濾波 acc, gyro, mag
# for sensor in ['acc', 'gyro', 'mag']:
#     for axis in ['x', 'y', 'z']:
#         col = f'{sensor}_{axis}'
#         final_export_df[col] = lowpass_filter(final_export_df[col])

# 畫圖
# 抓出世界座標下的加速度分量
# timestamps = final_export_df['AppTimestamp(s)'].values

# # 世界座標加速度
# acc_x_world = final_export_df['acc_x'].values
# acc_y_world = final_export_df['acc_y'].values
# acc_z_world = final_export_df['acc_z'].values

# # 世界座標角速度
# gyro_x_world = final_export_df['gyro_x'].values
# gyro_y_world = final_export_df['gyro_y'].values
# gyro_z_world = final_export_df['gyro_z'].values

# # 世界座標磁力計
# mag_x_world = final_export_df['mag_x'].values
# mag_y_world = final_export_df['mag_y'].values
# mag_z_world = final_export_df['mag_z'].values

# -----------------------------------------------------------------------------------------------
# 抓出世界座標下的加速度分量
timestamps = final_export_df[(final_export_df['AppTimestamp(s)'] >= start_acc)]['AppTimestamp(s)'].values

# 世界座標加速度
acc_x_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= start_acc)]['acc_x'].values
acc_y_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= start_acc)]['acc_y'].values
acc_z_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= start_acc)]['acc_z'].values

# 世界座標角速度
gyro_x_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= start_acc)]['gyro_x'].values
gyro_y_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= start_acc)]['gyro_y'].values
gyro_z_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= start_acc)]['gyro_z'].values

# 世界座標磁力計
mag_x_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= start_acc)]['mag_x'].values
mag_y_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= start_acc)]['mag_y'].values
mag_z_world = final_export_df[(final_export_df['AppTimestamp(s)'] >= start_acc)]['mag_z'].values



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
plt.legend(loc="upper right")
plt.grid(True)

# 第二張圖：角速度
plt.subplot(3, 1, 2)
plt.plot(timestamps, gyro_x_world, label='gyro_x_calibrated')
plt.plot(timestamps, gyro_y_world, label='gyro_y_calibrated')
plt.plot(timestamps, gyro_z_world, label='gyro_z_calibrated')
plt.title('Calibrated Angular Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Angular Velocity (rad/s)')
plt.legend(loc="upper right")
plt.grid(True)

# 第三張圖：磁力計
plt.subplot(3, 1, 3)
plt.plot(timestamps, mag_x_world, label='mag_x_calibrated')
plt.plot(timestamps, mag_y_world, label='mag_y_calibrated')
plt.plot(timestamps, mag_z_world, label='mag_z_calibrated')
plt.title('Calibrated Magnetometer Readings')
plt.xlabel('Time (s)')
plt.ylabel('Magnetic Field (μT)')
plt.legend(loc="upper right")
plt.grid(True)

plt.tight_layout()
plt.show()

# final_export_df.to_csv(f'{index}/IMU_calibrated3_temp3.csv', index=False)
# final_export_df.to_csv('IMU_calibrated3_temp2.csv', index=False)

