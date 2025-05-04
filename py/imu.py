
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


if __name__ == '__main__':
    index = 'C1'
    # 讀取感測器資料
    acce_df = pd.read_csv(f'{index}/ACCE.csv')
    gyro_df = pd.read_csv(f'{index}/GYRO.csv')
    magn_df = pd.read_csv(f'{index}/MAGN.csv')

    # 定義對齊函數：最近時間點
    def align_nearest(target_df, source_df, target_time_col, source_time_col, value_cols):
        aligned_data = []
        source_times = source_df[source_time_col].values
        source_values = source_df[value_cols].values

        for t in target_df[target_time_col].values:
            nearest_idx = np.argmin(np.abs(source_times - t))
            aligned_data.append(source_values[nearest_idx])

        aligned_df = pd.DataFrame(aligned_data, columns=[f"{col}" for col in value_cols])
        return pd.concat([target_df.reset_index(drop=True), aligned_df], axis=1)

    # 對齊 GYRO 與 MAGN 至 ACCE 的時間軸
    aligned_df = align_nearest(acce_df, gyro_df, "SensorTimestamp(s)", "SensorTimestamp(s)", ["gyro_x", "gyro_y", "gyro_z"])
    aligned_df = align_nearest(aligned_df, magn_df, "SensorTimestamp(s)", "SensorTimestamp(s)", ["mag_x", "mag_y", "mag_z"])

    # timestamps = aligned_df['SensorTimestamp(s)'].values        # 時間戳記
    # acc_data = aligned_df[['acc_x', 'acc_y', 'acc_z']].values    # 加速度 (ax, ay, az)
    # gyro_data = aligned_df[['gyro_x', 'gyro_y', 'gyro_z']].values   # 陀螺儀 (gx, gy, gz)

    # dynamic_acc = remove_gravity(acc_data, gyro_data, timestamps)

    # # 覆蓋 ax, ay, az
    # aligned_df.iloc[1:, [aligned_df.columns.get_loc('acc_x'),
    #                      aligned_df.columns.get_loc('acc_y'),
    #                      aligned_df.columns.get_loc('acc_z')]] = dynamic_acc
    
    # # 去掉第一列
    # aligned_df = aligned_df.iloc[1:].reset_index(drop=True)

    # 匯出同步後資料
    aligned_df.to_csv(f'{index}/IMU_50Hz.csv', index=False)
