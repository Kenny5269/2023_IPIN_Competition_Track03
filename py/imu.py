
import pandas as pd
import numpy as np


if __name__ == '__main__':
    # 讀取感測器資料
    acce_df = pd.read_csv('ACCE.csv')
    gyro_df = pd.read_csv('GYRO.csv')
    magn_df = pd.read_csv('MAGN.csv')

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

    # 匯出同步後資料
    aligned_df.to_csv('IMU_50Hz.csv', index=False)
