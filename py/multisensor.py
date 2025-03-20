
import pandas as pd
import numpy as np



# 定義通用對齊函數（根據 AppTimestamp 做最近時間點對齊）
def align_nearest_app_time(target_df, source_df, target_time_col, source_time_col, value_cols):
    aligned_data = []
    source_times = source_df[source_time_col].values
    source_values = source_df[value_cols].values

    for t in target_df[target_time_col].values:
        nearest_idx = np.argmin(np.abs(source_times - t))
        aligned_data.append(source_values[nearest_idx])

    aligned_df = pd.DataFrame(aligned_data, columns=[f"{col}" for col in value_cols])
    return pd.concat([target_df.reset_index(drop=True), aligned_df], axis=1)

if __name__ == '__main__':

    file_read = 'T53_R2'
    # 讀取資料
    aligned_df = pd.read_csv(f'./{file_read}/IMU_50Hz.csv')
    ble_df = pd.read_csv(f'./{file_read}/BLE4_merged.csv')
    wifi_df = pd.read_csv(f'./{file_read}/WIFI_merged.csv')
    posi_df = pd.read_csv(f'./{file_read}/POSI.csv')

    # 對齊 BLE 資料
    ble_cols = [col for col in ble_df.columns if col != "AppTimestamp(s)"]
    aligned_ble = align_nearest_app_time(aligned_df, ble_df, "AppTimestamp(s)", "AppTimestamp(s)", ble_cols)

    # 對齊 WiFi 資料
    wifi_cols = [col for col in wifi_df.columns if col != "AppTimestamp(s)"]
    aligned_wifi = align_nearest_app_time(aligned_ble, wifi_df, "AppTimestamp(s)", "AppTimestamp(s)", wifi_cols)

    # 對齊 POSI 資料
    posi_cols = [col for col in posi_df.columns if col != "AppTimestamp(s)"]
    aligned_full = align_nearest_app_time(aligned_wifi, posi_df, "AppTimestamp(s)", "AppTimestamp(s)", posi_cols)

    # NAN補值
    rssi_cols = [col for col in aligned_full.columns if 'rssi' in col]
    aligned_full[rssi_cols] = aligned_full[rssi_cols].fillna(-110)

    #wifi_rss_cols = [col for col in aligned_full.columns if 'rss' in col]
    #aligned_full[wifi_rss_cols] = aligned_full[wifi_rss_cols].fillna(-110)

    freq_cols = [col for col in aligned_full.columns if 'freq' in col]
    aligned_full[freq_cols] = aligned_full[freq_cols].fillna(-1)
    # 匯出結果
    aligned_full.to_csv(f'./{file_read}/MultiSensor_50Hz.csv', index=False)
