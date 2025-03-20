
import pandas as pd


if __name__ == '__main__':
    index = 'T53_R2'

    # 讀取檔案
    all_columns_df = pd.read_csv("ALL_column_names_unique.csv")
    multi_sensor_df = pd.read_csv(f'./{index}/MultiSensor_50Hz.csv')

    # 擷取從第二列開始的前480個不重複欄位名稱
    selected_column_names = all_columns_df.iloc[1:481, 0].drop_duplicates().tolist()

    # 找出缺少的欄位
    missing_columns = [col for col in selected_column_names if col not in multi_sensor_df.columns]

    # 一次建立一個新的 DataFrame 補上缺少欄位（全部先填 NaN）
    missing_df = pd.DataFrame(columns=missing_columns, index=multi_sensor_df.index)

    # 合併原始資料與缺少欄位
    multi_sensor_df = pd.concat([multi_sensor_df, missing_df], axis=1)

    # 使用 .fillna() 補值
    multi_sensor_df.loc[:, multi_sensor_df.columns.str.contains('rssi', case=False)] = \
        multi_sensor_df.loc[:, multi_sensor_df.columns.str.contains('rssi', case=False)].fillna(-110)

    multi_sensor_df.loc[:, multi_sensor_df.columns.str.contains('freq', case=False)] = \
        multi_sensor_df.loc[:, multi_sensor_df.columns.str.contains('freq', case=False)].fillna(-1)

    # 移除 "unique_feature_columns" 欄位（如果有）
    if "unique_feature_columns" in multi_sensor_df.columns:
        multi_sensor_df.drop(columns=["unique_feature_columns"], inplace=True)

    # 欄位分類排序
    original_columns = [col for col in multi_sensor_df.columns if not any(k in col.lower() for k in ['rssi', 'freq'])]
    wifi_rssi_columns = sorted([col for col in multi_sensor_df.columns if 'wifi_rssi' in col.lower()])
    wifi_freq_columns = sorted([col for col in multi_sensor_df.columns if 'wifi_freq' in col.lower()])
    ble_rssi_columns = sorted([col for col in multi_sensor_df.columns if 'ble_rssi' in col.lower()])

    # 合併欄位順序並重排
    new_column_order = original_columns + wifi_rssi_columns + wifi_freq_columns + ble_rssi_columns

    # 將 Latitude, Longitude, floor_ID 移到最後
    location_columns = ['Latitude_degrees', 'Longitude_degrees', 'floor_ID']
    new_column_order = [col for col in new_column_order if col not in location_columns]
    new_column_order += [col for col in location_columns if col in multi_sensor_df.columns]

    # 套用新欄位順序
    multi_sensor_df = multi_sensor_df[new_column_order]

    # 匯出 CSV
    multi_sensor_df.to_csv(f'./training_data/{index}_MultiSensor_50Hz.csv', index=False)
