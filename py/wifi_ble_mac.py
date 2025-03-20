import pandas as pd

if __name__ == '__main__':
    index = 53

    # ============ WiFi 數據處理 ============
    wifi_df = pd.read_csv('WIFI.csv')  # 修改為你的檔案路徑

    pivot_wifi_rssi = wifi_df.pivot_table(index='AppTimestamp(s)', columns='MAC', values='RSSI', aggfunc='first')
    pivot_wifi_freq = wifi_df.pivot_table(index='AppTimestamp(s)', columns='MAC', values='Frequency', aggfunc='first')

    pivot_wifi_rssi.columns = [f'wifi_rssi_{mac}' for mac in pivot_wifi_rssi.columns]
    pivot_wifi_freq.columns = [f'wifi_freq_{mac}' for mac in pivot_wifi_freq.columns]

    merged_wifi_df = pd.concat([pivot_wifi_rssi, pivot_wifi_freq], axis=1).reset_index()
    merged_wifi_df.to_csv('WIFI_merged.csv', index=False)

    wifi_rssi_columns = list(pivot_wifi_rssi.columns)
    wifi_freq_columns = list(pivot_wifi_freq.columns)
    wifi_columns_df = pd.DataFrame({
        'wifi_rssi_columns': wifi_rssi_columns + [''] * (len(wifi_freq_columns) - len(wifi_rssi_columns)) if len(wifi_freq_columns) > len(wifi_rssi_columns) else wifi_rssi_columns,
        'wifi_freq_columns': wifi_freq_columns + [''] * (len(wifi_rssi_columns) - len(wifi_freq_columns)) if len(wifi_rssi_columns) > len(wifi_freq_columns) else wifi_freq_columns
    })
    wifi_columns_df.to_csv('WIFI_column_names.csv', index=False)

    # ============ BLE 數據處理 ============
    ble_df = pd.read_csv('BLE4.csv')  # 修改為你的檔案路徑

    pivot_ble_rssi = ble_df.pivot_table(index='AppTimestamp(s)', columns='MAC', values='RSSI', aggfunc='first')
    pivot_ble_rssi.columns = [f'ble_rssi_{mac}' for mac in pivot_ble_rssi.columns]

    merged_ble_df = pivot_ble_rssi.reset_index()
    merged_ble_df.to_csv('BLE4_merged.csv', index=False)

    ble_rssi_columns = list(pivot_ble_rssi.columns)
    ble_columns_df = pd.DataFrame({'ble_rssi_columns': ble_rssi_columns})
    ble_columns_df.to_csv('BLE4_column_names.csv', index=False)

    # ============ 合併所有欄位名稱輸出 ============
    all_columns = wifi_rssi_columns + wifi_freq_columns + ble_rssi_columns
    all_columns_df = pd.DataFrame({'all_feature_columns': all_columns})
    all_columns_df.to_csv(f'ALL_column_names_{index}.csv', index=False)

    print("整理完成！已輸出 WIFI_merged.csv、WIFI_column_names.csv、BLE4_merged.csv、BLE4_column_names.csv、ALL_column_names.csv")
