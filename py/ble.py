
import pandas as pd


if __name__ == '__main__':
    # 步驟1：讀取 BLE 原始數據
    ble_df = pd.read_csv('BLE4.csv')  # 修改為你的檔案路徑

    # 步驟2：建立 RSSI 的 pivot table
    pivot_rssi = ble_df.pivot_table(index='AppTimestamp(s)', columns='MAC', values='RSSI', aggfunc='first')

    # 步驟3：重命名欄位名稱為 MAC_rssi 格式
    pivot_rssi.columns = [f'ble_rssi_{mac}' for mac in pivot_rssi.columns]

    # 步驟4：合併並輸出結果
    merged_ble_df = pivot_rssi.reset_index()
    merged_ble_df.to_csv('BLE4_merged.csv', index=False)

    print("整理完成！輸出檔案為 BLE4_merged.csv")
