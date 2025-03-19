
import pandas as pd


if __name__ == '__main__':
    # 步驟1：讀取原始 WiFi 數據檔案
    wifi_df = pd.read_csv('WIFI.csv')  # 修改為你檔案的路徑

    # 步驟2：建立 RSS 和 Frequency 的 pivot table
    pivot_rss = wifi_df.pivot_table(index='AppTimestamp(s)', columns='MAC', values='RSSI', aggfunc='first')
    pivot_freq = wifi_df.pivot_table(index='AppTimestamp(s)', columns='MAC', values='Frequency', aggfunc='first')

    # 步驟3：重命名欄位名稱為 MAC_rssi 和 MAC_freq 格式
    pivot_rss.columns = [f'wifi_rssi_{mac}' for mac in pivot_rss.columns]
    pivot_freq.columns = [f'wifi_freq_{mac}' for mac in pivot_freq.columns]

    # 步驟4：合併兩張表格
    merged_df = pd.concat([pivot_rss, pivot_freq], axis=1).reset_index()

    # 步驟5：輸出為 CSV 檔案
    merged_df.to_csv('WIFI_merged.csv', index=False)  # 最終輸出檔案

    print("整理完成！輸出檔案為 WIFI_merged.csv")
