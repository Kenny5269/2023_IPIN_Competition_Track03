import pandas as pd
import os
import glob

# === 設定資料夾路徑（請根據實際情況修改） ===
folder_path = './all_column_csv_files'  # 放置 ALL_column_names_*.csv 的資料夾路徑

# === 1. 搜尋所有匹配的 CSV 檔案 ===
csv_files = glob.glob(os.path.join(folder_path, 'ALL_column_names_*.csv'))

# === 2. 合併所有欄位 ===
all_features = pd.DataFrame()

for file in csv_files:
    df = pd.read_csv(file)
    if 'all_feature_columns' in df.columns:
        all_features = pd.concat([all_features, df[['all_feature_columns']]], axis=0)
    else:
        print(f"檔案 {file} 缺少 all_feature_columns 欄位，已略過。")

# === 3. 去重並儲存 ALL_column_names_unique.csv ===
unique_features = all_features['all_feature_columns'].drop_duplicates().reset_index(drop=True)
unique_features_df = pd.DataFrame({'unique_feature_columns': unique_features})
unique_features_path = os.path.join(folder_path, 'ALL_column_names_unique.csv')
unique_features_df.to_csv(unique_features_path, index=False)

# === 4. 根據 prefix 分類並儲存 ===
wifi_rssi = unique_features[unique_features.str.startswith('wifi_rssi_')].reset_index(drop=True)
wifi_freq = unique_features[unique_features.str.startswith('wifi_freq_')].reset_index(drop=True)
ble_rssi  = unique_features[unique_features.str.startswith('ble_rssi_')].reset_index(drop=True)

# 儲存分類檔案
wifi_rssi.to_frame(name='wifi_rssi_columns').to_csv(os.path.join(folder_path, 'unique_wifi_rssi.csv'), index=False)
wifi_freq.to_frame(name='wifi_freq_columns').to_csv(os.path.join(folder_path, 'unique_wifi_freq.csv'), index=False)
ble_rssi.to_frame(name='ble_rssi_columns').to_csv(os.path.join(folder_path, 'unique_ble_rssi.csv'), index=False)

print("所有整理已完成,輸出4個檔案")
print("- ALL_column_names_unique.csv")
print("- unique_wifi_rssi.csv")
print("- unique_wifi_freq.csv")
print("- unique_ble_rssi.csv")
