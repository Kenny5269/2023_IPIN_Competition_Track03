import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def kalman_filter_rssi_masked(data, process_var=1.0, measure_var=9.0, init_error=1.0):      # 只對有收到rssi的時間點做濾波
    data = np.array(data)
    valid_mask = data > -100  # -100 視為無效數值
    valid_data = data[valid_mask]

    if len(valid_data) == 0:
        return data.tolist()

    # 初始化
    x = valid_data[0]
    P = init_error
    Q = process_var
    R = measure_var
    filtered_valid = []

    for z in valid_data:
        P = P + Q
        K = P / (P + R)
        x = x + K * (z - x)
        P = (1 - K) * P
        filtered_valid.append(x)

    # 把濾波結果填回原陣列
    result = data.copy()
    result[valid_mask] = filtered_valid
    return result.tolist()

trial = 'T27'
repitition = 'R4'
input_file = 'T53_R2'

file = open('index.txt')
index = []
for line in file.read().splitlines():
    index.append(line)
file.close
print(index)

top = 10
for name in index:
    wifi_df = pd.read_csv(f'{name}/WIFI_merged2_top{top}.csv')
    filtered_df = wifi_df.copy()
    rssi_cols = [col for col in wifi_df.columns if col.startswith("wifi_rssi_")]

    filtered_df[rssi_cols] = wifi_df[rssi_cols].apply(lambda col: kalman_filter_rssi_masked(col.values))

    filtered_df.to_csv(f'{name}/WIFI_merged_filtered_top{top}.csv', index=False)

# timestamps = wifi_df['AppTimestamp(s)'].values

# plt.rcParams['figure.figsize'] = [25.6, 14.4]

# num = 1

# for i, col_name in enumerate(rssi_cols):
#     # print(i)
#     rssi_ori = wifi_df[col_name].values
#     rssi_filtered = filtered_df[col_name].values

#     # plt.figure(figsize=(25.6, 14.4))
#     plt.subplot(5, 1, (i%5)+1)
#     plt.plot(timestamps, rssi_ori, label='rssi origin')
#     plt.plot(timestamps, rssi_filtered, label='rssi filtered')
#     plt.title(col_name)
#     plt.xlabel('Time (s)')
#     plt.ylabel('rssi')
#     plt.legend(loc="upper right")
#     plt.grid(True)

#     if i % 5 == 4:
#         plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=1)
#         # plt.savefig(f'figure/{trial}/wifi_rssi_filtered/{repitition}/{num}.png')
#         plt.savefig(f'figure/{input_file}/wifi_rssi_filtered/{num}.png')
#         num += 1
#         plt.clf()
#         # plt.figure()
#         # plt.show()

#     # plt.subplot(2, 1, 2)
#     # plt.plot(timestamps, rssi_filtered, label='rssi filtered')
#     # plt.title('rssi filtered')
#     # plt.xlabel('Time (s)')
#     # plt.ylabel('rssi')
#     # plt.legend()
#     # plt.grid(True)

#     # plt.tight_layout()
#     # plt.show()

# plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=1)
# # plt.savefig(f'figure/{trial}/wifi_rssi_filtered/{repitition}/{num}.png')
# plt.savefig(f'figure/{input_file}/wifi_rssi_filtered/{num}.png')
# # plt.show()

# filtered_df.to_csv(f'{input_file}/WIFI_merged_filtered.csv', index=False)
