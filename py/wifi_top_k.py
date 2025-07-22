import pandas as pd


file = open('index.txt')
index = []
for line in file.read().splitlines():
    index.append(line)
file.close
print(index)
# 路徑
# index = 'T1_R1'
# out_path = "/mnt/data/WIFI_top6perTimestamp.csv"

top = 10
for name in index:
    # 讀檔
    df = pd.read_csv(f'{name}/WIFI.csv')

    # 保留原始讀入順序（避免 groupby 排序影響）
    df["_orig_order"] = range(len(df))

    # 依 AppTimestamp(s) 分群，每群保留前 6 列（依原始順序）
    # sort=False 保留群組出現順序；group_keys=False 讓 head 回傳平坦結果
    filtered = (
        df.sort_values("_orig_order")
        .groupby("AppTimestamp(s)", sort=False, group_keys=False)
        .head(top)
    )

    # 移除輔助欄位
    filtered = filtered.drop(columns="_orig_order")

    # 輸出
    filtered.to_csv(f'{name}/WIFI_top{top}.csv', index=False)

print(f"Done")
