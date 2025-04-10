import os
import pickle
import numpy as np
from geopy.distance import distance, geodesic
from geopy import Point
import matplotlib.pyplot as plt

STEP_THRESHOLD = 1.2
FIXED_STEP_LENGTH = 0.7

if __name__ == '__main__':
    trial_dict = {}

    root_dir = "aligned_trials"

    for trial_name in os.listdir(root_dir):
        trial_path = os.path.join(root_dir, trial_name)
        if not os.path.isdir(trial_path):
            continue

        trial_data = []
        for fname in sorted(os.listdir(trial_path)):
            if fname.endswith(".pkl"):
                with open(os.path.join(trial_path, fname), "rb") as f:
                    trial_data.append(pickle.load(f))
        trial_dict[trial_name] = trial_data

    # aligned_data_train = trial_dict['trial_01'] + trial_dict['trial_02'] + trial_dict['trial_03'] \
    #                 + trial_dict['trial_04'] + trial_dict['trial_05'] + trial_dict['trial_21'] \
    #                     + trial_dict['trial_22'] + trial_dict['trial_23'] + trial_dict['trial_24'] \
    #                         + trial_dict['trial_25'] + trial_dict['trial_26'] + trial_dict['trial_27']

    aligned_data_train = trial_dict['all_trial']

    aligned_data = trial_dict['test_trial04']

    

    # 計算 PDR vs Ground Truth 的誤差（地理距離）
    num = 0
    wifi_init_errors = [
        geodesic((d["gt_lat"], d["gt_lon"]), (d["init_lat"], d["init_lon"])).meters
        for d in aligned_data
    ]
    wifi_init_rmse = np.sqrt(np.mean(np.square(wifi_init_errors)))
    wifi_init_mean_error = np.mean(wifi_init_errors)

    pdr_errors = []
    for d in aligned_data:
        if num == 0:
            prev_pdr_lat = d["pdr_lat"]
            prev_pdr_lon = d["pdr_lon"]
            num += 1
            continue
        pdr_errors.append(geodesic((d["gt_lat"], d["gt_lon"]), (prev_pdr_lat, prev_pdr_lon)).meters)
        prev_pdr_lat = d["pdr_lat"]
        prev_pdr_lon = d["pdr_lon"]
    pdr_rmse = np.sqrt(np.mean(np.square(pdr_errors)))
    pdr_mean_error = np.mean(pdr_errors)

    fused_errors = [
        geodesic((d["gt_lat"], d["gt_lon"]), (d["fused_lat"], d["fused_lon"])).meters
        for d in aligned_data
    ]
    fused_rmse = np.sqrt(np.mean(np.square(fused_errors)))
    fused_mean_error = np.mean(fused_errors)

    # 可視化：軌跡圖
    plt.figure(figsize=(8, 6))
    gt_coords_origin = [(d["gt_lat2"], d["gt_lon2"]) for d in aligned_data_train]
    gt_coords = [(d["gt_lat"], d["gt_lon"]) for d in aligned_data]
    init_coords = [(d["init_lat"], d["init_lon"]) for d in aligned_data]
    pdr_coords = [(d["pdr_lat"], d["pdr_lon"]) for d in aligned_data]
    #pdr_tra = [(d["pdr_trajectory"]) for d in aligned_data]
    pdr_tra = [pt for d in aligned_data for pt in d["pdr_trajectory"]]
    fused_coords = [(d["fused_lat"], d["fused_lon"]) for d in aligned_data]
    # print(len(gt_coords))
    # print(len(pdr_coords))


    gt_lats_ori, gt_lons_ori = zip(*gt_coords_origin)
    gt_lats, gt_lons = zip(*gt_coords)
    init_lats, init_lons = zip(*init_coords)
    pdr_lats, pdr_lons = zip(*pdr_coords)
    tra_lats, tra_lons = zip(*pdr_tra)
    fused_lat, fused_lon = zip(*fused_coords)


    #plt.plot(gt_lons_ori, gt_lats_ori, label="Ground Truth origin", marker="o")
    plt.plot(gt_lons, gt_lats, label="Ground Truth", marker="o")
    plt.plot(init_lons, init_lats, label="Wi-Fi Init", marker="x")
    #plt.plot(pdr_lons, pdr_lats, label="IMU PDR", marker="^")
    #plt.plot(tra_lons, tra_lats, label="IMU PDR", marker="^")
    #plt.plot(fused_lon, fused_lat, label="Wifi PDR Fused", marker="^")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("PDR 推估軌跡 vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f'wifi_init_rmse = {wifi_init_rmse}, wifi_init_mean_error = {wifi_init_mean_error}')
    print(f'pdr_rmse = {pdr_rmse}, pdr_mean_error = {pdr_mean_error}')
    print(f'fused_rmse = {fused_rmse}, fused_mean_error = {fused_mean_error}')
    #print(gt_coords)

