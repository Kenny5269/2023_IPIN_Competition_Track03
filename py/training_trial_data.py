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

    aligned_data = trial_dict['trial_01']

    

    # 計算 PDR vs Ground Truth 的誤差（地理距離）
    pdr_errors = [
        geodesic((d["gt_lat"], d["gt_lon"]), (d["init_lat"], d["init_lon"])).meters
        for d in aligned_data
    ]
    pdr_rmse = np.sqrt(np.mean(np.square(pdr_errors)))
    pdr_mean_error = np.mean(pdr_errors)

    # 可視化：軌跡圖
    plt.figure(figsize=(8, 6))
    gt_coords = [(d["gt_lat2"], d["gt_lon2"]) for d in aligned_data]
    init_coords = [(d["init_lat"], d["init_lon"]) for d in aligned_data]
    pdr_coords = [(d["pdr_lat"], d["pdr_lon"]) for d in aligned_data]
    #pdr_tra = [(d["pdr_trajectory"]) for d in aligned_data]
    pdr_tra = [pt for d in aligned_data for pt in d["pdr_trajectory"]]
    # print(len(gt_coords))
    # print(len(pdr_coords))


    gt_lats, gt_lons = zip(*gt_coords)
    init_lats, init_lons = zip(*init_coords)
    pdr_lats, pdr_lons = zip(*pdr_coords)
    tra_lats, tra_lons = zip(*pdr_tra)


    plt.plot(gt_lons, gt_lats, label="Ground Truth", marker="o")
    plt.plot(init_lons, init_lats, label="Wi-Fi Init", marker="x")
    #plt.plot(pdr_lons, pdr_lats, label="IMU PDR", marker="^")
    #plt.plot(tra_lons, tra_lats, label="IMU PDR", marker="^")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("PDR 推估軌跡 vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(pdr_rmse, pdr_mean_error)
    #print(gt_coords)

