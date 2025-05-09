import os
import pickle
import numpy as np
from geopy.distance import distance, geodesic
from geopy import Point
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot_figure_train(aligned_data_train, aligned_data_test):
    # 可視化：軌跡圖
    plt.figure(figsize=(8, 6))
    gt_coords_origin = [(d["gt_lat_ori"], d["gt_lon_ori"]) for d in aligned_data_train if d["gt_lat_ori"] is not None]
    gt_coords = [(d["gt_lat"], d["gt_lon"]) for d in aligned_data_test if d["gt_lat"] is not None]
    wifi_coords = [(d["gt_lat_temp"], d["gt_lon_temp"]) for d in aligned_data_test if d["gt_lat_temp"] is not None and d["rssi_vector"] is not None]
    #print(len(gt_coords))
    #init_coords = [(d["init_lat"], d["init_lon"]) for d in aligned_data_test if d["gt_lat"] is not None]
    #print(len(init_coords))
    #pdr_coords = [(d["pdr_lat"], d["pdr_lon"]) for d in aligned_data_test if d["pdr_lat"] is not None]
    #pdr_tra = [(d["pdr_trajectory"]) for d in aligned_data]
    #pdr_tra = [pt for d in aligned_data_test if d["pdr_trajectory"] is not None for pt in d["pdr_trajectory"]]
    # fused_coords = [(d["fused_lat"], d["fused_lon"]) for d in aligned_data_test if d["gt_lat"] is not None and d["fused_lat"] is not None]
    # knn_coords = [(d["knn_lat"], d["knn_lon"]) for d in aligned_data_test if d["knn_lat"] is not None]
    # wifi_coords_test = [(d["wifi_lat"], d["wifi_lon"]) for d in aligned_data_test if d["wifi_lat"] is not None]
        
    #print(len(pdr_coords))
    # print(len(gt_coords))
    # print(len(pdr_coords))
    # print(len(fused_coords))


    gt_lats_ori, gt_lons_ori = zip(*gt_coords_origin)
    gt_lats, gt_lons = zip(*gt_coords)
    wifi_lats, wifi_lons = zip(*wifi_coords)
    #init_lats, init_lons = zip(*init_coords)
    #pdr_lats, pdr_lons = zip(*pdr_coords)
    #tra_lats, tra_lons = zip(*pdr_tra)
    # fused_lat, fused_lon = zip(*fused_coords)
    # knn_lat, knn_lon = zip(*knn_coords)
    # wifi_lats_test, wifi_lons_test = zip(*wifi_coords_test)


    plt.plot(gt_lons_ori, gt_lats_ori, label="Ground Truth origin", marker="o")
    plt.plot(gt_lons, gt_lats, label="Ground Truth", marker="o")
    # plt.plot(wifi_lons, wifi_lats, label="Wi-Fi Point", marker="o")
    #plt.plot(init_lons, init_lats, label="Wi-Fi Init", marker="x")
    #plt.plot(pdr_lons[105], pdr_lats[105], label="IMU PDR", marker="^")
    #plt.plot(tra_lons, tra_lats, label="IMU PDR", marker="^")
    # plt.plot(fused_lon, fused_lat, label="Wifi PDR Fused", marker="^")
    # plt.plot(knn_lon, knn_lat, label="KNN", marker="o")
    # plt.plot(wifi_lons_test, wifi_lats_test, label="Wi-Fi", marker="^")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("PDR vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_figure_test(aligned_data_train, aligned_data_test):
    fused_check = False
    knn_check = False
    wifi_check = False
    # 可視化：軌跡圖
    plt.figure(figsize=(8, 6))
    # gt_coords_origin = [(d["gt_lat_ori"], d["gt_lon_ori"]) for d in aligned_data_train if d["gt_lat_ori"] is not None]
    gt_coords = [(d["gt_lat"], d["gt_lon"]) for d in aligned_data_test if d["gt_lat"] is not None]
    # wifi_coords = [(d["gt_lat_temp"], d["gt_lon_temp"]) for d in aligned_data_test if d["gt_lat_temp"] is not None and d["rssi_vector"] is not None]
    #print(len(gt_coords))
    #init_coords = [(d["init_lat"], d["init_lon"]) for d in aligned_data_test if d["gt_lat"] is not None]
    #print(len(init_coords))
    #pdr_coords = [(d["pdr_lat"], d["pdr_lon"]) for d in aligned_data_test if d["pdr_lat"] is not None]
    #pdr_tra = [(d["pdr_trajectory"]) for d in aligned_data]
    #pdr_tra = [pt for d in aligned_data_test if d["pdr_trajectory"] is not None for pt in d["pdr_trajectory"]]
    # fused_coords = [(d["fused_lat"], d["fused_lon"]) for d in aligned_data_test if d["gt_lat"] is not None and d["fused_lat"] is not None]
    fused_coords = []
    knn_coords = []
    wifi_coords_test = []
    # knn_coords = [(d["knn_lat"], d["knn_lon"]) for d in aligned_data_test if d["knn_lat"] is not None]
    # wifi_coords_test = [(d["wifi_lat"], d["wifi_lon"]) for d in aligned_data_test if d["wifi_lat"] is not None]
    for i, d in enumerate(aligned_data_test):
        if fused_check:
            if d["fused_lat"] is not None:
                fused_coords.append((d["fused_lat"], d["fused_lon"]))
                fused_check = False
            continue
        if d["gt_lat"] is None:
            continue
        fused_check = True
    for i, d in enumerate(aligned_data_test):
        if knn_check:
            if d["knn_lat"] is not None:
                knn_coords.append((d["knn_lat"], d["knn_lon"]))
                knn_check = False
            continue
        if d["gt_lat"] is None:
            continue
        knn_check = True
    for i, d in enumerate(aligned_data_test):
        if wifi_check:
            if d["wifi_lat"] is not None:
                wifi_coords_test.append((d["wifi_lat"], d["wifi_lon"]))
                wifi_check = False
            continue
        if d["gt_lat"] is None:
            continue
        wifi_check = True
        
    #print(len(pdr_coords))
    # print(len(gt_coords))
    # print(len(pdr_coords))
    # print(len(fused_coords))


    # gt_lats_ori, gt_lons_ori = zip(*gt_coords_origin)
    gt_lats, gt_lons = zip(*gt_coords)
    # wifi_lats, wifi_lons = zip(*wifi_coords)
    #init_lats, init_lons = zip(*init_coords)
    #pdr_lats, pdr_lons = zip(*pdr_coords)
    #tra_lats, tra_lons = zip(*pdr_tra)
    fused_lat, fused_lon = zip(*fused_coords)
    knn_lat, knn_lon = zip(*knn_coords)
    wifi_lats_test, wifi_lons_test = zip(*wifi_coords_test)


    # plt.plot(gt_lons_ori, gt_lats_ori, label="Ground Truth origin", marker="o")
    plt.plot(gt_lons, gt_lats, label="Ground Truth", marker="o")
    #plt.plot(wifi_lons, wifi_lats, label="Wi-Fi Point", marker="o")
    #plt.plot(init_lons, init_lats, label="Wi-Fi Init", marker="x")
    #plt.plot(pdr_lons[105], pdr_lats[105], label="IMU PDR", marker="^")
    #plt.plot(tra_lons, tra_lats, label="IMU PDR", marker="^")
    plt.plot(fused_lon, fused_lat, label="Wifi PDR Fused", marker="^")
    # plt.plot(knn_lon, knn_lat, label="KNN", marker="o")
    # plt.plot(wifi_lons_test, wifi_lats_test, label="Wi-Fi", marker="^")

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("EKF vs Ground Truth")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def all_errors(aligned_data):
    num = 0
    wifi_init_errors = [
        geodesic((d["gt_lat"], d["gt_lon"]), (d["init_lat"], d["init_lon"])).meters
        for d in aligned_data if d["gt_lat"] is not None
    ]
    wifi_init_rmse = np.sqrt(np.mean(np.square(wifi_init_errors)))
    wifi_init_mean_error = np.mean(wifi_init_errors)
    wifi_init_std_error = np.std(wifi_init_errors)
    wifi_init_max_error = np.max(wifi_init_errors)

    pdr_errors = []
    for d in aligned_data:
        if num == 0 and d["pdr_lat"] is not None:
            prev_pdr_lat = d["pdr_lat"]
            prev_pdr_lon = d["pdr_lon"]
            num += 1
            continue
        if d["gt_lat"] is not None:
            pdr_errors.append(geodesic((d["gt_lat"], d["gt_lon"]), (prev_pdr_lat, prev_pdr_lon)).meters)
        if d["pdr_lat"] is not None:
            prev_pdr_lat = d["pdr_lat"]
            prev_pdr_lon = d["pdr_lon"]
    pdr_rmse = np.sqrt(np.mean(np.square(pdr_errors)))
    pdr_mean_error = np.mean(pdr_errors)
    pdr_std_error = np.std(pdr_errors)
    pdr_max_error = np.max(pdr_errors)

    fused_errors = [
        geodesic((d["gt_lat"], d["gt_lon"]), (d["fused_lat"], d["fused_lon"])).meters
        for d in aligned_data if d["gt_lat"] is not None
    ]
    fused_rmse = np.sqrt(np.mean(np.square(fused_errors)))
    fused_mean_error = np.mean(fused_errors)
    fused_std_error = np.std(fused_errors)
    fused_max_error = np.max(fused_errors)

    print(f'wifi_init_rmse = {wifi_init_rmse:.2f} m, wifi_init_mean_error = {wifi_init_mean_error:.2f} m, wifi_init_std_error = {wifi_init_std_error:.2f} m, wifi_init_max_error = {wifi_init_max_error:.2f} m')
    print(f'pdr_rmse = {pdr_rmse:.2f} m, pdr_mean_error = {pdr_mean_error:.2f} m, pdr_std_error = {pdr_std_error:.2f} m, pdr_max_error = {pdr_max_error:.2f} m')
    print(f'fused_rmse = {fused_rmse:.2f} m, fused_mean_error = {fused_mean_error:.2f} m, fused_std_error = {fused_std_error:.2f} m, fused_max_error = {fused_max_error:.2f} m')

def evaluate_errors(aligned_data, pred_key_lat="fused_lat", pred_key_lon="fused_lon", plot_cdf=True):
    dists = np.array([
        geodesic((d["gt_lat"], d["gt_lon"]), (d[pred_key_lat], d[pred_key_lon])).meters
        for d in aligned_data if d["gt_lat"] is not None
    ])
    rmse = np.sqrt(np.mean(dists ** 2))
    mean_error = np.mean(dists)
    std_error = np.std(dists)
    max_error = np.max(dists)

    if plot_cdf:
        sorted_errors = np.sort(dists)
        cdf = np.arange(len(dists)) / len(dists)
        plt.figure(figsize=(6, 4))
        plt.plot(sorted_errors, cdf, label="CDF of Error")
        plt.xlabel("Localization Error (m)")
        plt.ylabel("Cumulative Probability")
        plt.grid(True)
        plt.title("CDF of Localization Error")
        plt.tight_layout()
        plt.show()

    # print(f"RMSE: {rmse:.2f} m")
    # print(f"Mean Error: {mean_error:.2f} m")
    # print(f"Std Deviation: {std_error:.2f} m")
    # print(f"Max Error: {max_error:.2f} m")
    return {
        "rmse": rmse,
        "mean": mean_error,
        "std": std_error,
        "max": max_error,
        "all_errors": dists
    }


if __name__ == '__main__':
    trial_dict = {}

    root_dir = "aligned_trials"

    read_file = ['T2_R1', 'TEST1']

    # for trial_name in os.listdir(root_dir):
    #     trial_path = os.path.join(root_dir, trial_name)
    #     if not os.path.isdir(trial_path):
    #         continue

    #     trial_data = []
    #     for fname in sorted(os.listdir(trial_path)):
    #         if fname.endswith(".pkl"):
    #             with open(os.path.join(trial_path, fname), "rb") as f:
    #                 trial_data.append(pickle.load(f))
    #     trial_dict[trial_name] = trial_data

    for trial_name in read_file:
        trial_path = os.path.join(root_dir, trial_name)
        if not os.path.isdir(trial_path):
            continue

        trial_data = []
        # for fname in sorted(os.listdir(trial_path)):
        #     if fname.endswith(".pkl"):
        #         with open(os.path.join(trial_path, fname), "rb") as f:
        #             trial_data.append(pickle.load(f))

        with open(os.path.join(trial_path, 'all_data.pkl'), "rb") as f:
            trial_data = pickle.load(f)
        trial_dict[trial_name] = trial_data

    # aligned_data_train = trial_dict['T1_R1'] + trial_dict['T2_R1'] + trial_dict['T3_R1'] \
    #                 + trial_dict['T4_R1'] + trial_dict['T5_R1'] + trial_dict['T21_R1'] \
    #                     + trial_dict['T22_R1'] + trial_dict['T23_R1'] + trial_dict['T24_R1'] \
    #                         + trial_dict['T25_R1'] + trial_dict['T26_R1'] + trial_dict['T27_R1']

    aligned_data_train = trial_dict['TEST1']
    # plot_figure_train(aligned_data_train, aligned_data_train)

    # aligned_data_temp = trial_dict['temp_trial']

    # aligned_data_test1 = trial_dict['test_trial01']
    # aligned_data_test2 = trial_dict['test_trial02']
    # aligned_data_test3 = trial_dict['test_trial03']
    # aligned_data_test4 = trial_dict['test_trial04']

    # evaluate_errors(aligned_data_train)
    #all_errors(aligned_data_train)
    plot_figure_test(aligned_data_train, aligned_data_train)

    #evaluate_errors(aligned_data_test1)
    # all_errors(aligned_data_test1)
    # plot_figure(aligned_data_train, aligned_data_test1)

    # evaluate_errors(aligned_data_test2)
    # all_errors(aligned_data_test2)
    # plot_figure(aligned_data_train, aligned_data_test2)

    # evaluate_errors(aligned_data_test3)
    # all_errors(aligned_data_test3)
    # plot_figure(aligned_data_train, aligned_data_test3)

    # evaluate_errors(aligned_data_test4)
    # all_errors(aligned_data_test4)
    # plot_figure(aligned_data_train, aligned_data_test4)


