import os
import pickle
import numpy as np
from geopy.distance import distance, geodesic
from geopy import Point
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import warnings
warnings.filterwarnings("ignore")

# 假設基準點 (lat0, lon0) 為原點
def latlon_to_xy(lat, lon, lat0, lon0):
    R = 6371000  # 地球半徑（公尺）
    lat = np.radians(lat)
    lon = np.radians(lon)
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    x = R * (lon - lon0) * np.cos(lat0)
    y = R * (lat - lat0)
    return x, y

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
    pdr_check = False
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
    pdr_coords_test = []
    dists_knn = []
    dists_fused = []
    # knn_coords = [(d["knn_lat"], d["knn_lon"]) for d in aligned_data_test if d["knn_lat"] is not None]
    # wifi_coords_test = [(d["wifi_lat"], d["wifi_lon"]) for d in aligned_data_test if d["wifi_lat"] is not None]
    for i, d in enumerate(aligned_data_test):
        if fused_check:
            if d["wifi_lat"] is not None:
                # print(geodesic((lat, lon), (d["fused_lat"], d["fused_lon"])).meters)
                # print(f'{timestamp},{d["timestamp"]}')
                dists_fused.append(geodesic((lat, lon), (d["wifi_lat"], d["wifi_lon"])).meters)
                fused_check = False
            continue
        if d["gt_lat"] is None:
            continue
        lat = d["gt_lat"]
        lon = d["gt_lon"]
        # timestamp = d["timestamp"]
        fused_check = True
    # print(dists)
    
    rmse_fused = np.sqrt(np.mean(np.square(dists_fused)))
    mean_error_fused = np.mean(dists_fused)
    std_error_fused = np.std(dists_fused)
    max_error_fused = np.max(dists_fused)

    fused_check = False

    for i, d in enumerate(aligned_data_test):
        if fused_check:
            if d["wifi_lat"] is not None:
                # print(geodesic((lat, lon), (d["fused_lat"], d["fused_lon"])).meters)
                print(f'{timestamp},{d["timestamp"]}')
                dists_knn.append(geodesic((lat, lon), (d["wifi_lat"], d["wifi_lon"])).meters)
                fused_check = False
            continue
        if d["gt_lat"] is None:
            continue
        lat = d["gt_lat"]
        lon = d["gt_lon"]
        timestamp = d["timestamp"]
        fused_check = True
    # print(dists)
    
    rmse_knn = np.sqrt(np.mean(np.square(dists_knn)))
    mean_error_knn = np.mean(dists_knn)
    std_error_knn = np.std(dists_knn)
    max_error_knn = np.max(dists_knn)

    fused_check = False
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
    for i, d in enumerate(aligned_data_test):
        if pdr_check:
            if d["pdr_lat"] is not None:
                pdr_coords_test.append((d["pdr_lat"], d["pdr_lon"]))
            continue
        if d["gt_lat"] is None:
            continue
        pdr_check = True
        
    #print(len(pdr_coords))
    # print(len(gt_coords))
    # print(len(pdr_coords))
    # print(len(fused_coords))
    print(len(gt_coords))
    print(len(wifi_coords_test))
    print(len(fused_coords))


    # gt_lats_ori, gt_lons_ori = zip(*gt_coords_origin)
    gt_lats, gt_lons = zip(*gt_coords)
    # wifi_lats, wifi_lons = zip(*wifi_coords)
    #init_lats, init_lons = zip(*init_coords)
    #pdr_lats, pdr_lons = zip(*pdr_coords)
    #tra_lats, tra_lons = zip(*pdr_tra)
    fused_lat, fused_lon = zip(*fused_coords)
    knn_lat, knn_lon = zip(*knn_coords)
    wifi_lats_test, wifi_lons_test = zip(*wifi_coords_test)
    pdr_lats_test, pdr_lons_test = zip(*pdr_coords_test)

    ref_lat, ref_lon = gt_lats[0], gt_lons[0]  # 設定參考點
    gt_lats, gt_lons = latlon_to_xy(gt_lats, gt_lons, ref_lat, ref_lon)
    fused_lat, fused_lon = latlon_to_xy(fused_lat, fused_lon, ref_lat, ref_lon)
    knn_lat, knn_lon = latlon_to_xy(knn_lat, knn_lon, ref_lat, ref_lon)
    wifi_lats_test, wifi_lons_test = latlon_to_xy(wifi_lats_test, wifi_lons_test, ref_lat, ref_lon)

    fig, ax = plt.subplots()

    # plt.plot(gt_lons_ori, gt_lats_ori, label="Ground Truth origin", marker="o")
    ax.plot(gt_lats, gt_lons, label="Ground Truth", marker="o")
    #plt.plot(wifi_lons, wifi_lats, label="Wi-Fi Point", marker="o")
    #plt.plot(init_lons, init_lats, label="Wi-Fi Init", marker="x")
    #plt.plot(pdr_lons[105], pdr_lats[105], label="IMU PDR", marker="^")
    #plt.plot(tra_lons, tra_lats, label="IMU PDR", marker="^")

    # plt.plot(fused_lon, fused_lat, label="Wifi PDR Fused", marker="^")
    # plt.plot(knn_lon[0], knn_lat[0], label="KNN", marker="o")
    ax.plot(wifi_lats_test, wifi_lons_test, label="Wi-Fi", marker="^")
    # plt.plot(pdr_lons_test, pdr_lats_test, label="PDR", marker="^")

    # 標上每個點的編號
    # for i, (x, y) in enumerate(zip(xs, ys)):
    #     plt.text(x, y, str(i), fontsize=9, ha='center', va='bottom', color='blue')

    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.axis("equal")
    # plt.title("EKF vs Ground Truth")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    
    # ax.plot(xs, ys, 'o-')
    # ax.set_aspect('equal')  # 這行強制 x, y 軸單位長度一樣
    # ax.grid()
    # ax.title("EKF vs Ground Truth")
    # ax.set_xlabel("X (m)")
    # ax.set_ylabel("Y (m)")
    # ax.legend()
    # plt.show()

    ax.set_aspect('equal')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(10))
    ax.set_xlim(min(gt_lats + wifi_lats_test), max(gt_lats + wifi_lats_test))
    ax.set_ylim(min(gt_lons + wifi_lons_test), max(gt_lons + wifi_lons_test))
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True)
    ax.legend()
    plt.show()

    print(f'rmse_knn = {rmse_knn}')
    print(f'mean_error_knn = {mean_error_knn}')
    print(f'std_error_knn = {std_error_knn}')
    print(f'max_error_knn = {max_error_knn}')
    print(f'all_error_knn = {dists_knn}')
    
    print(f'rmse_fused = {rmse_fused}')
    print(f'mean_error_fused = {mean_error_fused}')
    print(f'std_error_fused = {std_error_fused}')
    print(f'max_error_fused = {max_error_fused}')
    print(f'all_error_fused = {dists_fused}')


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

    read_file = ['T1_R1', 'T2_R1', 'T3_R1', 'T4_R1', 'T5_R1', 'T21_R1', 'T22_R1', 'T23_R1', 'T24_R1', 'T25_R1', 'T26_R1', 'T27_R1', 'TEST1', 'TEST2', 'TEST3', 'TEST4']

    read_file_test = ['TEST1', 'TEST2', 'TEST3', 'TEST4']

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

    for trial_name in read_file_test:
        trial_path = os.path.join(root_dir, trial_name)
        if not os.path.isdir(trial_path):
            continue

        trial_data = []
        # for fname in sorted(os.listdir(trial_path)):
        #     if fname.endswith(".pkl"):
        #         with open(os.path.join(trial_path, fname), "rb") as f:
        #             trial_data.append(pickle.load(f))

        with open(os.path.join(trial_path, 'all_data_R123.pkl'), "rb") as f:
            trial_data = pickle.load(f)
        trial_dict[trial_name] = trial_data

    # aligned_data_train = trial_dict['T1_R1'] + trial_dict['T2_R1'] + trial_dict['T3_R1'] \
    #                 + trial_dict['T4_R1'] + trial_dict['T5_R1'] + trial_dict['T21_R1'] \
    #                     + trial_dict['T22_R1'] + trial_dict['T23_R1'] + trial_dict['T24_R1'] \
    #                         + trial_dict['T25_R1'] + trial_dict['T26_R1'] + trial_dict['T27_R1']
    aligned_data_test = trial_dict['TEST1']

    # aligned_data_train = trial_dict['T3_R1']
    # plot_figure_train(aligned_data_train, aligned_data_train)

    # aligned_data_temp = trial_dict['temp_trial']

    # aligned_data_test1 = trial_dict['test_trial01']
    # aligned_data_test2 = trial_dict['test_trial02']
    # aligned_data_test3 = trial_dict['test_trial03']
    # aligned_data_test4 = trial_dict['test_trial04']

    # evaluate_errors(aligned_data_train)
    #all_errors(aligned_data_train)
    plot_figure_test(aligned_data_test, aligned_data_test)

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


