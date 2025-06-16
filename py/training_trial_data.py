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
    # plt.figure(figsize=(8, 6))
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

    ref_lat, ref_lon = gt_lats_ori[0], gt_lons_ori[0]  # 設定參考點
    gt_lats_ori, gt_lons_ori = latlon_to_xy(gt_lats_ori, gt_lons_ori, ref_lat, ref_lon)
    gt_lats, gt_lons = latlon_to_xy(gt_lats, gt_lons, ref_lat, ref_lon)
    wifi_lats, wifi_lons = latlon_to_xy(wifi_lats, wifi_lons, ref_lat, ref_lon)

    fig, ax = plt.subplots()

    trial = 'T27'
    repitition = 'R4'
    interval = 1
    offset = 5
    width = 25.6
    height = 14.4


    # plt.plot(gt_lons_ori, gt_lats_ori, label="Ground Truth origin", marker="o", linewidth=0)
    # plt.plot(gt_lons, gt_lats, label="PDR Trajectory", marker="o")
    # plt.plot(wifi_lons, wifi_lats, label="Wi-Fi Point", marker="o", linewidth=0)

    #plt.plot(init_lons, init_lats, label="Wi-Fi Init", marker="x")
    #plt.plot(pdr_lons[105], pdr_lats[105], label="IMU PDR", marker="^")
    #plt.plot(tra_lons, tra_lats, label="IMU PDR", marker="^")
    # plt.plot(fused_lon, fused_lat, label="Wifi PDR Fused", marker="^")
    # plt.plot(knn_lon, knn_lat, label="KNN", marker="o")
    # plt.plot(wifi_lons_test, wifi_lats_test, label="Wi-Fi", marker="^")

    # plt.xlabel("Longitude")
    # plt.ylabel("Latitude")
    # plt.title("PDR vs Ground Truth")
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    
    ax.plot(gt_lats, gt_lons, label="PDR Trajectory", marker="o", color='#ff7f0e')
    ax.plot(wifi_lats, wifi_lons, label="Wi-Fi Point", marker="o", color='#d62728', linewidth=0)
    ax.plot(gt_lats_ori, gt_lons_ori, label="Ground Truth", marker="o", color='blue', linewidth=0)

    # 加上編號（GT點）
    for i, (x, y) in enumerate(zip(gt_lats_ori, gt_lons_ori)):
        if i < (len(gt_lats_ori)/2):
            ax.text(x, y + 0.5, f'{i+1},{len(gt_lats_ori)-i}', fontsize=9, ha='center', va='bottom', color='blue')
    # 加上編號（Pred點）
    for i, (x, y) in enumerate(zip(wifi_lats, wifi_lons)):
        ax.text(x, y - 0.5, f'{i+1}', fontsize=9, ha='center', va='top', color='#d62728')

    # ax.set_aspect('equal')
    ax.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax.set_xlim(min(gt_lats)-offset, max(gt_lats)+offset)
    ax.set_ylim(min(gt_lons)-offset, max(gt_lons)+offset)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.grid(True)
    ax.legend()
    fig.set_size_inches(width, height)
    fig.savefig(f'figure/{trial}/trajectory/{repitition}.png')
    plt.show()

def plot_figure_test(aligned_data_train, aligned_data_test, id, re):
    fused_check = False
    knn_check = False
    wifi_check = False
    pdr_check = False
    gt_check = False
    # 可視化：軌跡圖
    # plt.figure(figsize=(8, 6))
    # gt_coords_origin = [(d["gt_lat_ori"], d["gt_lon_ori"]) for d in aligned_data_train if d["gt_lat_ori"] is not None]
    gt_coords = [(d["gt_lat"], d["gt_lon"]) for d in aligned_data_test if d["gt_lat"] is not None]
    # wifi_coords = [(d["gt_lat_temp"], d["gt_lon_temp"]) for d in aligned_data_test if d["gt_lat_temp"] is not None and d["rssi_vector"] is not None]
    #print(len(gt_coords))
    #init_coords = [(d["init_lat"], d["init_lon"]) for d in aligned_data_test if d["gt_lat"] is not None]
    #print(len(init_coords))
    #pdr_coords = [(d["pdr_lat"], d["pdr_lon"]) for d in aligned_data_test if d["pdr_lat"] is not None]
    #pdr_tra = [(d["pdr_trajectory"]) for d in aligned_data]
    #pdr_tra = [pt for d in aligned_data_test if d["pdr_trajectory"] is not None for pt in d["pdr_trajectory"]]
    # fused_coords_all = [(d["fused_lat"], d["fused_lon"]) for d in aligned_data_test if d["fused_lat"] is not None]

    knn_coords = []
    wifi_coords = []
    fused_coords = []

    knn_coords_all = []
    wifi_coords_all = []
    fused_coords_all = []

    pdr_coords_test = []
    dists_knn = []
    dists_ekf_wifi = []
    dists_ekf_fused = []
    # knn_coords = [(d["knn_lat"], d["knn_lon"]) for d in aligned_data_test if d["knn_lat"] is not None]
    # wifi_coords_test = [(d["wifi_lat"], d["wifi_lon"]) for d in aligned_data_test if d["wifi_lat"] is not None]
    for i, d in enumerate(aligned_data_test):
        if d["knn_lat"] is not None:
            last_knn_lat, last_knn_lon = d["knn_lat"], d["knn_lon"]
        if d["gt_lat"] is not None:
            knn_coords.append((last_knn_lat, last_knn_lon))
            dists_knn.append(geodesic((last_knn_lat, last_knn_lon), (d["gt_lat"], d["gt_lon"])).meters)
        # timestamp = d["timestamp"]
    # print(dists)

    rmse_knn = np.sqrt(np.mean(np.square(dists_knn)))
    mean_knn = np.mean(dists_knn)
    median_knn = np.median(dists_knn)
    std_knn = np.std(dists_knn)
    max_knn = np.max(dists_knn)

    for i, d in enumerate(aligned_data_test):
        if d["wifi_lat"] is not None:
            last_wifi_lat, last_wifi_lon = d["wifi_lat"], d["wifi_lon"]
        if d["gt_lat"] is not None:
            wifi_coords.append((last_wifi_lat, last_wifi_lon))
            dists_ekf_wifi.append(geodesic((last_wifi_lat, last_wifi_lon), (d["gt_lat"], d["gt_lon"])).meters)
        # timestamp = d["timestamp"]

    # print(dists)
    
    rmse_ekf_wifi = np.sqrt(np.mean(np.square(dists_ekf_wifi)))
    mean_ekf_wifi = np.mean(dists_ekf_wifi)
    median_ekf_wifi = np.median(dists_ekf_wifi)
    std_ekf_wifi = np.std(dists_ekf_wifi)
    max_ekf_wifi = np.max(dists_ekf_wifi)

    for i, d in enumerate(aligned_data_test):
        if d["fused_lat"] is not None:
            last_fused_lat, last_fused_lon = d["fused_lat"], d["fused_lon"]
        if d["gt_lat"] is not None:
            fused_coords.append((last_fused_lat, last_fused_lon))
            dists_ekf_fused.append(geodesic((last_fused_lat, last_fused_lon), (d["gt_lat"], d["gt_lon"])).meters)
        # timestamp = d["timestamp"]

    # print(dists)
    
    rmse_ekf_fused = np.sqrt(np.mean(np.square(dists_ekf_fused)))
    mean_ekf_fused = np.mean(dists_ekf_fused)
    median_ekf_fused = np.median(dists_ekf_fused)
    std_ekf_fused = np.std(dists_ekf_fused)
    max_ekf_fused = np.max(dists_ekf_fused)

    for i, d in enumerate(aligned_data_test):
        # if gt_check:
        #     if d["knn_lat"] is not None:
        #         knn_coords_all.append((d["knn_lat"], d["knn_lon"]))
        #     if d["wifi_lat"] is not None:
        #         wifi_coords_all.append((d["wifi_lat"], d["wifi_lon"]))
        #     if d["fused_lat"] is not None:
        #         fused_coords_all.append((d["fused_lat"], d["fused_lon"]))
        # elif d["gt_lat"] is not None:
        #     gt_check = True
        if d["knn_lat"] is not None:
            knn_coords_all.append((d["knn_lat"], d["knn_lon"]))
        if d["wifi_lat"] is not None:
            wifi_coords_all.append((d["wifi_lat"], d["wifi_lon"]))
        if d["fused_lat"] is not None:
            fused_coords_all.append((d["fused_lat"], d["fused_lon"]))

    # for i, d in enumerate(aligned_data_test):
    #     if fused_check:
    #         if d["fused_lat"] is not None:
    #             fused_coords.append((d["fused_lat"], d["fused_lon"]))
    #             fused_check = False
    #         continue
    #     if d["gt_lat"] is None:
    #         continue
    #     fused_check = True
    # for i, d in enumerate(aligned_data_test):
    #     if knn_check:
    #         if d["knn_lat"] is not None:
    #             knn_coords.append((d["knn_lat"], d["knn_lon"]))
    #         continue
    #     if d["gt_lat"] is None:
    #         continue
    #     knn_check = True
    # for i, d in enumerate(aligned_data_test):
    #     if wifi_check:
    #         if d["wifi_lat"] is not None:
    #             wifi_coords_test.append((d["wifi_lat"], d["wifi_lon"]))
    #             wifi_check = False
    #         continue
    #     if d["gt_lat"] is None:
    #         continue
    #     wifi_check = True
    # for i, d in enumerate(aligned_data_test):
    #     if pdr_check:
    #         if d["pdr_lat"] is not None:
    #             pdr_coords_test.append((d["pdr_lat"], d["pdr_lon"]))
    #         continue
    #     if d["gt_lat"] is None:
    #         continue
    #     pdr_check = True

    # for i, d in enumerate(aligned_data_test):
    #     if d["fused_lat"] is not None:
    #         last_fused_lat, last_fused_lon = d["fused_lat"], d["fused_lon"]
    #     if d["gt_lat"] is not None:
    #         fused_coords.append((last_fused_lat, last_fused_lon))

        
    #print(len(pdr_coords))
    # print(len(gt_coords))
    # print(len(pdr_coords))
    # print(len(fused_coords))

    # print(len(gt_coords))
    # print(len(wifi_coords_test))
    # print(len(fused_coords))


    # gt_lats_ori, gt_lons_ori = zip(*gt_coords_origin)
    gt_lats, gt_lons = zip(*gt_coords)
    # wifi_lats, wifi_lons = zip(*wifi_coords)
    #init_lats, init_lons = zip(*init_coords)
    #pdr_lats, pdr_lons = zip(*pdr_coords)
    #tra_lats, tra_lons = zip(*pdr_tra)
    knn_lats, knn_lons = zip(*knn_coords)
    wifi_lats, wifi_lons = zip(*wifi_coords)
    fused_lats, fused_lons = zip(*fused_coords)
    knn_lats_all, knn_lons_all = zip(*knn_coords_all)
    wifi_lats_all, wifi_lons_all = zip(*wifi_coords_all)
    fused_lats_all, fused_lons_all = zip(*fused_coords_all)
    # pdr_lats_test, pdr_lons_test = zip(*pdr_coords_test)

    ref_lat, ref_lon = gt_lats[0], gt_lons[0]  # 設定參考點
    gt_lats, gt_lons = latlon_to_xy(gt_lats, gt_lons, ref_lat, ref_lon)
    knn_lats, knn_lons = latlon_to_xy(knn_lats, knn_lons, ref_lat, ref_lon)
    wifi_lats, wifi_lons = latlon_to_xy(wifi_lats, wifi_lons, ref_lat, ref_lon)
    fused_lats, fused_lons = latlon_to_xy(fused_lats, fused_lons, ref_lat, ref_lon)
    knn_lats_all, knn_lons_all = latlon_to_xy(knn_lats_all, knn_lons_all, ref_lat, ref_lon)
    wifi_lats_all, wifi_lons_all = latlon_to_xy(wifi_lats_all, wifi_lons_all, ref_lat, ref_lon)
    fused_lats_all, fused_lons_all = latlon_to_xy(fused_lats_all, fused_lons_all, ref_lat, ref_lon)
    # fig, ax = plt.subplots()
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()
    fig5, ax5 = plt.subplots()
    fig6, ax6 = plt.subplots()
    fig7, ax7 = plt.subplots()

    # plt.plot(gt_lons_ori, gt_lats_ori, label="Ground Truth origin", marker="o")
    # ax.plot(gt_lats, gt_lons, label="Ground Truth", marker="o", linewidth=0)
    #plt.plot(wifi_lons, wifi_lats, label="Wi-Fi Point", marker="o")
    #plt.plot(init_lons, init_lats, label="Wi-Fi Init", marker="x")
    #plt.plot(pdr_lons[105], pdr_lats[105], label="IMU PDR", marker="^")
    #plt.plot(tra_lons, tra_lats, label="IMU PDR", marker="^")

    # ax.plot(fused_lat, fused_lon, label="EKF Fused", marker="^", linewidth=0)
    # ax.plot(knn_lat, knn_lon, label="KNN", marker="o", linewidth=0)
    # ax.plot(wifi_lats_test, wifi_lons_test, label="EKF Wi-Fi", marker="^", linewidth=0)
    # plt.plot(pdr_lons_test, pdr_lats_test, label="PDR", marker="^")
    # --------------------------------------------------------------------------------------------------------------------------------
    R = re
    interval = 2
    offset = 5
    width = 25.6
    height = 14.4
    ax1.plot(gt_lats, gt_lons, label="Ground Truth", marker="o", linewidth=0, color='b')
    ax1.plot(knn_lats_all, knn_lons_all, label="KNN Trajectory", color='#ff7f0e')
    ax1.plot([0,0], [0,0], label="Error", color='b')
    # 加上編號（GT點）
    for i, (x, y) in enumerate(zip(gt_lats, gt_lons)):
        ax1.plot([gt_lats[i], knn_lats[i]], [gt_lons[i], knn_lons[i]], color='b')
        # FOR TEST1 and TEST2
        if i == 12 or i == 15 or i == 18 or i == 21:
            ax1.text(x, y + 0.5, f'{i+1},{i+3}', fontsize=9, ha='center', va='bottom', color='blue')
            continue
        elif i == 14 or i == 17 or i == 20 or i == 23:
            continue
        ax1.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')

        # FOR TEST3 and TEST4
        # if i == 4 or i == 7 or i == 10 or i == 13:
        #     ax1.text(x, y + 0.5, f'{i+1},{i+3}', fontsize=9, ha='center', va='bottom', color='blue')
        #     continue
        # elif i == 6 or i == 9 or i == 12 or i == 15:
        #     continue
        # ax1.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')
    # 加上編號（Pred點）
    # for i, (x, y) in enumerate(zip(knn_lats, knn_lons)):
    #     ax1.text(x, y - 0.5, f'{i+1}', fontsize=9, ha='center', va='top', color='red')
    ax1.set_title('2D position estimation vs GT')
    ax1.set_aspect('equal')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax1.set_xlim(min(gt_lats)-offset, max(gt_lats)+offset)
    ax1.set_ylim(min(gt_lons)-offset, max(gt_lons)+offset)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True)
    ax1.legend(loc="upper left")
    fig1.set_size_inches(width, height)
    fig1.savefig(f'figure/{id}/trajectory/{R}/GT_KNN.png')

    ax2.plot(gt_lats, gt_lons, label="Ground Truth", marker="o", linewidth=0, color='b')
    ax2.plot(wifi_lats_all, wifi_lons_all, label="EKF Wi-Fi Trajectory", color='#ff7f0e')
    ax2.plot([0,0], [0,0], label="Error", color='b')
    # 加上編號（GT點）
    for i, (x, y) in enumerate(zip(gt_lats, gt_lons)):
        ax2.plot([gt_lats[i], wifi_lats[i]], [gt_lons[i], wifi_lons[i]], color='b')
        # FOR TEST1 and TEST2
        if i == 12 or i == 15 or i == 18 or i == 21:
            ax2.text(x, y + 0.5, f'{i+1},{i+3}', fontsize=9, ha='center', va='bottom', color='blue')
            continue
        elif i == 14 or i == 17 or i == 20 or i == 23:
            continue
        ax2.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')

        # FOR TEST3 and TEST4
        # if i == 4 or i == 7 or i == 10 or i == 13:
        #     ax2.text(x, y + 0.5, f'{i+1},{i+3}', fontsize=9, ha='center', va='bottom', color='blue')
        #     continue
        # elif i == 6 or i == 9 or i == 12 or i == 15:
        #     continue
        # ax2.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')
    # 加上編號（Pred點）
    # for i, (x, y) in enumerate(zip(wifi_lats, wifi_lons)):
    #     ax2.text(x, y - 0.5, f'{i+1}', fontsize=9, ha='center', va='top', color='red')
    ax2.set_title('2D position estimation vs GT')
    ax2.set_aspect('equal')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax2.set_xlim(min(gt_lats)-offset, max(gt_lats)+offset)
    ax2.set_ylim(min(gt_lons)-offset, max(gt_lons)+offset)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    ax2.legend(loc="upper left")
    fig2.set_size_inches(width, height)
    # fig2.savefig(f'figure/{id}/trajectory/{R}/GT_EKF_WIFI.png')

    ax3.plot(gt_lats, gt_lons, label="Ground Truth", marker="o", linewidth=0, color='b')
    ax3.plot(fused_lats_all, fused_lons_all, label="EKF Fused Trajectory", color='#ff7f0e')
    ax3.plot([0,0], [0,0], label="Error", color='b')
    # ax3.plot(wifi_lats_all, wifi_lons_all, label="EKF Wi-Fi", marker="o", linewidth=0)
    # 加上編號（GT點）
    for i, (x, y) in enumerate(zip(gt_lats, gt_lons)):
        ax3.plot([gt_lats[i], fused_lats[i]], [gt_lons[i], fused_lons[i]], color='b')
        # FOR TEST1 and TEST2
        if i == 12 or i == 15 or i == 18 or i == 21:
            ax3.text(x, y + 0.5, f'{i+1},{i+3}', fontsize=9, ha='center', va='bottom', color='blue')
            continue
        elif i == 14 or i == 17 or i == 20 or i == 23:
            continue
        ax3.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')

        # FOR TEST3 and TEST4
        # if i == 4 or i == 7 or i == 10 or i == 13:
        #     ax3.text(x, y + 0.5, f'{i+1},{i+3}', fontsize=9, ha='center', va='bottom', color='blue')
        #     continue
        # elif i == 6 or i == 9 or i == 12 or i == 15:
        #     continue
        # ax3.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')
    # 加上編號（Pred點）
    # for i, (x, y) in enumerate(zip(wifi_lats_all, wifi_lons_all)):
    #     ax3.text(x, y - 0.5, f'{i+1}', fontsize=9, ha='center', va='top', color='red')
    ax3.set_title('2D position estimation vs GT')
    ax3.set_aspect('equal')
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax3.set_xlim(min(gt_lats)-offset, max(gt_lats)+offset)
    ax3.set_ylim(min(gt_lons)-offset, max(gt_lons)+offset)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.grid(True)
    ax3.legend(loc="upper left")
    fig3.set_size_inches(width, height)
    fig3.savefig(f'figure/{id}/trajectory/{R}/GT_EKF_FUSED.png')
    # plt.show()

    # 誤差折線圖 fig4, fig5
    ax4.plot(range(len(dists_knn)), dists_knn, marker='o', linestyle='-')
    ax4.set_title('Position Error (KNN)')
    ax4.set_xlabel('sample number')
    ax4.set_ylabel('Error (m)')
    ax4.set_ylim(0, max(dists_knn)+5)
    ax4.grid(True)
    # ax4.legend()
    fig4.set_size_inches(width, height)
    fig4.savefig(f'figure/{id}/All_Position_Error/{R}/GT_KNN.png')

    ax5.plot(range(len(dists_ekf_fused)), dists_ekf_fused, marker='o', linestyle='-')
    ax5.set_title('Position Error (EKF)')
    ax5.set_xlabel('sample number')
    ax5.set_ylabel('Error (m)')
    ax5.set_ylim(0, max(dists_ekf_fused)+5)
    ax5.grid(True)
    # ax5.legend()
    fig5.set_size_inches(width, height)
    fig5.savefig(f'figure/{id}/All_Position_Error/{R}/GT_EKF_FUSED.png')

    # CDF圖 (累積分布函數)
    percentile_to_mark = 75

    sorted_errors_knn = np.sort(dists_knn)
    cdf_knn = np.arange(1, len(sorted_errors_knn) + 1) / len(sorted_errors_knn)
    perc_value_knn = np.percentile(dists_knn, percentile_to_mark)
    ax6.plot(sorted_errors_knn, cdf_knn * 100, marker='o', linestyle='-', label='CDF')
    ax6.axhline(percentile_to_mark, color='red', linestyle='--', label=f'P{percentile_to_mark}')
    ax6.plot([perc_value_knn, perc_value_knn], [0, percentile_to_mark], color='red') 
    # ax6.axvline(perc_value_knn, color='red')
    ax6.text(perc_value_knn + 0.5, percentile_to_mark - 5, f'{perc_value_knn:.2f} m', color='red')
    # 自定 y 軸刻度（在原有基礎上加 perc_y）
    ax6_yticks = list(ax6.get_yticks())
    if percentile_to_mark not in ax6_yticks:
        ax6_yticks.append(percentile_to_mark)
        ax6_yticks = sorted(ax6_yticks)
    ax6.set_yticks(ax6_yticks)
    ax6.set_title('CDF - Position Errors distribution (KNN)')
    ax6.set_xlabel('Position Error (m)')
    ax6.set_ylabel('CDF (%)')
    ax6.set_ylim(0, 105)
    ax6.grid(True)
    ax6.legend(loc="upper left")
    fig6.set_size_inches(width, height)
    fig6.savefig(f'figure/{id}/CDF/{R}/GT_KNN.png')

    sorted_errors_ekf = np.sort(dists_ekf_fused)
    cdf_ekf = np.arange(1, len(sorted_errors_ekf) + 1) / len(sorted_errors_ekf)
    perc_value_ekf = np.percentile(dists_ekf_fused, percentile_to_mark)
    ax7.plot(sorted_errors_ekf, cdf_ekf * 100, marker='o', linestyle='-', label='CDF')
    ax7.axhline(percentile_to_mark, color='red', linestyle='--', label=f'P{percentile_to_mark}')
    ax7.plot([perc_value_ekf, perc_value_ekf], [0, percentile_to_mark], color='red')
    # ax7.axvline(perc_value_ekf, color='red')
    ax7.text(perc_value_ekf + 0.5, percentile_to_mark - 5, f'{perc_value_ekf:.2f} m', color='red')
    # 自定 y 軸刻度（在原有基礎上加 perc_y）
    ax7_yticks = list(ax7.get_yticks())
    if percentile_to_mark not in ax7_yticks:
        ax7_yticks.append(percentile_to_mark)
        ax7_yticks = sorted(ax7_yticks)
    ax7.set_yticks(ax7_yticks)
    ax7.set_title('CDF - Position Errors distribution (EKF)')
    ax7.set_xlabel('Position Error (m)')
    ax7.set_ylabel('CDF (%)')
    ax7.set_ylim(0, 105)
    ax7.grid(True)
    ax7.legend(loc="upper left")
    fig7.set_size_inches(width, height)
    fig7.savefig(f'figure/{id}/CDF/{R}/GT_EKF_FUSED.png')
    plt.show()

    # R = 'R123'
    # with open(f'figure/{id}/trajectory/{R}/GT_KNN.pkl', 'wb') as f:
    #     pickle.dump(fig1, f)
    # with open(f'figure/{id}/trajectory/{R}/GT_EKF_WIFI.pkl', 'wb') as f:
    #     pickle.dump(fig2, f)
    # with open(f'figure/{id}/trajectory/{R}/GT_EKF_FUSED.pkl', 'wb') as f:
    #     pickle.dump(fig3, f)
    # -----------------------------------------------------------------------------------------------------------------------------
    # ax.plot(gt_lats, gt_lons, label="Ground Truth", marker="o", linewidth=0)
    # ax.plot(fused_lat, fused_lon, label="EKF Fused", marker="o", linewidth=0)
    # # 加上編號（GT點）
    # for i, (x, y) in enumerate(zip(gt_lats, gt_lons)):
    #     ax.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')
    # # 加上編號（Pred點）
    # for i, (x, y) in enumerate(zip(fused_lat, fused_lon)):
    #     ax.text(x, y - 0.5, f'{i+1}', fontsize=9, ha='center', va='top', color='red')
    # ax.set_aspect('equal')
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.set_xlim(min(gt_lats)-10, max(gt_lats)+10)
    # ax.set_ylim(min(gt_lons)-10, max(gt_lons)+10)
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.grid(True)
    # ax.legend()
    # plt.show()

    # R = 'R123'
    # with open(f'figure/{id}/trajectory/{R}/GT_EKF_FUSED.pkl', 'wb') as f:
    #     pickle.dump(fig, f)
    print('--------------------------------------------------------')
    print(f'mean_knn = {mean_knn}')
    print(f'median_knn = {median_knn}')
    print(f'rmse_knn = {rmse_knn}')
    print(f'std_knn = {std_knn}')
    print(f'90th perc. = {np.percentile(dists_knn, 90)}')
    print(f'75th perc. = {np.percentile(dists_knn, 75)}')
    print('--------------------------------------------------------')
    # print(f'max_knn = {max_knn}')
    # print(f'all_knn = {dists_knn}')
    
    # print(f'rmse_ekf_wifi = {rmse_ekf_wifi}')
    # print(f'mean_ekf_wifi = {mean_ekf_wifi}')
    # print(f'std_ekf_wifi = {std_ekf_wifi}')
    # print(f'max_ekf_wifi = {max_ekf_wifi}')
    # print(f'all_ekf_wifi = {dists_ekf_wifi}')

    print(f'mean_ekf_fused = {mean_ekf_fused}')
    print(f'median_ekf_fused = {median_ekf_fused}')
    print(f'rmse_ekf_fused = {rmse_ekf_fused}')
    print(f'std_ekf_fused = {std_ekf_fused}')
    print(f'90th perc. = {np.percentile(dists_ekf_fused, 90)}')
    print(f'75th perc. = {np.percentile(dists_ekf_fused, 75)}')
    # print(f'max_ekf_fused = {max_ekf_fused}')
    # print(f'all_ekf_fused = {dists_ekf_fused}')

    # 儲存圖形物件（包含資料與設定）
    # R = 'R123'
    # file_name = 'GT_KNN'            # Ground Truth 與 knn 最近時間估計位置點
    # file_name = 'GT_EKF_WIFI'       # Ground Truth 與 EKF_WIFI 最近時間估計位置點
    # file_name = 'GT_EKF_FUSED'      # Ground Truth 與 EKF_FUSED 最近時間估計位置點
    # with open(f'figure/{id}/trajectory/{R}/{file_name}.pkl', 'wb') as f:
    #     pickle.dump(fig, f)
    
def plot_figure_test_pdr(aligned_data_train, aligned_data_test, id, re):
    fused_check = False
    knn_check = False
    wifi_check = False
    pdr_check = False
    # 可視化：軌跡圖
    # plt.figure(figsize=(8, 6))
    # gt_coords_origin = [(d["gt_lat_ori"], d["gt_lon_ori"]) for d in aligned_data_train if d["gt_lat_ori"] is not None]
    gt_coords = [(d["gt_lat"], d["gt_lon"]) for d in aligned_data_test if d["gt_lat"] is not None]
    # wifi_coords = [(d["gt_lat_temp"], d["gt_lon_temp"]) for d in aligned_data_test if d["gt_lat_temp"] is not None and d["rssi_vector"] is not None]
    #print(len(gt_coords))
    #init_coords = [(d["init_lat"], d["init_lon"]) for d in aligned_data_test if d["gt_lat"] is not None]
    #print(len(init_coords))
    #pdr_coords = [(d["pdr_lat"], d["pdr_lon"]) for d in aligned_data_test if d["pdr_lat"] is not None]
    #pdr_tra = [(d["pdr_trajectory"]) for d in aligned_data]
    #pdr_tra = [pt for d in aligned_data_test if d["pdr_trajectory"] is not None for pt in d["pdr_trajectory"]]
    fused_coords = [(d["fused_lat"], d["fused_lon"]) for d in aligned_data_test if d["fused_lat"] is not None]
    # fused_coords = []
    knn_coords = []
    wifi_coords_test = []
    pdr_coords_test = []
    dists_knn = []
    dists_ekf_wifi = []
    dists_ekf_fused = []
    # knn_coords = [(d["knn_lat"], d["knn_lon"]) for d in aligned_data_test if d["knn_lat"] is not None]
    # wifi_coords_test = [(d["wifi_lat"], d["wifi_lon"]) for d in aligned_data_test if d["wifi_lat"] is not None]
    for i, d in enumerate(aligned_data_test):
        if d["knn_lat"] is not None:
            print('knn')
            # last_knn_lat, last_knn_lon = d["knn_lat"], d["knn_lon"]
            knn_coords.append((d["knn_lat"], d["knn_lon"]))
        # if d["gt_lat"] is not None:
        #     knn_coords.append((last_knn_lat, last_knn_lon))
        #     dists_knn.append(geodesic((last_knn_lat, last_knn_lon), (d["gt_lat"], d["gt_lon"])).meters)
        # timestamp = d["timestamp"]
    # print(dists)

    # rmse_knn = np.sqrt(np.mean(np.square(dists_knn)))
    # mean_knn = np.mean(dists_knn)
    # std_knn = np.std(dists_knn)
    # max_knn = np.max(dists_knn)

    # for i, d in enumerate(aligned_data_test):
    #     if d["fused_lat"] is not None:
    #         print(i)
    #         break
    # for i, d in enumerate(aligned_data_test):
    #     if d["gt_lat"] is not None:
    #         print(i)
    #         break

    # for i, d in enumerate(aligned_data_test):
    #     if fused_check:
    #         if d["fused_lat"] is not None:
    #             fused_coords.append((d["fused_lat"], d["fused_lon"]))
    #             dists_ekf_fused.append(geodesic((d["fused_lat"], d["fused_lon"]), (last_gt_lat, last_gt_lon)).meters)
    #             fused_check = False
    #         continue
    #     if d["gt_lat"] is None:
    #         continue
    #     last_gt_lat = d["gt_lat"]
    #     last_gt_lon = d["gt_lon"]
    #     fused_check = True

    # print(dists)
    
    # rmse_ekf_fused = np.sqrt(np.mean(np.square(dists_ekf_fused)))
    # mean_ekf_fused = np.mean(dists_ekf_fused)
    # std_ekf_fused = np.std(dists_ekf_fused)
    # max_ekf_fused = np.max(dists_ekf_fused)

    # for i, d in enumerate(aligned_data_test):
    #     if fused_check:
    #         if d["fused_lat"] is not None:
    #             fused_coords.append((d["fused_lat"], d["fused_lon"]))
    #             fused_check = False
    #         continue
    #     if d["gt_lat"] is None:
    #         continue
    #     fused_check = True
    # for i, d in enumerate(aligned_data_test):
    #     if knn_check:
    #         if d["knn_lat"] is not None:
    #             knn_coords.append((d["knn_lat"], d["knn_lon"]))
    #         continue
    #     if d["gt_lat"] is None:
    #         continue
    #     knn_check = True
    # for i, d in enumerate(aligned_data_test):
    #     if wifi_check:
    #         if d["wifi_lat"] is not None:
    #             wifi_coords_test.append((d["wifi_lat"], d["wifi_lon"]))
    #             wifi_check = False
    #         continue
    #     if d["gt_lat"] is None:
    #         continue
    #     wifi_check = True
    # for i, d in enumerate(aligned_data_test):
    #     if pdr_check:
    #         if d["pdr_lat"] is not None:
    #             pdr_coords_test.append((d["pdr_lat"], d["pdr_lon"]))
    #         continue
    #     if d["gt_lat"] is None:
    #         continue
    #     pdr_check = True

    # for i, d in enumerate(aligned_data_test):
    #     if d["fused_lat"] is not None:
    #         last_fused_lat, last_fused_lon = d["fused_lat"], d["fused_lon"]
    #     if d["gt_lat"] is not None:
    #         fused_coords.append((last_fused_lat, last_fused_lon))

        
    #print(len(pdr_coords))
    # print(len(gt_coords))
    # print(len(pdr_coords))
    # print(len(fused_coords))

    # print(len(gt_coords))
    # print(len(wifi_coords_test))
    print(len(fused_coords))


    # gt_lats_ori, gt_lons_ori = zip(*gt_coords_origin)
    gt_lats, gt_lons = zip(*gt_coords)
    # wifi_lats, wifi_lons = zip(*wifi_coords)
    #init_lats, init_lons = zip(*init_coords)
    #pdr_lats, pdr_lons = zip(*pdr_coords)
    #tra_lats, tra_lons = zip(*pdr_tra)
    fused_lat, fused_lon = zip(*fused_coords)
    knn_lat, knn_lon = zip(*knn_coords)
    # wifi_lats_test, wifi_lons_test = zip(*wifi_coords_test)
    # pdr_lats_test, pdr_lons_test = zip(*pdr_coords_test)

    ref_lat, ref_lon = gt_lats[0], gt_lons[0]  # 設定參考點
    gt_lats, gt_lons = latlon_to_xy(gt_lats, gt_lons, ref_lat, ref_lon)
    fused_lat, fused_lon = latlon_to_xy(fused_lat, fused_lon, ref_lat, ref_lon)
    knn_lat, knn_lon = latlon_to_xy(knn_lat, knn_lon, ref_lat, ref_lon)
    # wifi_lats_test, wifi_lons_test = latlon_to_xy(wifi_lats_test, wifi_lons_test, ref_lat, ref_lon)

    # fig, ax = plt.subplots()
    # fig1, ax1 = plt.subplots()
    # fig2, ax2 = plt.subplots()
    fig3, ax3 = plt.subplots()

    # plt.plot(gt_lons_ori, gt_lats_ori, label="Ground Truth origin", marker="o")
    # ax.plot(gt_lats, gt_lons, label="Ground Truth", marker="o", linewidth=0)
    #plt.plot(wifi_lons, wifi_lats, label="Wi-Fi Point", marker="o")
    #plt.plot(init_lons, init_lats, label="Wi-Fi Init", marker="x")
    #plt.plot(pdr_lons[105], pdr_lats[105], label="IMU PDR", marker="^")
    #plt.plot(tra_lons, tra_lats, label="IMU PDR", marker="^")

    # ax.plot(fused_lat, fused_lon, label="EKF Fused", marker="^", linewidth=0)
    # ax.plot(knn_lat, knn_lon, label="KNN", marker="o", linewidth=0)
    # ax.plot(wifi_lats_test, wifi_lons_test, label="EKF Wi-Fi", marker="^", linewidth=0)
    # plt.plot(pdr_lons_test, pdr_lats_test, label="PDR", marker="^")
    # --------------------------------------------------------------------------------------------------------------------------------
    R = re
    interval = 2
    offset = 5
    width = 25.6
    height = 14.4
    '''
    ax1.plot(gt_lats, gt_lons, label="Ground Truth", marker="o", linewidth=0)
    ax1.plot(knn_lat, knn_lon, label="KNN", marker="o", linewidth=0)
    # 加上編號（GT點）
    for i, (x, y) in enumerate(zip(gt_lats, gt_lons)):
        # FOR TEST1 and TEST2
        # if i == 12 or i == 15 or i == 18 or i == 21:
        #     ax1.text(x, y + 0.5, f'{i+1},{i+3}', fontsize=9, ha='center', va='bottom', color='blue')
        #     continue
        # elif i == 14 or i == 17 or i == 20 or i == 23:
        #     continue
        # ax1.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')

        # FOR TEST3 and TEST4
        # if i == 4 or i == 7 or i == 10 or i == 13:
        #     ax1.text(x, y + 0.5, f'{i+1},{i+3}', fontsize=9, ha='center', va='bottom', color='blue')
        #     continue
        # elif i == 6 or i == 9 or i == 12 or i == 15:
        #     continue
        ax1.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')
    # 加上編號（Pred點）
    for i, (x, y) in enumerate(zip(knn_lat, knn_lon)):
        ax1.text(x, y - 0.5, f'{i+1}', fontsize=9, ha='center', va='top', color='red')
    ax1.set_aspect('equal')
    ax1.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax1.yaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax1.set_xlim(min(gt_lats)-offset, max(gt_lats)+offset)
    ax1.set_ylim(min(gt_lons)-offset, max(gt_lons)+offset)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.grid(True)
    ax1.legend()
    fig1.set_size_inches(width, height)
    # fig1.savefig(f'figure/{id}/trajectory/{R}/GT_KNN.png')

    ax2.plot(gt_lats, gt_lons, label="Ground Truth", marker="o", linewidth=0)
    ax2.plot(wifi_lats_test, wifi_lons_test, label="EKF Wi-Fi", marker="o", linewidth=0)
    # 加上編號（GT點）
    for i, (x, y) in enumerate(zip(gt_lats, gt_lons)):
        # FOR TEST1 and TEST2
        # if i == 12 or i == 15 or i == 18 or i == 21:
        #     ax2.text(x, y + 0.5, f'{i+1},{i+3}', fontsize=9, ha='center', va='bottom', color='blue')
        #     continue
        # elif i == 14 or i == 17 or i == 20 or i == 23:
        #     continue
        # ax2.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')

        # FOR TEST3 and TEST4
        # if i == 4 or i == 7 or i == 10 or i == 13:
        #     ax2.text(x, y + 0.5, f'{i+1},{i+3}', fontsize=9, ha='center', va='bottom', color='blue')
        #     continue
        # elif i == 6 or i == 9 or i == 12 or i == 15:
        #     continue
        ax2.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')
    # 加上編號（Pred點）
    for i, (x, y) in enumerate(zip(wifi_lats_test, wifi_lons_test)):
        ax2.text(x, y - 0.5, f'{i+1}', fontsize=9, ha='center', va='top', color='red')
    ax2.set_aspect('equal')
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax2.yaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax2.set_xlim(min(gt_lats)-offset, max(gt_lats)+offset)
    ax2.set_ylim(min(gt_lons)-offset, max(gt_lons)+offset)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.grid(True)
    ax2.legend()
    fig2.set_size_inches(width, height)
    # fig2.savefig(f'figure/{id}/trajectory/{R}/GT_EKF_WIFI.png')
    '''
    ax3.plot(gt_lats, gt_lons, label="Ground Truth", marker="o", linewidth=0)
    ax3.plot(fused_lat, fused_lon, label="EKF PDR", marker="o", linewidth=0)
    ax3.plot(knn_lat, knn_lon, label="First Point (KNN)", marker="o", linewidth=0)
    # 加上編號（GT點）
    for i, (x, y) in enumerate(zip(gt_lats, gt_lons)):
        # FOR TEST1 and TEST2
        if i == 12 or i == 15 or i == 18 or i == 21:
            ax3.text(x, y + 0.5, f'{i+1},{i+3}', fontsize=9, ha='center', va='bottom', color='blue')
            continue
        elif i == 14 or i == 17 or i == 20 or i == 23:
            continue
        ax3.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')

        # FOR TEST3 and TEST4
        # if i == 4 or i == 7 or i == 10 or i == 13:
        #     ax3.text(x, y + 0.5, f'{i+1},{i+3}', fontsize=9, ha='center', va='bottom', color='blue')
        #     continue
        # elif i == 6 or i == 9 or i == 12 or i == 15:
        #     continue
        # ax3.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')
    # 加上編號（Pred點）
    # for i, (x, y) in enumerate(zip(fused_lat, fused_lon)):
    #     ax3.text(x, y - 0.5, f'{i+1}', fontsize=9, ha='center', va='top', color='red')
    # ax3.set_aspect('equal')
    ax3.xaxis.set_major_locator(ticker.MultipleLocator(interval))
    ax3.yaxis.set_major_locator(ticker.MultipleLocator(interval))
    # ax3.set_xlim(min(gt_lats)-offset, max(gt_lats)+offset)
    # ax3.set_ylim(min(gt_lons)-offset, max(gt_lons)+offset)
    ax3.set_title('2D position estimation vs GT')
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.grid(True)
    ax3.legend(loc="upper left")
    fig3.set_size_inches(width, height)
    fig3.savefig(f'figure/{id}/trajectory/{R}/GT_EKF_PDR.png')
    plt.show()

    # R = 'R123'
    # with open(f'figure/{id}/trajectory/{R}/GT_KNN.pkl', 'wb') as f:
    #     pickle.dump(fig1, f)
    # with open(f'figure/{id}/trajectory/{R}/GT_EKF_WIFI.pkl', 'wb') as f:
    #     pickle.dump(fig2, f)
    # with open(f'figure/{id}/trajectory/{R}/GT_EKF_FUSED.pkl', 'wb') as f:
    #     pickle.dump(fig3, f)
    # -----------------------------------------------------------------------------------------------------------------------------
    # ax.plot(gt_lats, gt_lons, label="Ground Truth", marker="o", linewidth=0)
    # ax.plot(fused_lat, fused_lon, label="EKF Fused", marker="o", linewidth=0)
    # # 加上編號（GT點）
    # for i, (x, y) in enumerate(zip(gt_lats, gt_lons)):
    #     ax.text(x, y + 0.5, str(i+1), fontsize=9, ha='center', va='bottom', color='blue')
    # # 加上編號（Pred點）
    # for i, (x, y) in enumerate(zip(fused_lat, fused_lon)):
    #     ax.text(x, y - 0.5, f'{i+1}', fontsize=9, ha='center', va='top', color='red')
    # ax.set_aspect('equal')
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    # ax.set_xlim(min(gt_lats)-10, max(gt_lats)+10)
    # ax.set_ylim(min(gt_lons)-10, max(gt_lons)+10)
    # ax.set_xlabel('X (m)')
    # ax.set_ylabel('Y (m)')
    # ax.grid(True)
    # ax.legend()
    # plt.show()

    # R = 'R123'
    # with open(f'figure/{id}/trajectory/{R}/GT_EKF_FUSED.pkl', 'wb') as f:
    #     pickle.dump(fig, f)

    # print(f'rmse_knn = {rmse_knn}')
    # print(f'mean_knn = {mean_knn}')
    # print(f'std_knn = {std_knn}')
    # print(f'max_knn = {max_knn}')
    # print(f'all_knn = {dists_knn}')
    
    # print(f'rmse_ekf_wifi = {rmse_ekf_wifi}')
    # print(f'mean_ekf_wifi = {mean_ekf_wifi}')
    # print(f'std_ekf_wifi = {std_ekf_wifi}')
    # print(f'max_ekf_wifi = {max_ekf_wifi}')
    # print(f'all_ekf_wifi = {dists_ekf_wifi}')

    # print(f'rmse_ekf_fused = {rmse_ekf_fused}')
    # print(f'mean_ekf_fused = {mean_ekf_fused}')
    # print(f'std_ekf_fused = {std_ekf_fused}')
    # print(f'max_ekf_fused = {max_ekf_fused}')
    # print(f'all_ekf_fused = {dists_ekf_fused}')

    # 儲存圖形物件（包含資料與設定）
    # R = 'R123'
    # file_name = 'GT_KNN'            # Ground Truth 與 knn 最近時間估計位置點
    # file_name = 'GT_EKF_WIFI'       # Ground Truth 與 EKF_WIFI 最近時間估計位置點
    # file_name = 'GT_EKF_FUSED'      # Ground Truth 與 EKF_FUSED 最近時間估計位置點
    # with open(f'figure/{id}/trajectory/{R}/{file_name}.pkl', 'wb') as f:
    #     pickle.dump(fig, f)

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

    repitition = 'R1234'
    neighbors = 3

    name = 'TEST2'

    root_dir = "aligned_trials"

    read_file = ['T1_R1', 'T2_R1', 'T3_R1', 'T4_R1', 'T5_R1', 'T21_R1', 'T22_R1', 'T23_R1', 'T24_R1', 'T25_R1', 'T26_R1', 'T27_R1', 'TEST1', 'TEST2', 'TEST3', 'TEST4']

    read_file_R1234 = ['T1_R1', 'T1_R2', 'T1_R3', 'T1_R4',
                     'T2_R1', 'T2_R2', 'T2_R3', 'T2_R4',
                     'T3_R1', 'T3_R2', 'T3_R3', 'T3_R4',
                     'T4_R1', 'T4_R2', 'T4_R3', 'T4_R4',
                     'T5_R1', 'T5_R2', 'T5_R3', 'T5_R4',
                     'T21_R1', 'T21_R2', 'T21_R3', 'T21_R4',
                     'T22_R1', 'T22_R2', 'T22_R3', 'T22_R4',
                     'T23_R1', 'T23_R2', 'T23_R3', 'T23_R4',
                     'T24_R1', 'T24_R2', 'T24_R3', 'T24_R4',
                     'T25_R1', 'T25_R2', 'T25_R3', 'T25_R4',
                     'T26_R1', 'T26_R2', 'T26_R3', 'T26_R4',
                     'T27_R1', 'T27_R2', 'T27_R3', 'T27_R4']

    read_file_test = ['TEST1', 'TEST2', 'TEST3', 'TEST4']

    read_file_temp = ['TEST2']

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

    for trial_name in read_file_temp:
        trial_path = os.path.join(root_dir, trial_name)
        if not os.path.isdir(trial_path):
            continue

        trial_data = []
        # for fname in sorted(os.listdir(trial_path)):
        #     if fname.endswith(".pkl"):
        #         with open(os.path.join(trial_path, fname), "rb") as f:
        #             trial_data.append(pickle.load(f))

        with open(os.path.join(trial_path, f'distance/all_data_pdr_{repitition}_neighbors{neighbors}.pkl'), "rb") as f:
            trial_data = pickle.load(f)
        trial_dict[trial_name] = trial_data

    # aligned_data_train = trial_dict['T1_R1'] + trial_dict['T2_R1'] + trial_dict['T3_R1'] \
    #                 + trial_dict['T4_R1'] + trial_dict['T5_R1'] + trial_dict['T21_R1'] \
    #                     + trial_dict['T22_R1'] + trial_dict['T23_R1'] + trial_dict['T24_R1'] \
    #                         + trial_dict['T25_R1'] + trial_dict['T26_R1'] + trial_dict['T27_R1']

    aligned_data_test = trial_dict[name]

    # aligned_data_train = trial_dict['T27_R4']

    # plot_figure_train(aligned_data_train, aligned_data_train)

    # aligned_data_temp = trial_dict['temp_trial']

    # aligned_data_test1 = trial_dict['test_trial01']
    # aligned_data_test2 = trial_dict['test_trial02']
    # aligned_data_test3 = trial_dict['test_trial03']
    # aligned_data_test4 = trial_dict['test_trial04']

    # evaluate_errors(aligned_data_train)
    #all_errors(aligned_data_train)

    # plot_figure_test(aligned_data_test, aligned_data_test, name, repitition)
    plot_figure_test_pdr(aligned_data_test, aligned_data_test, name, repitition)

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


