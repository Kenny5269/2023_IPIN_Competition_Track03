import os
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    trial_dict = {}

    root_dir = "aligned_trials"

    read_file = ['T1_R1', 
                 'T2_R1', 
                 'T3_R1', 
                 'T4_R1', 
                 'T5_R1', 
                 'T21_R1', 
                 'T22_R1', 
                 'T23_R1', 
                 'T24_R1', 
                 'T25_R1', 
                 'T26_R1', 
                 'T27_R1']

    rssi_features = []
    gt_positions = []
    num = 0

    for trial_name in read_file:
        num += 1
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
        # trial_dict[trial_name] = trial_data

        for d in trial_data:
            if d["rssi_vector"] is not None and d["gt_lat_temp"] is not None:
                rssi = np.nan_to_num(d["rssi_vector"], nan=-100.0)
                rssi_features.append(rssi)
                gt_positions.append([d["gt_lat_temp"], d["gt_lon_temp"]])

    rssi_features = np.array(rssi_features)
    gt_positions = np.array(gt_positions)
    
    model = KNeighborsRegressor(n_neighbors=3)
    model.fit(rssi_features, gt_positions)

    with open('knn_model.pkl','wb') as f:
        pickle.dump(model,f)
    
    print(num)

    # wifi_model = train_wifi_model(rssi_features, gt_positions)