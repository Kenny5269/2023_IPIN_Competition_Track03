# IPIN 2023 Track 3 - PyTorch Implementation (Data Preprocessing + Model + Training + Evaluation + WiFi/BLE Fusion + Time Synchronization + Unknown AP/Beacon Handling)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# =========================
# 1. Data Preprocessing

import glob

# Helper function: collect all CSV files from a directory
def get_all_csv_files(directory_path):
    return sorted(glob.glob(os.path.join(directory_path, '*.csv')))

# Column name cleaning function to replace colons in WiFi/BLE identifiers

def clean_column_names(df):
    df.columns = [col.replace(':', '_') if 'wifi_rssi' in col or 'ble_rssi' in col or 'wifi_freq' in col else col for col in df.columns]
    return df

def clean_feature_names(a, b, c):
    a = [col.replace(':', '_') for col in a]
    b = [col.replace(':', '_') for col in b]
    c = [col.replace(':', '_') for col in c]
    return a, b, c

# Helper function: extract known WiFi/BLE columns from training data

# Mapping original MAC to sanitized column names (for reference or reverse mapping)
def generate_mac_column_mapping(columns):
    mapping = {}
    for col in columns:
        if 'wifi_rssi' in col or 'ble_rssi' in col:
            original = col.replace('_', ':', 5)  # replace only first 5 underscores
            mapping[original] = col
    return mapping

def extract_known_wifi_ble_columns(csv_file):
    data = pd.read_csv(csv_file)
    wifi_rssi_cols = [col for col in data.columns if 'wifi_rssi' in col]
    wifi_freq_cols = [col for col in data.columns if 'wifi_freq' in col]
    ble_cols = [col for col in data.columns if 'ble_rssi' in col]
    return wifi_rssi_cols, wifi_freq_cols, ble_cols

def feature_list(wifi_rssi_file, wifi_freq_file, ble_rssi_file):
    wifi_rssi_df = pd.read_csv(wifi_rssi_file)
    wifi_freq_df = pd.read_csv(wifi_freq_file)
    ble_rssi_df = pd.read_csv(ble_rssi_file)
    return wifi_rssi_df.iloc[:, 0].tolist(), wifi_freq_df.iloc[:, 0].tolist(), ble_rssi_df.iloc[:, 0].tolist()
# =========================

# =========================
# Updated Dataset class with optional WiFi Frequency feature

class IPINDataset(Dataset):
    def __init__(self, csv_file, sequence_length=100, known_wifi_rssi_cols=None, known_ble_cols=None, known_wifi_freq_cols=None, include_wifi_freq=True):
        if isinstance(csv_file, list):
            self.data = pd.concat([pd.read_csv(f) for f in csv_file], ignore_index=True)
        else:
            self.data = pd.read_csv(csv_file)
        self.data = clean_column_names(self.data)

        # Drop unavailable sensor features
        self.data = self.data.drop(columns=['PRES', 'HUMI', 'TEMP'], errors='ignore')

        # Identify feature columns
        imu_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
        '''
        wifi_rssi_cols = wifi_rssi_feature_list
        wifi_freq_cols = wifi_freq_feature_list
        ble_cols = ble_rssi_feature_list

        self.include_wifi_freq = include_wifi_freq

        if self.include_wifi_freq:
            wifi_cols = wifi_rssi_cols + wifi_freq_cols
        else:
            wifi_cols = wifi_rssi_cols
        '''
        
        
        all_wifi_cols = [col for col in self.data.columns if 'wifi_rssi' in col]
        all_ble_cols = [col for col in self.data.columns if 'ble_rssi' in col]

        self.include_wifi_freq = include_wifi_freq
        if known_wifi_rssi_cols is not None:
            wifi_cols = [col for col in all_wifi_cols if col in known_wifi_rssi_cols]
        else:
            wifi_cols = all_wifi_cols

        if known_ble_cols is not None:
            ble_cols = [col for col in all_ble_cols if col in known_ble_cols]
        else:
            ble_cols = all_ble_cols

        if self.include_wifi_freq:
            wifi_freq_cols = [col for col in self.data.columns if 'wifi_freq' in col]
            if known_wifi_freq_cols is not None:
                wifi_freq_cols = [col for col in wifi_freq_cols if col in known_wifi_freq_cols]
            wifi_cols = wifi_cols + wifi_freq_cols
        
        

        # Normalize features
        epsilon = 1e-6
        self.data[imu_cols] = (self.data[imu_cols] - self.data[imu_cols].mean()) / (self.data[imu_cols].std() + epsilon)
        self.data[wifi_cols] = (self.data[wifi_cols] - self.data[wifi_cols].mean()) / (self.data[wifi_cols].std() + epsilon)
        self.data[ble_cols] = (self.data[ble_cols] - self.data[ble_cols].mean()) / (self.data[ble_cols].std() + epsilon)

        self.feature_cols = imu_cols + wifi_cols + ble_cols

        # Debug check for NaN in features or labels
        print("NaN in features:", self.data[self.feature_cols].isna().sum().sum())
        print("NaN in Latitude_degrees/Longitude_degrees:", self.data[['Latitude_degrees', 'Longitude_degrees']].isna().sum())

        # Extract sequences
        self.sequences = []
        self.labels = []
        for i in range(0, len(self.data) - sequence_length):
            seq = self.data.iloc[i:i+sequence_length][self.feature_cols].values
            label = self.data.iloc[i+sequence_length][['Latitude_degrees', 'Longitude_degrees']].values
            self.sequences.append(seq)
            self.labels.append(label)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)

# =========================
# 2. Transformer Model (Multi-modal Support)
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TransformerPositionModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=3, dim_feedforward=128):
        super().__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.output_fc(x)

# =========================
# 3. Training Pipeline
# =========================

def train_model(model, dataloader, epochs=20, lr=1e-3):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in dataloader:
            inputs, targets = batch
            outputs = model(inputs)
            # print(inputs)
            # print(outputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

# =========================
# 4. Inference & Evaluation
# =========================

def predict(model, sequence_tensor):
    model.eval()
    with torch.no_grad():
        output = model(sequence_tensor.unsqueeze(0))
    return output.squeeze().numpy()

def evaluate_model(model, dataloader):
    model.eval()
    total_error = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            preds = model(inputs)
            error = torch.norm(preds - targets, dim=1)
            total_error.extend(error.cpu().numpy())
    mean_error = np.mean(total_error)
    median_error = np.median(total_error)
    print(f"Evaluation - Mean Error: {mean_error:.2f}, Median Error: {median_error:.2f}")
    return total_error

def plot_error_histogram(errors):
    plt.hist(errors, bins=30)
    plt.title("Localization Error Histogram")
    plt.xlabel("Error (m)")
    plt.ylabel("Count")
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # =========================
    # Example Usage (replace paths with your file)
    # =========================
    #wifi_rssi_feature, wifi_freq_feature, ble_rssi_feature = feature_list('./all_column_csv_files/unique_wifi_rssi.csv', './all_column_csv_files/unique_wifi_freq.csv', './all_column_csv_files/unique_ble_rssi.csv')
    #wifi_rssi_feature, wifi_freq_feature, ble_rssi_feature = clean_feature_names(wifi_rssi_feature, wifi_freq_feature, ble_rssi_feature)
    #print(wifi_rssi_feature+wifi_freq_feature)
    csv_list = get_all_csv_files('./temp_data/')
    dataset = IPINDataset(csv_list)

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = TransformerPositionModel(input_dim=dataset[0][0].shape[1])
    train_model(model, dataloader)
    errors = evaluate_model(model, dataloader)
    plot_error_histogram(errors)
    sample_input, _ = dataset[0]
    pred = predict(model, sample_input)
    print("Predicted position:", pred)
