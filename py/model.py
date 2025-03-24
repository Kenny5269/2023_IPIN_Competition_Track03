# IPIN 2023 Track 3 - PyTorch Implementation (Data Preprocessing + Model + Training + Evaluation + WiFi/BLE Fusion + Time Synchronization + Unknown AP/Beacon Handling)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import time

# =========================
# 1. Data Preprocessing

from torch.utils.data import random_split

def split_dataset(dataset, train_ratio=0.8, batch_size=32):
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

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
# 2. Dataset

# Utility for label normalization
class LabelNormalizer:
    def __init__(self, mean_lat, std_lat, mean_lon, std_lon):
        self.mean_lat = mean_lat
        self.std_lat = std_lat
        self.mean_lon = mean_lon
        self.std_lon = std_lon

    def normalize(self, lat, lon):
        norm_lat = (lat - self.mean_lat) / self.std_lat
        norm_lon = (lon - self.mean_lon) / self.std_lon
        return norm_lat, norm_lon

    def denormalize(self, norm_lat, norm_lon):
        lat = norm_lat * self.std_lat + self.mean_lat
        lon = norm_lon * self.std_lon + self.mean_lon
        return lat, lon

class IPINDataset(Dataset):
    def __init__(self, csv_file, sequence_length=100, known_wifi_rssi_cols=None, known_ble_cols=None, known_wifi_freq_cols=None, include_wifi_freq=True, normalize_labels=True):
        if isinstance(csv_file, list):
            self.data = pd.concat([pd.read_csv(f) for f in csv_file], ignore_index=True)
        else:
            self.data = pd.read_csv(csv_file)
        self.data = clean_column_names(self.data)
        self.data = self.data.drop(columns=['PRES', 'HUMI', 'TEMP'], errors='ignore')

        imu_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z', 'mag_x', 'mag_y', 'mag_z']
        all_wifi_cols = [col for col in self.data.columns if 'wifi_rssi' in col]
        all_ble_cols = [col for col in self.data.columns if 'ble_rssi' in col]

        self.include_wifi_freq = include_wifi_freq
        wifi_cols = [col for col in all_wifi_cols if col in known_wifi_rssi_cols] if known_wifi_rssi_cols else all_wifi_cols
        ble_cols = [col for col in all_ble_cols if col in known_ble_cols] if known_ble_cols else all_ble_cols

        if self.include_wifi_freq:
            wifi_freq_cols = [col for col in self.data.columns if 'wifi_freq' in col]
            if known_wifi_freq_cols:
                wifi_freq_cols = [col for col in wifi_freq_cols if col in known_wifi_freq_cols]
            wifi_cols += wifi_freq_cols

        epsilon = 1e-6
        self.data[imu_cols] = (self.data[imu_cols] - self.data[imu_cols].mean()) / (self.data[imu_cols].std() + epsilon)
        self.data[wifi_cols] = (self.data[wifi_cols] - self.data[wifi_cols].mean()) / (self.data[wifi_cols].std() + epsilon)
        self.data[ble_cols] = (self.data[ble_cols] - self.data[ble_cols].mean()) / (self.data[ble_cols].std() + epsilon)

        self.feature_cols = imu_cols + wifi_cols + ble_cols

        # Label normalization setup
        self.normalize_labels = normalize_labels
        if self.normalize_labels:
            self.mean_lat = self.data['Latitude_degrees'].mean()
            self.std_lat = self.data['Latitude_degrees'].std()
            self.mean_lon = self.data['Longitude_degrees'].mean()
            self.std_lon = self.data['Longitude_degrees'].std()
            self.label_normalizer = LabelNormalizer(self.mean_lat, self.std_lat, self.mean_lon, self.std_lon)
        print("NaN in features:", self.data[self.feature_cols].isna().sum().sum())
        print("NaN in Latitude/Longitude:", self.data[['Latitude_degrees', 'Longitude_degrees']].isna().sum())

        all_features = self.data[self.feature_cols].values
        all_labels = self.data[['Latitude_degrees', 'Longitude_degrees']].values

        if self.normalize_labels:
            all_labels[:, 0], all_labels[:, 1] = self.label_normalizer.normalize(all_labels[:, 0], all_labels[:, 1])

        num_samples = len(all_features) - sequence_length
        print(num_samples)
        self.sequences = np.zeros((num_samples, sequence_length, all_features.shape[1]), dtype=np.float32)
        self.labels = np.zeros((num_samples, all_labels.shape[1]), dtype=np.float32)
        for i in range(num_samples):
            self.sequences[i] = all_features[i:i+sequence_length]
            self.labels[i] = all_labels[i+sequence_length]
            #print(i)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.from_numpy(self.sequences[idx]), torch.from_numpy(self.labels[idx])


# =========================
# 3. Model

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
            nn.Linear(d_model, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, x):
        x = self.input_fc(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.output_fc(x)

# =========================
# 4. Training & Evaluation

def train_model(model, dataloader, device, epochs=20, lr=1e-3):
    # # model.to(device)  # <-- removed, now moved to example usage
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    loss_list = []
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(dataloader)
        loss_list.append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Time: {time.time() - start_time:.4f}s")
    plot_loss_curve(loss_list)


def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = phi2 - phi1
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def evaluate_model(model, dataloader, device, label_normalizer=None):
    model.eval()
    total_error = []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            preds = model(inputs).cpu().numpy()
            targets = targets.cpu().numpy()
            if label_normalizer:
                for i in range(len(preds)):
                    preds[i][0], preds[i][1] = label_normalizer.denormalize(preds[i][0], preds[i][1])
                    targets[i][0], targets[i][1] = label_normalizer.denormalize(targets[i][0], targets[i][1])
            for p, t in zip(preds, targets):
                total_error.append(haversine_distance(p[0], p[1], t[0], t[1]))
    print(f"Mean Error: {np.mean(total_error):.2f} m, Median Error: {np.median(total_error):.2f} m")
    return total_error

# Predict function that returns denormalized prediction

def plot_loss_curve(loss_list):
    plt.figure(figsize=(8, 4))
    plt.plot(loss_list, marker='o', linestyle='-', alpha=0.8)
    plt.title('Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.show()

def collect_predictions(model, dataloader, device, label_normalizer=None):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            preds = model(inputs).cpu().numpy()
            targets = targets.cpu().numpy()
            if label_normalizer:
                for i in range(len(preds)):
                    pred_lat, pred_lon = label_normalizer.denormalize(preds[i][0], preds[i][1])
                    true_lat, true_lon = label_normalizer.denormalize(targets[i][0], targets[i][1])
                    all_preds.append([pred_lat, pred_lon])
                    all_targets.append([true_lat, true_lon])
            else:
                all_preds.extend(preds.tolist())
                all_targets.extend(targets.tolist())
    return np.array(all_preds), np.array(all_targets)


def plot_error_histogram(train_errors, test_errors):
    plt.figure(figsize=(10, 5))
    plt.hist(train_errors, bins=50, alpha=0.6, label='Train Error')
    plt.hist(test_errors, bins=50, alpha=0.6, label='Test Error')
    plt.xlabel('Haversine Error (meters)')
    plt.ylabel('Count')
    plt.title('Prediction Error Distribution')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_error_trend(errors, dataset_type="Test"):
    plt.figure(figsize=(10, 4))
    plt.plot(errors, marker='o', linestyle='-', alpha=0.7)
    plt.title(f'{dataset_type} Set Error per Sample')
    plt.xlabel('Sample Index')
    plt.ylabel('Haversine Error (meters)')
    plt.grid(True)
    plt.show()

def plot_prediction_scatter(preds, targets):
    plt.figure(figsize=(8, 6))
    plt.scatter(targets[:, 1], targets[:, 0], label='True', marker='o', alpha=0.6)
    plt.scatter(preds[:, 1], preds[:, 0], label='Predicted', marker='x', alpha=0.6)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('True vs Predicted Locations')
    plt.legend()
    plt.grid(True)
    plt.show()

def predict(model, input_tensor, device, label_normalizer=None):
    model.eval()
    input_tensor = input_tensor.to(device).unsqueeze(0)  # Add batch dim
    with torch.no_grad():
        pred = model(input_tensor).squeeze().cpu().numpy()
    if label_normalizer:
        pred_lat, pred_lon = label_normalizer.denormalize(pred[0], pred[1])
        return pred_lat, pred_lon
    return pred

if __name__ == '__main__':
    # =========================
    # Example Usage (replace paths with your file)

    # Select device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda'
    print(f"Using device: {device}")

    # Load dataset
    csv_list = get_all_csv_files('./py/training_data/')
    wifi_rssi_cols, wifi_freq_cols, ble_cols = extract_known_wifi_ble_columns(csv_list[0])
    dataset = IPINDataset(csv_list, sequence_length=100, known_wifi_rssi_cols=wifi_rssi_cols, known_wifi_freq_cols=wifi_freq_cols, known_ble_cols=ble_cols)

    # Split into train and test set
    train_loader, test_loader = split_dataset(dataset, train_ratio=0.8, batch_size=128)
    print('ok')

    # Initialize model
    model = TransformerPositionModel(input_dim=dataset[0][0].shape[1], d_model=128, nhead=4, dim_feedforward=256, num_layers=3)
    model.to(device)

    # Train and evaluate
    train_model(model, train_loader, device, epochs=20, lr=1e-3)
    train_errors = evaluate_model(model, train_loader, device, label_normalizer=dataset.label_normalizer)
    test_errors = evaluate_model(model, test_loader, device, label_normalizer=dataset.label_normalizer)

    # Collect predictions and targets for scatter plot
    all_preds, all_targets = collect_predictions(model, test_loader, device, label_normalizer=dataset.label_normalizer)
    plot_prediction_scatter(all_preds, all_targets)

    plot_error_histogram(train_errors, test_errors)
    plot_error_trend(train_errors, dataset_type='Train')
    plot_error_trend(test_errors, dataset_type='Test')

    # Sample prediction
    sample_input, _ = dataset[0]
    pred_lat, pred_lon = predict(model, sample_input, device, label_normalizer=dataset.label_normalizer)
    print("Predicted latitude/longitude:", pred_lat, pred_lon)
    