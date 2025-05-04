
import pandas as pd
import numpy as np

# Read the calibrated IMU data
imu_df = pd.read_csv('IMU_calibrated.csv')

# Assume static interval is 40~60 seconds to estimate initial orientation
static_start = 40
static_end = 60

# Select the static data
static_range = imu_df[(imu_df['AppTimestamp(s)'] >= static_start) & (imu_df['AppTimestamp(s)'] <= static_end)]

# Calculate mean acceleration during static period (to estimate gravity vector)
mean_acc = static_range[['acc_x', 'acc_y', 'acc_z']].mean().values
gravity_vector = mean_acc / np.linalg.norm(mean_acc)  # Normalize to unit vector

# Define world frame Z-axis pointing upward
world_z = np.array([0, 0, 1])

# Compute rotation matrix to align phone's gravity vector to world Z-axis
def rotation_matrix_from_vectors(vec1, vec2):
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)  # Already aligned
    kmat = np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s ** 2))
    return rotation_matrix

R_align = rotation_matrix_from_vectors(gravity_vector, world_z)

# Rotate acceleration data
acc_data = imu_df[['acc_x', 'acc_y', 'acc_z']].values
acc_world = (R_align @ acc_data.T).T

# Rotate gyroscope data
gyro_data = imu_df[['gyro_x', 'gyro_y', 'gyro_z']].values
gyro_world = (R_align @ gyro_data.T).T

# Rotate magnetometer data
mag_data = imu_df[['mag_x', 'mag_y', 'mag_z']].values
mag_world = (R_align @ mag_data.T).T

# Save the rotated data back to the DataFrame
imu_df['acc_x_world'] = acc_world[:, 0]
imu_df['acc_y_world'] = acc_world[:, 1]
imu_df['acc_z_world'] = acc_world[:, 2]

imu_df['gyro_x_world'] = gyro_world[:, 0]
imu_df['gyro_y_world'] = gyro_world[:, 1]
imu_df['gyro_z_world'] = gyro_world[:, 2]

imu_df['mag_x_world'] = mag_world[:, 0]
imu_df['mag_y_world'] = mag_world[:, 1]
imu_df['mag_z_world'] = mag_world[:, 2]

# Save the fully aligned IMU data
imu_df.to_csv('IMU_aligned_full.csv', index=False)
