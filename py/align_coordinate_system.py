
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

# Rotate all acceleration data to world frame
acc_data = imu_df[['acc_x', 'acc_y', 'acc_z']].values
acc_world = (R_align @ acc_data.T).T  # Note the transpose

# Save the rotated accelerations into new columns
imu_df['acc_x_world'] = acc_world[:, 0]
imu_df['acc_y_world'] = acc_world[:, 1]
imu_df['acc_z_world'] = acc_world[:, 2]

# Save the aligned IMU data
imu_df.to_csv('IMU_aligned.csv', index=False)
