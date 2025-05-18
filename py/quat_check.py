import numpy as np
from scipy.spatial.transform import Rotation as R

q_ref = [0.010828, -0.43064, -0.90152, 0.041131]
q_est = [-0.07552, 0.897911, -0.43114, -0.04663]

# dot_product = np.abs(np.dot(q_ref, q_est))  # 內積越接近 1 表示角度越接近
# angle_diff = 2 * np.arccos(dot_product)
# angle_diff_deg = np.degrees(angle_diff)
# print(angle_diff_deg)

# dot = np.dot(q_ref, q_est)
# dot = np.clip(abs(dot), 0, 1)
# angle_diff_deg = np.degrees(2 * np.arccos(dot))
# print(angle_diff_deg)

# q_ref, q_est 都為 [x, y, z, w]
r_ref = R.from_quat(q_ref)
r_est = R.from_quat(q_est)

# 查看旋轉矩陣（每個 column 是一個軸在世界中的方向）
print("z_ref =", r_ref.apply([0, 0, 1]))  # 手機「上方」方向
print("z_est =", r_est.apply([0, 0, 1]))  # 你算的方向