import numpy as np

q_ref = [0.898533981086575, 0.42904782, 0.03004867, -0.08747417]
q_est = [0.99997031079557, -0.00570420851817289, -0.00509270589565691, 0.000950725644482185]

dot_product = np.abs(np.dot(q_ref, q_est))  # 內積越接近 1 表示角度越接近
angle_diff = 2 * np.arccos(dot_product)
angle_diff_deg = np.degrees(angle_diff)
print(angle_diff_deg)