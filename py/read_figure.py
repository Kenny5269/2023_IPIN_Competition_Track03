import pickle
import matplotlib.pyplot as plt

test_id = 'TEST1'
R = 'R123'
# file_name = 'GT_KNN'
# file_name = 'GT_EKF_WIFI'
file_name = 'GT_EKF_FUSED'

with open(f'figure/{test_id}/trajectory/{R}/{file_name}.pkl', 'rb') as f:
    fig = pickle.load(f)

f,ax = plt.subplots()

plt.show()
