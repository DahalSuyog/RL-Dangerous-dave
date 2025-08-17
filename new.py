'''import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)'''
import numpy as np
data = np.load("rewards\\rnd_ddave_1742510376\\rewards.npy")
print(data)
np.savetxt('E:/data.txt', data, fmt='%.6f')
