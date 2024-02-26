import numpy as np  # 数值计算的库
import random
import h5py
import os
import matplotlib.pyplot as plt

##  读取h5py文件 ############################################################
file = h5py.File(f'/mnt/nfs/chenxinyan/DI-engine/bipedalwalker_data/bipedalwalker_normal_default_collect/processed_train.hdf5',"r")

print(f'length: {(file["obs"].shape)}')


# 数据处理
data = file['obs'][:500]
datan = file['next_obs'][:500]
l = data.shape[1]

for i in range(l):
    idata = data[:, i]
    idatan = datan[:, i]
    plt.figure()
    plt.plot(np.array(idata))
    plt.plot(np.array(idatan), alpha=0.5)
    plt.legend(['obs', 'next_obs'])
    plt.savefig(f'./obs_realdata_{i}.png')

# # ### 创建、修改h5py文件 #############################################################
