import numpy as np  # 数值计算的库
import random
import h5py
import os

##  读取h5py文件 ############################################################
file = h5py.File(f'/mnt/nfs/chenxinyan/DI-engine/bipedalwalker_data/bipedalwalker_normal_smoother_collect/processed_train.hdf5',"r")
# file = h5py.File(f'/mnt/nfs/chenxinyan/DI-engine/bipedalwalker_data/bipedalwalker_normal_smoother_collect/single_instance.hdf5',"r")

print(f'length: {(file["obs"].shape)}')

size = 2000
indexbox = []
for _ in range(size):
    indexbox.append(random.randint(0, file["obs"].shape[0]-1))

# 数据处理
processed_train_dict = {}
for key in file.keys():
    # print(key)
    # print(file[key][0])
    # print(file[key][50])
    # print(file[key].shape)
    processed_train_dict[key] = []

for idx in indexbox:
    for key in file.keys():
        processed_train_dict[key].append(np.array(file[key][idx]))

for key in file.keys():
    processed_train_dict[key] = np.stack(processed_train_dict[key])


# # ### 创建、修改h5py文件 #############################################################

f = h5py.File(f'/mnt/nfs/chenxinyan/DI-engine/bipedalwalker_data/bipedalwalker_normal_smoother_collect/few_instance_larger.hdf5', "w")
for key in processed_train_dict:
    dset = f.create_dataset(key, data=processed_train_dict[key])
f.close()
