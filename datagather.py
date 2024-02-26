import numpy as np  # 数值计算的库
import h5py
import os
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# # # ### Part.1 h5py背景信息加入，合并 #############################################################

# bg_dict = {                                   # [normal, hardcore, friction]
#     'bipedalwalker_hardcore_default_collect':   [0, 1, 2.5],
#     'bipedalwalker_hardcore_smooth_collect':    [0, 1, 1],
#     'bipedalwalker_hardcore_smoother_collect':  [0, 1, 0.1],
#     'bipedalwalker_hardcore_random_collect':    [0, 1, -1],   # representing unknown
#     'bipedalwalker_normal_default_collect':     [1, 0, 2.5],
#     'bipedalwalker_normal_smooth_collect':      [1, 0, 1],
#     'bipedalwalker_normal_smoother_collect':    [1, 0, 0.1],
#     'bipedalwalker_normal_random_collect':      [1, 0, -1],
# }

# train_dict = {}
# test_dict = {}



###  读取h5py文件 ############################################################
# for dir in os.listdir('./bipedalwalker_data'):
#     print(f'------------------{dir}-------------------')
#     for hdfile in os.listdir(f'./bipedalwalker_data/{dir}'):
#         if not '.hdf5' in hdfile:
#             continue
#         f = h5py.File(f'./bipedalwalker_data/{dir}/{hdfile}',"r")
#         if 'random' in dir:
#             if dir in test_dict:
#                 test_dict[dir].append(f)
#             else:
#                 test_dict[dir] = [f]
#         else:
#             if dir in train_dict:
#                 train_dict[dir].append(f)
#             else:
#                 train_dict[dir] = [f]


# # 数据处理
# processed_train_dict = {}
# processed_test_dict = {}
# for dir in train_dict:
#     for file in train_dict[dir]:
#         for key in file.keys():
#             if key in processed_train_dict:
#                 processed_train_dict[key] = np.concatenate((processed_train_dict[key], np.array(file[key])), axis=0)
#             else:
#                 processed_train_dict[key] = np.array(file[key])
#         bg = np.array(bg_dict[dir])
#         bg = bg[np.newaxis, :].repeat(file['action'].shape[0], axis=0)
#         if 'background' in processed_train_dict:
#             processed_train_dict['background'] = np.concatenate((processed_train_dict['background'], bg), axis=0)
#         else:
#             processed_train_dict['background'] = bg

# for dir in test_dict:
#     for file in test_dict[dir]:
#         for key in file.keys():
#             if key in processed_test_dict:
#                 processed_test_dict[key] = np.concatenate((processed_test_dict[key], np.array(file[key])), axis=0)
#             else:
#                 processed_test_dict[key] = np.array(file[key])
#         bg = np.array(bg_dict[dir])
#         bg = bg[np.newaxis, :].repeat(file['action'].shape[0], axis=0)
#         if 'background' in processed_test_dict:
#             processed_test_dict['background'] = np.concatenate((processed_test_dict['background'], bg), axis=0)
#         else:
#             processed_test_dict['background'] = bg



# # # ### 创建、修改h5py文件 #############################################################

# f = h5py.File('./bipedalwalker_data/friction_known.hdf5', "w")
# for key in processed_train_dict:
#     dset = f.create_dataset(key, data=processed_train_dict[key])
# f.close()

# f = h5py.File('./bipedalwalker_data/friction_unknown.hdf5', "w")
# for key in processed_test_dict:
#     dset = f.create_dataset(key, data=processed_test_dict[key])
# f.close()





# # # ### Part.2 分离friction_known的train, test = 0.8/0.2 #############################################################


source = h5py.File('./bipedalwalker_data/bipedalwalker_normal_smoother_collect/processed.hdf5',"r")
for key in source.keys():
    print(f'{key}: {source[key].shape}')
block = int(source['action'].shape[0] * 0.8)

train = h5py.File('./bipedalwalker_data/bipedalwalker_normal_smoother_collect/processed_train.hdf5', "w")
test = h5py.File('./bipedalwalker_data/bipedalwalker_normal_smoother_collect/processed_eval.hdf5', "w")
for key in source.keys():
    dset1 = train.create_dataset(key, data=source[key][:block])
    dset2 = test.create_dataset(key, data=source[key][block:])
train.close()
test.close()



############ 看看是否norm了obs

# source = h5py.File('./bipedalwalker_data/friction_known.hdf5',"r")
# print(f"obs: {source['obs'].shape}")

mmean = np.array([ 0.16703084, -0.00234181 , 0.390704,   -0.01001118,  0.3062675,  -0.01050948,
                    0.14360847, -0.06224813,  0.32798332,  0.36005434, -0.0191548,  -0.30569798,
                    -0.07250575,  0.45160165,  0.38210124,  0.3834068,   0.39315137,  0.42927817,
                    0.48456225,  0.5228259,   0.5992432,   0.75041467,  0.9561498,   0.9924783 ])
sstd = np.array([0.27602437, 0.02943488, 0.24448505, 0.06426995, 0.68457866, 0.7287007,
                 0.5446617,  0.80962706, 0.47370645, 0.46831095, 0.7259809 , 0.28212553,
                 0.69274896, 0.49801946, 0.06951486, 0.06732956, 0.07175232, 0.07613435,
                 0.08536533, 0.09403562, 0.11830232, 0.132318  , 0.09987015, 0.04131253])

# obs = np.array(source['obs'])


# for idx in range(obs.shape[1]):
#     value = obs[:,idx]
#     s = pd.DataFrame(value, columns = [str(idx)])
#     fig = plt.figure(figsize = (10,6))
#     s.hist(bins=100, alpha = 0.8)
#     plt.savefig(f'./tmp/{idx}_before.png')

# obs = (obs - mmean) / sstd

# for idx in range(obs.shape[1]):
#     value = obs[:,idx]
#     s = pd.DataFrame(value, columns = [str(idx)])
#     fig = plt.figure(figsize = (10,6))
#     s.hist(bins=100, alpha = 0.8)
#     plt.savefig(f'./tmp/{idx}_after.png')

# mmax = obs.mean(axis=0)
# mmin = obs.std(axis=0)
# print(mmax)
# print(mmin)