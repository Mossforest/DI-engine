import numpy as np  # 数值计算的库
import h5py
import os

# # # ### Part.1 h5py背景信息加入，合并 #############################################################

bg_dict = {                                   # [normal, hardcore, friction]
    'bipedalwalker_hardcore_default_collect':   [0, 1, 2.5],
    'bipedalwalker_hardcore_smooth_collect':    [0, 1, 1],
    'bipedalwalker_hardcore_smoother_collect':  [0, 1, 0.1],
    'bipedalwalker_hardcore_random_collect':    [0, 1, -1],   # representing unknown
    'bipedalwalker_normal_default_collect':     [1, 0, 2.5],
    'bipedalwalker_normal_smooth_collect':      [1, 0, 1],
    'bipedalwalker_normal_smoother_collect':    [1, 0, 0.1],
    'bipedalwalker_normal_random_collect':      [1, 0, -1],
}




##  读取h5py文件 ############################################################
for dir in os.listdir('./bipedalwalker_data'):
    if 'random' in dir:
        continue
    
    train_dict = {}
    print(f'------------------{dir}-------------------')
    for hdfile in os.listdir(f'./bipedalwalker_data/{dir}'):
        if not '.hdf5' in hdfile:
            continue
        f = h5py.File(f'./bipedalwalker_data/{dir}/{hdfile}',"r")
        if dir in train_dict:
            train_dict[dir].append(f)
        else:
            train_dict[dir] = [f]

    # 数据处理
    processed_train_dict = {}
    for dir in train_dict:
        for file in train_dict[dir]:
            for key in file.keys():
                if key in processed_train_dict:
                    processed_train_dict[key] = np.concatenate((processed_train_dict[key], np.array(file[key])), axis=0)
                else:
                    processed_train_dict[key] = np.array(file[key])
            bg = np.array(bg_dict[dir])
            bg = bg[np.newaxis, :].repeat(file['action'].shape[0], axis=0)
            if 'background' in processed_train_dict:
                processed_train_dict['background'] = np.concatenate((processed_train_dict['background'], bg), axis=0)
            else:
                processed_train_dict['background'] = bg


    # # ### 创建、修改h5py文件 #############################################################

    f = h5py.File(f'./bipedalwalker_data/{dir}/processed.hdf5', "w")
    for key in processed_train_dict:
        dset = f.create_dataset(key, data=processed_train_dict[key])
    f.close()


