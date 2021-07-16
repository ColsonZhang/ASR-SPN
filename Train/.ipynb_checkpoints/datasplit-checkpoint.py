"""
FileName: datasplit.py 
Purpose: split the data into the train-set and the test-set.
Author: ZhangShen
Time: 2021-07-06
Email: colson_zhang@foxmail.com
"""


# ===================================================================================================
# import modulse
# ===================================================================================================


import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import sys

# ===================================================================================================
# the parameters
# HERE IS THE OPTIONAL PARAMETERS  !!!!!!!!!!!!!
# ===================================================================================================

# file dir
data_dir = '../Dataset/Data_dtw_processed/'

# the labels
labels =        [ 'down',   'go',   'left',     'no',   'off',  'on',   'right',    'stop',     'up',   'yes']
labels_num =    [   0,      1,      2,          3,      4,      5,      6,          7,          8,      9 ]


# ===================================================================================================
# the functions 
# ===================================================================================================

def data_split( file_name , ratio = 0.4, ratio_train = 0.7):
    """
    :param tile_name: the file name.
    :param ratio: the ratio of the data used in the total data.
    :param ratio_train: the ratio of the train-data in the total data used.
    :return : the train-data and the test-data
    """
    datas = np.load(file_name, allow_pickle=True)
    total_num = round(ratio*np.shape(datas)[0])
    train_num = round(ratio_train*total_num)
    test_num = total_num - train_num
    
    train_data = datas[0:train_num]
    test_data = datas[train_num:total_num]
    
    return train_data,test_data


if __name__=="__main__":

    train_set = []
    test_set = []
    for lab in tqdm(labels):
        print('Processing lable: {}'.format(lab))
        file_name = data_dir + "/" + lab + '.npy'
        train_data,test_data = data_split(file_name, 0.4, 0.7)
        
        train_set.append(train_data)
        test_set.append(test_data)
        
    train_set = np.array(train_set, dtype = object)
    test_set = np.array(test_set, dtype = object)

    # 保存文件
    train_file =  data_dir + "/"  + 'train_toy.npy'
    test_file = data_dir + "/"  + 'test_toy.npy'
    np.save(train_file, train_set)
    np.save(test_file, test_set)