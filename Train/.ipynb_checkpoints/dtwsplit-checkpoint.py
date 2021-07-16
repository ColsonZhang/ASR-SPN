"""
FileName: dtwsplit.py 
Purpose: align the mfcc array with the template and split the array.
Author: ZhangShen
Time: 2021-07-11
Email: colson_zhang@foxmail.com
"""


# ===================================================================================================
# import modulse
# ===================================================================================================
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from numpy import random
from numpy.linalg import norm
from dtw import dtw
import sys
import os
import time
import pickle 

# ===================================================================================================
# the parameters
# HERE IS THE OPTIONAL PARAMETERS  !!!!!!!!!!!!!
# ===================================================================================================

# the input dir
data_dir = '../Dataset/Data_pre_processed/'
# the output dir
out_dir = '../Dataset/Data_dtw_processed/'

# the labels
labels =        [ 'down',   'go',   'left',     'no',   'off',  'on',   'right',    'stop',     'up',   'yes']
labels_num =    [   0,      1,      2,          3,      4,      5,      6,          7,          8,      9 ]


# ===================================================================================================
# the functions 
# ===================================================================================================
def get_template_mfcc(mfccs, parts=3):
    """
    :parameter mfccs : the mfccs set.
    :parameter parts : the class' num, default value = 3.
    :return template_mfcc: the template.
    :return num_split: the class' list.
    """
    
    # 获取一个随机模板
    num_total = mfccs.shape[0]
    index_random = random.randint(0, num_total-1)

    template_mfcc = mfccs[index_random]
    print("The template is the {} mfcc with the shape of {}".format(index_random, template_mfcc.shape))
    # print(template_mfcc.shape)
    # print(index_random)
    
    # 根据随机模板，生成PART段索引
    num_template = template_mfcc.shape[0]
    num_split = [round((i+1)*num_template/parts) for i in range(parts)]
    # print(num_split)
    
    return template_mfcc, num_split


def get_class(num, num_split):
    """
    :parameter num : the num to be classified.
    :parameter num_split: the list which the classification's basis.
    :return result: the class.
    """
    result = 0
    for i in num_split:
        if num >= i:
            result = result + 1
        else:
            break
    return result

def get_class_array(path_num, num_split):
    """
    :parameter path_num : the num-array to be classified.
    :parameter num_split: the list which the classification's basis.
    :return result: the class-array
    """
    result = np.array([], dtype=int)
    for i in path_num:
        tmp = get_class(i, num_split)
        result = np.append(result,tmp)
    return result

def translate_path(path):
    """
    :parameter path: the dtw path
    :return indexs:  the class of the array 
    """
    indexs = np.array([],dtype=int)
    
    for x, y in zip(path[0],path[1]):
        if indexs.shape[0] < y+1:
            indexs = np.append(indexs, x)
        else:
            indexs[y] = x
            
    return indexs

def mfcc_class(the_mfcc, template_mfcc, num_split):
    """
    :parameter the_mfcc: the mfcc to be classified
    :parameter template_mfcc: the template mfcc
    :return path_index: the mfcc's class index
    """
    
    # template_mfcc == path[0] ; the_mfcc == path[1]
    dist, cost, acc_cost, path = dtw( template_mfcc, the_mfcc, dist=lambda x, y: norm(x - y, ord=1))
    
    path_class = translate_path(path)
    path_index = get_class_array(path_class, num_split)
    
    return path_index


def divide_mfccs(mfccs, parts=3):
    """
    :parameter mfccs: the mfccs to be classified
    :parameter parts: the number of the kinds
    :return big_container: the mfccs classified container 
    """
    # initial the container
    big_container = []
    
    # get the template
    template_mfcc, num_split = get_template_mfcc(mfccs,parts)
    # print(template_mfcc.shape)
    
    # divide the mfccs
    for the_mfcc in tqdm(mfccs):
        mfcc_index = mfcc_class(the_mfcc, template_mfcc, num_split)
        
        container = []
        for i in range(parts):
            container.append(np.array([]))    
        
        for index, mfcc in zip(mfcc_index, the_mfcc):
            if container[index].shape[0] == 0:
                container[index] = mfcc
            else:
                container[index] = np.vstack((container[index], mfcc))
        
        big_container.append(container)         
        
    big_container = np.array(big_container, dtype=object)
    return big_container



if __name__ == '__main__':
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    for label in labels:
        path = data_dir + label + '.npy'
        save_path = out_dir + label + '.npy' 
        
        mfccs = np.load(path, allow_pickle=True)
        result = divide_mfccs(mfccs, parts=2)
        
        np.save(save_path, result)