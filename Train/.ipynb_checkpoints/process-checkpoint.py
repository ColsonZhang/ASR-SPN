import multiprocessing
from tqdm import tqdm
import numpy as np
import os
import sys

# max cores used for compution
cores = 16

# files path
data_dir =      '../../dataset/Data_processed_2/'
output_dir =    '../../dataset/Data_useful/'

labels =        [ 'down',   'go',   'left',     'no',   'off',  'on',   'right',    'stop',     'up',   'yes']
labels_num =    [   0,      1,      2,          3,      4,      5,      6,          7,          8,      9 ]

def process_data(arg):
    mfcc = arg[0]
    label = arg[1]
    mfcc = mfcc.T
    mfcc_res = []
    for index, frame in zip(range(len(mfcc)), mfcc):
        temp = np.append(frame, [int(index),label])
        mfcc_res.append(temp)
    mfcc_res = np.array(mfcc_res)
    return mfcc_res


if __name__ == '__main__':

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for lab, num in zip(labels, labels_num):
        print('Processing the label {} : {}'.format(num, lab))
        npy_name = data_dir + '/' + lab + '.npy'
        save_name = output_dir + '/' + lab + '.npy'

        the_mfccs = np.load(npy_name)
        mfccs_list = list(the_mfccs)
        label_list = [int(num)]*len(mfccs_list)
        arg_list = []
        for i in range(len(mfccs_list)):
            arg_list.append([mfccs_list[i],label_list[i]])

        pool = multiprocessing.Pool(processes=cores)
        with pool as workers:
            data_container = list(tqdm(workers.imap(process_data,arg_list),total=len(mfccs_list)))
        workers.close()
        workers.join()

        data_container = np.array(data_container)
        np.save(save_name, data_container)


