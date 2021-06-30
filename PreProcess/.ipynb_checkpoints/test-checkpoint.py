import numpy as np

# file-path

TRAIN_PATH = '../../dataset/trainset/'
labels = [ 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

# output_dir-path
data_dir = '../../Data_processed'


if __name__ == '__main__':
    path = data_dir + '/' + 'down.npy'
    mfcc_down = np.load(path)
    print(mfcc_down.shape)
    print(mfcc_down[0].shape)    
    