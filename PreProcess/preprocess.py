from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
import os

# file-path

TRAIN_PATH = '../../dataset/trainset/'
labels = [ 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

# output_dir-path
data_dir = '../../Data_processed'
# data_dir = './Data_processed'

# import parameters

fs = 16e3; # Known sample rate of the data set.

segmentDuration = 1
frameDuration = 0.025
hopDuration = 0.010

segmentSamples = round(segmentDuration*fs)
frameSamples = round(frameDuration*fs)
hopSamples = round(hopDuration*fs)
overlapSamples = frameSamples - hopSamples

FFTLength = 512
numBands = 50

# operation funcions

def pad_wave(y):
    numSamples  = y.shape[0]
    numToPadFront = np.floor( (segmentSamples - numSamples)/2 );
    numToPadBack = np.ceil( (segmentSamples - numSamples)/2 );

    numToPadFront = int(numToPadFront)
    numToPadBack = int(numToPadBack)

    num_pad = (numToPadFront, numToPadBack)
    y_pad = np.pad(y, num_pad, 'constant')
    
    return y_pad

def extract_mfcc(y):
    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=numBands, n_fft=FFTLength, hop_length=hopSamples, win_length=frameSamples )
    return mfcc

def get_mfcc(filename):
    y, sr = librosa.load(filename,sr=None)
    y_pad = pad_wave(y)
    the_mfcc = extract_mfcc(y_pad)
    return the_mfcc

def get_path_filenames(path):
    file_names = []
    pathname = Path(path)

    for cp in pathname.iterdir():
        term = Path(cp)
        file_names.append(term.name)

    return file_names
    
        


if __name__ == '__main__':

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for lab in labels:
        print('Processing the label:',lab)
        npy_name = data_dir + "/" + lab +'.npy'
        data_container = []
        path = TRAIN_PATH + lab + '/'
        files = get_path_filenames(path)
        for name in tqdm(files):
            file_name = path + name
            the_mfcc = get_mfcc(file_name)
            data_container.append(the_mfcc)
        data_container = np.array(data_container)
        np.save(npy_name, data_container)
