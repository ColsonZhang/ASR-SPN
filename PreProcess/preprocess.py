"""
FileName: preprocess.py 
Purpose: Transform the wave(point-detected) into the mfcc(encoded)
Author: ZhangShen
Time: 2021-06-29
Email: colson_zhang@foxmail.com
"""


# ===================================================================================================
# import modulse
# ===================================================================================================

from pathlib import Path
from tqdm import tqdm
import numpy as np
import librosa
import os
import multiprocessing
import sys

# ===================================================================================================
# the parameters
# HERE IS THE OPTIONAL PARAMETERS  !!!!!!!!!!!!!
# ===================================================================================================

# max cores used for compution
cores = 4 # recommand 32 cores 

# file-path

TRAIN_PATH = '../Dataset/dataset_toy/'
labels = [ 'down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'yes']

# output_dir-path
data_dir = '../Dataset/Data_pre_processed'

# important parameters

fs = 16e3; # Known sample rate of the data set.

segmentDuration = 1
frameDuration = 0.05  # 0.025
hopDuration = 0.020   # 0.010

segmentSamples = round(segmentDuration*fs)
frameSamples = round(frameDuration*fs)
hopSamples = round(hopDuration*fs)
overlapSamples = frameSamples - hopSamples

FFTLength = 1024 # 512
numBands = 13 # 50

# ===================================================================================================
# the process function of waves
# ===================================================================================================

def pad_wave(y):
    """
    :param y: the input wave.
    :return y_pad: the extended wave with the pad-operation.
    """
    numSamples  = y.shape[0]
    numToPadFront = np.floor( (segmentSamples - numSamples)/2 );
    numToPadBack = np.ceil( (segmentSamples - numSamples)/2 );

    numToPadFront = int(numToPadFront)
    numToPadBack = int(numToPadBack)

    num_pad = (numToPadFront, numToPadBack)
    y_pad = np.pad(y, num_pad, 'constant')
    
    return y_pad

def extract_wave(wave):
    """
    :param wave: the input wave .
    :return result: the extracted wave by the EndPoint-Detection 
    """
    # transform the wave-data's kind from float32 into shrot
    wave_encode = encode_wave(wave)
    
    # end point detection
    energy = calEnergy(wave_encode)
    zeroCrossingRate = calZeroCrossingRate(wave_encode)
    N = endPointDetect(energy, zeroCrossingRate)
    
    try:
        # extract the wave by the end point 
        result = np.array([])
        m = 0
        if len(N) == 2:
            temp = wave[N[m] * 256: N[m + 1] * 256]
            result = np.append(result,temp)
        elif len(N) == 1:
            temp = wave[N[m] * 256: ]
            result = np.append(result,temp)      
#         while m < len(N):
#             temp = wave[N[m] * 256: N[m + 1] * 256]
#             result = np.append(result,temp)
#             m = m + 2
    except:
        print("{}".format(N))
        result = np.array([])
    return result
# ===================================================================================================
# the process function of detectting the start-end points
# ===================================================================================================

def encode_wave(wave):
    """
    :param wave: the wave-input
    :return: the result of the encoded wave
    """
    result = (wave*2**15).astype(np.short)    
    return result

def calEnergy(wave_data):
    """
    :param wave_data: binary data of audio file
    :return: energy
    """
    energy = []
    sum = 0
    for i in range(len(wave_data)):
        sum = sum + (int(wave_data[i]) * int(wave_data[i]))
        if (i + 1) % 256 == 0:
            energy.append(sum)
            sum = 0
        elif i == len(wave_data) - 1:
            energy.append(sum)
    return energy


def calZeroCrossingRate(wave_data):
    """
    :param wave_data: binary data of audio file
    :return: ZeroCrossingRate
    """
    zeroCrossingRate = []
    sum = 0
    for i in range(len(wave_data)):
        sum = sum + np.abs(int(wave_data[i] >= 0) - int(wave_data[i - 1] >= 0))
        if (i + 1) % 256 == 0:
            zeroCrossingRate.append(float(sum) / 255)
            sum = 0
        elif i == len(wave_data) - 1:
            zeroCrossingRate.append(float(sum) / 255)
    return zeroCrossingRate


def endPointDetect(energy, zeroCrossingRate):
    """
    :param energy: energy
    :param zeroCrossingRate: zeroCrossingRate
    :return: data after endpoint detection
    """
    sum = 0
    for en in energy:
        sum = sum + en
    avg_energy = sum / len(energy)

    sum = 0
    for en in energy[:5]:
        sum = sum + en
    ML = sum / 5
    MH = avg_energy / 5  # high energy threshold
    ML = (ML + MH) / 5  # low energy threshold

    sum = 0
    for zcr in zeroCrossingRate[:5]:
        sum = float(sum) + zcr
    Zs = sum / 5  # zero crossing rate threshold

    A = []
    B = []
    C = []

    # MH is used for preliminary detection
    flag = 0
    for i in range(len(energy)):
        if len(A) == 0 and flag == 0 and energy[i] > MH:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 21 > A[len(A) - 1]:
            A.append(i)
            flag = 1
        elif flag == 0 and energy[i] > MH and i - 21 <= A[len(A) - 1]:
            A = A[:len(A) - 1]
            flag = 1

        if flag == 1 and energy[i] < MH:
            # if frame is too short, remove it
            if i - A[len(A) - 1] <= 2:
                A = A[:len(A) - 1]
            else:
                A.append(i)
            flag = 0

    # ML is used for second detection
    for j in range(len(A)):
        i = A[j]
        if j % 2 == 1:
            while i < len(energy) and energy[i] > ML:
                i = i + 1
            B.append(i)
        else:
            while i > 0 and energy[i] > ML:
                i = i - 1
            B.append(i)

    # zero crossing rate threshold is for the last step
    for j in range(len(B)):
        i = B[j]
        if j % 2 == 1:
            while i < len(zeroCrossingRate) and zeroCrossingRate[i] >= 3 * Zs:
                i = i + 1
            C.append(i)
        else:
            while i > 0 and zeroCrossingRate[i] >= 3 * Zs:
                i = i - 1
            C.append(i)
    return C

# ===================================================================================================
# the process function of extracting mfcc
# ===================================================================================================

def extract_mfcc(y):
    """
    :param y: the wave data.
    :return: the result of the mfcc.
    """
    mfcc = librosa.feature.mfcc(y=y, sr=fs, n_mfcc=numBands, n_fft=FFTLength, hop_length=hopSamples, win_length=frameSamples )
    return mfcc


def encode_mfcc(the_mfcc, dimension = 8):
    """
    :param the_mfcc: the mfcc-input; shape[0]=features' number; shape[1]=frames' number.
    :param dimension: the digitization's standard's dimension.
    :return: the result of the encoded mfcc.
    """
    result = np.asarray(the_mfcc)
    for i in range(the_mfcc.shape[0]):
        temp = the_mfcc[i,:]
        term_max = np.max(temp)
        term_min = np.min(temp)
        bins = np.linspace(term_min,term_max, dimension)
        which_bin = np.digitize(temp, bins=bins) - 1
        result[i,:] = which_bin
    
    return result

def get_mfcc(filename):
    """
    :parame y: the wave file name.
    :return: the result of the mfcc with the EndPoint-Detection and the Encode-MFCC .
    """
    y, sr = librosa.load(filename,sr=None)
    
    # y_extract = pad_wave(y)   # pad the wave
    y_extract = extract_wave(y) # extract the wave

    if len(y_extract)==0:
        the_mfcc = None
    else:
        the_mfcc = extract_mfcc(y_extract)

        # get the delta-data
        # the_mfcc = librosa.feature.delta(the_mfcc, order = 2, mode ='nearest')

        # encode the mfcc
        the_mfcc = the_mfcc.T
        the_mfcc = encode_mfcc(the_mfcc, dimension=4)
        # the_mfcc = encode_mfcc(the_mfcc)

    return the_mfcc

# ===================================================================================================
# the top function
# ===================================================================================================

def get_path_filenames(path):
    """
    :param path: the file's dir.
    :return: the file names in the dirs .
    """
    file_names = []
    pathname = Path(path)

    for cp in pathname.iterdir():
        term = Path(cp)
        file_names.append(term.name)

    return file_names
    

def parallel_process(file_name):
    """
    :param file_name: the file's name.
    :return: the MFCC .
    """
    the_mfcc = get_mfcc(file_name)
    return the_mfcc


if __name__ == '__main__':

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for lab in labels:
        print('Processing the label:',lab)
        npy_name = data_dir + "/" + lab +'.npy'
        
        # data_container = []
        path = TRAIN_PATH + lab + '/'
        files = get_path_filenames(path)
        files_list = [path+i for i in files]
        
        pool = multiprocessing.Pool(processes=cores)
        with pool as workers:
            data_container = list(tqdm(workers.imap(parallel_process,files_list),total=len(files_list)))
        workers.close()
        workers.join()

        # clear the None
        data_clear = []
        for i in data_container:
            if type(i) == type(np.array([])):
                data_clear.append(i)

        data_container = np.array(data_clear)
        np.save(npy_name, data_container)
