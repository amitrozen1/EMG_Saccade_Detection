import os
import numpy as np
from scipy.stats import zscore
#there are some general parameters in General.init
from General.init import *
from General.HelperFunctions import getPathsFromSessionFolder
from EDF.EdfAnalyzer import EdfAnalyzer
import matplotlib.pyplot as plt

def remove_first_values_from_all_array_values(arr):
    s = arr[0]
    for i in range(len(arr)):
        arr[i] = arr[i] - s
    return arr

#data path - the folder that contains session folders
# session folder name - e.g: 121222_1500
# electrode_number - the number of the electrode to extract the blinking vector from
def extract_blinking_vector(data_path, session, p,  electrode_number, downsampleInMs=1, rms_size_ms=100):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsampleInMs, forEEG=True)
    Y = zscore(Y, axis=1)

    Y = EdfAnalyzer.window_rms(Y, rms_size_ms, freq)
    #downsample:
    if (downsampleInMs != 1):
        Y = EdfAnalyzer.resample_by_averaging_every_n(Y, downsampleInMs, freq)
    return Y[electrode_number], EdfAnalyzer.getCallibrationTicks(chunks, "blink")

def main():
    # shoval, you can play with the windows values up to a point you smooth the signal enough but not too much.
    # too much = loosing valuable information.
    # 20ms ==> 50HZ
    
    vector, blinks_timing = extract_blinking_vector(data_path, "07122022_1230", A, 15, 20)
    vector = vector[blinks_timing[0]:blinks_timing[-1]]
    blinks_timing = remove_first_values_from_all_array_values(blinks_timing)

    plt.plot(vector)
    #plot blinking times as red ticks:
    for t in blinks_timing:
        plt.axvline(x=t, color='r')
    plt.show()