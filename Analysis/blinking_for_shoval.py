# Extracting the blinking vector from the EEG signal

import os
import numpy as np
from scipy.stats import zscore
from scipy import signal
from General.init import *
from General.HelperFunctions import getPathsFromSessionFolder
from EDF.EdfAnalyzer import EdfAnalyzer, butter_bandpass_filter
import matplotlib.pyplot as plt
from General.HelperFunctions import GetSessions
from ComponentAnalysis.ComponentExtractor import GetSideEyeComponent
from General.PlotFunctions import savePlot, clearPlot, plotChunkData
# there are some general parameters in General.init

def remove_first_values_from_all_array_values(arr):
    s = arr[0]
    for i in range(len(arr)):
        arr[i] = arr[i] - s
    return arr

# data path - the folder that contains session folders
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
    
    return Y[electrode_number], EdfAnalyzer.getCallibrationTicks(chunks, "blink"), chunks

def extract_blinking_component(data_path, session, p, downsample_window_in_ms=1, rms_size_ms=100):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsample_window_in_ms, forEEG=True)
    
    Y = zscore(Y, axis=1)
    w, Y = EdfAnalyzer.ICA(Y, 16)

    Y = EdfAnalyzer.window_rms(Y, rms_size_ms, freq)
    #downsample:
    if (downsample_window_in_ms != 1):
        Y = EdfAnalyzer.resample_by_averaging_every_n(Y, downsample_window_in_ms, freq)

    #choosing best component for bliking:
    blinks_timing = EdfAnalyzer.getCallibrationTicks(chunks, "blink")
    vector_b = Y[:, blinks_timing[0]:blinks_timing[-1]]
    blinks_timing_corrected = remove_first_values_from_all_array_values(blinks_timing)
    print("blinks timing: ", blinks_timing)
    print("blinks timing corrected: ", blinks_timing_corrected)
    
    plotChunkData(vector_b, "Choose Blinking Component", blinks_timing_corrected)
    clearPlot()

    comp = int(input("Choose the best component for blinking, zero based: ")) 
    vector = Y[comp]
    
    #Sanity - check only this vector on blinking:
    plt.plot(vector[blinks_timing[0]:blinks_timing[-1]])
    #plot horizontal line that is the mean of the vector:
    plt.axhline(y=np.mean(vector), color='r', linestyle='-')
    #plot blinking times as red ticks:
    for t in blinks_timing_corrected:
        plt.axvline(x=t, color='r')
    plt.show()
    #clearPlot()

    return vector, chunks

# not in use for now
# def extract_blink_from_vector(vector):
#     """
#     Extracts one blink from the vector
#     Uses generic signal python functions to find the peaks of the blinks 
#     Chooses the first blink and returns it with its duration (using find_blink_duration function)
#     """
#     peaks, _ = signal.find_peaks(vector, height=0.5)
#     blinks_timing = find_blink_duration(vector, peaks)
#     start_time = blinks_timing[0][1]
#     end_time = blinks_timing[0][2]
#     blink = vector[start_time:end_time]
#     plt.figure(figsize=(9,5))
#     plt.rcParams['font.size'] = '16'
#     plt.xlabel("one blink", fontsize=16)
#     plt.plot(blink)
#     plt.show()
#     return vector[start_time:end_time]


def extract_main():
    # shoval, you can play with the windows values up to a point you smooth the signal enough but not too much.
    # too much = loosing valuable information.
    # 20ms ==> 50HZ

    sessions = GetSessions(data_path)
    print(sessions)
    session = sessions[0]

    vector, chunks = extract_blinking_component(data_path, session, A, 20, 100)
    # vector_mean = np.mean(vector)
    # for key in chunks.keys():
    #     chunk  =  chunks[key]
    #     plt.plot(vector[int(chunk.Start.Time) : int(chunk.End.Time)])
    #     plt.axhline(y=vector_mean, color='r', linestyle='-')
    #     plt.title(key)
    #     plt.show()
    #     clearPlot()

    return vector, chunks
