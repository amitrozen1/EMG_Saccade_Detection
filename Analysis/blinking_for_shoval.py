import os
import numpy as np
from scipy.stats import zscore
#there are some general parameters in General.init
from General.init import *
from General.HelperFunctions import getPathsFromSessionFolder
from EDF.EdfAnalyzer import EdfAnalyzer, butter_bandpass_filter
import matplotlib.pyplot as plt
from General.HelperFunctions import GetSessions
from ComponentAnalysis.ComponentExtractor import GetSideEyeComponent
from General.PlotFunctions import savePlot, clearPlot, plotChunkData

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
    clearPlot()

    return vector, chunks

def blink_peaks_detect(single_channel_signal):
    # using hillas alg to find the windows where there are blinks
    # gets a filtered signal as input
    # returns a list consisnting all peaks that were detected as blinks
    binary_blinks_list = []
    # todo - fill in the implementation
    # HILA USED time_stamps_df.loc['blink_trig', 'start'] time_stamps_df.loc['blink_trig', 'end']  
    return binary_blinks_list

# from HILA
def create_filter(sig, fft_order):
    """
        creates a filter of a pattern of a participant using FFT in order to recognize it in a signal
    :param sig: EMG signal of a participant
    :param fft_order: the requested number of Fourier coefficients to calculate for the filter
    :param time_stamps: a csv file that contains the triggers time stamps
    :return: a filter of a pattern
    """
    n = len(sig)
    fftsig = np.fft.fft(sig)
    np.put(fftsig, range(fft_order + 1, n), 0.0)
    win = np.fft.ifft(fftsig)
    win1 = np.real(np.fft.ifft(fftsig)) / np.sqrt(np.sum(np.square(win)))
    # plt.figure(figsize=(9,5))
    # plt.rcParams['font.size'] = '16'
    # plt.xlabel("fs*sec", fontsize=16)
    # plt.ylabel("Amplitude", fontsize=16)
    # plt.plot(win1)
    # plt.show()
    return win1

# from HILA
def convert_patten_to_spike_train(sig, filter, fs):
    """
        gets an EMG signal and a blinking pattern filter and converts the signal to a spike train of blinks
    :param sig: EMG signal of a participant
    :param filter: a blinking pattern filter
    :param fs: sampling frequency of the measuring device
    :param time_stamps: a csv file that contains the triggers time stamps
    :return: a spike train of blinks
    """
    N = len(sig)
    t = np.linspace(1 / fs, N, N)
    corr = signal.correlate(sig, filter, mode='same') / len(filter)
    peaks = signal.find_peaks(corr, height=0.01)
    spike_train = np.zeros_like(t)
    spike_train[peaks[0]] = 1
    # plt.figure(figsize=(8, 5))
    # plt.rcParams['font.size'] = '16'
    # plt.plot(t, np.real(corr))
    # plt.plot(peaks[0], peaks[1]['peak_heights'], '*')
    # plt.xlabel("fs*sec", fontsize=16)
    # plt.ylabel("cross-correlation", fontsize=16)
    # plt.show()
    # plt.close()
    # plt.show()
    # plt.close()
    # plt.figure()
    # plt.rcParams['font.size'] = '16'
    # plt.xlabel("fs*sec", fontsize=14)
    # plt.ylabel("occurrence of a spike", fontsize=14)
    # plt.stem(t, spike_train)
    # plt.show()
    # plt.close()
    return spike_train

def find_blink_duration(signal, binary_blinks_list):
    # for each blink find its begining and its ending
    # gets as input a list with the timing (binary, not duration but a specific moment) of the blinks in the signal and the signal
    window_length = 20
    blinks_duration = []
    for blink in binary_blinks_list:
        # where signal is the single channel signal - expecting a vector over time 
        minima = signal[blink]
        t_start = blink
        t_end = blink
        got_to_normal_flag = False
        i = 1
        while not got_to_normal_flag:
           new_minima = signal[blink - i*window_length]
           if new_minima < minima:
               minima = new_minima
               i += 1
           else:
               t_start = blink - i*window_length
               minima = signal[blink]
               got_to_normal_flag = True
        
        got_to_normal_flag = False
        while not got_to_normal_flag:
            new_minima = signal[blink + i*window_length]
            if new_minima < minima:
               minima = new_minima
               i += 1
            else:
               t_end = blink + i*window_length
               got_to_normal_flag = True
        
        blinks_duration.append([blink, t_start, t_end])

    return blinks_duration      

def rate_blinks(blinks_timing):
    shortest = 0
    longest = 0
    blinks_duration = []
    for blink in blinks_timing:
        length = blink[2] - blink[1]
        blinks_duration.append([blink, length])
        if length < shortest | shortest == 0:
            shortest = length
        if length > longest:
            longest = length
    rated_blinks = []
    for blink in blinks_duration:
        rate = (blink[1] - shortest)/(longest - shortest)
        rated_blinks.append([blink, rate])

    return rated_blinks

def blink_detection(single_channel_signal, fs):
    # fill the low, high 
    filtered_signal = butter_bandpass_filter(single_channel_signal, lowcut=, highcut=, fs)
    # find in a binary way the blinks
    binary_blinks_list = blink_peaks_detect(filtered_signal)
    # after finding all peaks of blinking find their duration
    blinks_timing = find_blink_duration(filtered_signal, binary_blinks_list)
    # for each blink rate it between 0-1, zero for the shortest, 1 for the longest
    rated_blinks = rate_blinks(blinks_timing)
    
    return rated_blinks

def main():
    # shoval, you can play with the windows values up to a point you smooth the signal enough but not too much.
    # too much = loosing valuable information.
    # 20ms ==> 50HZ

    sessions = GetSessions(data_path)
    session = sessions[0]

    vector, chunks = extract_blinking_component(data_path, session, A, 20, 100)
    vector_mean = np.mean(vector)
    for key in chunks.keys():
        chunk  =  chunks[key]
        plt.plot(vector[int(chunk.Start.Time) : int(chunk.End.Time)])
        plt.axhline(y=vector_mean, color='r', linestyle='-')
        plt.title(key)
        plt.show()
        clearPlot()