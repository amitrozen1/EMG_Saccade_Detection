import os
from General.HelperFunctions import GetSessions, cleanSpace, reshape_num_channels, RemoveRowFromMatrix
import numpy as np
from General.init import *
from tqdm import tqdm
from General.HelperFunctions import getPathsFromSessionFolder, remove_rows_from_matrix
from EDF.EdfAnalyzer import EdfAnalyzer
from General.HelperFunctions import get_review_data
from ComponentAnalysis.ComponentExtractor import open_w_if_exisis, load_special_components_if_exists, get_special_component_indexes, save_new_w
import pandas as pd
import matplotlib.pyplot as plt
from General.PlotFunctions import plotDataFromBothChunks, clearPlot
from analysis_decisions_py.cleaning_signal import remove_emg_artifacts, uniform_continuity_clean
from Hilas_component_classification.classifying_ica_components import wavelet_denoising, whiten, center
from ComponentAnalysis.ReviewComponents_batch2 import get_chosen_comps
from General.PlotFunctions import clearPlot
from scipy import signal
from scipy.stats import zscore

default_rms_size_ms = 100
default_downsample_in_ms = 50   # 50 hz

SMILE = 0
ANGRY = 1

def get_vector_len(vec):
    if ( len(vec.shape) == 1):
        return vec.shape[0]
    return vec.shape[1]

def get_minimal_vecs_len(vec1, vec2):
    return min(get_vector_len(vec1), get_vector_len(vec2))

def get_folder(session,p):
    os.makedirs(session, exist_ok=True)
    return fr".\{session}_{p}"

def get_special_component_names():
    return ["Nose", "Eye", "Zygomaticus", "Corrugator"]

def fix_signal(independentComponents, rms_size_ms, downsample_window_ms, freq):
    independentComponents = remove_emg_artifacts(independentComponents, 3)
    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsample_window_ms, freq)
    return independentComponents

def load_first_reading_data(session, p, rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsample_window_ms)
    #elecs_to_remove, is_dark = get_review_data(session, data_path)
    elecs_to_remove = [[], []]
    #Y = remove_rows_from_matrix(Y, elecs_to_remove[p])
    # Y = zscore(Y, axis=1)

    Y = wavelet_denoising(Y, freq, 'db15')
    Y, _ = center(Y)
    Y = zscore(Y, axis=1)

    #w, independentComponents = EdfAnalyzer.ICA_picard(Y)
    Y = whiten(Y) #no whiten again.

    w = open_w_if_exisis(session, p, dwt=True)
    # w = None
    if ( w is None):
        w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove[p]))
        #save_new_w(session, p, w)
    else:
        independentComponents = np.matmul(w,Y)

    special_comps_index = load_special_components_if_exists(data_path, session, p, dwt=True)
    if (special_comps_index is None):
        special_comps_index = get_special_component_indexes(w, elecs_to_remove[p])

    print(f"special_comps_index: {special_comps_index}")
    #special_comps_index = get_chosen_comps(session, p)
    #special_comps_index = get_special_component_indexes(w, elecs_to_remove[p])

    # default_heatmap_with_channels_removal(w, elecs_to_remove[p], selected=special_comps_index, 
    #                                 save_at=get_folder(session,p),
    #                                 save_different_figures=True,
    #                                 extra_headers=get_special_component_names())
    # clearPlot()
    
    # default_heatmap_with_channels_removal(w, elecs_to_remove[p],
    #                         save_at=get_folder(session,p), 
    #                         save_different_figures=False,
    #                         save_whole_heatmap=True)
    # clearPlot()

    #independentComponents = independentComponents[special_comps_index, :]
    independentComponents = remove_emg_artifacts(independentComponents)
    #independentComponents = uniform_continuity_clean(independentComponents, 1*freq , 1, 2)
    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    #independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsample_window_ms, freq)
    independentComponents = signal.resample(independentComponents, int(independentComponents.shape[1] / freq * 20), axis=1)
    #independentComponents = [calculate_waveform_length(independentComponents[i], downsample_window_ms, freq) for i in range(independentComponents.shape[0])]
    #independentComponents = zscore(independentComponents, axis=1)
    # if p == B:
    #     max_comp = np.max(independentComponents[[0,1,3],:], axis=0)
    # else:
    max_comp = np.max(independentComponents, axis=0)
    independentComponents = np.vstack((independentComponents, max_comp))

    chunks = EdfAnalyzer.combine_callibration_ticks(chunks)
    #get first reading chunk:
    for key in chunks:
        action = key.split("_")[0].lower()
        #action = key.lower()

        if "smile" in action:
            smile_s = int(chunks[key].Start.Time)
            smile_e = int(chunks[key].End.Time)
        
        if "angry" in action:
            angry_s = int(chunks[key].Start.Time)
            angry_e = int(chunks[key].End.Time)
        
        if "blink" in action:
            blink_s = int(chunks[key].Start.Time)
            blink_e = int(chunks[key].End.Time)

        if "reading" in action:
            reading_s = int(chunks[key].Start.Time)
            reading_e = int(chunks[key].End.Time)
            

    #independentComponents = zscore(independentComponents, axis=1)

    smile_data = independentComponents[:, smile_s:smile_e]
    
    angry_data = independentComponents[:, angry_s:angry_e]
    blink_data = independentComponents[:, blink_s:blink_e]
    reading_data = independentComponents[:, reading_s:reading_e]
    return smile_data, angry_data, blink_data, reading_data

def plot(data_a, data_b, title):
    plotDataFromBothChunks(data_a, data_b, [], [], title, data_a.shape[0])
    clearPlot()


def calculate_zcr(emg_signal, window_size_ms, sampling_rate):
    """
    Calculate the Zero Crossing Rate (ZCR) of an EMG signal with a given window size.

    Parameters:
    emg_signal (np.array): The EMG signal as a NumPy array.
    window_size_ms (int): The window size in milliseconds.
    sampling_rate (int): The sampling rate of the signal in Hz.

    Returns:
    np.array: An array of ZCR values for each window.
    """
    # Convert window size from milliseconds to number of samples
    window_size = int((window_size_ms / 1000) * sampling_rate)
    
    # Initialize the list to store ZCR values
    zcr_values = []
    
    # Calculate ZCR for each window
    for start in range(0, len(emg_signal) - window_size + 1, window_size):
        window = emg_signal[start:start + window_size]
        # Count zero crossings
        zero_crossings = np.sum(np.abs(np.diff(np.sign(window)))) / 2
        zcr_values.append(zero_crossings / window_size)  # Normalize by window size

    return np.array(zcr_values)


def calculate_waveform_length(emg_signal, window_size_ms, sampling_rate):
    """
    Calculate the Waveform Length (WL) of an EMG signal within specified window sizes.

    Parameters:
    emg_signal (np.array): The EMG signal as a NumPy array.
    window_size_ms (int): The window size in milliseconds.
    sampling_rate (int): The sampling rate of the signal in Hz.

    Returns:
    np.array: An array of Waveform Length values for each window.
    """
    # Convert window size from milliseconds to number of samples
    window_size = int((window_size_ms / 1000) * sampling_rate)
    
    # Initialize the list to store Waveform Length values
    waveform_lengths = []
    
    # Calculate Waveform Length for each window
    for start in range(0, len(emg_signal) - window_size + 1, window_size):
        window = emg_signal[start:start + window_size]
        # Calculate the waveform length in the window
        wl = np.sum(np.abs(np.diff(window)))
        waveform_lengths.append(wl)

    return np.array(waveform_lengths)


def main_plot_read():
    sessions_list = GetSessions(data_path)[1:]
    for session in tqdm(sessions_list):
        print(session)
        smiles =   [[],[]]
        angrys =   [[],[]]
        blinks =   [[],[]]
        readings = [[],[]]
        for p in [A,B]:
            smile_data, angry_data, blink_data, reading_data = load_first_reading_data(session, p)
            smiles[p] = smile_data
            angrys[p] = angry_data
            blinks[p] = blink_data
            readings[p] = reading_data

            anger_comp = np.argmax(np.mean(angry_data, axis=1) - np.mean(smile_data, axis=1))+1
            smile_comp = np.argmax(np.mean(smile_data, axis=1) - np.mean(angry_data, axis=1))+1
            print(f"{session}_{p} Anger: {anger_comp}, Smile: {smile_comp} ")
            print(f"4: {np.max(angry_data[0]) - np.max(smile_data[0])}")
        
        plot(smiles[A], smiles[B], f"Smile {session}")
        plot(angrys[A], angrys[B], f"Angry {session}")
        # plot(blinks[A], blinks[B], f"Blink {session}")
        plot(readings[A], readings[B], f"Reading {session}")

def main_check_pearson_to_comp():
    sessions_list = GetSessions(data_path)[0:2]
    corrs = [[],[],[],[]]
    for session in tqdm(sessions_list):
        for p in [A,B]:
            smile_data, angry_data, blink_data, reading_data = load_first_reading_data(session, p)
            for i in range(4):
                corrs[i].append(np.corrcoef(reading_data[-1], reading_data[i])[0,1])
    
    headers = ["nose", "eye", "zygomaticus", "corrugator"]
    #save to csv
    df = pd.DataFrame(corrs, columns = headers)
    df.to_csv("pearson_each_to_max_comp.csv", index=False)
