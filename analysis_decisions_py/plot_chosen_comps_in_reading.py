import os
from General.HelperFunctions import GetSessions, cleanSpace, reshape_num_channels, RemoveRowFromMatrix
import numpy as np
from scipy.stats import zscore
from General.init import *
from tqdm import tqdm
import re
from General.HelperFunctions import getPathsFromSessionFolder, remove_rows_from_matrix
from EDF.EdfAnalyzer import EdfAnalyzer
from ComponentAnalysis.ComponentExtractor import GetCorrugatorComponent, GetNoseComponent, GetSideEyeComponent, GetZygomaticusComponent
from General.HelperFunctions import get_review_data
from ComponentAnalysis.ComponentExtractor import open_w_if_exisis, load_special_components_if_exists, get_special_component_indexes, save_new_w
from General.CorrolationMethods import get_max_xcorr_1d, get_max_xcorr_pearson_1d, windowed_lagged_cross_correlation, dtai_dist_dtw_pearson_over_seconds, cross_correlation_max_pearson, cross_correlation_max_pearson_ndim
import pandas as pd
import matplotlib.pyplot as plt
from General.PlotFunctions import plotDataFromBothChunks, clearPlot
from General.tam.tam_correlation import tam_corr
from analysis_decisions_py.cleaning_signal import remove_emg_artifacts
from Hilas_component_classification.classifying_ica_components import wavelet_denoising, whiten, center
from ComponentAnalysis.ReviewComponents_batch2 import get_chosen_comps

default_rms_size_ms = 400
default_downsample_in_ms = 100 # 50 hz

SMILE = 0
ANGRY = 1

def get_vector_len(vec):
    if ( len(vec.shape) == 1):
        return vec.shape[0]
    return vec.shape[1]

def get_minimal_vecs_len(vec1, vec2):
    return min(get_vector_len(vec1), get_vector_len(vec2))

def load_first_reading_data(session, p, rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsample_window_ms)
    elecs_to_remove, is_dark = get_review_data(session, data_path)
    #elecs_to_remove = [[], []]
    Y = remove_rows_from_matrix(Y, elecs_to_remove[p])
    Y = zscore(Y, axis=1)

    #Y = wavelet_denoising(Y, freq, 'db4')
    #Y = zscore(Y, axis=1)
    #Y, _ = center(Y)
    #Y = whiten(Y) #no whiten again.


    w = open_w_if_exisis(session, p, dwt=False)
    
    if ( w is None):
        w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove[p]))
        #save_new_w(session, p, w)
    else:
        independentComponents = np.dot(w, Y)

    special_comps_index = load_special_components_if_exists(data_path, session, p)
    if (special_comps_index is None):
        special_comps_index = get_special_component_indexes(w, elecs_to_remove[p])

    #special_comps_index = get_chosen_comps(session, p)

    independentComponents = independentComponents[special_comps_index, :]
    independentComponents = remove_emg_artifacts(independentComponents)
    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsample_window_ms, freq)

    chunks = EdfAnalyzer.combine_callibration_ticks(chunks)
    #get first reading chunk:
    for key in chunks:
        action = key.split("_")[0].lower()

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
            


    smile_data = independentComponents[:, smile_s:smile_e]
    angry_data = independentComponents[:, angry_s:angry_e]
    blink_data = independentComponents[:, blink_s:blink_e]
    reading_data = independentComponents[:, reading_s:reading_e]
    return smile_data, angry_data, blink_data, reading_data

def plot(data_a, data_b, title):
    plotDataFromBothChunks(data_a, data_b, [], [], title, 4)
    clearPlot()

def main_plot_read():
    sessions_list = ["06122022_1215"] #GetSessions(data_path)
    for session in tqdm(sessions_list):
        smiles = [[],[]]
        angrys = [[],[]]
        blinks = [[],[]]
        readings = [[],[]]
        for p in [A,B]:
            smile_data, angry_data, blink_data, reading_data = load_first_reading_data(session, p)
            smiles[p] = smile_data
            angrys[p] = angry_data
            blinks[p] = blink_data
            readings[p] = reading_data
        
        plot(smiles[A], smiles[B], f"Smile {session}")
        plot(angrys[A], angrys[B], f"Angry {session}")
        plot(blinks[A], blinks[B], f"Blink {session}")
        plot(readings[A], readings[B], f"Reading {session}")