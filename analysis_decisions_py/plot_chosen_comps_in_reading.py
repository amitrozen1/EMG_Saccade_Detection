import os
from General.HelperFunctions import GetSessions, cleanSpace, reshape_num_channels, RemoveRowFromMatrix
import numpy as np
from General.init import *
from tqdm import tqdm
from General.HelperFunctions import getPathsFromSessionFolder, remove_rows_from_matrix
from EDF.EdfAnalyzer import EdfAnalyzer
from General.HelperFunctions import get_review_data
from ComponentAnalysis.ComponentExtractor import open_w_if_exisis, load_special_components_if_exists, get_special_component_indexes, save_new_w, GetComponentByElecrodesNumber
import pandas as pd
import matplotlib.pyplot as plt
from General.PlotFunctions import plotDataFromBothChunks, clearPlot
from analysis_decisions_py.cleaning_signal import remove_emg_artifacts, uniform_continuity_clean
from Hilas_component_classification.classifying_ica_components import wavelet_denoising, whiten, center
from ComponentAnalysis.ReviewComponents_batch2 import get_chosen_comps
from General.PlotFunctions import clearPlot
from scipy import signal
from scipy.stats import zscore

default_rms_size_ms = 50
default_downsample_in_ms = 5   # 50 hz

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
    #Y = zscore(Y, axis=1)

    Y = wavelet_denoising(Y, freq, 'db15')
    Y, _ = center(Y)

    #w, independentComponents = EdfAnalyzer.ICA_picard(Y)
    Y = whiten(Y) #no whiten again.

    w = open_w_if_exisis(session, p, dwt=True)
    w = None
    if ( w is None):
        w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove[p]))
        #save_new_w(session, p, w)
    else:
        independentComponents = np.matmul(w,Y)

    special_comps_index = load_special_components_if_exists(data_path, session, p, dwt=True)
    #special_comps_index = None
    if (special_comps_index is None):
        special_comps_index = get_special_component_indexes(w, elecs_to_remove[p])

    special_comps_index += [GetComponentByElecrodesNumber(w, elecs_to_remove[p], [2])]
    special_comps_index += [GetComponentByElecrodesNumber(w, elecs_to_remove[p], [5])]
    special_comps_index += [GetComponentByElecrodesNumber(w, elecs_to_remove[p], [4])]

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

    independentComponents = independentComponents[special_comps_index, :]
    #independentComponents = remove_emg_artifacts(independentComponents)
    #independentComponents = uniform_continuity_clean(independentComponents, 1*freq , 1, 2)
    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    #independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsample_window_ms, freq)
    hz = int(1000/downsample_window_ms)
    independentComponents = signal.resample(independentComponents, int(independentComponents.shape[1] / freq * hz), axis=1)
    #independentComponents = [calculate_waveform_length(independentComponents[i], downsample_window_ms, freq) for i in range(independentComponents.shape[0])]
    #independentComponents = zscore(independentComponents, axis=1)
    # if p == B:
    #     max_comp = np.max(independentComponents[[0,1,3],:], axis=0)
    # else:

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

def plot(data_a, data_b, title, extra_titles=None):
    plotDataFromBothChunks(data_a, data_b, [], [], title, data_a.shape[0], extra_titles=extra_titles)
    clearPlot()


import logging
def init_logger():
    logger = logging.getLogger("bad_noses")
    logger.setLevel("INFO")
    fh = logging.FileHandler(fr"logs\bad_noses.log")
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

#logger = init_logger()

# def log_bad_noses(smile_data, angry_data, session, p):
#     smile_corrs = []
#     angry_corrs = []
#     for i in range(1, len(smile_data)):
#         smile_corrs.append(np.corrcoef(smile_data[0], smile_data[i])[0,1])
#         angry_corrs.append(np.corrcoef(angry_data[0], angry_data[i])[0,1])
    
#     p = "A" if p == 0 else "B"
#     log = f"Session: {session}, P: {p}, Smile: {smile_corrs}, Angry: {angry_corrs}"
#     if (np.max(smile_corrs) > 0.5 or np.max(angry_corrs) > 0.5):
#         logger.log(logging.INFO, log)
#     else:
#         logger.log(logging.ERROR, log)

def get_excluded_sessions():
    return ["10052023_1430", "09052023_1215", "08052023_1430", "30112023_1200"]
    #first reading of 10052023_1430 - A is not good. all the rest is ok.

#08051430 - remove 8 near nose electrode.


def check_component_i(smile_data, angry_data, i, threshold=0.25):
    smile_i = smile_data[i]
    angry_i = angry_data[i]

    for j in range(0, len(smile_data)):
        if ( j == i):
            continue
        if (np.corrcoef(smile_i, smile_data[j])[0,1] > threshold):
            return True
        if (np.corrcoef(angry_i, angry_data[j])[0,1] > threshold):
            return True
    
    return False

def get_component_validity(smile_data, angry_data, threshold=0.25):
    return [check_component_i(smile_data, angry_data, i, threshold) for i in range(1, len(smile_data))]

def get_component_validity_title(smile_data, angry_data, threshold=0.25):
    bools = get_component_validity(smile_data, angry_data, threshold)
    return [f"OK_{i}" if bools[i] else f"BAD_{i}" for i in range(len(bools))]


def load_components(session, p, rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsample_window_ms)
    elecs_to_remove = [[], []]

    Y = wavelet_denoising(Y, freq, 'db15')
    Y, _ = center(Y)

    Y = whiten(Y)

    w = open_w_if_exisis(session, p, dwt=True)
    if ( w is None):
        w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove[p]))
        #save_new_w(session, p, w)
    else:
        independentComponents = np.matmul(w,Y)

    special_comps_index = load_special_components_if_exists(data_path, session, p, dwt=True)
    if (special_comps_index is None):
        special_comps_index = get_special_component_indexes(w, elecs_to_remove[p])

    special_comps_index += [GetComponentByElecrodesNumber(w, elecs_to_remove[p], [2])]
    special_comps_index += [GetComponentByElecrodesNumber(w, elecs_to_remove[p], [5])]
    special_comps_index += [GetComponentByElecrodesNumber(w, elecs_to_remove[p], [4])]

    independentComponents = independentComponents[special_comps_index, :]

    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    hz = int(1000/downsample_window_ms)
    independentComponents = signal.resample(independentComponents, int(independentComponents.shape[1] / freq * hz), axis=1)
    
    chunks = EdfAnalyzer.combine_callibration_ticks(chunks)
    start = int(chunks["smile"].Start.Time)
    end = int(chunks["smile"].End.Time)

    return independentComponents[:, start:end]

def pearson_vec_for_nose(session, p):
    independentComponents = load_components(session, p)
    nose = independentComponents[0]
    return [np.corrcoef(nose, independentComponents[i])[0,1] for i in range(1, len(independentComponents))]

def get_ok_sessions():
    if current_batch == 1:
        sessions =  GetSessions(data_path)
        sessions.remove("20122022_1415")
    else:
        #return ["10052023_1430", "09052023_1215", "08052023_1430"]
        excluded_list = ["25122023_1400", "25122023_1000", "21122023_1230", "18042023_1415", "25122023_1200", "30032023_1100", "30112023_1400", "22052023_1430"]
        extra = ["17052023_1430", "06122023_1000"]
        excluded_list += extra
        # can be improved "21122023_1500", "17052023_1430", "11122023_1400", "06122023_1000	"
        sessions = GetSessions(data_path)
        for session in excluded_list:
            if session in sessions:
                sessions.remove(session)
    return sessions

def main_nose_corrs():
    sessions_list = get_ok_sessions()
    df = pd.DataFrame(columns = ["particiapant", "Eye", "Zygomaticus", "Corrugator", "2_neg", "5_pos", "4_neg", "batch"])
    try:
        for session in tqdm(sessions_list):
            for p in [A,B]:
                corrs = np.arctan(pearson_vec_for_nose(session, p))
                df = df.append({"particiapant": f"{session}_{p}", "Eye": corrs[0], "Zygomaticus": corrs[1], "Corrugator": corrs[2], "2_neg": corrs[3], "5_pos": corrs[4], "4_neg": corrs[5], "batch": 2}, ignore_index=True)
    except Exception as e:
        print(f"Error in session: {session}, p: {p}")
        print(e)
    finally:
        df.to_csv(os.path.join(data_path, "Results_2024", "nose_corrs_batch2_onlySmiles.csv"), index=False)

    
def main_plot_read():
    sessions_list = GetSessions(data_path)[10:12]
    #sessions_list = get_excluded_sessions() #["30112023_1200", "30032023_1100", "20042023_1015"]
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
            #log_bad_noses(smile_data, angry_data, session, p)

            anger_comp = np.argmax(np.mean(angry_data, axis=1) - np.mean(smile_data, axis=1))+1
            smile_comp = np.argmax(np.mean(smile_data, axis=1) - np.mean(angry_data, axis=1))+1
            print(f"{session}_{p} Anger: {anger_comp}, Smile: {smile_comp} ")
            print(f"4: {np.max(angry_data[0]) - np.max(smile_data[0])}")
        
        extra_titles = [get_component_validity_title(smiles[A], angrys[A]), get_component_validity_title(smiles[B], angrys[B])]
        plot(smiles[A], smiles[B], f"Smile {session}", extra_titles)
        plot(angrys[A], angrys[B], f"Angry {session}", extra_titles)
        plot(blinks[A], blinks[B], f"Blink {session}", extra_titles)
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
