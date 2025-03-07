import os
from General.HelperFunctions import GetSessions, cleanSpace
import numpy as np
from General.init import *
from tqdm import tqdm
from General.HelperFunctions import getPathsFromSessionFolder, remove_rows_from_matrix
from EDF.EdfAnalyzer import EdfAnalyzer
from ComponentAnalysis.ComponentExtractor import open_w_if_exisis, open_whiten_if_exisis, load_special_components_if_exists, get_special_component_indexes, GetLipCornerDownPuller, GetComponentByElecrodesNumber
from scipy import signal
from ComponentAnalysis.ReviewComponents_batch2 import GetSessionsBatch2, open_w_if_exists_batch2, get_bad_electrode, get_chosen_comps
from Hilas_component_classification.classifying_ica_components import wavelet_denoising, whiten, center
from General.CorrolationMethods import get_max_xcorr_1d, get_max_xcorr_pearson_1d, windowed_lagged_cross_correlation
import pandas as pd
from ComponentAnalysis.ComponentExtractor import get_special_component_names

def get_vector_len(vec):
    if ( len(vec.shape) == 1):
        return vec.shape[0]
    return vec.shape[1]

def get_minimal_vecs_len(vec1, vec2):
    return min(get_vector_len(vec1), get_vector_len(vec2))

def whose_reading(chunks, chunk_key):
    calibration_phases = [-1]*9
    reading_order = [A, A, B, B, A, A, B, B]
    reading_order = calibration_phases + reading_order
    ind = list(chunks.keys()).index(chunk_key)
    return reading_order[ind]

def load_dyad_data(session, chunk_key, downsampleWindowInMs, rms_size_ms):
    vectors = [[],[]]
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    for p in [A,B]:
        Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], 1)
        
        chunk = chunks[chunk_key]
        start = int(chunk.Start.Time)
        end = int(chunk.End.Time)

        Y = Y[:, start:end]
        
        Y = wavelet_denoising(Y, freq, 'db15')
        Y, _ = center(Y)

        whiten_mat = open_whiten_if_exisis(session, p, dwt=True)
        if (whiten_mat is None):
            Y = whiten(Y)
        else:
            Y = np.dot(whiten_mat, Y)

        w = open_w_if_exisis(session, p, dwt=True)
        
        if ( w is None):
            w, independentComponents = EdfAnalyzer.ICA(Y, 16, whiten=False)
        else:
            independentComponents = np.matmul(w,Y)

        independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
        hz = int(1000/downsampleWindowInMs)
        independentComponents = signal.resample(independentComponents, int(independentComponents.shape[1] / freq * hz), axis=1)

        #select components:
        special_comps_index = load_special_components_if_exists(data_path, session, p, dwt=True)
        if (special_comps_index is None):
            special_comps_index = get_special_component_indexes(w, [])
        
        #special_comps_index += get_neg_pos_indexes(independentComponents, chunks)
        special_comps_index += [GetComponentByElecrodesNumber(w, [], [2])]
        special_comps_index += [GetComponentByElecrodesNumber(w, [], [5])]
        special_comps_index +=  [GetLipCornerDownPuller(w, [])]

        independentComponents = independentComponents[special_comps_index, :]

        if "Reading" in chunk_key:
            if whose_reading(chunks, chunk_key) == p:
                vectors[1] = independentComponents
            else:
                vectors[0] = independentComponents
        else:
            vectors[p] = independentComponents
    
    return vectors


def compute_single_corrolations(vectors, downsampleWindowInMs, window_size_in_ms, maxium_lag_in_ms, chunk_key):
    max_len = get_minimal_vecs_len(vectors[0][0], vectors[1][0])
    maxium_lag_in_ms = int(maxium_lag_in_ms)
    window_size_in_ms = int(window_size_in_ms)

    numOfPixelsForXCorrWindow = window_size_in_ms//downsampleWindowInMs
    numOfPixelsForXCorrStep = 500//downsampleWindowInMs

    direction = "one-direction" if "Reading" in chunk_key else "bi-direction"

    max_corrs = [[] for i in range(7)]
    for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow
            for comp in range(7):
                #Listener then reader.
                max_corr, max_ind = get_max_xcorr_pearson_1d(vectors[0][comp], vectors[1][comp], maxium_lag_in_ms,
                                                              downsampleWindowInMs, s, e, direction)
                max_corrs[comp].append(max_corr)
    return [np.mean(comp) for comp in max_corrs]


def compute_corr_and_psuedo_corr_in_group(session_list, chunk_key, downsampleWindowInMs, windows, lags, rms_size_ms):
    
    # real_corrs = {}
    # pseudo_corrs = {}
    # for window in windows:
    #     real_corrs[window] = {}
    #     pseudo_corrs[window] = {}
    #     for lag in lags:
    #         real_corrs[window][lag] = []
    #         pseudo_corrs[window][lag] = []

    loaded_listener_data = []
    loaded_reader_data = []

    print("load")
    for session in tqdm(session_list):
        try:
            vectors = load_dyad_data(session, chunk_key, downsampleWindowInMs, rms_size_ms)
            loaded_listener_data.append(vectors[A])
            loaded_reader_data.append(vectors[B])
        except Exception as e:
            print(f"Error in session {session}: {e}")
    

    corrs = []

    for window in windows:
        for lag in lags:
            print(f"real window: {window}, lag: {lag}")
            for i in range(len(session_list)):
                try:
                    corrs.append(compute_single_corrolations([loaded_listener_data[i], loaded_reader_data[i]], downsampleWindowInMs, window, lag, chunk_key))
                except Exception as e:
                    print(f"Error in session {session_list[i]}: {e}")
            df = pd.DataFrame(corrs)
            df.columns = get_special_component_names(True)            
            df.to_csv(fr"C:\Liron\neuroscience\PHD\First year experiment - EMG\Paper\Review\surrogate\real_corrs_window_{window}_maxlag_{lag}.csv", index=False)
            corrs = []

    corrs = []
    loaded_reader_data = [loaded_reader_data[-1]] + loaded_reader_data[:-1]
    for window in windows:
        for lag in lags:
            for i in tqdm(range(len(session_list))):
                print(f"pseudo window: {window}, lag: {lag}")
                try:
                    corrs.append(compute_single_corrolations([loaded_listener_data[i], loaded_reader_data[i]], downsampleWindowInMs, window, lag, chunk_key))
                except Exception as e:
                    print(f"Error in session {session_list[i]}: {e}")
            df = pd.DataFrame(corrs)
            df.columns = get_special_component_names(True)            
            df.to_csv(fr"C:\Liron\neuroscience\PHD\First year experiment - EMG\Paper\Review\surrogate\pseudo_corrs_window_{window}_maxlag_{lag}.csv", index=False)
            corrs = []
    
    del loaded_listener_data
    del loaded_reader_data

    return True

def main(chunk_key = "Listening_12.ogg_2"):
    session_list = GetSessions(data_path)
    downsampleWindowInMs = 5
    rms_size_ms = 50
    
    windows = [1000, 2000, 3000, 4000, 5000]
    lags = [1000, 2000, 3000, 4000, 5000]

    compute_corr_and_psuedo_corr_in_group(session_list, chunk_key, downsampleWindowInMs, windows, lags, rms_size_ms)

def gen_heatmap(base_path = fr"C:\Liron\neuroscience\PHD\First year experiment - EMG\Paper\Review\surrogate", chunk_key = "Listening_12.ogg_2"):
    windows = [1000, 2000, 3000, 4000, 5000]
    lags = [1000, 2000, 3000, 4000, 5000]

    base_path = os.path.join(base_path, chunk_key)

    for comp in get_special_component_names(True):
        heatmap = []

        for window in windows:
            t_values_in_window = []
            for lag in lags:
                real_df = pd.read_csv(os.path.join(base_path, fr"real_corrs_window_{window}_maxlag_{lag}.csv"))
                pseudo_df = pd.read_csv(os.path.join(base_path, fr"pseudo_corrs_window_{window}_maxlag_{lag}.csv"))
                #compute t-value per component of real > pseudo
                from scipy.stats import ttest_ind
                #for comp in range(7):
                # #   t_values.append(ttest_ind(real_df.iloc[:,comp], pseudo_df.iloc[:,comp], alternative='greater')[0])
                # for comp in ["Nose", "Eye"]:
                #     t_values.append(ttest_ind(real_df[comp], pseudo_df[comp], alternative='greater')[0])
                # t_values_in_window.append(np.mean(t_values))
                t_values_in_window.append(ttest_ind(real_df[comp], pseudo_df[comp], alternative='greater')[0])
            heatmap.append(t_values_in_window)
            t_values_in_window = []
        
        #plot heatmap
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.heatmap(heatmap, annot=True)
        #titles:
        plt.title("T-values of real > pseudo for component: " + comp)
        plt.xlabel("Lag")
        plt.ylabel("Window")
        #tickes:
        plt.xticks(np.arange(len(lags)), lags)
        plt.yticks(np.arange(len(windows)), windows)
        plt.show()