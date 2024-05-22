import os
from General.HelperFunctions import GetSessions, cleanSpace, reshape_num_channels, RemoveRowFromMatrix
import numpy as np
from scipy.stats import zscore
from General.init import *
from General.HelperFunctions import getPathsFromSessionFolder, remove_rows_from_matrix
from EDF.EdfAnalyzer import EdfAnalyzer
from General.HelperFunctions import get_review_data
from ComponentAnalysis.ComponentExtractor import open_w_if_exisis, load_special_components_if_exists, get_special_component_indexes, save_new_w
from General.PlotFunctions import plotDataFromBothChunks, clearPlot
from analysis_decisions_py.validation_of_correlation_via_smile_anger import compute_single_corrolations
from analysis_decisions_py.cleaning_signal import remove_emg_artifacts

rms_size_ms = 400
downsampleWindowInMs = 50 # 50 hz

SMILE = 0
ANGRY = 1

def get_vector_len(vec):
    if ( len(vec.shape) == 1):
        return vec.shape[0]
    return vec.shape[1]

def get_minimal_vecs_len(vec1, vec2):
    return min(get_vector_len(vec1), get_vector_len(vec2))

def load_signle_data_smile_and_angry(session, p):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsampleWindowInMs)
    elecs_to_remove, is_dark = get_review_data(session, data_path)
    
    Y = remove_rows_from_matrix(Y, elecs_to_remove[p])
    Y = zscore(Y, axis=1)

    w = open_w_if_exisis(session, p)
    
    if ( w is None):
        w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove[p]))
        save_new_w(session, p, w)
    else:
        independentComponents = np.dot(w, Y)

    special_comps_index = load_special_components_if_exists(data_path, session, p)
    if (special_comps_index is None):
        special_comps_index = get_special_component_indexes(w, elecs_to_remove[p])

    independentComponents = independentComponents[special_comps_index, :]
    independentComponents = remove_emg_artifacts(independentComponents)
    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsampleWindowInMs, freq)

    smile_s = int(chunks["smile__0"].Start.Time)
    smile_e = int(chunks["smile__2"].End.Time)

    angry_s = int(chunks["angry__0"].Start.Time)
    angry_e = int(chunks["angry__2"].End.Time)

    smile_data = independentComponents[:, smile_s:smile_e]
    angry_data = independentComponents[:, angry_s:angry_e]

    return [smile_data, angry_data]

def load_general_dyad_data(sessions, p_s):
    vectors = []
    for i in range(2):
        chunk_data = load_signle_data_smile_and_angry(sessions[i], p_s[i])
        vectors.append(chunk_data)
    return vectors

def load_real_dyad_data(session):
    return load_general_dyad_data([session, session], [A,B])

def plot_couples_graph_and_xcorr(session):
    vectors = load_real_dyad_data(session)
    window_size = 1000
    lag = window_size*2

    smile_corrs = compute_single_corrolations([vectors[A][SMILE], vectors[B][SMILE]], "smile", window_size, lag)
    print(smile_corrs)
    plotDataFromBothChunks(vectors[A][SMILE], vectors[B][SMILE], [], [], "smile", 4)
    clearPlot()
    angry_corrs = compute_single_corrolations([vectors[A][ANGRY], vectors[B][ANGRY]], "angry", window_size, lag)
    print(angry_corrs)
    plotDataFromBothChunks(vectors[A][ANGRY], vectors[B][ANGRY], [], [], "angry", 4)
    clearPlot()


