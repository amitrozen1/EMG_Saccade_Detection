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
from General.CorrolationMethods import get_max_xcorr_1d, get_max_xcorr_pearson_1d, windowed_lagged_cross_correlation


rms_size_ms = 100
downsampleWindowInMs = 20


def main_corrolation(current_session, independentComponents, chunks, windowForCorrInMs, numOfPixelsForXCorrStep):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]
    numOfPixelsForXCorrWindow = windowForCorrInMs//downsampleWindowInMs
    for key in chunks[0]:
        chunk_data = [[],[]]
        action = key.split("_")[0]

        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]

        mat1 = chunk_data[A][0:4]
        mat2 = chunk_data[B][0:4]

        temporal_max1 = chunk_data[A][4]
        temporal_max2 = chunk_data[B][4]
        temporal_correlations = []
        temporal_lags = []

        chosen_max1 = chunk_data[A][5]
        chosen_max2 = chunk_data[B][5]
        chosen_correlations = []
        chosen_lags = []

        max_lag_ms = windowForCorrInMs
        #jumps = numOfPixelsForXCorrWindow
        max_len = min(mat1.shape[1], mat2.shape[1])
        corolations = []
        lags = []
        current_reader = A
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow # todo: run again.
            #numOfPixelsForXCorrWindow
            if (action.lower() == "reading" and current_reader == A):
                max_temporal_corr, max_temporal_ind = get_max_xcorr_1d(temporal_max2, temporal_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
                max_chosen_corr, max_chosen_ind = get_max_xcorr_1d(chosen_max2, chosen_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
            else:    
                max_temporal_corr, max_temporal_ind = get_max_xcorr_1d(temporal_max1, temporal_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
                max_chosen_corr, max_chosen_ind = get_max_xcorr_1d(chosen_max1, chosen_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
    return results

def get_vector_len(vec):
    if ( len(vec.shape) == 1):
        return vec.shape[0]
    return vec.shape[1]

def get_minimal_vecs_len(vec1, vec2):
    return min(get_vector_len(vec1), get_vector_len(vec2))

def extract_the_mean_peak_value_of_specific_story_and_participant(sessions, p_s, chunk_key, window_size_in_ms, maxium_lag_in_ms):
    vectors = []
    for i in range(2):
        session = sessions[i]
        p = p_s[i]

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

        independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
        independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsampleWindowInMs, freq)

        special_comps_index = load_special_components_if_exists(data_path, session, p)
        if (special_comps_index is None):
            special_comps_index = get_special_component_indexes(w, elecs_to_remove[p])

        #total_max = np.max(independentComponents, axis=0)
        independentComponents = independentComponents[special_comps_index, :]
        #special_max = np.max(independentComponents, axis=0)

        chunk = chunks[chunk_key]
        chunk_start = int(chunk.Start.Time)
        chunk_end = int(chunk.End.Time)
        chunk_data = independentComponents[:, chunk_start:chunk_end]
        vectors.append(chunk_data)
    
    #max_lag = maxium_lag_in_ms//downsampleWindowInMs
    max_len = get_minimal_vecs_len(vectors[0][0], vectors[1][0])
    numOfPixelsForXCorrWindow = window_size_in_ms//downsampleWindowInMs
    numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow//2
    max_corrs = [[] for i in range(4)]
    for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow
            for comp in range(4):
                max_corr, max_ind = get_max_xcorr_1d(vectors[0][comp], vectors[1][comp], maxium_lag_in_ms, downsampleWindowInMs, s, e, chunk_key.split("_")[0])
                max_corrs[comp].append(max_corr)

    #return mean and ignore nan values:
    return [np.mean(comp) for comp in max_corrs]

window_sizes = np.arange(1,6.1)*1000
max_lags = np.arange(1,6.1)*1000
num_dyads = 15
REAL_DYADS = 0
PSEUDO_DYADS = 1
#allocate a matrix the size of len(windoe_sizes) x len(max_lags) X 4 X 2 X 15

window_lag_correlation_matrix = np.zeros((len(window_sizes), len(max_lags), 4, 2 , num_dyads))


def load_signle_data(session, p, chunk_key):
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

    #total_max = np.max(independentComponents, axis=0)
    independentComponents = independentComponents[special_comps_index, :]
    #special_max = np.max(independentComponents, axis=0)

    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsampleWindowInMs, freq)

    chunk = chunks[chunk_key]
    chunk_start = int(chunk.Start.Time)
    chunk_end = int(chunk.End.Time)
    chunk_data = independentComponents[:, chunk_start:chunk_end]

    return chunk_data

def load_dyad_data(sessions, p_s, chunk_key):
    vectors = []
    for i in range(2):
        chunk_data = load_signle_data(sessions[i], p_s[i], chunk_key)
        vectors.append(chunk_data)

    return vectors

def compute_single_corrolations(vectors, chunk_key, window_size_in_ms, maxium_lag_in_ms):
    max_len = get_minimal_vecs_len(vectors[0][0], vectors[1][0])
    maxium_lag_in_ms = int(maxium_lag_in_ms)
    window_size_in_ms = int(window_size_in_ms)

    numOfPixelsForXCorrWindow = window_size_in_ms//downsampleWindowInMs
    numOfPixelsForXCorrStep = 500//downsampleWindowInMs
    tMax = maxium_lag_in_ms//downsampleWindowInMs
    tInc = numOfPixelsForXCorrWindow//10 
    if (tInc == 0):
        tInc = 1
    max_corrs = [[] for i in range(4)]
    for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow
            for comp in range(4):
                max_corr, max_ind = get_max_xcorr_pearson_1d(vectors[0][comp], vectors[1][comp], maxium_lag_in_ms,
                                                              downsampleWindowInMs, s, e, chunk_key.split("_")[0])
                max_corrs[comp].append(max_corr)
    return [np.mean(comp) for comp in max_corrs]
    # max_corrs = []
    # for comp in range(4):
    #     max_corr = windowed_lagged_cross_correlation(vectors[0][comp], vectors[1][comp],
    #                                                 numOfPixelsForXCorrWindow,
    #                                                 numOfPixelsForXCorrStep, 
    #                                                 tMax,tInc)
    #     max_corrs.append(max_corr)
    # return max_corrs
    
def compute_corrolations_for_dyad_from_vectors(vectors, dyad_index, dyad_type, chunk_key):
    for i, window_size in enumerate(window_sizes):
        for j, max_lag in enumerate(max_lags):
            corrolations = compute_single_corrolations(vectors, chunk_key, window_size, max_lag)
            window_lag_correlation_matrix[i, j, :, dyad_type, dyad_index] = corrolations

def compute_corrolations_for_dyad(sessions, p_s, chunk_key, dyad_index):
    dyad_type = REAL_DYADS if sessions[0] == sessions[1] else PSEUDO_DYADS
    vectors = load_dyad_data(sessions, p_s, chunk_key)
    compute_corrolations_for_dyad_from_vectors(vectors, dyad_index, dyad_type, chunk_key)

def compute_corrolations_for_all_dyads(sessions_list, chunk_key):
    #real_dyads:
    print("Computing real dyads")
    for i, session in enumerate(sessions_list[0:num_dyads]):
        print("Computing real dyad: ", i)
        compute_corrolations_for_dyad([session, session], [A,B], chunk_key, i)
    
    #pseudo_dyads:
    print("Computing pseudo dyads")
    sessions_pool_1 = sessions_list[0:num_dyads]
    #cyclic shift session_pool_1:
    sessions_pool_2 = sessions_list[0:num_dyads]
    sessions_pool_2.append(sessions_pool_2.pop(0))
    
    for i in tqdm(range(len(sessions_pool_1))):
        session_1 = sessions_pool_1[i]
        session_2 = sessions_pool_2[i]
        compute_corrolations_for_dyad([session_1, session_2], [A,B], chunk_key, i)


def one_dyad_versus_many(session, p, sessions_list, chunk_key):
    vectors_real = load_dyad_data([session, session], [A,B], chunk_key)
    compute_corrolations_for_dyad_from_vectors(vectors_real, 0, REAL_DYADS, chunk_key)

    sessions_list.pop(sessions_list.index(session))

    for i, session in enumerate(sessions_list[0:num_dyads]):
        print("Computing pseudo dyad: ", i)
        vectors_pseudo = [vectors_real[p], load_signle_data(session, A, chunk_key)]
        compute_corrolations_for_dyad_from_vectors(vectors_pseudo, i, PSEUDO_DYADS, chunk_key)




def main_one_dyad():
    sessions = GetSessions(data_path)
    # one_dyad_versus_many(sessions[1], A, sessions, "Listening_12.ogg_2")
    # np.save(f"window_lag_correlation_matrix_one_dyad_{sessions[1]}.npy", window_lag_correlation_matrix)
    window_lag_correlation_matrix = np.load("window_lag_correlation_matrix_one_dyad_06122022_1215.npy")

    from scipy.stats import ttest_1samp

    window_lag_correlation_matrix_ttest = np.zeros((len(window_sizes), len(max_lags)))

    for j in range(len(window_sizes)):
        for k in range(len(max_lags)):
            real_values = window_lag_correlation_matrix[j, k, :, REAL_DYADS, 0]
            real_values = np.mean(real_values)
            pseudo_values = window_lag_correlation_matrix[j, k, :, PSEUDO_DYADS, :]
            pseudo_values = np.mean(pseudo_values, axis = 0)
            #fisher z transform:
            real_value = np.arctanh(real_values)
            pseudo_values = np.arctanh(pseudo_values)
            #t-test:
            t, p = ttest_1samp(pseudo_values, real_value)
            #print("Component: ", i, " Window size: ", window_sizes[j], " Max lag: ", max_lags[k], " t-value: ", t, " p-value: ", p)
            window_lag_correlation_matrix_ttest[j, k] = t
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(window_lag_correlation_matrix_ttest, xticklabels=max_lags, yticklabels=window_sizes)
    plt.show()
    plt.cla()

def main():
    #first stage.
    # sessions_list = GetSessions(data_path)
    # compute_corrolations_for_all_dyads(sessions_list, "Reading_2.png_4")
    # np.save("window_lag_correlation_matrix_step05_50hz_read.npy", window_lag_correlation_matrix)

    #second stage.
    window_lag_correlation_matrix = np.load("window_lag_correlation_matrix_new_pearson_corr_step05_50hz_read.npy")
    from scipy.stats import ttest_ind

    window_lag_correlation_matrix_ttest = np.zeros((len(window_sizes), len(max_lags),4))

    for i in range(4):
        for j in range(len(window_sizes)):
            for k in range(len(max_lags)):
                real_values = window_lag_correlation_matrix[j, k, i, REAL_DYADS, :]
                pseudo_values = window_lag_correlation_matrix[j, k, i, PSEUDO_DYADS, :]
                #fisher z transform:
                real_values = np.arctanh(real_values)
                pseudo_values = np.arctanh(pseudo_values)
                #t-test:
                t, p = ttest_ind(real_values, pseudo_values)
                print("Component: ", i, " Window size: ", window_sizes[j], " Max lag: ", max_lags[k], " t-value: ", t, " p-value: ", p)
                window_lag_correlation_matrix_ttest[j, k, i] = t
    
    #draw a heatmap for each one of the 4 components:
    import seaborn as sns
    import matplotlib.pyplot as plt
    for i in range(4):
        plt.figure()
        sns.heatmap(window_lag_correlation_matrix_ttest[:,:,i], xticklabels=max_lags, yticklabels=window_sizes)
        plt.title("Component: " + str(i))
        plt.show()
        plt.cla()

    # chosen_session = sessions_list.pop(15)
    # real_value = extract_the_mean_peak_value_of_specific_story_and_participant([chosen_session, chosen_session], [A,B], "Listening_12.ogg_2", 2000, 2000)
    # print(real_value)

    # computed_values = [[] for i in range(4)]
    # for session in tqdm(sessions_list[0:15]):
    #     compute_value = extract_the_mean_peak_value_of_specific_story_and_participant([chosen_session, session], [A,B], "Listening_16.ogg_4", 3000, 3000)
    #     for i in range(4):
    #         computed_values[i].append(compute_value[i])
    #     #computed_values.append(compute_value)

    # print(computed_values)

    # print("Real values are: ")
    # print(real_value)
    # print("Computed values means are: ")
    # print(np.mean(computed_values, axis=1))
    
    # print("TTEST")
    # #t-test real value should be bigger:
    # from scipy.stats import ttest_1samp
    # for i in range(4):
    #     print(i)
    #     t, p = ttest_1samp(computed_values[i], real_value[i])
    #     print("t-value: ", -1*t) # we want to know if the real value is bigger than the computed value.
    #     # negative t-value means that the real value is bigger than the computed value.

    # #ttest of one value against an array:
    
    # #todo: make a functoin that already compute data over all windows and lags once loaded a couple and then saves that.
    



#dict_keys(['smile__0', 'smile__1', 'smile__2', 'angry__0', 'angry__1', 'angry__2', 'blink__0', 'blink__1', 'blink__2', 'Reading_2.png_4', 'Reading_4.png_4', 'Reading_7.png_3', 'Reading_15.png_3', 'Reading_9.png_2', 'Reading_13.png_2', 'Reading_14.png_1', 'Reading_1.png_1', 'Listening_12.ogg_2', 'Listening_6.ogg_2', 'Listening_3.ogg_3', 'Listening_11.ogg_3', 'Listening_5.ogg_1', 'Listening_10.ogg_1', 'Listening_8.ogg_4', 'Listening_16.ogg_4'])