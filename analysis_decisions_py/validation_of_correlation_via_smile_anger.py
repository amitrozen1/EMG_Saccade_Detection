import os
from General.HelperFunctions import GetSessions
import numpy as np
from scipy.stats import zscore
from General.init import *
from tqdm import tqdm
from General.HelperFunctions import getPathsFromSessionFolder, remove_rows_from_matrix
from EDF.EdfAnalyzer import EdfAnalyzer
from General.HelperFunctions import get_review_data
from ComponentAnalysis.ComponentExtractor import open_w_if_exisis, load_special_components_if_exists, get_special_component_indexes, save_new_w
from General.CorrolationMethods import get_max_xcorr_1d, get_max_xcorr_pearson_1d, windowed_lagged_cross_correlation, dtai_dist_dtw_pearson_over_seconds, cross_correlation_max_pearson, cross_correlation_max_pearson_ndim
import pandas as pd
import matplotlib.pyplot as plt
from General.PlotFunctions import plotDataFromBothChunks, clearPlot
from analysis_decisions_py.cleaning_signal import remove_emg_artifacts
import ast
from ComponentAnalysis.ReviewComponents_batch2 import GetSessionsBatch2, open_w_if_exists_batch2, get_bad_electrode, get_chosen_comps, is_no_review_folder


default_rms_size_ms = 400
default_downsample_in_ms = 50 # 50 hz

SMILE = 0
ANGRY = 1

def convert_str_to_list(s):
    return ast.literal_eval(s)

def get_vector_len(vec):
    if ( len(vec.shape) == 1):
        return vec.shape[0]
    return vec.shape[1]

def get_minimal_vecs_len(vec1, vec2):
    return min(get_vector_len(vec1), get_vector_len(vec2))

def load_single_gyro_data_smile_and_angry(session, p, rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms, compare_to = "angry"):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    Y, chunks, freq, inertialData, inertialFreq = EdfAnalyzer.ReadWithInertial(path_and_corrections[p][0], path_and_corrections[p][1], downsample_window_ms, just_accelorometer=False)

    if ( len(inertialData) == 3):
        return None

    inertialData = inertialData[0:3] #gyro only

    inertialData = zscore(inertialData, axis=1)
    #inertialData = remove_emg_artifacts(inertialData)

    inertialData = EdfAnalyzer.window_rms(inertialData, rms_size_ms, inertialFreq)
    inertialData = EdfAnalyzer.resample_by_averaging_every_n(inertialData, downsample_window_ms, inertialFreq)

    max_row = np.max(inertialData, axis=0)
    #combined_with_rms = np.sqrt(np.sum(np.square(inertialData), axis=0))

    smile_s = int(chunks["smile__0"].Start.Time)
    smile_e = int(chunks["smile__2"].End.Time)

    compare_to_s = int(chunks[f"{compare_to}__0"].Start.Time)
    compare_to_e = int(chunks[f"{compare_to}__2"].End.Time)

    smile_data = inertialData[:, smile_s:smile_e]
    compare_to_data = inertialData[:, compare_to_s:compare_to_e]

    return [smile_data, compare_to_data]

def load_single_data_smile_and_angry_batch2(session, p, rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms, compare_to = "angry"):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))

    Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsample_window_ms)
    elecs_to_remove = get_bad_electrode(session, p)
    Y = remove_rows_from_matrix(Y, elecs_to_remove)
    Y = zscore(Y, axis=1)

    w = open_w_if_exists_batch2(session, p)
    
    if ( w is None):
        w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove))
        #save new w.. but it supposed to exist anyway..
    else:
        independentComponents = np.dot(w, Y)

    special_comps_index = get_chosen_comps(session, p)

    independentComponents = independentComponents[special_comps_index, :]
    independentComponents = remove_emg_artifacts(independentComponents)
    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsample_window_ms, freq)

    smile_s = int(chunks["smile__0"].Start.Time)
    smile_e = int(chunks["smile__2"].End.Time)

    compare_to_s = int(chunks[f"{compare_to}__0"].Start.Time)
    compare_to_e = int(chunks[f"{compare_to}__2"].End.Time)

    smile_data = independentComponents[:, smile_s:smile_e]
    compare_to_data = independentComponents[:, compare_to_s:compare_to_e]

    return [smile_data, compare_to_data] 


def load_signle_data_smile_and_angry(session, p, rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms, compare_to = "angry"):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsample_window_ms)
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
    independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsample_window_ms, freq)

    smile_s = int(chunks["smile__0"].Start.Time)
    smile_e = int(chunks["smile__2"].End.Time)

    compare_to_s = int(chunks[f"{compare_to}__0"].Start.Time)
    compare_to_e = int(chunks[f"{compare_to}__2"].End.Time)

    smile_data = independentComponents[:, smile_s:smile_e]
    compare_to_data = independentComponents[:, compare_to_s:compare_to_e]

    return [smile_data, compare_to_data]

def load_general_dyad_data(sessions, p_s, rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms, compare_to = "angry", data_type="emg"):
    vectors = []
    for i in range(2):
        if (data_type == "emg"):
            chunk_data = load_single_data_smile_and_angry_batch2(sessions[i], p_s[i], rms_size_ms, downsample_window_ms, compare_to)
        elif (data_type == "gyro"):
            chunk_data = load_single_gyro_data_smile_and_angry(sessions[i], p_s[i], rms_size_ms, downsample_window_ms, compare_to)
            if chunk_data is None:
                return None
        vectors.append(chunk_data)
    return vectors

def load_real_dyad_data(session, rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms, compare_to = "angry", data_type="emg"):
    return load_general_dyad_data([session, session], [A,B], rms_size_ms, downsample_window_ms, compare_to, data_type)

def compute_single_corrolations(vectors, chunk_key, window_size_in_ms, maxium_lag_in_ms, downsample_window_ms = default_downsample_in_ms):
    max_len = get_minimal_vecs_len(vectors[0][0], vectors[1][0])
    maxium_lag_in_ms = int(maxium_lag_in_ms)
    window_size_in_ms = int(window_size_in_ms)

    numOfPixelsForXCorrWindow = window_size_in_ms//downsample_window_ms
    numOfPixelsForXCorrStep = 500//downsample_window_ms
    tMax = maxium_lag_in_ms//downsample_window_ms
    tInc = numOfPixelsForXCorrWindow//10 

    if (tInc == 0):
        tInc = 1
    
    max_corrs = [[] for i in range(4)]
    # for p in [A,B]:
    #     vectors[p] = zscore(vectors[p], axis=1)
    
    # for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
    #         e = s + numOfPixelsForXCorrWindow
    #         for comp in range(4):
    #             max_corr, max_ind = get_max_xcorr_1d(vectors[0][comp], vectors[1][comp], maxium_lag_in_ms,
    #                                                           downsampleWindowInMs, s, e, chunk_key.split("_")[0])
    #             # max_corr = windowed_lagged_cross_correlation(vectors[0][comp], vectors[1][comp],
    #             #                                                       numOfPixelsForXCorrWindow, numOfPixelsForXCorrStep, 
    #             #                                                       tMax, tInc)
    #             max_corrs[comp].append(max_corr)

    # for comp in range(4):
    #     max_corr = dtai_dist_dtw_pearson_over_seconds(vectors[0][comp], vectors[1][comp], tMax)
    #     max_corrs[comp].append(max_corr)
    # for comp in range(4):
    #     a = vectors[0][comp]
    #     b = vectors[1][comp]
    #     # compute simple pearson correlation:
    #     min_length = get_minimal_vecs_len(a, b)
    #     a = a[:min_length]
    #     b = b[:min_length]
    #     #corr = np.corrcoef(a, b)[0, 1]
    #     #corr = tam_corr(a, b, tMax)
    #     #max_corrs[comp].append(corr)
    #     corr = cross_correlation_max_pearson(a, b, tMax)
    #     max_corrs[comp].append(np.arctan(corr))
    min_length = get_minimal_vecs_len(vectors[0][0], vectors[1][0])
    a = vectors[0][:, :min_length]
    b = vectors[1][:, :min_length]
    _, _, max_corrs = cross_correlation_max_pearson_ndim(a, b, tMax)

    return [np.mean(comp) for comp in max_corrs]

def compute_single_corrolations_2(vectors, chunk_key, window_size_in_ms, maxium_lag_in_ms, downsample_window_ms = default_downsample_in_ms):
    max_len = get_minimal_vecs_len(vectors[0][0], vectors[1][0])
    maxium_lag_in_ms = int(maxium_lag_in_ms)
    window_size_in_ms = int(window_size_in_ms)

    numOfPixelsForXCorrWindow = window_size_in_ms//downsample_window_ms
    numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow
    tMax = maxium_lag_in_ms//downsample_window_ms
    tInc = numOfPixelsForXCorrWindow//10 

    mat1 = np.max(vectors[0], axis=0)
    mat2 = np.max(vectors[1], axis=0)

    if (tInc == 0):
        tInc = 1
    
    max_corrs = []
    for p in [A,B]:
        vectors[p] = zscore(vectors[p], axis=1)
    
    for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow
            max_corr, max_ind = get_max_xcorr_pearson_1d(mat1, mat2, maxium_lag_in_ms,
                                                            downsample_window_ms, s, e, chunk_key.split("_")[0])
            max_corrs.append(max_corr)

    return max_corrs



def compute_correlation_row(session, rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms, compare_to = "angry"):
    vectors = load_real_dyad_data(session, rms_size_ms, downsample_window_ms, compare_to)
    window_size = 1000
    lag = window_size*2

    smile_corrs = compute_single_corrolations([vectors[A][SMILE], vectors[B][SMILE]], "smile", window_size, lag, downsample_window_ms)
    angry_corrs = compute_single_corrolations([vectors[A][ANGRY], vectors[B][ANGRY]], "angry", window_size, lag, downsample_window_ms)
    smile_anger_corrs = compute_single_corrolations([vectors[A][SMILE], vectors[B][ANGRY]], "smile_anger", window_size, lag, downsample_window_ms)
    anger_smile_corrs = compute_single_corrolations([vectors[A][ANGRY], vectors[B][SMILE]], "smile_anger", window_size, lag, downsample_window_ms)
    return [smile_corrs, angry_corrs, smile_anger_corrs, anger_smile_corrs]


def compute_correlation_row_2(session, rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms, compare_to = "angry", window_size_ms=3000, max_lag_ms=3000, data_type="emg"):
    vectors = load_real_dyad_data(session, rms_size_ms, downsample_window_ms, compare_to, data_type)
    if vectors is None:
        return None
    smile_corrs = compute_single_corrolations_2([vectors[A][SMILE], vectors[B][SMILE]], "smile", window_size_ms, max_lag_ms, downsample_window_ms)
    angry_corrs = compute_single_corrolations_2([vectors[A][ANGRY], vectors[B][ANGRY]], "angry", window_size_ms, max_lag_ms, downsample_window_ms)
    smile_anger_corrs = compute_single_corrolations_2([vectors[A][SMILE], vectors[B][ANGRY]], "smile_anger", window_size_ms, max_lag_ms, downsample_window_ms)
    anger_smile_corrs = compute_single_corrolations_2([vectors[A][ANGRY], vectors[B][SMILE]], "smile_anger", window_size_ms, max_lag_ms, downsample_window_ms)
    return [smile_corrs, angry_corrs, smile_anger_corrs, anger_smile_corrs]





def plot_couples_graph_and_xcorr(session, compare_to = "angry"):
    vectors = load_real_dyad_data(session, compare_to=compare_to)
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
    smile_anger_corrs = compute_single_corrolations([vectors[A][SMILE], vectors[B][ANGRY]], "smile_anger", window_size, lag)
    print(smile_anger_corrs)
    plotDataFromBothChunks(vectors[A][SMILE], vectors[B][ANGRY], [], [], "smile_anger", 4)
    clearPlot()
    anger_smile_corrs = compute_single_corrolations([vectors[A][ANGRY], vectors[B][SMILE]], "smile_anger", window_size, lag)
    print(anger_smile_corrs)
    plotDataFromBothChunks(vectors[A][ANGRY], vectors[B][SMILE], [], [], "anger_smile", 4)
    clearPlot()

def plot_sessions(compare_to = "angry"):
    sessions = GetSessions(data_path)[1:]
    for session in tqdm(sessions):
        plot_couples_graph_and_xcorr(session, compare_to)


def compute_t(smile_vector, anger_smile_vector, smile_anger_vector):
    from scipy.stats import ttest_ind
    real_list = smile_vector.values.tolist()
    pseudo_list = anger_smile_vector.values.tolist() + smile_anger_vector.values.tolist()
    t_stat, p_val = ttest_ind(real_list, pseudo_list, equal_var=False)
    return t_stat

def compute_pseudo_and_real_dyads(rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms, compare_to = "angry"):
    headers = ["session", "smile_corr", "angry_corr", "smile_anger_corr", "anger_smile_corr"]
    mean_comps_csv_file_path = os.path.join(data_path,"anger_smile_results","compute_rms_and_ds", f"cross_pearson_rms_{rms_size_ms}_ds{downsample_window_ms}_artifacts_removed_bestLag_all_comps_min_40.csv")

    if (not os.path.exists(mean_comps_csv_file_path)):
        sessions = GetSessions(data_path)
        all_corrs = []
        comp_corrs = [[],[],[],[]]
        for session in tqdm(sessions):
            corrs = compute_correlation_row(session, rms_size_ms, downsample_window_ms, compare_to)
            meancorrs = [f"{session}"] + [np.mean(c) for c in corrs]
            for i in range(4):
                comp_corrs[i].append([f"{session}"] + corrs[i])
            
            all_corrs.append(meancorrs)
        
        df = pd.DataFrame(all_corrs)
        df.columns = headers
        df.to_csv(mean_comps_csv_file_path)
        
        for i in range(4):
            df_i = pd.DataFrame(comp_corrs[i])
            df_i.columns = headers
            df_i.to_csv(os.path.join(data_path,"anger_smile_results","compute_rms_and_ds", f"cross_pearson_rms_{rms_size_ms}_ds{downsample_window_ms}_artifacts_removed_bestLag_comp_{i}_min_40.csv"))    
    else:
        df = pd.read_csv(mean_comps_csv_file_path)
        df = df.drop(columns=["Unnamed: 0"])

    corr_columns = headers[1:]
    df_corr = df[corr_columns]

    # Normalize the correlation values by row
    df_normalized = (df_corr - df_corr.mean(axis=1).values.reshape(-1, 1)) / df_corr.std(axis=1).values.reshape(-1, 1)

    # Add back the session information to the normalized dataframe
    df_normalized['session'] = df['session']

    t_stats = compute_t(df_normalized["smile_corr"], df_normalized["smile_anger_corr"], df_normalized["anger_smile_corr"])
    return t_stats


def compute_pseudo_and_real_dyads_2(rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms, compare_to = "angry", window_size_ms=3000, max_lag_ms=3000, data_type="emg"):
    headers = ["session", "smile_corr", "angry_corr", "smile_anger_corr", "anger_smile_corr"]

    string_to_add_if_gyro = "" if data_type == "emg" else "_gyro"

    mean_comps_csv_file_path = os.path.join(data_path,"anger_smile_results","compute_rms_and_ds", f"cross_pearson_rms_{rms_size_ms}_ds{downsample_window_ms}_window_{window_size_ms//1000}_lag_{max_lag_ms//1000}{string_to_add_if_gyro}_artifacts_removed_bestLag_max_comp.csv")

    mistaken_saved_as_arr = False

    if (not os.path.exists(mean_comps_csv_file_path)):
        sessions = GetSessionsBatch2()  # GetSessions(data_path)
        all_corrs = []
        for session in tqdm(sessions):
            corrs = compute_correlation_row_2(session, rms_size_ms, downsample_window_ms, compare_to, window_size_ms , max_lag_ms, data_type)
            if ( corrs is None):
                continue
            meancorrs = [f"{session}"] + [np.mean(c) for c in corrs]           
            all_corrs.append(meancorrs)
        
        df = pd.DataFrame(all_corrs)
        df.columns = headers
        df.to_csv(mean_comps_csv_file_path)
   
    else:
        df = pd.read_csv(mean_comps_csv_file_path)
        df = df.drop(columns=["Unnamed: 0"])

    corr_columns = headers[1:]
    df_corr = df[corr_columns]

    if  (mistaken_saved_as_arr):
        #each cell is an array - replace it with the mean of the array:
        for i in range(df_corr.shape[0]):
            for j in range(df_corr.shape[1]):
                df_corr.iloc[i,j] = np.mean(convert_str_to_list(df_corr.iloc[i,j]))
        
    # Normalize the correlation values by row
    df_normalized = (df_corr - df_corr.mean(axis=1).values.reshape(-1, 1)) / df_corr.std(axis=1).values.reshape(-1, 1)

    # Add back the session information to the normalized dataframe
    df_normalized['session'] = df['session']

    t_stats = compute_t(df_normalized["smile_corr"], df_normalized["smile_anger_corr"], df_normalized["anger_smile_corr"])
    return t_stats


def plot_boxplot(graph_name = "all_comps.csv", rms_size_ms = default_rms_size_ms, downsample_window_ms = default_downsample_in_ms):
    #path = os.path.join(data_path,"anger_smile_results", f"cross_pearson_rms_{default_rms_size_ms}_ds{default_downsample_in_ms}_artifacts_removed_bestLag_all_comps.csv")
    #path = os.path.join(data_path,"anger_smile_results", "pearson_dtw_all_comps.csv")
    path = os.path.join(data_path,"anger_smile_results","compute_rms_and_ds_blink", f"cross_pearson_rms_{rms_size_ms}_ds{downsample_window_ms}_artifacts_removed_bestLag_all_comps.csv")
    df = pd.read_csv(path)
    #df = df.drop(columns=["session"])

    #skip first column:
    df = df.drop(columns=["Unnamed: 0"])
    
    corr_columns = ['smile_corr', 'angry_corr', 'smile_anger_corr', 'anger_smile_corr']
    df_corr = df[corr_columns]

    # Normalize the correlation values by row
    df_normalized = (df_corr - df_corr.mean(axis=1).values.reshape(-1, 1)) / df_corr.std(axis=1).values.reshape(-1, 1)

    # Add back the session information to the normalized dataframe
    df_normalized['session'] = df['session']

    # combine smile_anger_corr and anger_smile_corr into one array:
    pseudo_list = df_normalized['smile_anger_corr'].values.tolist()
    pseudo_list.extend(df_normalized['anger_smile_corr'].values.tolist())
    real_list = df_normalized['smile_corr'].values.tolist()

    import seaborn as sns
    long_df = pd.melt(df_normalized, id_vars="session", var_name="correlation_type", value_name="correlation_value")

    # Create the boxplot
    plt.figure(figsize=(10, 8))
    sns.boxplot(x='correlation_type', y='correlation_value', data=long_df)
    plt.title('Comparison of Correlation Types')
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.show()

    clearPlot()
    #exit()

    # for col in df.columns:
    #     df[col] = df[col].apply(lambda x: np.arctan(x))
    
    # print(df.head())
    # df.boxplot()
    # plt.title("all_comps")
    # plt.show()
    # clearPlot()


    for i in range(4):
        #path = os.path.join(data_path,"anger_smile_results", f"cross_pearson_rms_{default_rms_size_ms}_ds{default_downsample_in_ms}_artifacts_removed_bestLag_comp_{i}.csv")
        path = os.path.join(data_path,"anger_smile_results","compute_rms_and_ds_blink", f"cross_pearson_rms_{rms_size_ms}_ds{downsample_window_ms}_artifacts_removed_bestLag_comp_{i}.csv")
        df = pd.read_csv(path)
        #df = df.drop(columns=["session"])
        #skip first column:
        df = df.drop(columns=["Unnamed: 0"])
        corr_columns = ['smile_corr', 'angry_corr', 'smile_anger_corr', 'anger_smile_corr']
        df_corr = df[corr_columns]

        # Normalize the correlation values by row
        df_normalized = (df_corr - df_corr.mean(axis=1).values.reshape(-1, 1)) / df_corr.std(axis=1).values.reshape(-1, 1)

        # Add back the session information to the normalized dataframe
        df_normalized['session'] = df['session']


        import seaborn as sns
        long_df = pd.melt(df_normalized, id_vars="session", var_name="correlation_type", value_name="correlation_value")

        # Create the boxplot
        plt.figure(figsize=(10, 8))
        sns.boxplot(x='correlation_type', y='correlation_value', data=long_df)
        plt.title('Comparison of Correlation Types')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.show()

        clearPlot()
        # for col in df.columns:
        #     df[col] = df[col].apply(lambda x: np.arctan(x))
        # print(df.head())
        # df.boxplot()
        # plt.title(f"comp {i}")
        # plt.show()
        # clearPlot()

    #build anew df with max of each comp for each state:
    # df_max = pd.DataFrame()

    # for i in range(4):
    #     path = os.path.join(data_path,"anger_smile_results", f"cross_pearson_rms_{rms_size_ms}_ds{downsampleWindowInMs}_artifacts_removed_bestLag_comp_{i}.csv")
    #     df = pd.read_csv(path)
    #     df = df.drop(columns=["session"])
    #     #skip first column:
    #     df = df.drop(columns=["Unnamed: 0"])
    #     if i==0:
    #         df_max = df
    #     else:
    #         df_max = df_max.combine(df, np.maximum)
    
    # print(df_max.head())
    # df_max.boxplot()
    # plt.title("max")
    # plt.show()
    # clearPlot()

def check_best_rms_ds(data_type="emg"):
    #data_tyoe is either "emg" or "gyro"
    rms_options = [100, 200, 400]
    downsample_options = [20, 40, 50]

    #store t stat matrix: 
    result_mat = np.zeros((len(rms_options), len(downsample_options)))

    for rms_size_ms in tqdm(rms_options):
        for downsampleWindowInMs in tqdm(downsample_options):
            t_stat = compute_pseudo_and_real_dyads_2(rms_size_ms, downsampleWindowInMs, data_type=data_type)
            result_mat[rms_options.index(rms_size_ms), downsample_options.index(downsampleWindowInMs)] = t_stat
    
    #save result_mat
    path = os.path.join(data_path, "anger_smile_results", "compute_rms_and_ds", "result_mat_max_comp.npy")
    np.save(path, result_mat)

    #plot matrix heatmap:
    plt.imshow(result_mat, cmap="viridis")
    plt.colorbar()
    plt.xlabel("downsampleWindowInMs")
    plt.ylabel("rms_size_ms")
    plt.xticks(range(len(downsample_options)), downsample_options)
    plt.yticks(range(len(rms_options)), rms_options)
    plt.title("T-statistic for different rms and downsample options")
    plt.show()

def check_best_window_lag(data_type="emg"):
    #data_tyoe is either "emg" or "gyro"
    windows = [1000, 2000, 3000]
    lags = [1000, 2000, 3000]

    #store t stat matrix: 
    result_mat = np.zeros((len(windows), len(lags)))

    for window_ms in tqdm(windows):
        for lag_ms in tqdm(lags):
            t_stat = compute_pseudo_and_real_dyads_2(rms_size_ms=400, downsample_window_ms=50, compare_to="angry", window_size_ms=window_ms, max_lag_ms=lag_ms, data_type=data_type)
            result_mat[windows.index(window_ms), lags.index(lag_ms)] = t_stat
    
    #save result_mat
    path = os.path.join(data_path, "anger_smile_results", "compute_rms_and_ds", "result_mat_max_comp_windw_lag.npy")
    np.save(path, result_mat)

    #plot matrix heatmap:
    plt.imshow(result_mat, cmap="viridis")
    plt.colorbar()
    plt.xlabel("max_lag")
    plt.ylabel("window")
    plt.xticks(range(len(lags)), lags)
    plt.yticks(range(len(windows)), windows)
    plt.title("T-statistic for different rms and downsample options")
    plt.show()
