from General.init import *
from General.CorrolationMethods import  get_max_xcorr_pearson_1d
from revision_analysis.synchrony_funcs import get_max_xcorr_CCC_1d
import numpy as np
from statsmodels.tsa.stattools import grangercausalitytests

def corr_headers(append_with=""):
    headers = []
    for i in range(16):
        headers += [f"cluster_{i}_corr{append_with}", f"cluster_{i}_lag{append_with}"]
    return headers

def add_mean_component_headers_listening():
    headers = []
    for i in range(16):
        headers += [f"cluster_{i}_mean_comp"]
    return headers

def add_mean_component_headers_reading():
    headers = []
    for i in range(16):
        headers += [f"cluster_{i}_mean_comp_reader", f"cluster_{i}_mean_comp_listener"]
    return headers

def corelation_lags_measurments(correlations, lags):
    if len(correlations) == 0:
        return [np.nan, np.nan]
    
    mean_corr = np.mean(correlations)
    mean_lag = np.mean(lags)
    return [mean_corr, mean_lag]

#todo - maybe add mean comp of ok participant even if in one participant the component is nan?

def add_mean_comps(chunk_data, current_reader):
    mean_comps = []
    for i in range(16):
        if chunk_data[i] is not np.nan:
            if current_reader == A:
                mean_comps += [np.mean(chunk_data[i][A]), np.mean(chunk_data[i][B])]
            elif current_reader == B:
                mean_comps += [np.mean(chunk_data[i][B]), np.mean(chunk_data[i][A])]
            elif current_reader == "actress":
                mean_comps += [np.mean(chunk_data[i])]
        else:
            if current_reader == "actress":
                mean_comps += [np.nan]
            else:
                mean_comps += [np.nan, np.nan]
    return mean_comps

def add_mean_comps_both_listeners(chunk_data):
    mean_comps = []
    for i in range(16):
        for p in [A,B]:
            if chunk_data[i][p] is not np.nan:
                mean_comps.append(np.mean(chunk_data[i][p]))
            else:
                mean_comps.append(np.nan)

    return mean_comps

def add_mean_comps_both_listeners_headers():
    headers = []
    for i in range(16):
        for p in [A,B]:
            headers.append(f"cluster_{i}_mean_comp_{p}")
    return headers

def get_headers_both_listeners():
    return corr_headers() + add_mean_comps_both_listeners_headers()

def corr_one_lag_headers():
    headers = []
    for i in range(16):
        headers += [f"cluster_{i}_corr_optimal_lag"]
    return headers

def get_headers_reading():
    #["session", "chunk", "action"] + 
    return corr_headers() + add_mean_component_headers_reading() # removed corr_one_lag_headers() from middle

def chunk_to_combi(chunks, p, combination):
    combi_stacked = []
    for i in combination:
        if chunks[i] is not np.nan:
            combi_stacked.append(list(chunks[i][p]))
    if len(combi_stacked) == 0:
        return np.nan
    return np.mean(combi_stacked, axis=0)

def main_correlation_of_combinations_reading(current_session, both_comps, both_chunks, window_ms, max_lag_ms, downsample_factor_ms):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]

    special_combinations = dict()
    
    special_combinations["Interest"] = [5, 12, 13]
    special_combinations["Happiness"] = [5,6,9]
    special_combinations["disgust"] =   [8, 11, 2]
    special_combinations["sadness"] = [14, 15, 1, 3]

    numOfPixelsForXCorrWindow = window_ms//downsample_factor_ms
    numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow

    n = 16

    for key in both_chunks[0]:
        chunk_data = [np.nan]*16
        action = key.split("_")[0]
        current_reader = None
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)
        else:
            continue

        lengths = [0,0]
        for i in range(n):
            chunks_i = [[],[]]
            for participant in [A,B]:
                chunk = both_chunks[participant][key]
                chunk_start = int(chunk.Start.Time)
                chunk_end = int(chunk.End.Time)
                if both_comps[participant][i] is not np.nan:
                    chunks_i[participant] = both_comps[participant][i][chunk_start:chunk_end]
                    lengths[participant] = len(chunks_i[participant])
                else:   
                    chunks_i[participant] = np.nan
            if ((chunks_i[A] is not np.nan) and (chunks_i[B] is not np.nan)):
                chunk_data[i] = chunks_i


        max_len = min(lengths)

        corrs = [[] for i in range(len(special_combinations))]
        lags =  [[] for i in range(len(special_combinations))]

        
        for s in range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep):
            e = s + numOfPixelsForXCorrWindow
            direction = "one-direction"
            for i, combi in enumerate(special_combinations):
                mat1 = chunk_to_combi(chunk_data, A, special_combinations[combi])
                mat2 = chunk_to_combi(chunk_data, B, special_combinations[combi])
                if np.isnan(mat1).any() or np.isnan(mat2).any():
                    continue
                if ( current_reader == A):
                    #todo - fix this is I return to reader's assumption.
                    # should I have one best lag in a window for all comps? if so - return all corrs arr and determine max here.
                    max_corr, max_ind = get_max_xcorr_pearson_1d(mat2, mat1, max_lag_ms,
                                                        downsample_factor_ms, s, e, direction)
                else:
                    max_corr, max_ind = get_max_xcorr_pearson_1d(mat1, mat2, max_lag_ms,
                                                        downsample_factor_ms, s, e, direction)
                corrs[i].append(max_corr)
                lags[i].append(max_ind)
        
        cor_results = []
        for i in range(len(special_combinations)):
            cor_results += corelation_lags_measurments(corrs[i], lags[i])
        
        current_result = [current_session, key, action] + \
        cor_results +\
        [np.mean(chunk_to_combi(chunk_data, A, special_combinations[combi])) for combi in special_combinations] +\
        [np.mean(chunk_to_combi(chunk_data, B, special_combinations[combi])) for combi in special_combinations]
    
        results.append(current_result)
    
    return results




def get_headers_reading_combinations():
    headers = []
    combinations = ["interest", "happiness", "disgust", "sadness"]
    for combi in combinations:
        headers += [f"{combi}_corr", f"{combi}_lag"]

    headers += [f"{combi}_mean_A" for combi in combinations]
    headers += [f"{combi}_mean_B" for combi in combinations]

    return headers



def main_correlation_reading_GC(current_session, both_comps, both_chunks, window_ms, max_lag_ms, downsample_factor_ms):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]

    n = 16

    max_lag = int(max_lag_ms//downsample_factor_ms)

    for key in both_chunks[0]:
        chunk_data = [np.nan]*16
        action = key.split("_")[0]
        current_reader = None
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)
        else:
            continue

        lengths = [0,0]
        for i in range(n):
            chunks_i = [[],[]]
            for participant in [A,B]:
                chunk = both_chunks[participant][key]
                chunk_start = int(chunk.Start.Time)
                chunk_end = int(chunk.End.Time)
                if both_comps[participant][i] is not np.nan:
                    chunks_i[participant] = both_comps[participant][i][chunk_start:chunk_end]
                    lengths[participant] = len(chunks_i[participant])
                else:   
                    chunks_i[participant] = np.nan
            chunk_data[i] = chunks_i

        max_len = min(lengths)

        cor_results = [np.nan]*32

        for i in range(n):
            mat1 = chunk_data[i][A]
            mat2 = chunk_data[i][B]
            if mat1 is np.nan or mat2 is np.nan:
                continue
            
            mat1 = mat1[:max_len]
            mat2 = mat2[:max_len]

            #reader in second column:
            if ( current_reader == A):
                x = np.column_stack([mat2, mat1])
            else:
                x = np.column_stack([mat1, mat2])
            
            f_value, p_value, _, _ = grangercausalitytests(x, [max_lag], True, verbose=False)[max_lag][0]['ssr_ftest']
            cor_results[2*i] = f_value
            cor_results[2*i+1] = p_value

        current_result = [current_session, key, action] + \
            cor_results +\
            add_mean_comps(chunk_data, current_reader)
        
        results.append(current_result)
    return results

def get_headers_reading_GC():
    headers = []
    for i in range(16):
        headers += [f"cluster_{i}_GCF", f"cluster_{i}_GCP"]
    return headers + add_mean_component_headers_reading()


def main_corrolation_of_single_comps_reading(current_session, both_comps, both_chunks, window_ms, max_lag_ms, downsample_factor_ms):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]

    numOfPixelsForXCorrWindow = window_ms//downsample_factor_ms
    numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow

    n = 16

    for key in both_chunks[0]:
        chunk_data = [np.nan]*16
        action = key.split("_")[0]
        current_reader = None
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)
        else:
            continue

        lengths = [0,0]
        for i in range(n):
            chunks_i = [[],[]]
            for participant in [A,B]:
                chunk = both_chunks[participant][key]
                chunk_start = int(chunk.Start.Time)
                chunk_end = int(chunk.End.Time)
                if both_comps[participant][i] is not np.nan:
                    chunks_i[participant] = both_comps[participant][i][chunk_start:chunk_end]
                    lengths[participant] = len(chunks_i[participant])
                else:   
                    chunks_i[participant] = np.nan
            # if ((chunks_i[A] is not np.nan) and (chunks_i[B] is not np.nan)):
            #     chunk_data[i] = chunks_i
            chunk_data[i] = chunks_i


        max_len = min(lengths)

        corrs = [[] for i in range(n)]
        lags =  [[] for i in range(n)]

        # corrs_one_lag = [[] for i in range(n)]

        for s in range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep):
            e = s + numOfPixelsForXCorrWindow
            direction = "one-direction"
            # all_corr_vecs_within_s = [[] for i in range(n)]
            for i in range(n):
                mat1 = chunk_data[i][A]
                mat2 = chunk_data[i][B]
                if mat1 is np.nan or mat2 is np.nan:
                    # all_corr_vecs_within_s[i] = [np.nan]
                    continue

                if ( current_reader == A):
                    #todo - fix this is I return to reader's assumption.
                    # should I have one best lag in a window for all comps? if so - return all corrs arr and determine max here.
                    max_corr, max_ind, corr_vec = get_max_xcorr_pearson_1d(mat2, mat1, max_lag_ms,
                                                        downsample_factor_ms, s, e, direction, True)
                else:
                    max_corr, max_ind, corr_vec = get_max_xcorr_pearson_1d(mat1, mat2, max_lag_ms,
                                                        downsample_factor_ms, s, e, direction, True)
                # all_corr_vecs_within_s[i] = corr_vec
                corrs[i].append(max_corr)
                lags[i].append(max_ind)
            
            # m = max(len(row) for row in all_corr_vecs_within_s)
            # all_corr_vecs_within_s_padded = np.array([row + [np.nan] * (m - len(row)) for row in all_corr_vecs_within_s])
            # ind = np.argmax(np.nanmean(all_corr_vecs_within_s_padded, axis=0))
            # for i in range(n):
            #     if chunk_data[i] is not np.nan:
            #         corrs_one_lag[i].append(all_corr_vecs_within_s[i][ind])

        cor_results = []
        for i in range(n):
            cor_results += corelation_lags_measurments(corrs[i], lags[i])
        
        # for i in range(n):
        #     if chunk_data[i] is not np.nan:
        #         cor_results += [np.mean(corrs_one_lag[i])]
        #     else:
        #         cor_results += [np.nan]

        current_result = [current_session, key, action] + \
            cor_results +\
            add_mean_comps(chunk_data, current_reader)
        
        results.append(current_result)
    return results

def main_corrolation_of_single_comps_listeninig_participants(current_session, both_comps, both_chunks, window_ms, max_lag_ms, downsample_factor_ms):
    results = []
    numOfPixelsForXCorrWindow = window_ms//downsample_factor_ms
    numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow

    n = 16

    for key in both_chunks[0]:
        chunk_data = [np.nan]*16
        action = key.split("_")[0]
        current_reader = None
        if ( action.lower() != "listening" ):
            continue

        lengths = [0,0]
        for i in range(n):
            chunks_i = [[],[]]
            for participant in [A,B]:
                chunk = both_chunks[participant][key]
                chunk_start = int(chunk.Start.Time)
                chunk_end = int(chunk.End.Time)
                if both_comps[participant][i] is not np.nan:
                    chunks_i[participant] = both_comps[participant][i][chunk_start:chunk_end]
                    lengths[participant] = len(chunks_i[participant])
                else:   
                    chunks_i[participant] = np.nan
            chunk_data[i] = chunks_i


        max_len = min(lengths)

        corrs = [[] for i in range(n)]
        lags =  [[] for i in range(n)]

        for s in range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep):
            e = s + numOfPixelsForXCorrWindow
            direction = "bi-directional"
            for i in range(n):
                if chunk_data[i][A] is np.nan or chunk_data[i][B] is np.nan:
                    continue
                mat1 = chunk_data[i][A]
                mat2 = chunk_data[i][B]
                max_corr, max_ind = get_max_xcorr_pearson_1d(mat1, mat2, max_lag_ms,
                                                    downsample_factor_ms, s, e, direction, weighted_corr=False, lowess=False)
                corrs[i].append(max_corr)
                lags[i].append(max_ind)

        cor_results = []
        for i in range(n):
            cor_results += corelation_lags_measurments(corrs[i], lags[i])

        current_result = [current_session, key, action] + \
            cor_results +\
            add_mean_comps_both_listeners(chunk_data)
        
        results.append(current_result)
    return results


def main_corrolation_of_single_comps_listening(current_session, both_comps, both_chunks, actress_chunks, window_ms, max_lag_ms, downsample_factor_ms):
    results = []
    numOfPixelsForXCorrWindow = window_ms//downsample_factor_ms
    numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow
    n = 16
    current_reader = "actress"

    for key in both_chunks[0]:
        chunk_A = [np.nan]*16
        chunk_B = [np.nan]*16

        action = key.split("_")[0]
        if action.lower() != "listening":
            continue
        story = key.split("_")[1].split(".")[0]
        actress_data = actress_chunks[story].data

        lengths = [0,0]
        for i in range(n):
            for participant, chunk_p in zip([A,B], [chunk_A, chunk_B]):
                chunk = both_chunks[participant][key]
                chunk_start = int(chunk.Start.Time)
                chunk_end = int(chunk.End.Time)
                if both_comps[participant][i] is not np.nan:
                    chunk_p[i] = both_comps[participant][i][chunk_start:chunk_end]
                    lengths[participant] = int(chunk_end-chunk_start)
                else:   
                    chunk_p[i] = np.nan
        
        max_len = min(lengths)
        max_len = min(max_len, len(actress_data[0]))


        #actress smile data:
        actress_smile_data = np.mean(np.vstack([actress_data[5], actress_data[9]]), axis=0)
        A_smiles = np.nan
        if chunk_A[5] is not np.nan and chunk_A[9] is not np.nan:
            A_smiles = np.mean(np.vstack([chunk_A[5], chunk_A[9]]), axis=0)
        B_smiles = np.nan
        if chunk_B[5] is not np.nan and chunk_B[9] is not np.nan:
            B_smiles = np.mean(np.vstack([chunk_B[5], chunk_B[9]]), axis=0)

        corrs_A = [[] for i in range(n)]
        smile_corrs_A = []
        smile_corrs_B = []
        smile_lags_A = []
        smile_lags_B = []
        corrs_B = [[] for i in range(n)]
        lags_A = [[] for i in range(n)]
        lags_B =  [[] for i in range(n)]

        direction = "one-direction"

        for s in range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep):
            e = s + numOfPixelsForXCorrWindow
            for i in range(n):
                if actress_data[i] is np.nan:
                    continue

                matA = chunk_A[i]
                matB = chunk_B[i]
                matActress = actress_data[i]
                #todo - fix this is I return to reader's assumption.
                # should I have one best lag in a window for all comps? if so - return all corrs arr and determine max here.
                #reder is the second.
                if matA is not np.nan:
                    max_corr, max_ind = get_max_xcorr_pearson_1d(matA, matActress, max_lag_ms,
                                                        downsample_factor_ms, s, e, direction)
                    corrs_A[i].append(max_corr)
                    lags_A[i].append(max_ind)
                if matB is not np.nan:
                    max_corr, max_ind = get_max_xcorr_pearson_1d(matB, matActress, max_lag_ms,
                                                        downsample_factor_ms, s, e, direction)
                    corrs_B[i].append(max_corr)
                    lags_B[i].append(max_ind)
            
            #smiles:
            if A_smiles is not np.nan:
                max_corr, max_ind = get_max_xcorr_pearson_1d(A_smiles, actress_smile_data, max_lag_ms,
                                                    downsample_factor_ms, s, e, direction)
                smile_corrs_A.append(max_corr)
                smile_lags_A.append(max_ind)
            
            if B_smiles is not np.nan:
                max_corr, max_ind = get_max_xcorr_pearson_1d(B_smiles, actress_smile_data, max_lag_ms,
                                                    downsample_factor_ms, s, e, direction)
                smile_corrs_B.append(max_corr)
                smile_lags_B.append(max_ind)

        cor_results_A = []
        cor_results_B = []
        for i in range(n):
            cor_results_A += corelation_lags_measurments(corrs_A[i], lags_A[i])
            cor_results_B += corelation_lags_measurments(corrs_B[i], lags_B[i])


        smile_corr_A = np.mean(smile_corrs_A) if len(smile_corrs_A) > 0 else np.nan
        smile_lag_A = np.mean(smile_lags_A) if len(smile_lags_A) > 0 else np.nan
        smile_corr_B = np.mean(smile_corrs_B) if len(smile_corrs_B) > 0 else np.nan
        smile_lag_B = np.mean(smile_lags_B) if len(smile_lags_B) > 0 else np.nan
        smile_mean_A = np.mean(A_smiles) if A_smiles is not np.nan else np.nan
        smile_mean_B = np.mean(B_smiles) if B_smiles is not np.nan else np.nan

        cor_results_A += [smile_corr_A, smile_lag_A]
        cor_results_B += [smile_corr_B, smile_lag_B]

        current_result_A = [current_session, key, action, "A"] + \
            cor_results_A +\
            add_mean_comps(chunk_A, current_reader) + [smile_mean_A]
        current_result_B = [current_session, key, action, "B"] + \
            cor_results_B +\
            add_mean_comps(chunk_B, current_reader) + [smile_mean_B]
        
        results.append(current_result_A)
        results.append(current_result_B)
    
    return results

def get_headers_listening():
    return corr_headers() + ["smile_corr", "smile_lag"] + add_mean_component_headers_listening() + ["smile_mean"]