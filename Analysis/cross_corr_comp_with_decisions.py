import os
import numpy as np
from General.init import *
import pandas as pd
from General.StatisticsFunctions import cross_rqa
from General.CorrolationMethods import dtai_dist_dtw_pearson_over_seconds, get_max_xcorr_2d, get_max_xcorr_1d, get_max_pearson_xcorr_2d, get_max_xcorr_pearson_1d,  cross_correlation_max_pearson_ndim, cross_correlation_max_pearson
import re
from ComponentAnalysis.ComponentExtractor import get_special_component_names
from Analysis.analysis_runner import run_emg_analysis, run_emg_analysis_batch2

incluce_var = False
rms_size_ms = 100
downsampleWindowInMs = 50
windowForCorrInMs = 3000
max_lag_ms = 3000
numOfPixelsForXCorrWindow = windowForCorrInMs//downsampleWindowInMs
#overlap : numOfPixelsForXCorrStep = (windowForCorrInMs//2) // downsampleWindowInMs
numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow
reader_assumption = True

def get_vector_len(vec):
    if ( len(vec.shape) == 1):
        return vec.shape[0]
    return vec.shape[1]

def get_minimal_vecs_len(vec1, vec2):
    return min(get_vector_len(vec1), get_vector_len(vec2))

def make_vectors_comperable(a,b):
    min_length = get_minimal_vecs_len(a, b)
    if ( len(a.shape) == 1):
        return a[:min_length], b[:min_length]
    return a[:, :min_length], b[:, :min_length]

def corelation_lags_measurments(correlations, lags):
    m = [np.mean(correlations), np.mean(lags), np.max(correlations)]
    if (incluce_var):
        return m + [np.var(correlations), np.var(lags)]
    return m

def corelation_lags_measurments_headers(title):
    m = [f"{title}_mean_corr", f"{title}_mean_lag", f"{title}_max_corr"]
    if (incluce_var):
        return m + [f"{title}_var_corr", f"{title}_var_lag"]
    return m

def add_mean_component_headers():
    return [f"mean_{comp}_{p}" for comp in get_special_component_names() for p in ["A", "B"]]
    #['mean_Nose_A', 'mean_Nose_B', 'mean_Eye_A', 'mean_Eye_B', 'mean_Zygomaticus_A', 'mean_Zygomaticus_B', 'mean_Corrugator_A', 'mean_Corrugator_B']

def add_mean_components(mat_a, mat_b):
    mats = [np.mean(mat_a, axis=1), np.mean(mat_b, axis=1)]
    size = len(mats[0])
    return [mats[p][i] for i in range(size) for p in [A,B]]

def wrap_in_default_headers(headers):
    return ["session", "key","action"] + headers + ['whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']

def get_headers():
    return corelation_lags_measurments_headers("chosen_max_comp") +\
    ["lags_under_500", "lags_under_1000"] + \
    corelation_lags_measurments_headers("all_comps") + \
    ["lags_under_500_all_comps"] + \
    ["mean_chosen_max_A", "mean_chosen_max_B"] + \
    ["var_chosen_max_A", "var_chosen_max_B"] + \
    add_mean_component_headers()

def get_main_corrolation_with_mean_comp_headers():
    return corelation_lags_measurments_headers("chosen_max_comps") + \
    corelation_lags_measurments_headers("chosen_sum_comps") + \
    ["mean_chosen_max1", "mean_chosen_max2"] + \
    ["mean_sum_max1", "mean_sum_max2"] + \
    add_mean_component_headers()

def get_main_corrolation_with_rms_comp_headers():
    return corelation_lags_measurments_headers("chosen_max_comps") + \
    corelation_lags_measurments_headers("chosen_sum_comps") + \
    ["mean_chosen_max_A", "mean_chosen_max_B"] + \
    ["mean_rms_max_A", "mean_rms_max_B"] + \
    add_mean_component_headers()

def get_main_corrolation_of_single_comps_headers():
    comps_names = get_special_component_names()
    return corelation_lags_measurments_headers(comps_names[0]) + \
    corelation_lags_measurments_headers(comps_names[1]) + \
    corelation_lags_measurments_headers(comps_names[2]) + \
    corelation_lags_measurments_headers(comps_names[3]) + \
    add_mean_component_headers()

def get_main_corrolation_with_actress_compare_headers():
    return corelation_lags_measurments_headers("chosen_comps") + \
    ["mean_chosen_max_A", "mean_chosen_max_B"] + \
    ["lags_under_500"] + \
    corelation_lags_measurments_headers("chosen_actress_A") + \
    corelation_lags_measurments_headers("chosen_actress_B") + \
    ["actress_lags_under_500_A", "actress_lags_under_500_B"] +\
    corelation_lags_measurments_headers("all_comps") + \
    corelation_lags_measurments_headers("chosen_actress_2d_A") + \
    corelation_lags_measurments_headers("chosen_actress_2d_B") + \
    ["lags_under_500_with_actress_2d_A", "lags_under_500_with_actress_2d_B"] + \
    ["mean_actress_" + comp for comp in get_special_component_names()] + \
    add_mean_component_headers()

def main_corrolation(current_session, independentComponents, chunks):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]

    for key in chunks[0]:
        chunk_data = [[],[]]
        action = key.split("_")[0]

        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]
        
        mean_comps_measurements = add_mean_components(chunk_data[A][0:4], chunk_data[B][0:4])

        if (reader_assumption and action.lower() == "reading" ):
            #take rows 0,1,3 - avoid zygomaticus.
            mat1 = chunk_data[A][[0,1,3]]
            mat2 = chunk_data[B][[0,1,3]]
        else:
            mat1 = chunk_data[A][0:4]
            mat2 = chunk_data[B][0:4]

        temporal_max1 = np.max(mat1, axis=0)
        temporal_max2 = np.max(mat2, axis=0)
        temporal_correlations = []
        temporal_lags = []
        
        lags_under_500 = 0
        lags_under_1000 = 0

        lags_under_500_all_comps = 0

        N=0

        max_len = min(mat1.shape[1], mat2.shape[1])
        corolations = []
        lags = []
        current_reader = A
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow # todo: run again.
            #numOfPixelsForXCorrWindow
            if ( reader_assumption and action.lower() == "reading" and current_reader == A):
                max_corr, max_ind = get_max_pearson_xcorr_2d(mat2, mat1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action, max_ver=True)
                max_temporal_corr, max_temporal_ind = get_max_xcorr_pearson_1d(temporal_max2, temporal_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
            else:    
                max_corr, max_ind = get_max_pearson_xcorr_2d(mat1, mat2, max_lag_ms,
                                                        downsampleWindowInMs, s, e, action, max_ver=True)
                max_temporal_corr, max_temporal_ind = get_max_xcorr_pearson_1d(temporal_max1, temporal_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
            corolations.append(max_corr)
            lags.append(max_ind)

            temporal_correlations.append(max_temporal_corr)
            temporal_lags.append(max_temporal_ind)

            N=N+1
            if (max_temporal_ind < 500):
                lags_under_500 = lags_under_500 + 1
            if (max_temporal_ind < 1000):
                lags_under_1000 = lags_under_1000 + 1
            
            if ( max_ind < 500 ):
                lags_under_500_all_comps = lags_under_500_all_comps + 1

            # chosen_correlations.append(max_chosen_corr)
            # chosen_lags.append(max_chosen_ind)
        
    
        current_result = [current_session, key, action] + \
            corelation_lags_measurments(temporal_correlations, temporal_lags) + \
            [lags_under_500/N, lags_under_1000/N] + \
            corelation_lags_measurments(corolations, lags) + \
            [lags_under_500_all_comps/N] + \
            [np.mean(temporal_max1) , np.mean(temporal_max2)] + \
            [np.var(temporal_max1) , np.var(temporal_max2)] + \
            mean_comps_measurements
            
        
        results.append(current_result)
    return results



def main_corrolation_with_mean_comp(current_session, independentComponents, chunks):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]

    for key in chunks[0]:
        chunk_data = [[],[]]
        action = key.split("_")[0]

        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]
        
        mean_comps_measurements = add_mean_components(chunk_data[A][0:4], chunk_data[B][0:4])

        if (reader_assumption and action.lower() == "reading" ):
            #take rows 0,1,3 - avoid zygomaticus.
            mat1 = chunk_data[A][[0,1,3]]
            mat2 = chunk_data[B][[0,1,3]]
        else:
            mat1 = chunk_data[A][0:4]
            mat2 = chunk_data[B][0:4]

        chosen_max1 = np.max(mat1, axis=0)
        chosen_max2 = np.max(mat2, axis=0)

        sum_max1 = np.sum(mat1, axis=0)
        sum_max2 = np.sum(mat2, axis=0)
        
        chosen_correlations = []
        chosen_lags = []

        max_len = min(mat1.shape[1], mat2.shape[1])
        sum_corolations = []
        sum_lags = []
        current_reader = A
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow # todo: run again.
            #numOfPixelsForXCorrWindow
            skipReaderAssumption = False
            if ( (not skipReaderAssumption) and action.lower() == "reading" and current_reader == A):
                max_chosen_corr, max_chosen_ind = get_max_xcorr_pearson_1d(chosen_max2, chosen_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
                sum_corr, sum_ind = get_max_xcorr_pearson_1d(sum_max2, sum_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
            else:    
                max_chosen_corr, max_chosen_ind = get_max_xcorr_pearson_1d(chosen_max1, chosen_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
                sum_corr, sum_ind = get_max_xcorr_pearson_1d(sum_max1, sum_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
            
            chosen_correlations.append(max_chosen_corr)
            chosen_lags.append(max_chosen_ind)
            
            sum_corolations.append(sum_corr)
            sum_lags.append(sum_ind)
        
        current_result = [current_session, key, action] + \
            corelation_lags_measurments(chosen_correlations, chosen_lags) + \
            corelation_lags_measurments(sum_corolations, sum_lags) + \
            [np.mean(chosen_max1), np.mean(chosen_max2)] + \
            [np.mean(sum_max1), np.mean(sum_max2)] + \
            mean_comps_measurements
        
        results.append(current_result)
    return results



def main_corrolation_with_rms_comp(current_session, independentComponents, chunks):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]

    for key in chunks[0]:
        chunk_data = [[],[]]
        action = key.split("_")[0]

        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]
        
        mean_comps_measurements = add_mean_components(chunk_data[A][0:4], chunk_data[B][0:4])

        if (reader_assumption and action.lower() == "reading" ):
            #take rows 0,1,3 - avoid zygomaticus.
            mat1 = chunk_data[A][[0,1,3]]
            mat2 = chunk_data[B][[0,1,3]]
        else:
            mat1 = chunk_data[A][0:4]
            mat2 = chunk_data[B][0:4]

        chosen_max1 = np.max(mat1, axis=0)
        chosen_max2 = np.max(mat2, axis=0)

        rms_max1 = np.sqrt(np.mean(mat1**2, axis=0))
        rms_max2 = np.sqrt(np.mean(mat2**2, axis=0))
        
        chosen_correlations = []
        chosen_lags = []

        max_len = min(mat1.shape[1], mat2.shape[1])
        sum_corolations = []
        sum_lags = []
        current_reader = A
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow # todo: run again.
            #numOfPixelsForXCorrWindow
            skipReaderAssumption = False
            if ( (not skipReaderAssumption) and action.lower() == "reading" and current_reader == A):
                max_chosen_corr, max_chosen_ind = get_max_xcorr_pearson_1d(chosen_max2, chosen_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
                sum_corr, sum_ind = get_max_xcorr_pearson_1d(rms_max2, rms_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
            else:    
                max_chosen_corr, max_chosen_ind = get_max_xcorr_pearson_1d(chosen_max1, chosen_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
                sum_corr, sum_ind = get_max_xcorr_pearson_1d(rms_max1, rms_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
            
            chosen_correlations.append(max_chosen_corr)
            chosen_lags.append(max_chosen_ind)
            
            sum_corolations.append(sum_corr)
            sum_lags.append(sum_ind)
        
        current_result = [current_session, key, action] + \
            corelation_lags_measurments(chosen_correlations, chosen_lags) + \
            corelation_lags_measurments(sum_corolations, sum_lags) + \
            [np.mean(chosen_max1), np.mean(chosen_max2)] + \
            [np.mean(rms_max1), np.mean(rms_max2)] + \
            mean_comps_measurements
        
        results.append(current_result)
    return results




def main_corrolation_of_single_comps(current_session, independentComponents, chunks):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]

    for key in chunks[0]:
        chunk_data = [[],[]]
        action = key.split("_")[0]

        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]
        
        mean_comps_measurements = add_mean_components(chunk_data[A][0:4], chunk_data[B][0:4])


        mat1 = chunk_data[A][0:4]
        mat2 = chunk_data[B][0:4]

        max_len = min(mat1.shape[1], mat2.shape[1])

        corrs = [[],[],[],[]]
        lags = [[],[],[],[]]

        current_reader = A
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow # todo: run again.
            #numOfPixelsForXCorrWindow
            skipReaderAssumption = False
            if ( (not skipReaderAssumption) and action.lower() == "reading" and current_reader == A):
                for i in range(4):
                    max_corr, max_ind = get_max_xcorr_pearson_1d(mat2[i], mat1[i], max_lag_ms,
                                                        downsampleWindowInMs, s, e, action)
                    corrs[i].append(max_corr)
                    lags[i].append(max_ind)
            else:
                for i in range(4):
                    max_corr, max_ind = get_max_xcorr_pearson_1d(mat1[i], mat2[i], max_lag_ms,
                                                        downsampleWindowInMs, s, e, action)
                    corrs[i].append(max_corr)
                    lags[i].append(max_ind)

        current_result = [current_session, key, action] + \
            corelation_lags_measurments(corrs[0], lags[0] ) + \
            corelation_lags_measurments(corrs[1], lags[1] ) + \
            corelation_lags_measurments(corrs[2], lags[2] ) + \
            corelation_lags_measurments(corrs[3], lags[3] ) + \
            mean_comps_measurements
        
        results.append(current_result)
    return results


def main_correlation_with_actress_compare(current_session, independentComponents, chunks, actress_chunks):
    results = []

    for key in chunks[0]:
        chunk_data = [[],[]]
        action = key.split("_")[0]
        if (action.lower() != "listening"):
            continue

        story_number = key.split("_")[1].split(".")[0]

        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]
        
        mean_comps_measurements = add_mean_components(chunk_data[A][0:4], chunk_data[B][0:4])
        # mat1 = chunk_data[A][0:4, :]
        # mat2 = chunk_data[B][0:4, :]

        #select no zygomaticus to compare synchrony.

        mat_actress = actress_chunks[story_number].data[0:4, :]

        no_zyg_A = chunk_data[A][[0,1,3], :]
        no_zyg_B = chunk_data[B][[0,1,3], :]

        max_no_zyg_A = np.max(no_zyg_A, axis=0).flatten()
        max_no_zyg_B = np.max(no_zyg_B, axis=0).flatten()
        max_A = np.max(chunk_data[A][0:4,:], axis=0).flatten()
        max_B = np.max(chunk_data[B][0:4,:], axis=0).flatten()

        actress_no_zygomaticus = mat_actress[[0,1,3], :]
        # chosen_max_actress = np.max(mat_actress, axis=0)
        max_actress_no_zygomaticus = np.max(actress_no_zygomaticus, axis=0)

        lags_under_500 = 0
        lags_under_500_2d = 0
        lags_under_500_with_actress = [0,0]
        lags_under_500_with_actress_2d = [0,0]
        N=0
        N_actress = [0,0]

        correlations2d = []
        lags2d = []

        corrolations = []
        lags = []

        chosen_actress_correlations = [[],[]]
        chosen_actress_correlations_2d = [[],[]]
        chosen_actress_lags = [[],[]]
        chosen_actress_lags_2d = [[],[]]


        max_len = min(len(max_no_zyg_A), len(max_no_zyg_B))
        max_len_actress = [min(len(mat), mat_actress.shape[1]) for mat in [max_no_zyg_A, max_no_zyg_B]]

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow # todo: run again.  
            corr, lag = get_max_xcorr_pearson_1d(max_A, max_B, max_lag_ms,
                                                downsampleWindowInMs, s, e, action)
            corr2d, lag2d = get_max_pearson_xcorr_2d(no_zyg_A, no_zyg_B, max_lag_ms,
                                                downsampleWindowInMs, s, e, action)

            corrolations.append(corr)
            lags.append(lag)

            correlations2d.append(corr2d)
            lags2d.append(lag2d)

            N=N+1

            if (lag < 500):
                lags_under_500 = lags_under_500 + 1
            
            if (lag2d < 500):
                lags_under_500_2d = lags_under_500_2d + 1

            for p, mat in zip([A,B], [max_no_zyg_A, max_no_zyg_B]):
                if e > max_len_actress[p]:
                    continue
                
                #actress is the reader.
                cor, ind = get_max_xcorr_pearson_1d(mat, max_actress_no_zygomaticus, max_lag_ms,
                                                downsampleWindowInMs, s, e, "reading")

                chosen_actress_correlations[p].append(cor)
                chosen_actress_lags[p].append(ind)

                N_actress[p] = N_actress[p] + 1
                if (ind < 500):
                    lags_under_500_with_actress[p] = lags_under_500_with_actress[p] + 1
            
            for p, mat in zip([A,B], [no_zyg_A, no_zyg_B]):
                if e > max_len_actress[p]:
                    continue

                cor, ind = get_max_pearson_xcorr_2d(mat, mat_actress, max_lag_ms,
                                                downsampleWindowInMs, s, e, "reading")
                
                chosen_actress_correlations_2d[p].append(cor)
                chosen_actress_lags_2d[p].append(ind)

                if (ind < 500):
                    lags_under_500_with_actress_2d[p] = lags_under_500_with_actress_2d[p] + 1
                
                #actress is the reader.
                cor, ind = get_max_pearson_xcorr_2d(mat, mat_actress, max_lag_ms,
                                                downsampleWindowInMs, s, e, "reading")

                chosen_actress_correlations_2d[p].append(cor)
                chosen_actress_lags_2d[p].append(ind)

                if (ind < 500):
                    lags_under_500_with_actress_2d[p] = lags_under_500_with_actress_2d[p] + 1
        
        
        current_result = [current_session, key, action] + \
            corelation_lags_measurments(corrolations, lags) + \
            [np.mean(max_A), np.mean(max_B)] + \
            [lags_under_500/N] + \
            corelation_lags_measurments(chosen_actress_correlations[A], chosen_actress_lags[A]) + \
            corelation_lags_measurments(chosen_actress_correlations[B], chosen_actress_lags[B]) + \
            [lags_under_500_with_actress[p]/N_actress[p] for p in [A,B]] + \
            corelation_lags_measurments(correlations2d, lags2d) + \
            corelation_lags_measurments(chosen_actress_correlations_2d[A], chosen_actress_lags_2d[A]) + \
            corelation_lags_measurments(chosen_actress_correlations_2d[B], chosen_actress_lags_2d[B]) + \
            [lags_under_500_with_actress_2d[p]/N_actress[p] for p in [A,B]] + \
            [np.mean(mat_actress[i]) for i in range(4)] + \
            mean_comps_measurements
        
        results.append(current_result)
    return results

def main(new_val_for_max_lag = 3000):
    global max_lag_ms
    max_lag_ms = new_val_for_max_lag
    run_emg_analysis(main_corrolation, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}_after_pre_reg_dwt_db6_2dmaxver.csv"))

def main_2():
    global incluce_var
    incluce_var = True
    run_emg_analysis(main_corrolation_with_mean_comp, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_main_corrolation_with_mean_comp_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_with_mean_of_corr_comp_VAR_fixedlag.csv"))

def main_3():
    run_emg_analysis(main_corrolation_of_single_comps, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_main_corrolation_of_single_comps_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_corrEachComp.csv"))

def main_rms():
    global incluce_var
    incluce_var = True
    run_emg_analysis(main_corrolation_with_rms_comp, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_main_corrolation_with_rms_comp_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_with_mean_of_corr_comp_VAR.csv"))

def main_with_actress_compare(batch2 = False):
    if (batch2):
        data_path = "C:\\Liron\\DataEmg\\new_batch"
        run_emg_analysis_batch2(main_correlation_with_actress_compare, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_main_corrolation_with_actress_compare_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}_after_prereg_au6_allcomps_with_actress_2d.csv"), with_actress=True)
    else:
        run_emg_analysis(main_correlation_with_actress_compare, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_main_corrolation_with_actress_compare_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}_remove-artifacts_nooverlap_actress-compare_no-zyg-for-sync_hardremove_correctlag3_lagCount_actress_lag.csv"), with_actress=True, hard_noise_remove_for_actress=True)

def main_batch2(new_val_for_max_lag = 3000):
    global max_lag_ms
    max_lag_ms = new_val_for_max_lag
    data_path = "C:\\Liron\\DataEmg\\new_batch"
    run_emg_analysis_batch2(main_corrolation, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}_after_prereg_dwt_2dmaxver.csv"))
