import os
import numpy as np
from General.init import *
import pandas as pd
from General.StatisticsFunctions import cross_rqa
from General.CorrolationMethods import dtai_dist_dtw_pearson_over_seconds, get_max_xcorr_2d, get_max_xcorr_1d, get_max_pearson_xcorr_2d, get_max_xcorr_pearson_1d,  cross_correlation_max_pearson_ndim, cross_correlation_max_pearson
import re
from ComponentAnalysis.ComponentExtractor import get_special_component_names
from Analysis.analysis_runner import run_emg_analysis


rms_size_ms = 100
downsampleWindowInMs = 20
windowForCorrInMs = 3000
max_lag_ms = 2000
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
    return [np.mean(correlations), np.mean(lags), 
            np.max(correlations), np.var(correlations), np.var(lags)]

def corelation_lags_measurments_headers(title):
    return [f"{title}_mean_corr", f"{title}_mean_lag", 
            f"{title}_max_corr", f"{title}_var_corr", f"{title}_var_lag"]

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
    return corelation_lags_measurments_headers("chosen_comps") + \
    corelation_lags_measurments_headers("all_max_comp") + \
    corelation_lags_measurments_headers("chosen_max_comp") +\
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

        temporal_max1 = chunk_data[A][4]
        temporal_max2 = chunk_data[B][4]
        temporal_correlations = []
        temporal_lags = []

        if (reader_assumption and action.lower() == "reading" ):
            chosen_max1 = np.max(mat1, axis=0)
            chosen_max2 = np.max(mat2, axis=0)
        else:
            chosen_max1 = chunk_data[A][5]
            chosen_max2 = chunk_data[B][5]
        chosen_correlations = []
        chosen_lags = []

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
            skipReaderAssumption = False
            if ( (not skipReaderAssumption) and action.lower() == "reading" and current_reader == A):
                max_corr, max_ind = get_max_pearson_xcorr_2d(mat2, mat1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
                max_temporal_corr, max_temporal_ind = get_max_xcorr_pearson_1d(temporal_max2, temporal_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
                max_chosen_corr, max_chosen_ind = get_max_xcorr_pearson_1d(chosen_max2, chosen_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
            else:    
                max_corr, max_ind = get_max_pearson_xcorr_2d(mat1, mat2, max_lag_ms,
                                                        downsampleWindowInMs, s, e, action)
                max_temporal_corr, max_temporal_ind = get_max_xcorr_pearson_1d(temporal_max1, temporal_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
                max_chosen_corr, max_chosen_ind = get_max_xcorr_pearson_1d(chosen_max1, chosen_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
            corolations.append(max_corr)
            lags.append(max_ind)

            temporal_correlations.append(max_temporal_corr)
            temporal_lags.append(max_temporal_ind)

            chosen_correlations.append(max_chosen_corr)
            chosen_lags.append(max_chosen_ind)
        
        # tMax = max_lag_ms//downsampleWindowInMs

        # mat1, mat2 = make_vectors_comperable(mat1, mat2)
        # _, lags, corolations = cross_correlation_max_pearson_ndim(mat1, mat2, tMax)

        # temporal_max1, temporal_max2 = make_vectors_comperable(temporal_max1, temporal_max2)
        # temporal_correlations, temporal_lags = cross_correlation_max_pearson(temporal_max1, temporal_max2, tMax)

        # chosen_max1, chosen_max2 = make_vectors_comperable(chosen_max1, chosen_max2)
        # chosen_correlations, chosen_lags = cross_correlation_max_pearson(chosen_max1, chosen_max2, tMax)

        # corolations = np.array(corolations)
        # lags = np.array(lags)
        # temporal_correlations = np.array(temporal_correlations)
        # temporal_lags = np.array(temporal_lags)
        # chosen_correlations = np.array(chosen_correlations)
        # chosen_lags = np.array(chosen_lags)
    
        current_result = [current_session, key, action] + \
            corelation_lags_measurments(corolations, lags) + \
            corelation_lags_measurments(temporal_correlations, temporal_lags) + \
            corelation_lags_measurments(chosen_correlations, chosen_lags) + \
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


def main():
    run_emg_analysis(main_corrolation, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap.csv"))

def main_2():
    run_emg_analysis(main_corrolation_with_mean_comp, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_main_corrolation_with_mean_comp_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_with_mean_of_corr_comp_VAR.csv"))

def main_3():
    run_emg_analysis(main_corrolation_of_single_comps, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_main_corrolation_of_single_comps_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_corrEachComp.csv"))

def main_rms():
    run_emg_analysis(main_corrolation_with_rms_comp, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_main_corrolation_with_rms_comp_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{int(windowForCorrInMs/1000)}_maxlag_{int(max_lag_ms/1000)}_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_with_mean_of_corr_comp_VAR.csv"))