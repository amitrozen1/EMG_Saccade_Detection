import os
import numpy as np
from General.init import *
import pandas as pd
from General.StatisticsFunctions import cross_rqa
from General.CorrolationMethods import dtai_dist_dtw_pearson_over_seconds, get_max_xcorr_2d, get_max_xcorr_1d, get_max_pearson_xcorr_2d, get_max_xcorr_pearson_1d
import re
from ComponentAnalysis.ComponentExtractor import get_special_component_names
from Analysis.analysis_runner import run_emg_analysis

rms_size_ms = 100
downsampleWindowInMs = 50
windowForCorrInMs = 1000
numOfPixelsForXCorrWindow = windowForCorrInMs//downsampleWindowInMs

#todo: delete:
numOfPixelsForXCorrStep = (windowForCorrInMs//2) // downsampleWindowInMs

def get_rqa_results(mat1, mat2):
    rqa = [cross_rqa(mat1[i], mat2[i]) for i in range(mat1.shape[0])]
    mean_recurrence_rate = np.mean([r.recurrence_rate for r in rqa])
    mean_determinism = np.mean([r.determinism for r in rqa])
    mean_longest_diagonal_line = np.mean([r.longest_diagonal_line for r in rqa])/mat1.shape[1]
    mean_avg_diagonal_line = np.mean([r.average_diagonal_line for r in rqa])
    mean_laminarity = np.mean([r.laminarity for r in rqa])
    mean_trapping_time = np.mean([r.trapping_time for r in rqa])
    mean_number_of_diagonal_lines = np.mean([r.number_of_diagonal_lines(2) for r in rqa]) / (mat1.shape[0] * mat1.shape[1])
    mean_entropy_diagonal_lines = np.mean([r.entropy_diagonal_lines for r in rqa])
    mean_entropy_vertical_lines = np.mean([r.entropy_vertical_lines for r in rqa])

    return [mean_recurrence_rate, mean_determinism, mean_longest_diagonal_line, mean_avg_diagonal_line, mean_laminarity, mean_trapping_time, mean_number_of_diagonal_lines, mean_entropy_diagonal_lines, mean_entropy_vertical_lines]


def corelation_lags_measurments(correlations, lags):
    return [np.arctan(np.mean(correlations)), np.mean(lags), 
            np.arctan(np.max(correlations))]

def corelation_lags_measurments_headers(title):
    return [f"{title}_mean_corr", f"{title}_mean_lag", 
            f"{title}_max_corr"]

def add_mean_component_headers():
    return [f"mean_{comp}_{p}" for comp in get_special_component_names() for p in ["A", "B"]]
    #['mean_Nose_A', 'mean_Nose_B', 'mean_Eye_A', 'mean_Eye_B', 'mean_Zygomaticus_A', 'mean_Zygomaticus_B', 'mean_Corrugator_A', 'mean_Corrugator_B']

def add_mean_components(mat_a, mat_b):
    mats = [np.mean(mat_a, axis=1), np.mean(mat_b, axis=1)]
    size = len(mats[0])
    return [mats[i][p] for i in range(size) for p in [A,B]]

def wrap_in_default_headers(headers):
    return ["session", "key","action"] + headers + ['whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']

def get_headers():
    return corelation_lags_measurments_headers("chosen_comps") + \
    corelation_lags_measurments_headers("all_max_comp") + \
    corelation_lags_measurments_headers("chosen_max_comp") +\
    ["dtw_pearson_over_seconds", "dtw_dist", "temporal_dtw_pearson_over_seconds", "temporal_dtw_dist", "chosen_dtw_pearson_over_seconds", "chosen_dtw_dist"] +\
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

        if (False): #action.lower() == "reading" ):
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

        if (False): #action.lower() == "reading" ):
            chosen_max1 = np.max(mat1, axis=0)
            chosen_max2 = np.max(mat2, axis=0)
        else:
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
            skipReaderAssumption = False
            if ( (not skipReaderAssumption) and action.lower() == "reading" and current_reader == A):
                max_corr, max_ind = get_max_xcorr_2d(mat2, mat1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
                max_temporal_corr, max_temporal_ind = get_max_xcorr_1d(temporal_max2, temporal_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
                max_chosen_corr, max_chosen_ind = get_max_xcorr_1d(chosen_max2, chosen_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
            else:    
                max_corr, max_ind = get_max_xcorr_2d(mat1, mat2, max_lag_ms,
                                                        downsampleWindowInMs, s, e, action)
                max_temporal_corr, max_temporal_ind = get_max_xcorr_1d(temporal_max1, temporal_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
                max_chosen_corr, max_chosen_ind = get_max_xcorr_1d(chosen_max1, chosen_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
            corolations.append(max_corr)
            lags.append(max_ind)

            temporal_correlations.append(max_temporal_corr)
            temporal_lags.append(max_temporal_ind)

            chosen_correlations.append(max_chosen_corr)
            chosen_lags.append(max_chosen_ind)


        # last_30_percent_start_index = int(len(corolations) * 0.7)
            
        # used for:
        #   np.arctan(np.mean(corolations[last_30_percent_start_index:])), 
        #   np.mean(lags[last_30_percent_start_index:]),

        dtw_pearson_over_seconds, dist = dtai_dist_dtw_pearson_over_seconds(mat1, 
                                                    mat2,
                                                    max_lag_ms // downsampleWindowInMs)
        
        temporal_dtw_pearson_over_seconds, temporal_dist = dtai_dist_dtw_pearson_over_seconds(temporal_max1,
                                                    temporal_max2,
                                                    max_lag_ms // downsampleWindowInMs)
        
        chosen_dtw_pearson_over_seconds, chosen_dist = dtai_dist_dtw_pearson_over_seconds(chosen_max1,
                                                    chosen_max2,
                                                    max_lag_ms // downsampleWindowInMs)          
                                                                                

        # max_index_for_lag_norm = np.argmax(corolations)
            # used for:
            # np.arctan(corolations[max_index_for_lag_norm]/lags[max_index_for_lag_norm]),
        
        corolations = np.array(corolations)
        lags = np.array(lags)
        temporal_correlations = np.array(temporal_correlations)
        temporal_lags = np.array(temporal_lags)
        chosen_correlations = np.array(chosen_correlations)
        chosen_lags = np.array(chosen_lags)

        # all_normed_max_corr = np.arctan(np.max(corolations/(lags+0.001)))
        current_result = [current_session, key, action] + \
            corelation_lags_measurments(corolations, lags) + \
            corelation_lags_measurments(temporal_correlations, temporal_lags) + \
            corelation_lags_measurments(chosen_correlations, chosen_lags) + \
            [dtw_pearson_over_seconds, dist] + \
            [temporal_dtw_pearson_over_seconds, temporal_dist] + \
            [chosen_dtw_pearson_over_seconds, chosen_dist] + \
            add_mean_components(mat1, mat2)
        
        results.append(current_result)
    
    return results

def main():
    run_emg_analysis(main_corrolation, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{windowForCorrInMs}_step_{numOfPixelsForXCorrStep}_pixels_new_comps+means.csv"))
