import os
import numpy as np
from General.init import *
import pandas as pd
from General.StatisticsFunctions import cross_rqa
from General.CorrolationMethods import dtai_dist_dtw_pearson_over_seconds, get_max_xcorr_2d, get_max_xcorr_1d
import re
from ComponentAnalysis.ComponentExtractor import GetCorrugatorComponent, GetNoseComponent, GetSideEyeComponent, GetZygomaticusComponent
from Analysis.analysis_runner import run_emg_analysis

rms_size_ms = 100
downsampleWindowInMs = 20
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

def wrap_in_default_headers(headers):
    return ["session", "key","action"] + headers + ['whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']

def get_headers():
    return corelation_lags_measurments_headers("chosen_comps") + \
    corelation_lags_measurments_headers("all_max_comp") + \
    corelation_lags_measurments_headers("chosen_max_comp") +\
    ["dtw_pearson_over_seconds", "dtw_dist", "temporal_dtw_pearson_over_seconds", "temporal_dtw_dist", "chosen_dtw_pearson_over_seconds", "chosen_dtw_dist"]

def main_corrolation(current_session, independentComponents, chunks):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]
    for key in chunks[0]:
        chunk_data = [[],[]]

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
        action = key.split("_")[0]
        corolations = []
        lags = []
        current_reader = A
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow # todo: run again.
            #numOfPixelsForXCorrWindow
            if ( action.lower() == "reading" and current_reader == A):
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
            [chosen_dtw_pearson_over_seconds, chosen_dist]
        
        results.append(current_result)
    
    return results

def regex_split_to_numbers_with_all_possible_seprators(s):
    if pd.isna(s):
        return []
    return [ int(x) for x in re.split(r"[^0-9.]", s) if x != "" ]

review_df = None

def get_review_data(session):
    global review_df
    if review_df is None:
        review_df = pd.read_csv(os.path.join(data_path, "batch1_review.csv"))
    
    row = review_df[review_df["session_name"] == session]
    bad_electrodes = [row[f"bad_electrodes_{p}"].values[0] for p in ["A", "B"]]
    #convert into two lists:
    bad_electrodes = [regex_split_to_numbers_with_all_possible_seprators(x) for x in bad_electrodes]
    bad_electrodes = [np.array(x)-1 for x in bad_electrodes]
    is_dark = row["is_dark"].values[0] == "Yes"

    return bad_electrodes, is_dark


def get_special_component_indexes(w, list_of_faulty_channels):
    #should replace with HILA's code.
    special_comps_getters = [GetNoseComponent, GetSideEyeComponent, GetZygomaticusComponent, GetCorrugatorComponent]
    return [getter(w, list_of_faulty_channels) for getter in special_comps_getters]

def main():
    run_emg_analysis(main_corrolation, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, wrap_in_default_headers(get_headers()), os.path.join(data_path,"results_2024", f"RMS_{rms_size_ms}_downsample_{downsampleWindowInMs}_window_{windowForCorrInMs}_step_{numOfPixelsForXCorrStep}_pixels_just_xcorr_plus_temporal_hilas_electordes_try_fix_overlap.csv"))
