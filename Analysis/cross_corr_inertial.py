from General.init import *
from Analysis.analysis_runner import run_inertial_analysis
import os
from General.CorrolationMethods import dtai_dist_dtw_pearson_over_seconds, get_max_xcorr_1d
import numpy as np

downsample_window_in_ms = 20
rms_window_in_ms = 100
windowForCorrInMs = 1000
numOfPixelsForXCorrWindow = windowForCorrInMs//downsample_window_in_ms
numOfPixelsForXCorrStep = (windowForCorrInMs//2) // downsample_window_in_ms

def wrap_in_default_headers(headers):
    return ["session", "key","action"] + headers + ['whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']

def get_headers():
    return corelation_lags_measurments_headers("inertial_max_comp") + \
        corelation_lags_measurments_headers("inertial_rms_comp") + \
        ["inertial_dtw_max_comp", "inertial_dtw_dist_max_comp"] + \
        ["inertial_dtw_rms_comp", "inertial_dtw_dist_rms_comp"]


def corelation_lags_measurments(correlations, lags):
    return [np.arctan(np.mean(correlations)), np.mean(lags), 
            np.arctan(np.max(correlations))]

def corelation_lags_measurments_headers(title):
    return [f"{title}_mean_corr", f"{title}_mean_lag", 
            f"{title}_max_corr"]

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

        mat1_max = chunk_data[A][0]
        mat2_max = chunk_data[B][0]

        max_comp_max_corrs = np.array([])
        max_comp_max_corr_lags = np.array([])

        mat1_rms = chunk_data[A][1]
        mat2_rms = chunk_data[B][1]

        rms_comp_max_corrs = np.array([])
        rms_comp_max_corr_lags = np.array([])

        max_lag_ms = windowForCorrInMs
        #jumps = numOfPixelsForXCorrWindow
        if len(mat1_max.shape) == 1:
            max_len = min(mat1_max.shape[0], mat2_max.shape[0])
        else:
            max_len = min(mat1_max.shape[1], mat2_max.shape[1])
        
        action = key.split("_")[0]
        current_reader = A
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow
            if ( action.lower() == "reading" and current_reader == A):
                max_comp_max_corr, max_comp_max_lag = get_max_xcorr_1d(mat2_max, mat1_max, max_lag_ms,
                                                    downsample_window_in_ms, s, e, action)
                rms_comp_max_corr, rms_comp_max_lag = get_max_xcorr_1d(mat2_rms, mat1_rms, max_lag_ms,
                                                    downsample_window_in_ms, s, e, action)
            else:    
                max_comp_max_corr, max_comp_max_lag = get_max_xcorr_1d(mat1_max, mat2_max, max_lag_ms,
                                                    downsample_window_in_ms, s, e, action)
                rms_comp_max_corr, rms_comp_max_lag = get_max_xcorr_1d(mat1_rms, mat2_rms, max_lag_ms,
                                                    downsample_window_in_ms, s, e, action)

            max_comp_max_corrs = np.append(max_comp_max_corrs, max_comp_max_corr)
            max_comp_max_corr_lags = np.append(max_comp_max_corr_lags, max_comp_max_lag)

            rms_comp_max_corrs = np.append(rms_comp_max_corrs, rms_comp_max_corr)
            rms_comp_max_corr_lags = np.append(rms_comp_max_corr_lags, rms_comp_max_lag)



        dtw_max_comp, dist_max_comp = dtai_dist_dtw_pearson_over_seconds(mat1_max, 
                                                    mat2_max,
                                                    max_lag_ms // downsample_window_in_ms)
        
        dtw_rms_comp, dist_rms_comp = dtai_dist_dtw_pearson_over_seconds(mat1_rms,
                                                    mat2_rms,
                                                    max_lag_ms // downsample_window_in_ms)

        current_result = [current_session, key, action] + \
            corelation_lags_measurments(max_comp_max_corrs, max_comp_max_corr_lags) + \
            corelation_lags_measurments(rms_comp_max_corrs, rms_comp_max_corr_lags) + \
            [dtw_max_comp, dist_max_comp] + \
            [dtw_rms_comp, dist_rms_comp]
        
        results.append(current_result)
    
    return results

def main():
    headers = wrap_in_default_headers(get_headers())
    analysis_path = os.path.join(data_path, "results_2024")
    csv_out_path = os.path.join(analysis_path, "inertial_correlations_just_acc_corrected.csv")
    run_inertial_analysis(main_corrolation, rms_window_in_ms, downsample_window_in_ms, windowForCorrInMs, numOfPixelsForXCorrStep, headers, csv_out_path)