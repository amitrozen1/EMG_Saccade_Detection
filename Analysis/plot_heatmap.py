
import numpy as np
from General.init import *
from General.CorrolationMethods import get_max_xcorr_2d, get_max_xcorr_1d, get_max_pearson_xcorr_2d, get_max_xcorr_pearson_1d
from matplotlib.colors import Normalize
import seaborn as sns
import matplotlib.pyplot as plt
from General.PlotFunctions import savePlot, clearPlot
from Analysis.analysis_runner import run_emg_analysis
from General.HelperFunctions import stack_rows_with_pad
from scipy.stats import zscore

rms_size_ms = 100
downsampleWindowInMs = 20
windowForCorrInMs = 1000
numOfPixelsForXCorrWindow = windowForCorrInMs//downsampleWindowInMs

#todo: delete:
numOfPixelsForXCorrStep = (windowForCorrInMs//2) // downsampleWindowInMs

def plot_heatmap(vectors, key, extratitle):
    vectors = zscore(vectors, axis=0)
    sns.heatmap(vectors, cmap='viridis_r')
    plt.xlabel("lag")
    plt.ylabel("epoch")
    plt.title(f"{key}_{extratitle}")
    plt.show()
    clearPlot()
    


def main_corrolation(current_session, independentComponents, chunks):
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

        all_2d_corr_vecs = np.array([])

        temporal_max1 = chunk_data[A][4]
        temporal_max2 = chunk_data[B][4]

        all_temporal_corr_vecs = np.array([])

        chosen_max1 = chunk_data[A][5]
        chosen_max2 = chunk_data[B][5]

        all_chosen_temporal_corr_vecs = np.array([])

        max_lag_ms = windowForCorrInMs
        #jumps = numOfPixelsForXCorrWindow
        max_len = min(mat1.shape[1], mat2.shape[1])
        action = key.split("_")[0]

        current_reader = A
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep)): #not using jumps.
            e = s + numOfPixelsForXCorrWindow
            
            skipReaderAssumption = True
            if ( (not skipReaderAssumption) and action.lower() == "reading" and current_reader == A):
                xcorrVec2d = get_max_xcorr_2d(mat2, mat1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action, shouldReturnXcorrVec=True)
                xcorr_vec_temporal_all_max = get_max_xcorr_1d(temporal_max2, temporal_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action, shouldReturnXcorrVec=True)
                xcorr_vec_temporal_chosen_max = get_max_xcorr_1d(chosen_max2, chosen_max1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action, shouldReturnXcorrVec=True)
            else:    
                xcorrVec2d = get_max_xcorr_2d(mat1, mat2, max_lag_ms,
                                                        downsampleWindowInMs, s, e, action, shouldReturnXcorrVec=True)
                xcorr_vec_temporal_all_max = get_max_xcorr_1d(temporal_max1, temporal_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action, shouldReturnXcorrVec=True)
                xcorr_vec_temporal_chosen_max = get_max_xcorr_1d(chosen_max1, chosen_max2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action, shouldReturnXcorrVec=True)
        
            # todo: fix the bug here.
            all_2d_corr_vecs = stack_rows_with_pad(all_2d_corr_vecs, xcorrVec2d)
            all_temporal_corr_vecs = stack_rows_with_pad(all_temporal_corr_vecs, xcorr_vec_temporal_all_max)
            all_chosen_temporal_corr_vecs = stack_rows_with_pad(all_chosen_temporal_corr_vecs, xcorr_vec_temporal_chosen_max)
    
        #plot sns heatmap for all_2d_corr_vecs, all_temporal_corr_vecs, all_chosen_temporal_corr_vecs
        plot_heatmap(all_2d_corr_vecs, key, "all_2d_corr_vecs")
        plot_heatmap(all_temporal_corr_vecs, key, "all_temporal_corr_vecs")
        plot_heatmap(all_chosen_temporal_corr_vecs, key, "all_chosen_temporal_corr_vecs")

    return []

def main():
    run_emg_analysis(main_corrolation, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, [], None)