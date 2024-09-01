from SVM.svm_gpu_new import run, load_data
from General.HelperFunctions import GetSessions
from General.init import *
from tqdm import tqdm
from EDF.EdfAnalyzer import window_rms_single
from matplotlib import pyplot as plt
from General.PlotFunctions import clearPlot
from General.CorrolationMethods import  get_max_xcorr_pearson_1d
import numpy as np
from PsychData.LoadChoicesJson import loadChoiceFromSession
import pandas as pd
import os

rms_size_ms = 50
downsampleWindowInMs = 5
windowForCorrInMs = 3000
max_lag_ms = 3000
numOfPixelsForXCorrWindow = windowForCorrInMs//downsampleWindowInMs
numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow
reader_assumption = True


def plot_together(data_a, data_b, session, key):
    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    # Plot data_a in the first subplot
    axs[0].plot(data_a, label='Data A')
    axs[0].set_title('Data A')
    axs[0].set_ylabel('Value')
    axs[0].legend()

    # Plot data_b in the second subplot
    axs[1].plot(data_b, label='Data B', color='orange')
    axs[1].set_title('Data B')
    axs[1].set_ylabel('Value')
    axs[1].legend()

    # Set the x-axis label for the bottom subplot
    axs[1].set_xlabel('Index')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.title(f"{session}_{key}")

    # Show the plot
    plt.show()
    clearPlot()

def corr_func(chunks_a, chunks_b, session):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]

    for key in chunks_a:
        action = key.split("_")[0]
        current_reader = None
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)

        mat1 = window_rms_single(chunks_a[key].data, rms_size_ms)
        mat2 = window_rms_single(chunks_b[key].data, rms_size_ms)

        max_len = min(mat1.shape[0], mat2.shape[0])
        
        corrs, lags = [], []

        for s in range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep):
            e = s + numOfPixelsForXCorrWindow # todo: run again.
            #numOfPixelsForXCorrWindow
            skipReaderAssumption = False
            direction = "bi-directional"
            if ( action.lower() == "reading"):
                direction = "one-direction"

            if ( (not skipReaderAssumption) and action.lower() == "reading" and current_reader == A):
                #todo - fix this is I return to reader's assumption.
                max_corr, max_ind = get_max_xcorr_pearson_1d(mat2, mat1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, direction, weighted_corr=False, lowess=False)
            else:
                max_corr, max_ind = get_max_xcorr_pearson_1d(mat1, mat2, max_lag_ms,
                                                    downsampleWindowInMs, s, e, direction, weighted_corr=False, lowess=False)
            corrs.append(max_corr)
            lags.append(max_ind)
        curr_res = [session, key, action] + [np.nanmean(corrs), np.nanmean(lags)] +\
            [np.nanmax(corrs), np.nanmax(lags)] + [np.nanmean(mat1), np.nanmean(mat2)] +\
            [np.nanmax(mat1), np.nanmax(mat2)]
        results.append(curr_res)
    
    return results

def get_headers():
    return ["session", "key", "action", "mean_corr", "mean_lag", "max_corr", "max_lag", "mean_smile_A", "mean_smile_B", "max_smile_A", "max_smile_B"]


def run_analysis(svm_type="SVC"):
    all_correlations = []
    for session in tqdm(GetSessions(data_path)):
        chunks_a = load_data(session, A, svm_type)
        chunks_b = load_data(session, B, svm_type)
        corr_results = corr_func(chunks_a, chunks_b, session)
        choices = loadChoiceFromSession(data_path, session)
        for i in range(len(corr_results)):
            result = corr_results[i]
            key = result[1]
            key_splits = key.split("_")
            if ( len(key_splits) > 1):
                story_number = key_splits[1]
                if ( story_number != ''):
                    story_number = story_number.split('.')[0]
                    for choice in choices:
                        if ( story_number == choice[0] ):
                            corr_results[i] = result + choice[1:]
                            break
        all_correlations += corr_results 
        
    
    headers = get_headers() + ['whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']   
    df = pd.DataFrame(all_correlations)
    df.columns = headers #+ ["confidence_A", "confidence_B"]
    df.to_csv(os.path.join(data_path,"results_2024", "smiles", "batch1_svr.csv"), index = False)


def plot_sessions(svm_type="SVC"):
    for session in GetSessions(data_path)[2:]:
        chunks_a = load_data(session, A, svm_type)
        chunks_b = load_data(session, B, svm_type)

        for key in chunks_a:
            data_a = window_rms_single(chunks_a[key].data, rms_size_ms)
            data_b = window_rms_single(chunks_b[key].data, rms_size_ms)
            plot_together(data_a, data_b, session, key)

#plot_sessions("SVR")
run_analysis("SVR")