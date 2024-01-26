import os
from General.HelperFunctions import GetSessions, cleanSpace, reshape_num_channels, RemoveRowFromMatrix
from EDF.ComponentValidator import GetNoiseComponentsBySort
import numpy as np
from EDF.SharedIcaRunner import LoadComponents
from scipy.stats import zscore
from matplotlib.colors import Normalize
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from General import HtmlTablePlot
import pandas as pd
from PsychData.AnalyzeQuestionnaire import GetStatsRowsFromSession, GetStatsHeaderWithouSession
from PsychData.LoadChoicesJson import loadChoiceFromSession
from sklearn.metrics import classification_report
from tqdm import tqdm
from General.StatisticsFunctions import cross_rqa
from EDF.BlankRemover import EmgBlankRemover
from General.CorrolationMethods import get_peak_xcorr_generic, dtai_dist_dtw_pearson, dtai_dist_dtw_pearson_over_seconds, dtai_dist_dtw_pearson_over_seconds_blank_removed, get_max_xcorr_2d
import cupy as xp


#data_path = r'/mnt/c/Liron/DataEmg/Done'
data_path = r"C:\Liron\DataEmg\Try"
tempfilename = "lastplot.png"

def SavePlotToTempFile():
    plt.savefig(tempfilename)
    return tempfilename


num_channels = 16
A = 0
B = 1
downsampleWindowInMs = 50 # 50 == 20HZ. 20MS == 50HZ. 10 == 100HZ. 
orig_freq = 4000

freq = orig_freq // (4*downsampleWindowInMs)
windowForCorrInMs = 1000
numOfPixelsForXCorrWindow = windowForCorrInMs//downsampleWindowInMs

#todo: delete:
numOfPixelsForXCorrStep = 1000//downsampleWindowInMs #50 // downsampleWindowInMs # 100ms


def GetElectrodeFromComponent(data_path, session, component):
    w_path = os.path.join(data_path, session, "weightsAfterSharedICA.csv")
    w = np.loadtxt(w_path, delimiter=",")
    components_to_electrodes = np.argmax(np.absolute(np.linalg.inv(w)), axis=0)[::-1]
    return components_to_electrodes[int(component)]

def GenerateExcelHeader(statsFileName):
    header = ["session", "story", "isChoice", "rtChoice", "isOther", "ChoseTogether"]
    header+= [f"maxCorr_{i+1}" for i in range(0,16)]
    header += ["meanCorr", "maxComponentCorr", "mostSynchedElectrode"]
    statsHeaders = GetStatsHeaderWithouSession(statsFileName)
    for p in ["A", "B"]:
        header += [f"{x}_{p}" for x in statsHeaders]
    return header

def SessionToCorrelateOneComponentOverLagsNoDirectionsResultHeaders():
    return ["session", "key","action","mean_corr_equal_lag", "equal_lag",
            "mean_chosen_lag_ms", "new_corr", "mean_corr","mean_var_corr",
             "1-mean_determinism", "1-mean_recurrence_rate", "1-mean_longest_diagonal_line", "1-mean_avg_diagonal_line"]

def SessionToCorrelateOverLagsNoDirections_PEAK_Headers():
    return ["session", "key","action","mean_peak_corr", "var_peak_corr",
            "mean_peak_lag", "var_peak_lag", "dtw_pearson_seconds_norm", "dtw_norm_blank_removed"]


def GetMatrixForCorr(mat, s=0, e=0):
    arr = []
    if ( e == 0 ):
        arr = mat
    else:
        arr = mat[:, s:e]
    
    m =  np.mean(arr, axis=1)
    return np.array([arr[i, :] - m[i] for i in range(len(m))])

def get_matrix_reduced_part_of_mean(a, k=0.3):
    return ((a.transpose() - (k * np.mean(a, axis=1)))/np.std(a,axis=1)).transpose()


def RemoveNoiseComponentsFromComponentsMatrix(indenpendentComponents, session):
    rows_to_remove = GetNoiseComponentsBySort(data_path, session) #returns sorted.
    for p in [A,B]:
        rows_removed = 0
        for row in rows_to_remove:
            # comes sorted ? 
            indenpendentComponents[p] = RemoveRowFromMatrix(indenpendentComponents[p] , row - rows_removed)
            rows_removed = rows_removed + 1 #so you delete the right row.

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

def get_rqa_results_headers():
    return ["mean_recurrence_rate", "mean_determinism", "mean_longest_diagonal_line", "mean_avg_diagonal_line", "mean_laminarity", "mean_trapping_time", "mean_number_of_diagonal_lines", "mean_entropy_diagonal_lines", "mean_entropy_vertical_lines"]

def get_headers_27_08():
    return ["session", "key","action"] + ["mean_corr", "mean_lag", "max_corr", "max_lag", "min_corr", "min_lag", "norm_max_corr", "max_corr_with_lag_norm", "mean_last_30_corr", "mean_last_30_lag", "dtw_pearson_seconds_norm", "dtw_dist"]

def get_headers_23_08():
    return ["session", "key","action", "mean_A", "mean_B", "var_A", "var_B",
            "mean_peak_corr", "var_peak_corr", "mean_peak_lag", "var_peak_lag", "mean_pearson_peak_corr", "var_pearson_peak_corr",
            "mean_pearson_peak_lag", "var_pearson_peak_lag", "dtw_pearson_seconds_norm", 
            "dtw_dist"] + get_rqa_results_headers()


def SessionCorrelation27_08(current_session, independentComponents, chunks):
    results = []
    reading_order = [A, A, B, B, A, A, B, B]
    for key in chunks[0]:
        chunk_data = [[],[]]

        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]

        mat1 = chunk_data[A]
        mat2 = chunk_data[B]
        max_lag_ms = windowForCorrInMs
        jumps = numOfPixelsForXCorrWindow
        max_len = min(mat1.shape[1], mat2.shape[1])
        action = key.split("_")[0]
        corolations = []
        lags = []
        current_reader = A
        if ( action.lower() == "reading" ):
            current_reader = reading_order.pop(0)

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , jumps)):
            e = s+numOfPixelsForXCorrWindow
            max_corr = 0
            max_ind = 0
            if ( action.lower() == "reading" and current_reader == A):
                max_corr, max_ind = get_max_xcorr_2d(mat2, mat1, max_lag_ms,
                                                    downsampleWindowInMs, s, e, action)
            else:    
                max_corr, max_ind = get_max_xcorr_2d(mat1, mat2, max_lag_ms,
                                                        downsampleWindowInMs, s, e, action)
            corolations.append(max_corr)
            lags.append(max_ind)

        last_30_percent_start_index = int(len(corolations) * 0.7)

        dtw_pearson_over_seconds, dist = dtai_dist_dtw_pearson_over_seconds(mat1, 
                                                    mat2,
                                                    max_lag_ms // downsampleWindowInMs)
        max_index_for_lag_norm = np.argmax(corolations)
        corolations = np.array(corolations)
        lags = np.array(lags)
        all_normed_max_corr = np.arctan(np.max(corolations/(lags+0.001)))
        current_result = [current_session, key, action,
                          np.arctan(np.mean(corolations)), np.mean(lags), 
                          np.arctan(np.max(corolations)), np.max(lags),
                          np.arctan(np.min(corolations)), np.min(lags),
                          np.arctan(corolations[max_index_for_lag_norm]/lags[max_index_for_lag_norm]),
                          all_normed_max_corr,
                          np.arctan(np.mean(corolations[last_30_percent_start_index:])), 
                          np.mean(lags[last_30_percent_start_index:]),
                          dtw_pearson_over_seconds, dist]
        results.append(current_result)
    
    return results


def SessionCorrelation23_08(current_session, independentComponents, chunks): #, non_zscore_components):
    RemoveNoiseComponentsFromComponentsMatrix(independentComponents, current_session)
    
    results = []
    for key in chunks[0]:
        chunk_data = [[],[]]
        non_zscore_chunks = [[],[]]

        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]

        mat1 = chunk_data[A]
        mat2 = chunk_data[B]

        mean_A_non_zscore = np.mean(mat1)
        mean_B_non_zscore = np.mean(mat2)
        var_A_non_zscore = np.var(mat1)
        var_B_non_zscore = np.var(mat2)

        max_len = min(mat1.shape[1], mat2.shape[1])
        max_corr_all = np.array([])
        max_corr_ind = np.array([])
        pearson_max_corr_all = np.array([])
        pearson_max_corr_ind = np.array([])

        max_lag_ms = windowForCorrInMs
        jumps = numOfPixelsForXCorrWindow

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , jumps)):
            e = s+numOfPixelsForXCorrWindow
            max_corr, max_ind = get_peak_xcorr_generic(mat1, mat2, max_lag_ms,
                                                        downsampleWindowInMs, s, e, "mult")
            max_corr_all = np.append(max_corr_all, max_corr)
            max_corr_ind = np.append(max_corr_ind, max_ind)

            pearson_max_corr, pearson_max_ind = get_peak_xcorr_generic(mat1, mat2, max_lag_ms,
                                                        downsampleWindowInMs, s, e, "pearson")
            
            pearson_max_corr_all = np.append(pearson_max_corr_all, pearson_max_corr)
            pearson_max_corr_ind = np.append(pearson_max_corr_ind, pearson_max_ind)
    
        dtw_pearson_over_seconds, dist = dtai_dist_dtw_pearson_over_seconds(mat1, 
                                                            mat2, 
                                                            numOfPixelsForXCorrWindow)
        
        pearson_max_corr_all = np.arctan(pearson_max_corr_all)

        current_result = [current_session, key, key.split("_")[0],
                          mean_A_non_zscore, mean_B_non_zscore,
                          var_A_non_zscore, var_B_non_zscore,
                          np.mean(max_corr_all), np.var(max_corr_all),
                          np.mean(max_corr_ind), np.var(max_corr_ind), 
                          np.mean(pearson_max_corr_all), np.var(pearson_max_corr_all),
                          np.mean(pearson_max_corr_ind), np.var(pearson_max_corr_ind),
                          dtw_pearson_over_seconds, dist]

        rqa_results = get_rqa_results(mat1, mat2)
        current_result += rqa_results

        results.append(current_result)

    return results      
        


def SessionToCorrelateOverLagsNoDirections_PEAK(current_session, independentComponents, chunks, overlap=False):
    rows_to_remove = GetNoiseComponentsBySort(data_path, current_session) #returns sorted.
    for p in [A,B]:
        rows_removed = 0
        for row in rows_to_remove:
            # comes sorted ? 
            independentComponents[p] = RemoveRowFromMatrix(independentComponents[p] , row - rows_removed)
            rows_removed = rows_removed + 1 #so you delete the right row.

    blank_remover = EmgBlankRemover(independentComponents, chunks)
    
    results = []
    for key in chunks[0]:
        chunk_data = [[],[]]
        

        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]
        
        mat1 = chunk_data[A]
        mat2 = chunk_data[B]

        jumps = numOfPixelsForXCorrWindow
        if ( overlap ):
            jumps = jumps//2

        max_len = min(mat1.shape[1], mat2.shape[1])
        max_corr_all = np.array([])
        max_corr_ind = np.array([])
        max_lag_ms = windowForCorrInMs

        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , jumps)):
            e = s+numOfPixelsForXCorrWindow
            max_corr, max_ind = get_peak_xcorr(mat1, mat2, max_lag_ms,numOfPixelsForXCorrStep, 
                                               downsampleWindowInMs, s, e)
            max_corr_all = np.append(max_corr_all, max_corr)
            max_corr_ind = np.append(max_corr_ind, max_ind)
        
        #dtw, _ = min_dtw(mat1, mat2)
        #dtw = dtai_dist_dtw(mat1, mat2, max_lag_ms)
        #dtw_pearson = dtai_dist_dtw_pearson(mat1, mat2, max_lag_ms//downsampleWindowInMs)
        dtw_pearson_over_seconds, dist = dtai_dist_dtw_pearson_over_seconds(mat1, 
                                                                      mat2, 
                                                                      numOfPixelsForXCorrWindow)
        
        dtw_norm_sec_blank_removed = dtai_dist_dtw_pearson_over_seconds_blank_removed(mat1, 
                                                                                      mat2, 
                                                                                      numOfPixelsForXCorrWindow, 
                                                                                      blank_remover)

        current_result = [current_session, key, key.split("_")[0],
                          np.mean(max_corr_all), np.var(max_corr_all),
                          np.mean(max_corr_ind), np.var(max_corr_ind),
                          dtw_pearson_over_seconds, dtw_norm_sec_blank_removed]
    
        results.append(current_result)

    return results



def SessionToCorrelateOneComponentOverLagsNoDirections(current_session, independentComponents, chunks, overlap=False):
    '''
    return a list of - [["key", "mean_corr","mean_var_corr", "4_highest_mean", "max_corr_comp", "max_comp"]]
    '''
    
    rows_to_remove = GetNoiseComponentsBySort(data_path, current_session) #returns sorted.
    for p in [A,B]:
        rows_removed = 0
        for row in rows_to_remove:
            # comes sorted ? 
            independentComponents[p] = RemoveRowFromMatrix(independentComponents[p] , row - rows_removed)
            rows_removed = rows_removed + 1 #so you delete the right row.

    #blank_remover = EmgBlankRemover(independentComponents, chunks)
    
    results = []
    for key in chunks[0]:
        chunk_data = [[],[]]
        
        extra_offset = [0, numOfPixelsForXCorrWindow]

        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]
        
        mat1 = chunk_data[A] #get_matrix_reduced_part_of_mean(chunk_data[A])
        mat2 = chunk_data[B] #get_matrix_reduced_part_of_mean(chunk_data[B])

        maxValueForKey = np.array([])
        max_xcorr_ind = np.array([])
        mean_and_then_max_xcorr = np.array([])
        mean_and_then_max_xcorr_ind = np.array([])

        jumps = numOfPixelsForXCorrWindow
        if ( overlap ):
            jumps = jumps//2
        
        max_len = min(mat1.shape[1], mat2.shape[1])
        for i, s in enumerate(range(0, max_len - numOfPixelsForXCorrWindow , jumps)):
            e = s+numOfPixelsForXCorrWindow
            corrmat = np.array([])
            for lag_ind in np.arange(0,11): #00 - 1500 ms lag = range(0,16) 
                #the sliding window (that slides through A) and the window that moves in matB are the same?
                lag = int(lag_ind * numOfPixelsForXCorrStep) # 100 ms at a lag
                vec = np.empty((mat1.shape[0],0), float)
                mat1_in_window = mat1[:, s:e]
                if ( e + lag < mat2.shape[1]):
                    mat2_in_window = mat2[:, (s+lag):(e+lag)]
                    #if ( blank_remover.shouldAnalyzeCurrentWindow([mat1_in_window, mat2_in_window])):
                    vec = np.mean((mat1_in_window)*(mat2_in_window), axis=1)
                if ( s - lag >= 0 and e-lag > 0):
                    mat2_in_window = mat2[:, (s-lag):(e-lag)]
                    #if ( blank_remover.shouldAnalyzeCurrentWindow([mat1_in_window, mat2_in_window])):
                    vec = np.append(vec, np.mean((mat1_in_window)*(mat2_in_window), axis=1))
                if ( vec.any()):
                    corrmat = np.append(corrmat, vec)
            if ( corrmat.any() ):
                corrmat = reshape_num_channels(corrmat, mat1.shape[0])
                maxValueForKey = np.append(maxValueForKey, np.max(corrmat, axis=1))
                mean_columns = np.mean(corrmat, axis=0)
                mean_and_then_max_xcorr = np.append(mean_and_then_max_xcorr, np.max(mean_columns))
                mean_and_then_max_xcorr_ind = np.append(mean_and_then_max_xcorr_ind, (np.argmax(mean_columns)//2)*numOfPixelsForXCorrStep)
                max_xcorr_ind = np.append(max_xcorr_ind, (np.argmax(corrmat, axis=1)//2)*numOfPixelsForXCorrStep)
        
        maxValueForKey = reshape_num_channels(maxValueForKey, mat1.shape[0]) #16xNumWindows
        max_xcorr_ind = reshape_num_channels(max_xcorr_ind, mat1.shape[0]) #16xNumWindows

        new_corr = np.arctan(np.mean(np.max(maxValueForKey, axis=0)))
        maxValueCorr = np.mean(maxValueForKey, axis=1) #16x1
        maxValueForKeyVar = np.arctan(np.mean(np.var(maxValueForKey, axis=1)))
        

        electrode = GetElectrodeFromComponent(data_path, current_session, np.argmax(maxValueCorr))

        current_result = [current_session, key, key.split("_")[0], np.mean(mean_and_then_max_xcorr), 
                          np.mean(mean_and_then_max_xcorr_ind),
                          np.mean(max_xcorr_ind), new_corr,  np.mean(maxValueCorr), maxValueForKeyVar]

        #rqa:
        rqa = [cross_rqa(mat1[i], mat2[i]) for i in range(mat1.shape[0])]
        mean_determinism = np.mean([r.determinism for r in rqa])
        mean_recurrence_rate = np.mean([r.recurrence_rate for r in rqa])
        mean_longest_diagonal_line = np.mean([r.longest_diagonal_line for r in rqa])/mat1.shape[1]
        mean_avg_diagonal_line = np.mean([r.average_diagonal_line for r in rqa])/mat1.shape[1]
        current_result += [1 - mean_determinism, 1 - mean_recurrence_rate, 1 - mean_longest_diagonal_line, 1 - mean_avg_diagonal_line]

        results.append(current_result)
    
    return results


def SessoinToHtmlCorrAndPlot(current_session, independentComponents, chunks ):
    output_dic = HtmlTablePlot.initHtmlDictionary(["session", "key", "lag", "meanCorr", "maxMeanCompCor", "maxMeanCompArg","plot"])
    print("Go over chunks")
    for key in chunks[0]:
        chunk_data = [[],[]]
        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time)
            chunk_end = int(chunk.End.Time)
            chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]
        
        for lag_ind in np.arange(0,11): #00 - 1000 ms lag
            lag = lag_ind * numOfPixelsForXCorrWindow # 100 ms at a lag
            end = min(chunk_data[A].shape[1], chunk_data[B].shape[1])
            mat1 = chunk_data[B][:, :end-lag]
            mat2 = chunk_data[A][:, lag:end]

            corrMat = np.array([])
            shape1 = min(mat1.shape[1], mat2.shape[1])
            for s in range(0, shape1 - numOfPixelsForXCorrWindow , numOfPixelsForXCorrWindow):
                e = s + numOfPixelsForXCorrWindow
                #vec = np.sum((mat1[:, s:e] * mat2[:, s:e])/(e-s), axis=1)
                vec = np.array([pearsonr(mat1[i, s:e], mat2[i, s:e])[0] for i in range(mat1.shape[0])])
                corrMat = np.append(corrMat, vec)
            
            corrMat = np.reshape(corrMat, (num_channels, len(corrMat)//num_channels))
            matMeanCorr = np.mean(corrMat)
            meanByRows = np.mean(corrMat, axis=1)
            maxCompMean = max(meanByRows)
            maxCompArgMean = np.argmax(meanByRows)

            norm = Normalize(vmin=0, vmax=0.5)
            sns.heatmap(corrMat, norm=norm, cmap='viridis_r')
            plt.xlabel("Time")
            plt.ylabel("Components")
            plt.title(f"{key} A synchronize B, lag: {lag}")
            HtmlTablePlot.addRow(output_dic, [HtmlTablePlot.toTextHtmlCol(current_session), 
                HtmlTablePlot.toTextHtmlCol(key), HtmlTablePlot.toTextHtmlCol(lag),
                HtmlTablePlot.toTextHtmlCol(matMeanCorr), HtmlTablePlot.toTextHtmlCol(maxCompMean),
                HtmlTablePlot.toTextHtmlCol(maxCompArgMean) ,HtmlTablePlot.toImageHtmlColFromPath(SavePlotToTempFile())])
            
            plt.close()
            plt.cla()
    HtmlTablePlot.SaveHtmlFile(output_dic, os.path.join(data_path, current_session, "B_Corr_A_pearson.html"))


################################ MAIN #########################################

RunOverICs = True
RunOverCsv = True
csv_path = os.path.join(data_path, "XCorrResults", f"Rms{1000//downsampleWindowInMs}Hz_{windowForCorrInMs}_corr_lag_dtw_fisherd_1.csv")
#headers =  SessionToCorrelateOverLagsNoDirections_PEAK_Headers() + ['whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']
headers = get_headers_27_08() + ['whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']
#SessionToCorrelateOneComponentOverLagsNoDirectionsResultHeaders()

if ( __name__ == "__main__" and RunOverICs):
    sessions = GetSessions(data_path)

    all_correlations = []
    for current_session in tqdm(sessions):
        print(current_session)
        independentComponents, chunks = LoadComponents(data_path, current_session, downsampleWindowInMs)
        #non_zscore_components = [comp.copy() for comp in independentComponents]

        print("Zscore")
        for p in [A, B]:
            #independentComponents[p] = zscore(independentComponents[p], axis=1)
            #independentComponents[p] = xp.diff(xp.array(independentComponents[p]), axis=1).get()
            #(independentComponents[p] - 0.15 * np.mean(independentComponents[p], axis=1)[:,None])/(np.std(independentComponents[p], axis=1)[:,None])
            independentComponents[p] = zscore(independentComponents[p], axis=1)
            #independentComponents[p] = get_matrix_reduced_part_of_mean(independentComponents[p], 0.5)
        cleanSpace()
        print("finish z")

        #SessoinToHtmlCorrAndPlot(current_session, independentComponents, chunks)
        #corr_results = SessionToCorrelateOneComponentOverLagsNoDirections(current_session, independentComponents, chunks,False)
        #corr_results = SessionToCorrelateOverLagsNoDirections_PEAK(current_session, independentComponents, chunks, False)
        corr_results = SessionCorrelation27_08(current_session, independentComponents, chunks)#, non_zscore_components)
        choices = loadChoiceFromSession(data_path, current_session)
        for i in range(len(corr_results)):
            result = corr_results[i]
            key = result[1]
            story_number = key.split("_")[1]
            if ( story_number != ''):
                story_number = story_number.split('.')[0]
                for choice in choices:
                    if ( story_number == choice[0] ):
                        corr_results[i] = result + choice[1:]
                        break
        
        all_correlations += corr_results
        del independentComponents
        cleanSpace()
        print("finished session " + current_session)

        
    df = pd.DataFrame(all_correlations)
    df.columns = headers
    df.to_csv(csv_path)



if ( __name__ == "__main__" and RunOverCsv):
    file = pd.read_csv(csv_path)
    df = pd.DataFrame(file)

    reading_indexes = df["key"].str.contains("Read")  # fix that.
    reading_sessoins = df[reading_indexes]


    # look at the speaker and listener, and predict the listenr choice in the reading!!

    # cross pearson correlation 
    actual = []
    predicted = []
    for i in range(0,len(reading_sessoins), 2):
        if (True or reading_sessoins.iloc[i]["isChoiceA"] == reading_sessoins.iloc[i]["isChoiceB"]):
            if ( float(reading_sessoins.iloc[i]["mean_corr"]) - float(reading_sessoins.iloc[i+1]["mean_corr"]) > 0):
                predicted += [1,0]
            else:
                predicted += [0,1]
            
            for j in [i, i+1]:
                actual.append(int(reading_sessoins.iloc[j]["isChoiceA"]) * int (reading_sessoins.iloc[j]["isChoiceB"]))

    print(classification_report(actual, predicted))
