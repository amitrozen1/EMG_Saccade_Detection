import numpy as np
from General.HelperFunctions import GetSessions, cleanSpace, reshape_num_channels
import json
import cupy as xp
from tqdm import tqdm
import os
from EDF.EdfAnalyzer import Chunk, Annotation
from PsychData.LoadChoicesJson import loadChoiceFromSession
import pandas as pd
from scipy.stats import zscore, pearsonr
from General.StatisticsFunctions import cross_rqa


A=0
B=1

def getJsonFileName(participant):
    if ( participant == A):
        return "A.list"
    return "B.list"

def getDataFileName(particiapnt):
    if(particiapnt == A):
        return "A.csv"
    return "B.csv"

def getPairwiseZip(arr):
    return list(zip(arr[0::2], arr[1::2]))

def loadJsonToChunks(filename, downsample_window_in_ms = 0.25):
    chunks = dict()
    with open(filename, "r") as f: 
        data = json.load(f)
    
    for item in data:
        chunk        =   Chunk()
        start        =   Annotation() 
        end          =   Annotation()
        start.Time   =   int(item["start"]) // int(4 * downsample_window_in_ms)
        end.Time     =   int(item["end"])   // int(4 * downsample_window_in_ms)
        chunk.Start  =   start
        chunk.End    =   end
        key          =   str(item["key"])

        chunks[key] = chunk
    
    return chunks


def rms_downsample_single(a, window_size_in_ms):
    window_size = window_size_in_ms*4 # divide by 1000, and multiply by 4000
    rms_values = xp.array([])
    squared = xp.array(a) ** 2
    for i in range(0, len(a)-window_size , window_size):
        rms_values = xp.append(rms_values, xp.sqrt(xp.mean(squared[i:i+window_size])))
    return rms_values.get()


downsampleWindowInMs = 10 #10 # downsamples to 100HZ

orig_freq = 4000
freq = orig_freq // (4*downsampleWindowInMs)
windowForCorrInMs = 1000
numOfPixelsForXCorrWindow = int(windowForCorrInMs//downsampleWindowInMs)
numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow//10


def prepareStreamForXCorr(x, ratio = 0.3):
    return x - (ratio * np.mean(x))

def dtai_dist_dtw(mat1, mat2, max_window =1000, plot=False):
    from dtaidistance import dtw
    from dtaidistance import dtw_visualisation as dtwvis
    import matplotlib.pyplot as plt


    max_len = min(len(mat1), len(mat2))
    mat1 = mat1[:max_len]
    mat2 = mat2[:max_len]

    path = dtw.warping_path(mat1, mat2, window = max_window)
    p1 = [p[0] for p in path]
    p2 = [p[1] for p in path]

    r, _ = pearsonr(mat1[p1], mat2[p2])
    if ( plot ):
        dtwvis.plot_warping(mat1, mat2, path)
        plt.show()
        plt.cla()

    return r






def xCorrSmilesHeaders(state_to_analyze):
    return ["session", "key","action", f"{state_to_analyze}_mean_corr",f"{state_to_analyze}_mean_var_corr",
            f"{state_to_analyze}_mean_A", f"{state_to_analyze}_var_A", f"{state_to_analyze}_mean_B", f"{state_to_analyze}_var_B",
            f"{state_to_analyze}_rqa.recurrence_rate",f"{state_to_analyze}_rqa.determinisim",
            f"{state_to_analyze}_rqa.longest_diagonal_line", f"{state_to_analyze}_rqa.average_diagonal_line",
            f"1-{state_to_analyze}_rqa.recurrence_rate",f"1-{state_to_analyze}_rqa.determinisim",
            f"1-{state_to_analyze}_rqa.longest_diagonal_line", f"1-{state_to_analyze}_rqa.average_diagonal_line",
            f"{state_to_analyze}_dtw_corr"]

def xCorrSmiles(current_session, smileComponents, chunks, overlap=False):
    '''
    return a list of - [["key", "mean_corr","mean_var_corr"]]
    '''
    
    results = []
    for key in chunks[0]:
        chunk_data = [[],[]]
        extra_offset = [0, numOfPixelsForXCorrWindow]
        
        action = key.split("_")[0]

        for participant in [A,B]:
            chunk = chunks[participant][key]
            chunk_start = int(chunk.Start.Time) - extra_offset[participant]
            chunk_end = int(chunk.End.Time) + extra_offset[participant] # XXX - it was -extra
            chunk_data[participant] = smileComponents[participant][chunk_start:chunk_end]
        
        mat1 = chunk_data[A]
        mat2 = chunk_data[B]
        dtw_corr = dtai_dist_dtw(mat1, mat2, max_window = 15*numOfPixelsForXCorrStep)
    
        #max_len = min(mat1.shape[1], mat2.shape[1])
        maxValueForWindow = np.array([])

        jumps = numOfPixelsForXCorrWindow
        if ( overlap ):
            jumps = jumps//2

        for i, s in enumerate(range(0, len(mat1) - numOfPixelsForXCorrWindow , jumps)):
            e = s+numOfPixelsForXCorrWindow
            corrmat = np.array([])
            
            # if ( np.mean(mat1[s:e]) < -0.5 ):
            #     continue

            for lag_ind in np.arange(0,16): #00 - 1500 ms lag XXX - it was 11
                #the sliding window (that slides through A) and the window that moves in matB are the same?
                lag = int(lag_ind * numOfPixelsForXCorrStep) # 100 ms at a lag
                vec = np.array([])
                if ( e + lag < len(mat2) ):
                    vec = np.mean((mat1[s:e])*(mat2[(s+lag):(e+lag)]))
                if ( s - lag > 0 and e-lag > 0 and lag != 0):
                    vec = np.append(vec, np.mean((mat1[s:e])*(mat2[(s-lag):(e-lag)])))
                
                corrmat = np.append(corrmat, vec)

            maxValueForWindow = np.append(maxValueForWindow, np.max(corrmat))
        
        
        maxValueCorr = np.mean(maxValueForWindow) #16x1
        maxValueForWindowVar = np.arctan(np.var(maxValueForWindow))

        if ( len(maxValueForWindow) == 0 ):
            maxValueCorr = 0 
            maxValueForWindowVar = 0 
        

        #compute new cross occurence - 
        rqa = cross_rqa(mat1, mat2)

        results.append([current_session, key, action, np.arctan(maxValueCorr), maxValueForWindowVar, np.mean(mat1), np.var(mat1), np.mean(mat2), np.var(mat2), 
                        rqa.recurrence_rate , rqa.determinism, rqa.longest_diagonal_line/len(mat1), rqa.average_diagonal_line/len(mat1),
                        1 - rqa.recurrence_rate , 1 - rqa.determinism, 1 - rqa.longest_diagonal_line/len(mat1), 1 - rqa.average_diagonal_line/len(mat1)
                        , dtw_corr])
    return results


if __name__ == "__main__":

    data_folder = r"C:\Liron\DataEmg\Done"
    svmType = "svr"
    stateToAnalyze = "smile"


    csv_path = os.path.join(data_folder, "SmileXcorrResults", f"{stateToAnalyze}_{svmType}_downsample_to_{1000//downsampleWindowInMs}HZ_atan_zscore_with_avgAndLongDiag_with_dtw_delay_1500.csv")

    sessions = GetSessions(data_folder)

    all_correlations = []
    headers = xCorrSmilesHeaders(stateToAnalyze) + ['whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']

    for session in tqdm(sessions):
        svm_folder = os.path.join(data_folder, session, stateToAnalyze, svmType)
        chunks = [[], []]
        smileComponents = [[],[]]
        for p in [A,B]:
            chunks[p] = loadJsonToChunks(os.path.join(svm_folder, getJsonFileName(p)), downsampleWindowInMs)
            x = np.loadtxt(os.path.join(svm_folder, getDataFileName(p)))
            x = rms_downsample_single(x, downsampleWindowInMs)
            x = zscore(x) #prepareStreamForXCorr(x) #zscore(x) #
            smileComponents[p] = x
        
        cleanSpace()
        
        corr_results = xCorrSmiles(session, smileComponents, chunks)

        choices = loadChoiceFromSession(data_folder, session)
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
            len_diff = len(headers) - len(corr_results[i])
            if ( len_diff > 0 ):
                corr_results[i] = corr_results[i] + [0]*len_diff
        
        all_correlations += corr_results
        del smileComponents
        cleanSpace()
        print("finished session " + session)

    df = pd.DataFrame(all_correlations)
    df.columns = headers
    df.to_csv(csv_path)
