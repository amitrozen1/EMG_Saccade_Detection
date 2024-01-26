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
from scipy.signal import resample

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

def window_rms_signal(signal, window_size):
    return np.sqrt(np.convolve(signal**2, (np.ones(window_size)/float(window_size))))

def resample_signal(signal, freq, new_freq):
    return resample(signal, int(len(signal) * new_freq / freq))

def rms_resample_signal(signal, orig_freq, window_size):
    new_freq = 1000 / window_size
    return resample_signal(window_rms_signal(signal, window_size), orig_freq, new_freq)

def dictionary_from_lists_key_first(list_of_lists):
    result = dict()
    for item in list_of_lists:
        result[item[0]] = item[1:]
    return result

def GetHeadersForPsychOutput():
    return ["id", "whoseReading", "ChoiceA", "ChoiceB", "rtA", "rtB", "isOther", "choseTogether", "story0", "story1",
            "corr_story0", "corr_story1", "lag_story0", "lag_story1"]


def write_dict_to_csv_no_pandas(dic, folder, headers, name):
    with open(os.path.join(folder, name), "w") as f:
        f.write(",".join(headers) + "\n")
        for key in dic:
            f.write(key + "," + ",".join([str(item) for item in dic[key]]) + "\n")

def z_k_score(a, k=0.5):
    return (a - (k * np.mean(a)))/np.std(a)

def correlate(a_smile, b_smile, freq, windowInMs = 1000, maxLagInMs = 1000, jumpInMs = 1000 ):
    max_len = min(len(a_smile), len(b_smile))
    window = int(windowInMs * freq / 1000)
    maxLag = int(maxLagInMs * freq / 1000)
    jump = int(jumpInMs * freq / 1000)
    corrs = []
    lags = []
    for i in range(window, max_len - window - maxLag, jump):
        a = a_smile[(i-maxLag):(i+window+maxLag)]
        b = b_smile[i:(i+window)]
        ccArr = np.correlate(a,b, mode="valid")
        max_corr = np.max(ccArr)
        max_corr_lag = np.abs(np.argmax(ccArr) - maxLag)
        corrs.append(max_corr)
        lags.append(max_corr_lag)
    
    return np.mean(corrs)/window, np.mean(lags)

downsampleWindowInMs = 20 #10 # downsamples to 100HZ
orig_freq = 4000
freq = orig_freq // (4*downsampleWindowInMs)
windowForCorrInMs = 1000
numOfPixelsForXCorrWindow = int(windowForCorrInMs//downsampleWindowInMs)
numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow//10

if __name__ == "__main__":

    data_folder = r"C:\Liron\DataEmg\Try"
    svmType = "svr"
    stateToAnalyze = "smile"
    experiment_part = "listen" #"listen"

    sessions = GetSessions(data_folder)

    all_correlations = []
    headers = ['isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']

    for session in tqdm(sessions):
        svm_folder = os.path.join(data_folder, session, stateToAnalyze, svmType)
        ml_task_folder = os.path.join(svm_folder, f"MLTask_no_z_{experiment_part}ing_{freq}HZ_minus_half")
        #make folder if not exists:
        if not os.path.exists(ml_task_folder):
            os.makedirs(ml_task_folder)
        chunks = [[], []]
        smileComponents = [[],[]]
        for p in [A,B]:
            chunks[p] = loadJsonToChunks(os.path.join(svm_folder, getJsonFileName(p)), downsampleWindowInMs)
            x = np.loadtxt(os.path.join(svm_folder, getDataFileName(p)))
            x = rms_resample_signal(x, orig_freq, downsampleWindowInMs)
            x = z_k_score(x, 0.5) #zscore(x)
            smileComponents[p] = x
        
        cleanSpace()

        choices = loadChoiceFromSession(data_folder, session)
        choices = dictionary_from_lists_key_first(choices)
        # choice["story_id"] = [whoseReading, isChoiceA, isChoiceB, rtA, rtB, isOther, choseTogether]
        psych_data_dict = dict()

        story_pair_csv_arr = []
        stories_in_csv = []

        stories_mean_corr = []
        stories_mean_lag = []

        for chunk_key in chunks[0]:
            if experiment_part in chunk_key.lower():
                start_a = chunks[A][chunk_key].Start.Time
                end_a = chunks[A][chunk_key].End.Time
                start_b = chunks[B][chunk_key].Start.Time
                end_b = chunks[B][chunk_key].End.Time 
                arrA = smileComponents[A][start_a:end_a]
                arrB = smileComponents[B][start_b:end_b]
                story_pair_csv_arr.append(arrA)
                story_pair_csv_arr.append(arrB)
                corr,lag = correlate(arrA, arrB, freq)
                stories_mean_corr.append(corr)
                stories_mean_lag.append(lag)
                stories_in_csv.append(int(chunk_key.split("_")[1].split(".")[0]))
            #assuming listen story stored sequentially and chronologically.
            if ( len(story_pair_csv_arr) == 4):
                stories_in_csv_sorted = sorted(stories_in_csv)
                # to string with _ delimiter
                stories_in_csv_key = f"{session}_" + "_".join([str(x) for x in stories_in_csv_sorted])
                # save vectors
                df = pd.DataFrame(story_pair_csv_arr)
                df.to_csv(os.path.join(ml_task_folder, f"{stories_in_csv_key}.csv"), index=False, header=False)
                # save choices:
                #choices[str(stories_in_csv[1])][1:] = ChoiceA (0 if first story, 1 if second), ChoiceB, rtA, rtB, isOther, choseTogether
                psych_data_dict[stories_in_csv_key] = choices[str(stories_in_csv[1])] + stories_in_csv + stories_mean_corr + stories_mean_lag
                # write csv file and choices.
                story_pair_csv_arr = []
                stories_in_csv = []
                stories_mean_corr = []
                stories_mean_lag = []
        

        headers = GetHeadersForPsychOutput()
        write_dict_to_csv_no_pandas(psych_data_dict, ml_task_folder, headers, f"{experiment_part}ing_choices.csv")

# todo - implement one solution for both reading and listening - enter whoseReading.