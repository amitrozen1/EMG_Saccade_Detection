import numpy as np
from General.HelperFunctions import GetSessions, cleanSpace, particiapnt_to_char
import json
from tqdm import tqdm
import os
from EDF.EdfAnalyzer import Chunk, Annotation, ExtendedChunk
from scipy.signal import resample
from ComponentAnalysis.SingleComponentAnalysis import GetFacialComponentsElcrotdesAndIndexes
from EDF.SharedIcaRunner import LoadComponents
from scipy.stats import zscore

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
        chunk        =   ExtendedChunk()
        start        =   Annotation() 
        end          =   Annotation()
        start.Time   =   int(item["start"]) // int(4 * downsample_window_in_ms)
        end.Time     =   int(item["end"])   // int(4 * downsample_window_in_ms)
        chunk.Start  =   start
        chunk.End    =   end
        key          =   str(item["key"])
        if ( "isOther" in item ):
            chunk.isOther = bool(item["isOther"])
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
    return ["session", "LiveListenMean", "LiveListenVar", "RecordedListenMean",
            "RecordedListenVar"]


def write_dict_to_csv_no_pandas(dic, folder, headers, name):
    with open(os.path.join(folder, name), "w") as f:
        f.write(",".join(headers) + "\n")
        for key in dic:
            f.write(key + "," + ",".join([str(item) for item in dic[key]]) + "\n")


def convertToDict(mat_of_arrs):
    res = dict()
    for arr in mat_of_arrs:
        res[arr[0]] = arr[1:]
    return res

def convertToIsOtherOnly(dic):
    for key in dic:
        dic[key] = bool(dic[key][-2])

def get_story_num(key):
    return key.split("_")[1].split(".")[0]

def convertToExtendedChunks(data_folder, session, chunks):
    from PsychData.LoadChoicesJson import loadChoiceFromSession
    choices = convertToDict(loadChoiceFromSession(data_folder, session))
    convertToIsOtherOnly(choices)

    for key in chunks[0]:
        if ("listen" in key.lower()):
            for p in [A,B]:
                chunks[p][key] = ExtendedChunk(chunks[p][key], choices[get_story_num(key)])
        else:
            for p in [A,B]:
                chunks[p][key] = ExtendedChunk(chunks[p][key], False)
    return chunks


def convertToNoDownsampleTimes(chunks):
    for p in [A,B]:
        for key in chunks[p]:
            chunks[p][key].Start.Time = int(chunks[p][key].Start.Time*4000)
            chunks[p][key].End.Time = int(chunks[p][key].End.Time*4000)



def GetWholeCompnents(data_folder, session):
    independentComponents, chunks = LoadComponents(data_folder, session, 0, False)
    chunks = convertToExtendedChunks(data_folder, session, chunks)
    convertToNoDownsampleTimes(chunks)

    return independentComponents, chunks



NOSE = 0
SIDE_EYE = 1
ZYGOMATICUS = 2
CORRUGATOR = 3

def GetStream(data_folder, session, muscle):
    zygomaticus_index = GetFacialComponentsElcrotdesAndIndexes(data_folder, session)[muscle][0]
    independentComponents, chunks = LoadComponents(data_folder, session, 0, False)
    zygomaticus_stream = [independentComponents[p][zygomaticus_index, :] for p in [A,B]]
    chunks = convertToExtendedChunks(data_folder, session, chunks)
    convertToNoDownsampleTimes(chunks)
    return zygomaticus_stream, chunks

def GetZygomaticusStream(data_folder, session):
    return GetStream(data_folder, session, ZYGOMATICUS)

def GetCorrugatorStream(data_folder, session):
    return GetStream(data_folder, session, CORRUGATOR)

def GetSideEyeStream(data_folder, session):
    return GetStream(data_folder, session, SIDE_EYE)

def GetNoseStream(data_folder, session):
    return GetStream(data_folder, session, NOSE)


downsampleWindowInMs = 50 #10 # downsamples to 100HZ
orig_freq = 4000
freq = orig_freq // (4*downsampleWindowInMs)
windowForCorrInMs = 1000
numOfPixelsForXCorrWindow = int(windowForCorrInMs//downsampleWindowInMs)
numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow//10

if __name__ == "__main__":

    data_folder = r"C:\Liron\DataEmg\Try"
    svmType = "svc"
    stateToAnalyze = "smile"

    sessions = GetSessions(data_folder)

    all_correlations = []

    outputDict = dict()
    for session in tqdm(sessions):
        chunks = [[], []]
        components = [[],[]]
        svm_folder = os.path.join(data_folder, session, stateToAnalyze, svmType)

        #components, chunks = GetSideEyeStream(data_folder, session)

        for p in [A,B]:
            json_file_name = os.path.join(svm_folder, getJsonFileName(p))
            if ( not os.path.exists(json_file_name)):
                continue
            chunks[p] = loadJsonToChunks(json_file_name)#, downsampleWindowInMs)
            
            line_key = f"{session}_{particiapnt_to_char(p)}"
            nrows = 1 #components[p].shape[0]
            smile_dict = {"read": np.empty((nrows,0)), "listen":np.empty((nrows,0))}

            #change to load component.
            x = np.loadtxt(os.path.join(svm_folder, getDataFileName(p)))
            #x = rms_resample_signal(x, orig_freq, downsampleWindowInMs)
            #components[p] = zscore(components[p]) #, axis=1)
            components[p] = x

            reader = [A, A, B, B, A, A, B, B]
            for experiment_part in ["read", "listen"]:
                for chunk_key in chunks[p]:       
                    start = chunks[p][chunk_key].Start.Time
                    end = chunks[p][chunk_key].End.Time 
                    arr = components[p][start:end]#[:, start:end]
                    if experiment_part in chunk_key.lower():
                        if ( experiment_part == "read"):
                            current_reader = reader.pop(0)
                            if ( current_reader != p):
                                smile_dict[experiment_part] = np.append(smile_dict[experiment_part], arr)#, axis=1)
                        if ( experiment_part == "listen" and not chunks[p][chunk_key].isOther):
                            smile_dict[experiment_part] = np.append(smile_dict[experiment_part], arr)#, axis=1)
            
            outputDict[line_key] = [np.mean(smile_dict["read"]), np.var(smile_dict["read"]), np.mean(smile_dict["listen"]), np.var(smile_dict["listen"])]

    headers = GetHeadersForPsychOutput()
    write_dict_to_csv_no_pandas(outputDict, data_folder, headers, f"compare_smiles_svc.csv")