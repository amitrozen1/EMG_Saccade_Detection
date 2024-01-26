import numpy as np
from General.HelperFunctions import GetSessions, cleanSpace, particiapnt_to_char
import json
from tqdm import tqdm
import os
from EDF.EdfAnalyzer import Chunk, Annotation, ExtendedChunk
from scipy.signal import resample
from ComponentAnalysis.SingleComponentAnalysis import GetFacialComponentsElcrotdesAndIndexes
from EDF.SharedIcaRunner import SingleIca
from scipy.stats import zscore

A=0
B=1

NOSE = 0
SIDE_EYE = 1
ZYGOMATICUS = 2
CORRUGATOR = 3

muscles_names = ["Nose", "SideEye", "Zygomaticus", "Corrugator"]


def getPairwiseZip(arr):
    return list(zip(arr[0::2], arr[1::2]))

def dictionary_from_lists_key_first(list_of_lists):
    result = dict()
    for item in list_of_lists:
        result[item[0]] = item[1:]
    return result

def GetHeadersForPsychOutput():
    headers = ["session"]
    for muscle in muscles_names:
        for part in ["live", "recorded"]:
            for stat in ["mean", "var"]:
                headers.append(f"{muscle}_{part}_{stat}")
    
    return headers


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

    for key in chunks:
        if ("listen" in key.lower()):
            chunks[key] = ExtendedChunk(chunks[key], choices[get_story_num(key)])
        else:
            chunks[key] = ExtendedChunk(chunks[key], False)
    return chunks

def convertToNoDownsampleTimes(chunks):
    for key in chunks:
        chunks[key].Start.Time = int(chunks[key].Start.Time*4000)
        chunks[key].End.Time = int(chunks[key].End.Time*4000)

def GetCompnents(data_folder, session, p):
    chunks, w, components, freq = SingleIca(data_folder, session, p)
    chunks = convertToExtendedChunks(data_folder, session, chunks)
    convertToNoDownsampleTimes(chunks)

    return chunks, w, components, freq

def GetStreamIndex(w, muscle):
    muscle_index = GetFacialComponentsElcrotdesAndIndexes(w)[muscle][0]
    return muscle_index

def GetMusclesComponents(w, components, start, end):
    muscle_indexes = []
    for muscle in [NOSE, SIDE_EYE, ZYGOMATICUS, CORRUGATOR]:
        muscle_indexes += [GetStreamIndex(w, muscle)]
    return components[muscle_indexes, start:end]


def getMeanAndVarFromComponentsRecorder(components_recorder):
    live_means = np.mean(components_recorder["read"], axis=1)
    live_vars = np.var(components_recorder["read"], axis=1)
    recorded_means = np.mean(components_recorder["listen"], axis=1)
    recorded_vars = np.var(components_recorder["listen"], axis=1)

    output = []
    # muscle live\recorded mean\var
    for muscle in [NOSE, SIDE_EYE, ZYGOMATICUS, CORRUGATOR]:
        output += [live_means[muscle], live_vars[muscle], recorded_means[muscle], recorded_vars[muscle]]
    return output

# def GetMusclesMeanAndVar(w, components, start, end):
#     values = []
#     for muscle in [NOSE, SIDE_EYE, ZYGOMATICUS, CORRUGATOR]:
#         muscle_index = GetStreamIndex(w, muscle)
#         muscle_stream = components[muscle_index, start:end]
#         valus += [np.mean(muscle_stream), np.var(muscle_stream)]
#     return values

if __name__ == "__main__":

    data_folder = r"C:\Liron\DataEmg\Try"

    sessions = [GetSessions(data_folder)[8]]

    outputDict = dict()
    for session in tqdm(sessions):
        #components, chunks = GetSideEyeStream(data_folder, session)
        for p in [A,B]:
            chunks, w, components, freq = GetCompnents(data_folder, session, p)
            line_key = f"{session}_{particiapnt_to_char(p)}"
            nrows = 4
            components_recorder = {"read": np.empty((nrows,0)), "listen":np.empty((nrows,0))}

            reader = [A, A, B, B, A, A, B, B]
            for experiment_part in ["read", "listen"]:
                for chunk_key in chunks:       
                    start = chunks[chunk_key].Start.Time
                    end = chunks[chunk_key].End.Time 
                    
                    if experiment_part in chunk_key.lower():
                        if ( experiment_part == "read"):
                            current_reader = reader.pop(0)
                            if ( current_reader != p):
                                components_recorder[experiment_part] = np.append(components_recorder[experiment_part], GetMusclesComponents(w, components, start, end), axis=1)
                        if ( experiment_part == "listen" and not chunks[chunk_key].isOther):
                            components_recorder[experiment_part] = np.append(components_recorder[experiment_part], GetMusclesComponents(w, components, start, end), axis=1)
            
            outputDict[line_key] = getMeanAndVarFromComponentsRecorder(components_recorder)

    headers = GetHeadersForPsychOutput()
    write_dict_to_csv_no_pandas(outputDict, data_folder, headers, f"compare_face_comps_single_ica.csv")