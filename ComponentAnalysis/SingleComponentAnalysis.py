import os
from General.HelperFunctions import GetSessions, cleanSpace, reshape_num_channels, particiapnt_to_char
import numpy as np
from EDF.SharedIcaRunner import LoadComponents, LoadComponentsWithRegularRms
from scipy.stats import zscore
from General.StatisticsFunctions import pearson_rows
import pandas as pd
from PsychData.AnalyzeQuestionnaire import GetStatsRowsFromSession, GetStatsHeaderWithouSession
from PsychData.LoadChoicesJson import loadChoiceFromSession
from sklearn.metrics import classification_report
from tqdm import tqdm

num_channels = 16


def GetComponentByElectrodeNumber(w, electrode):
    components_to_electrodes = np.argmax(np.absolute(np.linalg.inv(w)), axis=0)[::-1]
    componentsArr = np.where(components_to_electrodes == electrode)[0]
    if ( len(componentsArr) == 0):
        return "Error", "Error"
    return componentsArr[0], electrode

def GetNoseComponent(w):
    res = GetComponentByElectrodeNumber(w, 8)
    if ( res[0] == "Error"):
        res = GetComponentByElectrodeNumber(w, 7)
    return res

def GetSideEyeComponent(w):
    return GetComponentByElectrodeNumber(w, 11)

def GetZygomaticusComponent(w):
    res =  GetComponentByElectrodeNumber(w, 4)
    if ( res[0] == "Error"):
        res = GetComponentByElectrodeNumber(w, 3)
    return res

def GetCorrugatorComponent(w):
    res = GetComponentByElectrodeNumber(w, 13)
    if ( res[0] == "Error"):
        res = GetComponentByElectrodeNumber(w, 12)
    return res


def GetFacialComponentsElcrotdesAndIndexes(w):
    '''
    return [ Nose, SideEye, Zygomaticus, Corrugator]
    '''
    return [GetNoseComponent(w), GetSideEyeComponent(w), 
            GetZygomaticusComponent(w), GetCorrugatorComponent(w)]


def GetFacialComponentsElcrotdesAndIndexes_old(data_path, session):
    '''
    return [ Nose, SideEye, Zygomaticus, Corrugator]
    '''

    w_path = os.path.join(data_path, session, "weightsAfterSharedICA.csv")
    w = np.loadtxt(w_path, delimiter=",")

    return GetFacialComponentsElcrotdesAndIndexes(w)


A = 0
B = 1


if ( __name__ == "__main__"):


    downsampleWindowInMs = 10 # downsamples to 100HZ
    orig_freq = 4000

    freq = orig_freq // (4*downsampleWindowInMs)
    windowForCorrInMs = 250
    numOfPixelsForXCorrWindow = windowForCorrInMs//downsampleWindowInMs
    numOfPixelsForXCorrStep = 100 // downsampleWindowInMs # 100ms
    data_path = r"C:\Liron\DataEmg\Done"
    csv_path = os.path.join(data_path, "four_components_mean_and_choices_NewReverseabsZscore.csv")

    sessions = GetSessions(data_path) # last one doesn't have nose component

    all_results = []
    headers = ["session","particiapnt", "key", "action", "nose_electrode", "eye_electrode", "zygomaticus_electrode", "corrugator_electrode",
                "mean_nose", "mean_eye", "mean_zygomaticus", "mean_corrugator", "var_nose", "var_eye", "var_zygomaticus", "var_corrugator",
                "my_choice", "other_choice", "my_rt", "other_rt", "choose_together", "is_reading", "is_other"]
    

    for session in tqdm(sessions):
        indexesAndElectrodes = GetFacialComponentsElcrotdesAndIndexes(data_path, session)
        indexes = [x[0] for x in indexesAndElectrodes]
        chosenElectrodes = [x[1] for x in indexesAndElectrodes]

        independentComponents, chunks = LoadComponents(data_path, session, doRms=False)
        for key in chunks[0]:
            for p in [A,B]:
                chunks[p][key].Start.Time = int(chunks[p][key].Start.Time)
                chunks[p][key].End.Time = int(chunks[p][key].Start.Time)
        
        #LoadComponents(data_path, session, downsampleWindowInMs, doRms=False)

        print("Zscore")
        for p in [A, B]:
            independentComponents[p] = independentComponents[p][indexes, :]
            independentComponents[p] = np.abs(zscore(independentComponents[p], axis=1))
        cleanSpace()

        choices = loadChoiceFromSession(data_path, session)
        particiapnt_choices_index = [2,3]
        particiapnt_rt_index = [4,5]

        for key in chunks[0]:
            chunk_data = [[],[]]
            
            for participant in [A,B]:
                chunk_results = [session, particiapnt_to_char(participant), key, key.split("_")[0]]
                chunk_results += chosenElectrodes

                chunk = chunks[participant][key]
                chunk_start = int(chunk.Start.Time)
                chunk_end = int(chunk.End.Time)
                chunk_data[participant] = independentComponents[participant][: , chunk_start:chunk_end]
                
                mean_comp = np.mean(chunk_data[participant], axis=1)
                chunk_results += mean_comp.tolist()
                var_comp  = np.var(chunk_data[participant], axis=1)
                chunk_results += var_comp.tolist()
                
                story_number = key.split("_")[1]
                if ( story_number!= ''): # smile angry blink have story number of ''
                    story_number = story_number.split('.')[0]
                    for choice in choices:
                        if ( choice[0] == story_number):
                            isReading = int((choice[1] - 1) == participant)
                            isOther = int(choice[6])
                            myChoice = int(choice[particiapnt_choices_index[participant]])
                            otherChoice = int(choice[(particiapnt_choices_index[::-1])[participant]])
                            myRt = float(choice[particiapnt_rt_index[participant]])
                            otherRt = float(choice[(particiapnt_rt_index[::-1])[participant]])
                            chooseTogether = myChoice*otherChoice

                            chunk_results += [myChoice, otherChoice, myRt, otherRt, chooseTogether, isReading, isOther]
                            break
                len_diff = len(headers) - len(chunk_results)
                if ( len_diff > 0 ):
                    chunk_results = chunk_results + [0]*len_diff
                
                all_results.append(chunk_results)
                    

    df = pd.DataFrame(all_results)
    df.columns = headers
    df.to_csv(csv_path)                

