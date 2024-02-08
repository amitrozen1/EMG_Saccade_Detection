import os
from General.HelperFunctions import GetSessions, cleanSpace, reshape_num_channels, RemoveRowFromMatrix
import numpy as np
from scipy.stats import zscore
from General.init import *
import pandas as pd
from PsychData.LoadChoicesJson import loadChoiceFromSession
from tqdm import tqdm
import re
from General.HelperFunctions import getPathsFromSessionFolder, remove_rows_from_matrix
from EDF.EdfAnalyzer import EdfAnalyzer
from ComponentAnalysis.ComponentExtractor import GetCorrugatorComponent, GetNoseComponent, GetSideEyeComponent, GetZygomaticusComponent

review_df = None

def regex_split_to_numbers_with_all_possible_seprators(s):
    if pd.isna(s):
        return []
    return [ int(x) for x in re.split(r"[^0-9.]", s) if x != "" ]

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

def run_emg_analysis(main_corrolation_func, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, headers, csv_out_path):
    sessions_list = GetSessions(data_path)
    all_correlations = []

    for session in tqdm(sessions_list):
        breaked = False
        print(session)
        elecs_to_remove, is_dark = get_review_data(session)

        both_componenets = [[],[]]
        both_chunks = [[],[]]

        path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
        for p in [A,B]:
            Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsampleWindowInMs)

            Y = remove_rows_from_matrix(Y, elecs_to_remove[p])
            Y = zscore(Y, axis=1)

            w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove[p]))
            independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
            independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsampleWindowInMs, freq)

            #create a new components:
            special_comps_index = get_special_component_indexes(w, elecs_to_remove[p])

            if (-1 in special_comps_index):
                print("Error in session " + session + " participant " + str(p))
                breaked = True
                continue

            total_max = np.max(independentComponents, axis=0)
            independentComponents = independentComponents[special_comps_index, :]
            special_max = np.max(independentComponents, axis=0)
            #for temporal analysis:
            independentComponents = np.vstack((independentComponents, total_max))
            independentComponents = np.vstack((independentComponents, special_max))

            both_componenets[p] = independentComponents
            both_chunks[p] = chunks

        if (breaked):
            continue
        
        corr_results = main_corrolation_func(session, both_componenets, both_chunks)

        choices = loadChoiceFromSession(data_path, session)
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
        print("finished session " + session)
    
    if (csv_out_path!= None):
        df = pd.DataFrame(all_correlations)
        df.columns = headers
        df.to_csv(csv_out_path)


def run_inertial_analysis(main_corrolation_func, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, headers, csv_out_path):
    sessions_list = GetSessions(data_path)
    all_correlations = []

    for session in tqdm(sessions_list):
        breaked = False
        print(session)

        both_componenets = [[],[]]
        both_chunks = [[],[]]

        path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
        for p in [A,B]:
            Y, chunks, freq, inertialData, inertialFreq = EdfAnalyzer.ReadWithInertial(path_and_corrections[p][0], path_and_corrections[p][1], downsampleWindowInMs, just_accelorometer=True)

            inertialData = zscore(inertialData, axis=1)
            inertialData = EdfAnalyzer.window_rms(inertialData, rms_size_ms, inertialFreq)
            inertialData = EdfAnalyzer.resample_by_averaging_every_n(inertialData, downsampleWindowInMs, inertialFreq)

            max_row = np.max(inertialData, axis=0)
            combined_with_rms = np.sqrt(np.sum(np.square(inertialData), axis=0))

            # inertialData = np.vstack((inertialData, max_row))
            # inertialData = np.vstack((inertialData, combined_with_rms))
            #only max and combined:
            inertialDataCombinations = np.vstack((max_row, combined_with_rms))

            both_componenets[p] = inertialDataCombinations
            both_chunks[p] = chunks

        if (breaked):
            continue

        corr_results = main_corrolation_func(session, both_componenets, both_chunks)

        choices = loadChoiceFromSession(data_path, session)
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
        cleanSpace()
        print("finished session " + session)
    
    df = pd.DataFrame(all_correlations)
    df.columns = headers
    df.to_csv(csv_out_path)

