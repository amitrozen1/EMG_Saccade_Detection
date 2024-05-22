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
from General.HelperFunctions import get_review_data
from ComponentAnalysis.ComponentExtractor import open_w_if_exisis, load_special_components_if_exists, get_special_component_indexes, save_new_w
from analysis_decisions_py.cleaning_signal import remove_emg_artifacts


def run_emg_analysis(main_corrolation_func, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, headers, csv_out_path):
    sessions_list = GetSessions(data_path)
    all_correlations = []

    for session in tqdm(sessions_list):
        breaked = False
        print(session)
        elecs_to_remove, is_dark = get_review_data(session, data_path)

        both_componenets = [[],[]]
        both_chunks = [[],[]]

        path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
        for p in [A,B]:
            Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsampleWindowInMs)

            Y = remove_rows_from_matrix(Y, elecs_to_remove[p])
            Y = zscore(Y, axis=1)

            w = open_w_if_exisis(session, p)
            
            if ( w is None):
                w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove[p]))
                save_new_w(session, p, w)
            else:
                independentComponents = np.dot(w, Y)

            independentComponents = remove_emg_artifacts(independentComponents)
            independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
            independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsampleWindowInMs, freq)

            #create a new components:
            special_comps_index = load_special_components_if_exists(data_path, session, p)
            if (special_comps_index is None):
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
            both_chunks[p] = EdfAnalyzer.combine_callibration_ticks(chunks)

        if (breaked):
            continue
        
        corr_results = main_corrolation_func(session, both_componenets, both_chunks)

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
            inertialData = remove_emg_artifacts(inertialData)

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


def run_gyro_analysis(main_corrolation_func, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, headers, csv_out_path):
    sessions_list = GetSessions(data_path)
    all_correlations = []

    for session in tqdm(sessions_list):
        breaked = False
        print(session)

        both_componenets = [[],[]]
        both_chunks = [[],[]]

        path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
        for p in [A,B]:
            Y, chunks, freq, inertialData, inertialFreq = EdfAnalyzer.ReadWithInertial(path_and_corrections[p][0], path_and_corrections[p][1], downsampleWindowInMs, just_accelorometer=False)

            if ( len(inertialData) == 3):
                breaked = True
                break

            inertialData = inertialData[0:3] #gyro only

            inertialData = zscore(inertialData, axis=1)
            #inertialData = remove_emg_artifacts(inertialData)

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