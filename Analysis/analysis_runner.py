import os
from General.HelperFunctions import GetSessions, cleanSpace
import numpy as np
from scipy.stats import zscore
from General.init import *
import pandas as pd
from PsychData.LoadChoicesJson import loadChoiceFromSession
from tqdm import tqdm
from General.HelperFunctions import getPathsFromSessionFolder, remove_rows_from_matrix
from EDF.EdfAnalyzer import EdfAnalyzer
from General.HelperFunctions import get_review_data
from ComponentAnalysis.ComponentExtractor import open_w_if_exisis, open_whiten_if_exisis, load_special_components_if_exists, get_special_component_indexes, GetLipCornerDownPuller, GetComponentByElecrodesNumber
from analysis_decisions_py.cleaning_signal import remove_emg_artifacts
from analysis_decisions_py.original_actress_readings import export_actress_chunked_data, export_actress_chunked_data_with_dwt
from scipy import signal
from ComponentAnalysis.ReviewComponents_batch2 import GetSessionsBatch2, open_w_if_exists_batch2, get_bad_electrode, get_chosen_comps, is_no_review_folder
from PsychData.Experiment_CSV import read_experiment_csv_batch2
from Hilas_component_classification.classifying_ica_components import wavelet_denoising, whiten, center

def run_emg_analysis_batch2(main_corrolation_func, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, headers, csv_out_path, with_actress=False, hard_noise_remove_for_actress = False):
    sessions = GetSessionsBatch2()
    all_correlations = []

    if (with_actress):
        actress_chunks = export_actress_chunked_data(downsampleWindowInMs, rms_size_ms, hard_noise_remove_for_actress)
    
    for session in tqdm(sessions):
        print(session)

        both_componenets = [[],[]]
        both_chunks = [[],[]]
        path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))

        for p in [A,B]:
            Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsampleWindowInMs)
            elecs_to_remove = get_bad_electrode(session, p)
            Y = remove_rows_from_matrix(Y, elecs_to_remove)
            Y = wavelet_denoising(Y, freq, 'db15')
            Y = zscore(Y, axis=1)

            w = open_w_if_exists_batch2(session, p)
            
            if ( w is None):
                w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove))
                #save new w.. but it supposed to exist anyway..
            else:
                independentComponents = np.dot(w, Y)

            independentComponents = remove_emg_artifacts(independentComponents)
            independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
            independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsampleWindowInMs, freq)

            special_comps_index = get_chosen_comps(session, p)         
            
            total_max = np.max(independentComponents, axis=0)
            independentComponents = independentComponents[special_comps_index, :]
            
            independentComponents = np.vstack((independentComponents, total_max))

            both_componenets[p] = independentComponents
            both_chunks[p] = EdfAnalyzer.combine_callibration_ticks(chunks)
        
        if ( with_actress ):
            corr_results = main_corrolation_func(session, both_componenets, both_chunks, actress_chunks)
        else:
            corr_results = main_corrolation_func(session, both_componenets, both_chunks)

        choices = read_experiment_csv_batch2(data_path, session)
        for i in range(len(corr_results)):
            result = corr_results[i]
            key = result[1]
            key_splits = key.split("_")
            if ( len(key_splits) > 1):
                story_number = key_splits[1]
                if ( story_number != ''):
                    story_number = story_number.split('.')[0]
                    for choice in choices:
                        uc_story_number = choice[0].split('.')[0]
                        if ( story_number == uc_story_number ):
                            corr_results[i] = result + choice[1:]
                            break
        
        all_correlations += corr_results
        del independentComponents
        cleanSpace()
        print("finished session " + session)
    
    if (csv_out_path!= None):
        df = pd.DataFrame(all_correlations)
        df.columns = headers + ["confidence_A", "confidence_B"]
        df.to_csv(csv_out_path)


def get_neg_pos_comps(independentComponents, chunks, n=3):
    smile_s = int(chunks["smile__0"].Start.Time)
    smile_e = int(chunks["smile__2"].End.Time)
    angry_s = int(chunks["angry__0"].Start.Time)
    angry_e = int(chunks["angry__2"].End.Time)

    smile_data = independentComponents[:, smile_s:smile_e]
    angry_data = independentComponents[:, angry_s:angry_e]

    neg_indexes = np.argsort(np.mean(angry_data, axis=1) - np.mean(smile_data, axis=1))[-n:]
    pos_indexes = np.argsort(np.mean(smile_data, axis=1) - np.mean(angry_data, axis=1))[-n:]

    neg_comp = np.mean(independentComponents[neg_indexes, :], axis=0)
    pos_comp = np.mean(independentComponents[pos_indexes, :], axis=0)

    return [neg_comp, pos_comp]
    
def get_neg_pos_indexes(independentComponents, chunks):
    key = "smile__2"
    smile_s = int(chunks[key].Start.Time)
    smile_e = int(chunks[key].End.Time)
    key = "angry__2"
    angry_s = int(chunks[key].Start.Time)
    angry_e = int(chunks[key].End.Time)
    
    smile_data = independentComponents[:, smile_s:smile_e]
    angry_data = independentComponents[:, angry_s:angry_e]

    neg_comp = np.argmax(np.mean(angry_data, axis=1) - np.mean(smile_data, axis=1))
    pos_comp = np.argmax(np.mean(smile_data, axis=1) - np.mean(angry_data, axis=1))

    return [neg_comp, pos_comp]

def get_ok_sessions():
    if current_batch == 1:
        sessions =  GetSessions(data_path)
        sessions.remove("20122022_1415")
    else:
        #return ["10052023_1430", "09052023_1215", "08052023_1430"]
        excluded_list = ["25122023_1400", "25122023_1000", "21122023_1230", "18042023_1415", "25122023_1200", "30032023_1100", "30112023_1400", "22052023_1430"]
        extra = ["17052023_1430", "06122023_1000"]
        excluded_list += extra
        # can be improved "21122023_1500", "17052023_1430", "11122023_1400", "06122023_1000	"
        sessions = GetSessions(data_path)
        for session in excluded_list:
            if session in sessions:
                sessions.remove(session)

        # 09052023_1630 finished battery in A before time!!!
        if "09052023_1630" in sessions:
            sessions.remove("09052023_1630")

    return sessions

def run_emg_analysis(main_corrolation_func, rms_size_ms, downsampleWindowInMs, windowForCorrInMs, numOfPixelsForXCorrStep, headers, csv_out_path, with_actress=False, hard_noise_remove_for_actress = False):
    more_comps = ["25122023_1000", "25122023_1400", "21122023_1230"] #["25122023_1000", "25122023_1200", "25122023_1400", "21122023_1230", "06122023_1000"]
    sessions_list = more_comps #get_ok_sessions() #GetSessions(data_path)
    #get_ok_sessions()

    all_correlations = []

    if (with_actress):
        actress_chunks = export_actress_chunked_data_with_dwt(downsampleWindowInMs, rms_size_ms)
    
    for session in tqdm(sessions_list):
        breaked = False
        print(session)
        #elecs_to_remove, is_dark = get_review_data(session, data_path)
        elecs_to_remove = [[],[]]

        both_componenets = [[],[]]
        both_chunks = [[],[]]
        special_comps_array = [[],[]]

        path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
        for p in [A,B]:
            Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsampleWindowInMs)

            #Y = remove_rows_from_matrix(Y, elecs_to_remove[p])
            Y = wavelet_denoising(Y, freq, 'db15')
            Y, _ = center(Y)
            #Y = zscore(Y, axis=1)
            
            whiten_mat = open_whiten_if_exisis(session, p, dwt=True)
            if (whiten_mat is None):
                Y = whiten(Y)
            else:
                Y = np.dot(whiten_mat, Y)

            w = open_w_if_exisis(session, p, dwt=True)
            
            if ( w is None):
                w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove[p]), whiten=False)
                #save_new_w(session, p, w)
            else:
                independentComponents = np.matmul(w,Y)


            # special_comps_index = get_chosen_comps(session, p)
            # independentComponents = independentComponents[special_comps_index, :]
        

            #independentComponents = remove_emg_artifacts(independentComponents)
            independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
            #independentComponents = EdfAnalyzer.resample_by_averaging_every_n(independentComponents, downsampleWindowInMs, freq)
            hz = int(1000/downsampleWindowInMs)
            independentComponents = signal.resample(independentComponents, int(independentComponents.shape[1] / freq * hz), axis=1)

            #select components:
            special_comps_index = load_special_components_if_exists(data_path, session, p, dwt=True)
            if (special_comps_index is None):
                special_comps_index = get_special_component_indexes(w, elecs_to_remove[p])
            
            #special_comps_index += get_neg_pos_indexes(independentComponents, chunks)
            special_comps_index += [GetComponentByElecrodesNumber(w, elecs_to_remove[p], [2])]
            special_comps_index += [GetComponentByElecrodesNumber(w, elecs_to_remove[p], [5])]
            special_comps_index +=  [GetLipCornerDownPuller(w, elecs_to_remove[p])]

            special_comps_array[p] = special_comps_index

            independentComponents = independentComponents[special_comps_index, :]

            both_chunks[p] = EdfAnalyzer.combine_callibration_ticks(chunks)
            both_componenets[p] = independentComponents
            

        if (breaked):
            continue
        
        if ( with_actress ):
            corr_results = main_corrolation_func(session, both_componenets, both_chunks, actress_chunks)
        else:
            corr_results = main_corrolation_func(session, both_componenets, both_chunks)

        if (current_batch == 1):
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
        else:
            choices = read_experiment_csv_batch2(data_path, session)
            for i in range(len(corr_results)):
                result = corr_results[i]
                key = result[1]
                key_splits = key.split("_")
                if ( len(key_splits) > 1):
                    story_number = key_splits[1]
                    if ( story_number != ''):
                        story_number = story_number.split('.')[0]
                        for choice in choices:
                            uc_story_number = choice[0].split('.')[0]
                            if ( story_number == uc_story_number ):
                                corr_results[i] = result + choice[1:] + [session in get_ok_sessions()] + [str(special_comps_array[A]), str(special_comps_array[B])]
                                break

        all_correlations += corr_results
        del independentComponents
        cleanSpace()
        print("finished session " + session)
    
    if current_batch == 2:
        headers += ["confidence_A", "confidence_B", "ok_electrodes", "special_comps_A", "special_comps_B"]

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