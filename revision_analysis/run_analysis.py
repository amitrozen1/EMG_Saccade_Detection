from General.init import *
from General.HelperFunctions import cleanSpace
from revision_analysis.data_loading import load_dyad_data, load_actress_data
from revision_analysis.compute_cross_corr import get_headers_reading, get_headers_listening, main_corrolation_of_single_comps_reading, main_corrolation_of_single_comps_listening, get_headers_both_listeners, main_corrolation_of_single_comps_listeninig_participants, main_correlation_of_combinations_reading, get_headers_reading_combinations, get_headers_reading_GC, main_correlation_reading_GC
from Analysis.analysis_runner import get_ok_sessions
import pandas as pd
from PsychData.LoadChoicesJson import loadChoiceFromSession
from PsychData.Experiment_CSV import read_experiment_csv_batch2, get_results_after_revision
from tqdm import tqdm

def wrap_in_default_headers(headers, action="reading"):
    if "reading" in action or action == "listening_participants" :
        return ["session", "key","action"] + headers + ['whoseReading', 'isChoiceA', 'isChoiceB', 'rtChoiceA', 'rtChoiceB', 'isOther', 'ChoseTogether']
    else:
        return ["session", "key","action", 'participant'] + headers + ['choice', 'rtChoice', 'isOther']

def run_analysis(rms_size_ms, ds_factor, window_ms, max_lag_ms, csv_out_path, action="reading"):
    all_correlations = []
    actress_chunks = None
    
    if action == "reading":
        headers = wrap_in_default_headers(get_headers_reading())
    elif action == "listening":
        headers = wrap_in_default_headers(get_headers_listening(), action=action)
        actress_chunks = load_actress_data(rms_size_ms, ds_factor)
    elif action == "listening_participants":
        headers = wrap_in_default_headers(get_headers_both_listeners(), action=action)
    elif action == "reading_combinations":
        headers = wrap_in_default_headers(get_headers_reading_combinations())
    elif action == "reading_GC":
        headers = wrap_in_default_headers(get_headers_reading_GC(), action=action)

    for session in tqdm(get_ok_sessions(action)):
        both_comps, both_chunks = load_dyad_data(session, ds_factor, rms_size_ms)
        if action == "reading":
            corr_results = main_corrolation_of_single_comps_reading(session, both_comps, both_chunks, window_ms, max_lag_ms, ds_factor)
        elif action == "listening":
            corr_results = main_corrolation_of_single_comps_listening(session, both_comps, both_chunks, actress_chunks, window_ms, max_lag_ms, ds_factor)
        elif action == "listening_participants":
            corr_results = main_corrolation_of_single_comps_listeninig_participants(session, both_comps, both_chunks, window_ms, max_lag_ms, ds_factor)
        elif action=="reading_combinations":
            corr_results = main_correlation_of_combinations_reading(session, both_comps, both_chunks, window_ms, max_lag_ms, ds_factor)
        elif action == "reading_GC":
            corr_results = main_correlation_reading_GC(session, both_comps, both_chunks, window_ms, max_lag_ms, ds_factor)
        
        if "reading" in action or action == "listening_participants":
            if (get_current_batch() == 1):
                choices = loadChoiceFromSession(get_data_path(), session)
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
                                    corr_results[i] = result + choice[1:8]
                                    break
            else:
                choices = read_experiment_csv_batch2(get_data_path(), session)
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
                                    corr_results[i] = result + choice[1:8]
                                    break
        else: #listening
            for i in range(len(corr_results)):
                key = corr_results[i][1]
                story = key.split("_")[1].split(".")[0]
                p = corr_results[i][3]
                choice = get_results_after_revision(get_data_path(), session, p, story)
                corr_results[i] += choice

        all_correlations += corr_results
        #del independentComponents
        cleanSpace()
        print("finished session " + session)
    
    # if current_batch == 2:
    #     headers += ["confidence_A", "confidence_B", "ok_electrodes", "special_comps_A", "special_comps_B"]

    
    df = pd.DataFrame(all_correlations)
    df.columns = headers
    if (csv_out_path!= None):
        df.to_csv(csv_out_path, index=False)
    else:
        return df

