from analysis_decisions_py.validation_of_correlation_via_smile_anger import load_real_dyad_data, compute_single_corrolations
from General.PlotFunctions import plotDataFromBothChunks, clearPlot
from tqdm import tqdm
from General.HelperFunctions import GetSessions
from General.init import *
from analysis_decisions_py.save_new_w_and_plot_chosen_comps import get_component_analysis_folder
import numpy as np
import os
import easygui
from General.HelperFunctions import particiapnt_to_char

SMILE = 0
ANGRY = 1

rms_size_ms = 400
downsampleWindowInMs = 50

def plot_couples_graph_and_xcorr(session):
    vectors = load_real_dyad_data(session)
    window_size = 1000
    lag = window_size*2

    smile_corrs = compute_single_corrolations([vectors[A][SMILE], vectors[B][SMILE]], "smile", window_size, lag)
    print(smile_corrs)
    plotDataFromBothChunks(vectors[A][SMILE], vectors[B][SMILE], [], [], "smile", 4)
    clearPlot()
    angry_corrs = compute_single_corrolations([vectors[A][ANGRY], vectors[B][ANGRY]], "angry", window_size, lag)
    print(angry_corrs)
    plotDataFromBothChunks(vectors[A][ANGRY], vectors[B][ANGRY], [], [], "angry", 4)
    clearPlot()


def get_int_list_input_from_user(prompt):
    if (prompt == ""):
        return []
    answer = easygui.enterbox(prompt)
    return [int(x) for x in str(answer).split()]

# problematic done : '05122022_1600', '07122022_1230', '13122022_1415', '14122022_1030', '14122022_1615',
# 14122022_1615 had a large noise comopnent in the beginning of smile 1 and I want to check how to overcome this before continuing.
problematic_sessions = [ '15122022_1430', '18122022_1115', '20122022_0830', '20122022_1030', '20122022_1215', '23112022_0830', '26122022_1015', '26122022_1430', '27122022_1215', '27122022_1415', '30112022_0830']

def plot_sessions_and_decide_on_components():
    sessions = problematic_sessions[1:]
    for session in tqdm(sessions):
        print(session)
        plot_couples_graph_and_xcorr(session)
        
        #get input list of ints from user:
        for p in [A,B]:
            p_char = particiapnt_to_char(p)
            which_comps_to_change = get_int_list_input_from_user(f"Enter your bad components for {p_char}: ")
            print(which_comps_to_change)
            if ( len(which_comps_to_change) != 0):
                folder = get_component_analysis_folder(data_path, session, p)
                #save list of replaced components:
                np.save(os.path.join(folder, "which_comps_to_change"), which_comps_to_change)
            else:
                #delete the file is exists:
                folder = get_component_analysis_folder(data_path, session, p)
                file = os.path.join(folder, "which_comps_to_change.npy")
                if ( os.path.exists(file) ):
                    os.remove(file)



def print_problematic_sessions():
    sessions = GetSessions(data_path)
    problematic = []
    for session in tqdm(sessions):
        for p in [A,B]:
            folder = get_component_analysis_folder(data_path, session, p)
            file = os.path.join(folder, "which_comps_to_change.npy")
            # if file exists:
            if ( os.path.exists(file) ):
                print(session)
                problematic.append(session)
                break
    
    print(problematic)