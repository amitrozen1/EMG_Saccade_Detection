import os
from General.HelperFunctions import GetSessions, cleanSpace, reshape_num_channels, RemoveRowFromMatrix, particiapnt_to_char
import numpy as np
from scipy.stats import zscore
from General.init import *
from tqdm import tqdm
from General.HelperFunctions import getPathsFromSessionFolder, remove_rows_from_matrix
from EDF.EdfAnalyzer import EdfAnalyzer
from ComponentAnalysis.ComponentExtractor import GetCorrugatorComponent, GetNoseComponent, GetSideEyeComponent, GetZygomaticusComponent
from General.HelperFunctions import get_review_data
from EDF.PlotHeatmap import default_heatmap_with_channels_removal
from General.PlotFunctions import clearPlot, plotChunkData
from General.HtmlTablePlot import initHtmlDictionary, SaveHtmlFile, addRow, toImageHtmlColFromPath, ColType, toTextHtmlCol

def get_special_component_indexes(w, list_of_faulty_channels):
    #should replace with HILA's code.
    special_comps_getters = [GetNoseComponent, GetSideEyeComponent, GetZygomaticusComponent, GetCorrugatorComponent]
    return [getter(w, list_of_faulty_channels) for getter in special_comps_getters]

def get_special_component_names():
    return ["Nose", "Eye", "Zygomaticus", "Corrugator"]

def get_component_analysis_folder(data_path, session, p ):
    return os.path.join(data_path, session, "IC_w_and_photos", particiapnt_to_char(p))

def save_new_w(session, p, w):
    folder = get_component_analysis_folder(data_path, session, p)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "w.npy"), w)

def open_w_if_exisis(session, p):
    try:
        return np.load(os.path.join(get_component_analysis_folder(data_path, session, p), "w.npy"))
    except:
        return None

def save_special_components(chosen_comps, session, p):
    file_path = os.path.join(get_component_analysis_folder(data_path, session, p), "components_chosen.txt")
    with open(file_path, "w") as file:
        file.write("\n".join(map(str, chosen_comps)))


def load_special_components_if_exists(data_path, session, p):
    folder = get_component_analysis_folder(data_path, session, p)
    file_path = os.path.join(folder, "components_chosen.txt")
    #if file exists:
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return list(map(int, file.read().split("\n")))
    return None

def do_single_session(session):
    elecs_to_remove, is_dark = get_review_data(session, data_path)
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    for p in [A,B]:
        print(elecs_to_remove[p])

        w = open_w_if_exisis(session, p)
        
        if ( w is None):
            Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1])
            Y = remove_rows_from_matrix(Y, elecs_to_remove[p])
            Y = zscore(Y, axis=1)
            w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove[p]))
            save_new_w(session, p, w)
        # else:
        #     independentComponents = np.dot(w, Y)
        chosen_components_index = load_special_components_if_exists(data_path, session, p)
        if (chosen_components_index is None):
            chosen_components_index = get_special_component_indexes(w, elecs_to_remove[p])
            save_special_components(chosen_components_index, session, p)
            
        
        print(chosen_components_index)
        default_heatmap_with_channels_removal(w, elecs_to_remove[p], selected=chosen_components_index, 
                                            save_at=get_component_analysis_folder(data_path, session, p), 
                                            save_different_figures=True,
                                            extra_headers=get_special_component_names())
        
        #save whole heatmap
        default_heatmap_with_channels_removal(w, elecs_to_remove[p],
                                    save_at=get_component_analysis_folder(data_path, session, p), 
                                    save_different_figures=False,
                                    save_whole_heatmap=True)
        print("end")


def get_int_list_input_from_user(prompt):
    if (prompt == ""):
        return []
    return [int(x) for x in input(prompt).split()]

        
def get_session_to_fix():
    return ["05122022_1600"]


def go_manually_and_edit_chosen_components():
    sessions = get_session_to_fix()
    for session in tqdm(sessions):
        for p in [A,B]:
            chosen_comps = get_int_list_input_from_user(f"Enter your suitable components for session {session} {particiapnt_to_char(p)}: ")
            if ( len(chosen_comps) == 0):
                continue
            save_special_components(chosen_comps, session, p)

def main():
    sessions_list = GetSessions(data_path)
    for session in tqdm(sessions_list):
        print("\n")
        print(session)
        print("\n")
        do_single_session(session)
    create_html()

def create_html():
    session_list = GetSessions(data_path)
    html_dict = initHtmlDictionary(["sesssion", "participant"] + get_special_component_names() +["whole_heatmap"])
    for session in session_list:
        for p in [A,B]:
            folder = get_component_analysis_folder(data_path, session, p)
            #prepare all image cols:
            image_paths = [os.path.join(folder, f + saved_images_extention) for f in get_special_component_names()]
            image_cols = [toImageHtmlColFromPath(f) for f in image_paths]
            whole_heatmap_path = os.path.join(folder, "whole_heatmap" + saved_images_extention)
            image_cols += [toImageHtmlColFromPath(whole_heatmap_path)]
            addRow(html_dict, [toTextHtmlCol(session), toTextHtmlCol(particiapnt_to_char(p))] +
                    image_cols)
    
    SaveHtmlFile(html_dict, os.path.join(data_path, "IC_w_and_photos_manually_fixed_2.html"))


def plot_selected_components():
    session_list = ["07122022_1230", "11122022_1500"]
    for session in session_list:
        elecs_to_remove, is_dark = get_review_data(session, data_path)
        for p in [A,B]:
            #load w:
            w = open_w_if_exisis(session, p)
            #load edf:
            path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
            Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsample_window_in_ms=50)
            Y = remove_rows_from_matrix(Y, elecs_to_remove[p])
            Y = zscore(Y, axis=1)
            components = np.dot(w, Y)
            #load chosen components:
            chosen_comps = load_special_components_if_exists(data_path, session, p)
            components = components[chosen_comps, :]

            #rms:
            components = EdfAnalyzer.window_rms(components, 100, freq)
            #downsample:
            components = EdfAnalyzer.resample_by_averaging_every_n(components, 50, freq)

            #get ticks:
            for state in ["smile", "angry", "blink"]:
                ticks = EdfAnalyzer.getCallibrationTicks(chunks, state)
                #fix ticks:
                s, e = ticks[0], ticks[-1]
                ticks = [t-s for t in ticks]
                #plot chunk data : 
                plotChunkData(components[:, s:e], f"{session}_{particiapnt_to_char(p)}_{state}_components",
                               ticks, get_special_component_names(), num_components = 4, show=True)
            

