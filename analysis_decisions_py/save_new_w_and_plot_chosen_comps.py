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
from General.HtmlTablePlot import initHtmlDictionary, SaveHtmlFile, addRow, toImageHtmlColFromPath, ColType, toTextHtmlCol, toLinkPhotoHtmlColFromPath, toImageHtmlColFromPathWithSrc
from ComponentAnalysis.ReviewComponents_batch2 import GetSessionsBatch2, get_w_path, get_bad_electrode, get_chosen_comps, is_no_review_folder
from Hilas_component_classification.classifying_ica_components import wavelet_denoising, center, whiten


def get_special_component_indexes(w, list_of_faulty_channels):
    #should replace with HILA's code.
    special_comps_getters = [GetNoseComponent, GetSideEyeComponent, GetZygomaticusComponent, GetCorrugatorComponent]
    return [getter(w, list_of_faulty_channels) for getter in special_comps_getters]

def get_special_component_names():
    return ["Nose", "Eye", "Zygomaticus", "Corrugator"]

def get_component_analysis_folder(data_path, session, p, no_review = False):
    if ( no_review ):
        return os.path.join(data_path, session, "DWT_all_comps_db6", particiapnt_to_char(p))
    return os.path.join(data_path, session, "IC_w_and_photos", particiapnt_to_char(p))

def save_new_w(session, p, w, data_path = data_path, no_review = False):
    folder = get_component_analysis_folder(data_path, session, p, no_review)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "w.npy"), w)

def open_w_if_exisis(session, p, no_review = False):
    try:
        return np.load(os.path.join(get_component_analysis_folder(data_path, session, p, no_review), "w.npy"))
    except:
        return None
    
def open_w_is_exists_batch2(session, p):
    try:
        return np.load(get_w_path(session, p))
    except:
        return None

def open_w_if_exists_batch2_dwt(session, p):
    try:
        base = get_component_analysis_folder(data_path, session, p, no_review = True)
        path = os.path.join(base, "w.npy")
        return np.load(path)
    except:
        return None

def save_new_w_batch2(session, p, w):
    w_path = get_w_path(session, p)
    folder = os.path.dirname(w_path)
    os.makedirs(folder, exist_ok=True)
    np.save(w_path, w)



def save_special_components(data_path, chosen_comps, session, p, no_review = False):
    file_path = os.path.join(get_component_analysis_folder(data_path, session, p, no_review), "components_chosen.txt")
    with open(file_path, "w") as file:
        file.write("\n".join(map(str, chosen_comps)))

def load_special_components_if_exists(data_path, session, p, no_review = False):
    folder = get_component_analysis_folder(data_path, session, p, no_review)
    file_path = os.path.join(folder, "components_chosen.txt")
    #if file exists:
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return list(map(int, file.read().split("\n")))
    return None

def do_single_session(session, data_path = data_path):
    elecs_to_remove, is_dark = get_review_data(session, data_path)
    elecs_to_remove = [[],[]]
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    for p in [A,B]:
        print(elecs_to_remove[p])

        #w = open_w_if_exisis(session, p)
        w=None
        
        if ( w is None):
            Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1])
            Y = remove_rows_from_matrix(Y, elecs_to_remove[p])
            Y = wavelet_denoising(Y, freq, 'db6')
            #Y = zscore(Y, axis=1)
            Y, _ = center(Y)
            Y = whiten(Y)
            w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove[p]), whiten=False)
            save_new_w(session, p, w, data_path, no_review = True)
        # else:
        #     independentComponents = np.dot(w, Y)
        chosen_components_index = load_special_components_if_exists(data_path, session, p, no_review=True)
        if (chosen_components_index is None):
            chosen_components_index = get_special_component_indexes(w, elecs_to_remove[p])
            save_special_components(data_path, chosen_components_index, session, p, no_review = True)
            
        
        print(chosen_components_index)
        default_heatmap_with_channels_removal(w, elecs_to_remove[p], selected=chosen_components_index, 
                                            save_at=get_component_analysis_folder(data_path, session, p, no_review=True), 
                                            save_different_figures=True,
                                            extra_headers=get_special_component_names())
        
        #save whole heatmap
        default_heatmap_with_channels_removal(w, elecs_to_remove[p],
                                    save_at=get_component_analysis_folder(data_path, session, p, no_review=True), 
                                    save_different_figures=False,
                                    save_whole_heatmap=True)
        print("end")


def do_single_session_no_review(session, data_path = data_path):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    for p in [A,B]:
        #w = open_w_if_exisis(session, p, no_review = True)
        w=None
        if ( w is None):
            Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1])
            #to remove:
            Y = wavelet_denoising(Y, freq, 'db15')
            #Y, _ = center(Y)
            #Y = whiten(Y)
            #end remove. uncomment next line.
            Y = zscore(Y, axis=1)
            w, independentComponents = EdfAnalyzer.ICA(Y, 16)
            save_new_w(session, p, w, data_path, no_review =True)
        # else:
        #     independentComponents = np.dot(w, Y)
        chosen_components_index = load_special_components_if_exists(data_path, session, p, no_review =True)
        if (chosen_components_index is None):
            chosen_components_index = get_special_component_indexes(w, [])
            save_special_components(data_path, chosen_components_index, session, p, no_review = True)
            
        save_at_folder = get_component_analysis_folder(data_path, session, p, no_review =True)
        print(chosen_components_index)
        default_heatmap_with_channels_removal(w, [], selected=chosen_components_index, 
                                            save_at= save_at_folder,
                                            save_different_figures=True,
                                            extra_headers=get_special_component_names())
        
        #save whole heatmap
        default_heatmap_with_channels_removal(w, [],
                                    save_at=save_at_folder,
                                    save_different_figures=False,
                                    save_whole_heatmap=True)
        print("end")


def do_single_session_batch2(session, data_path = data_path):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))
    for p in [A,B]:
        elecs_to_remove = get_bad_electrode(session, p)
        w = open_w_if_exists_batch2_dwt(session,p) #open_w_is_exists_batch2(session, p)
        
        if ( w is None):
            Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1])
            Y = remove_rows_from_matrix(Y, elecs_to_remove)
            Y = zscore(Y, axis=1)
            w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(elecs_to_remove))
            save_new_w_batch2(session, p, w)
        # else:
        #     independentComponents = np.dot(w, Y)
        
        chosen_components_index = get_chosen_comps(session, p)

        without_review = True #is_no_review_folder(session, p)

        save_special_components(data_path, chosen_components_index, session, p, no_review = without_review)
        
        default_heatmap_with_channels_removal(w, elecs_to_remove, selected=chosen_components_index, 
                                            save_at=get_component_analysis_folder(data_path, session, p, without_review),
                                            save_different_figures=True,
                                            extra_headers=get_special_component_names())
        
        #save whole heatmap
        default_heatmap_with_channels_removal(w, elecs_to_remove,
                                    save_at=get_component_analysis_folder(data_path, session, p, without_review), 
                                    save_different_figures=False,
                                    save_whole_heatmap=True)
        print("end")




def get_int_list_input_from_user(prompt):
    if (prompt == ""):
        return []
    return [int(x) for x in input(prompt).split()]

        
def get_session_to_fix():
    return ["05122022_1600"]


def go_manually_and_edit_chosen_components(data_path = data_path):
    sessions = get_session_to_fix()
    for session in tqdm(sessions):
        for p in [A,B]:
            chosen_comps = get_int_list_input_from_user(f"Enter your suitable components for session {session} {particiapnt_to_char(p)}: ")
            if ( len(chosen_comps) == 0):
                continue
            save_special_components(data_path, chosen_comps, session, p)

def main(data_path = data_path):
    sessions_list = GetSessions(data_path) #GetSessionsBatch2()
    for session in tqdm(sessions_list):
        print("\n")
        print(session)
        print("\n")
        #do_single_session_no_review(session, data_path)
        do_single_session(session, data_path)
        #do_single_session_batch2(session, data_path)
    
    #create_html_batch2(data_path)
    create_html_after_dwt(data_path)

def create_html(data_path = data_path, no_review=False):
    session_list = GetSessions(data_path)
    html_dict = initHtmlDictionary(["sesssion", "participant"] + get_special_component_names() +["whole_heatmap"])
    for session in tqdm(session_list):
        for p in [A,B]:
            folder = get_component_analysis_folder(data_path, session, p, no_review)
            #prepare all image cols:
            image_paths = [os.path.join(folder, f + saved_images_extention) for f in get_special_component_names()]
            image_cols = [toImageHtmlColFromPath(f) for f in image_paths]
            whole_heatmap_path = os.path.join(folder, "whole_heatmap" + saved_images_extention)
            image_cols += [toImageHtmlColFromPath(whole_heatmap_path)]
            addRow(html_dict, [toTextHtmlCol(session), toTextHtmlCol(particiapnt_to_char(p))] +
                    image_cols)
    
    SaveHtmlFile(html_dict, os.path.join(data_path, "IC_w_and_photos.html"))


def create_html_for_new_batch(data_path = data_path):
    session_list = GetSessions(data_path)
    html_dict = initHtmlDictionary(["sesssion", "participant"]+ [f"with_removal_{c}" for c in get_special_component_names()] + [f"no_removal_{c}" for c in get_special_component_names()] + ["heatmap_with_removal", "heatmap_no_removal"])
    for session in tqdm(session_list):
        for p in [A,B]:
            image_cols = []
            folder_no_removal = get_component_analysis_folder(data_path, session, p, True) #get_component_analysis_folder(data_path, session, p)
            folder_with_removal = get_component_analysis_folder(data_path, session, p, False)
            #prepare all image cols:

            for folder in [folder_with_removal, folder_no_removal]:
                image_paths = [os.path.join(folder, f + saved_images_extention) for f in get_special_component_names()]
                image_cols += [toImageHtmlColFromPathWithSrc(f) for f in image_paths]

            heatmap_no_removal = os.path.join(folder_no_removal, "whole_heatmap" + saved_images_extention)
            heatmap_with_removal = os.path.join(folder_with_removal, "whole_heatmap" + saved_images_extention)
            image_cols += [toImageHtmlColFromPathWithSrc(heatmap_with_removal), toImageHtmlColFromPathWithSrc(heatmap_no_removal)]
            addRow(html_dict, [toTextHtmlCol(session), toTextHtmlCol(particiapnt_to_char(p))] +
                    image_cols)
    
    SaveHtmlFile(html_dict, os.path.join(data_path, "review_with_and_without_removal_with_comps_src.html"))

def create_html_batch2(data_path = data_path):
    session_list = GetSessionsBatch2()
    html_dict = initHtmlDictionary(["sesssion", "participant"] + get_special_component_names() +["whole_heatmap"])
    for session in tqdm(session_list):
        for p in [A,B]:
            #without_review = is_no_review_folder(session, p)
            #folder = get_component_analysis_folder(data_path, session, p, without_review)
            folder = os.path.dirname(get_w_path(session, p))
            #prepare all image cols:
            image_paths = [os.path.join(folder, f + saved_images_extention) for f in get_special_component_names()]
            image_cols = [toImageHtmlColFromPath(f) for f in image_paths]
            whole_heatmap_path = os.path.join(folder, "whole_heatmap" + saved_images_extention)
            image_cols += [toImageHtmlColFromPath(whole_heatmap_path)]
            addRow(html_dict, [toTextHtmlCol(session), toTextHtmlCol(particiapnt_to_char(p))] +
                    image_cols)
    
    SaveHtmlFile(html_dict, os.path.join(data_path, "After_choices_batch_2_au6.html"))


def create_html_after_dwt(data_path = data_path):
    session_list = GetSessions(data_path) #GetSessionsBatch2()
    html_dict = initHtmlDictionary(["sesssion", "participant"] + get_special_component_names() +["whole_heatmap"])
    for session in tqdm(session_list):
        for p in [A,B]:
            folder = get_component_analysis_folder(data_path, session, p, no_review = True)
            #prepare all image cols:
            image_paths = [os.path.join(folder, f + saved_images_extention) for f in get_special_component_names()]
            image_cols = [toImageHtmlColFromPath(f) for f in image_paths]
            whole_heatmap_path = os.path.join(folder, "whole_heatmap" + saved_images_extention)
            image_cols += [toImageHtmlColFromPath(whole_heatmap_path)]
            addRow(html_dict, [toTextHtmlCol(session), toTextHtmlCol(particiapnt_to_char(p))] +
                    image_cols)
    
    SaveHtmlFile(html_dict, os.path.join(data_path, "after_dwt_and_whiten_db6_batch1.html"))


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
            

