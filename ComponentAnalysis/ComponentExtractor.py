import numpy as np
from numpy.linalg import inv
from scipy.stats import zscore
import os
from General.HelperFunctions import particiapnt_to_char
from General.init import *

def insert_zeros_in_removed_places(vector, electrodes_to_remove):
    for electrode in electrodes_to_remove:
        vector = np.insert(vector, electrode, 0)
    return vector


def find_first_match_index(vector, value):
    for i, v in enumerate(vector):
        if v == value:
            return i
    return -1

def find_match_indexes(vector, value):
    indexes = []
    for i, v in enumerate(vector):
        if v == value:
            indexes.append(i)
    return indexes

def find_first_match_index_with_prioritized_values(vector, values):
    for value in values:
        index = find_first_match_index(vector, value)
        if index != -1:
            return index
    return -1

def find_matched_index(vector, values, new_w):
    for value in values:
        indexes = find_match_indexes(vector, value)
        if ( len(indexes) == 0):
            continue
        if ( len(indexes) == 1):
            return indexes[0]
        
        scores = [zscore(new_w[:,index])[value-1] for index in indexes]
        return indexes[np.argmax(scores)]
    return -1

def find_matched_index_zscore_method(new_w, value):
    return np.argmax(zscore(new_w, axis=0)[value-1])

#todo - if there are two - how to choose the best one???
def GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, electrodes_channel_defintion):
    number_of_channels = 16 - len(removed_electrodes_zero_based)
        # calculations for the heatmap
    inverse = np.absolute(inv(w)) 
    new_w = np.zeros((16,number_of_channels))
    for i in range(number_of_channels):
        new_w[:,i] = insert_zeros_in_removed_places(inverse[:,i], removed_electrodes_zero_based)

    #value to send:
    chosen_electrode_definition = electrodes_channel_defintion[0]
    for value in electrodes_channel_defintion:
        zero_based_value = value - 1
        if ( zero_based_value in removed_electrodes_zero_based):
            continue
        else:
            chosen_electrode_definition = value
            break
    
    return find_matched_index_zscore_method(new_w, chosen_electrode_definition)

    ic_channel_to_electrode_number = np.argmax(new_w, axis=0) + 1
    ic_loc = find_matched_index(ic_channel_to_electrode_number, electrodes_channel_defintion, new_w)
    #find_first_match_index_with_prioritized_values(ic_channel_to_electrode_number, electrodes_channel_defintion)

    return ic_loc

def convert_comp_number_to_electrode_number(comp_number, w):
    inverse = np.absolute(inv(w)) 
    ic_channel_to_electrode_number = np.argmax(inverse, axis=0) + 1
    return ic_channel_to_electrode_number[comp_number]

def GetNoseComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, [9, 8]) #only 9

def GetSideEyeComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, [10,11,12]) # [10, 11, 12]

def GetZygomaticusComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, [6, 5, 7, 3, 4]) #  [5, 6, 3, 4]

def GetCorrugatorComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, [14, 13, 15, 16]) # [14, 13, 15, 16]

def GetOrbicularisOrisComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, [1])

def GetLipCornerDownPuller(w, removed_electrodes_zero_based):
    a = GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, [4])
    if ( a == -1):
        return 0
    return a


def get_special_component_indexes(w, list_of_faulty_channels=[]):
    #should replace with HILA's code.
    special_comps_getters = [GetNoseComponent, GetSideEyeComponent, GetZygomaticusComponent, GetCorrugatorComponent]
    return [getter(w, list_of_faulty_channels) for getter in special_comps_getters]

def get_special_component_names(withValenceComps = False):
    base = ["Nose", "Eye", "Zygomaticus", "Corrugator"]
    if (withValenceComps):
        base += ["NegComp", "PosComp"]
    base += ["LipCornerDownPuller"]
    return base

def get_component_analysis_folder(data_path, session, p, dwt=False):
    if (dwt):
        return os.path.join(data_path, session, "DWT_hila_db15", particiapnt_to_char(p))
    return os.path.join(data_path, session, "IC_w_and_photos", particiapnt_to_char(p))

def open_w_if_exisis(session, p, dwt=False):
    try:
        return np.load(os.path.join(get_component_analysis_folder(data_path, session, p, dwt), "w.npy"))
    except:
        return None

def open_whiten_if_exisis(session, p, dwt=False):
    try:
        return np.load(os.path.join(get_component_analysis_folder(data_path, session, p, dwt), "white.npy"))
    except:
        return None

def save_new_w(session, p, w):
    folder = get_component_analysis_folder(data_path, session, p)
    os.makedirs(folder, exist_ok=True)
    np.save(os.path.join(folder, "w.npy"), w)

def load_special_components_if_exists(data_path, session, p, dwt=False):
    folder = get_component_analysis_folder(data_path, session, p, dwt)
    file_path = os.path.join(folder, "components_chosen.txt")
    #if file exists:
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return list(map(int, file.read().split("\n")))
    return None