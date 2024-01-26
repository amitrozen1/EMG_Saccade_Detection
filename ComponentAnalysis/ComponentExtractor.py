import numpy as np
from numpy.linalg import inv

def insert_zeros_in_removed_places(vector, electrodes_to_remove):
    for electrode in electrodes_to_remove:
        vector = np.insert(vector, electrode, 0)
    return vector


def find_first_match_index(vector, value):
    for i, v in enumerate(vector):
        if v == value:
            return i
    return -1

def find_first_match_index_with_prioritized_values(vector, values):
    for value in values:
        index = find_first_match_index(vector, value)
        if index != -1:
            return index
    return -1

def GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, electrodes_channel_defintion):
    number_of_channels = 16 - len(removed_electrodes_zero_based)
        # calculations for the heatmap
    inverse = np.absolute(inv(w)) 
    new_w = np.zeros((16,number_of_channels))
    for i in range(number_of_channels):
        new_w[:,i] = insert_zeros_in_removed_places(inverse[:,i], removed_electrodes_zero_based)

    ic_channel_to_electrode_number = np.argmax(new_w, axis=0) + 1
    ic_loc = find_first_match_index_with_prioritized_values(ic_channel_to_electrode_number, electrodes_channel_defintion)

    return ic_loc

def GetNoseComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, [9, 8]) #only 9

def GetSideEyeComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, [10, 11, 12]) # 10 is better noisewise. flip the array.

def GetZygomaticusComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, [6, 5, 3, 4]) # maybe 3 is different. 3 and 6 sometime together.

def GetCorrugatorComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, [13, 14, 15, 16]) 

def GetOrbicularisOrisComponent(w, removed_electrodes_zero_based):
    return GetComponentByElecrodesNumber(w, removed_electrodes_zero_based, [1])