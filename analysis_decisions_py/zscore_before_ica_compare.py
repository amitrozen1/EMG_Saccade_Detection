# todo:
# 1. regular shared ICA + plot smiles and heatmaps
# 2. zscore before shared ICA + plot smiles and heatmaps

import os
from General.HelperFunctions import getPathsFromSessionFolder, cleanSpace, getICACsvFileNameWithPath 
import pyedflib
from EDF.EdfAnalyzer import EdfAnalyzer
import numpy as np
from scipy.stats import zscore
from EDF.PlotHeatmap import get_heatmap, get_participant_photo_path

A=0
B=1

num_channels = 16

def get_appended_signal_and_point_of_seperation(data_path, current_session, should_zscore_before_ica):

    paths = getPathsFromSessionFolder(os.path.join(data_path, current_session))
    
    signals = []
    chunks = [[],[]]

    print("Load Files")
    for p in [A,B]:
        pathProperties = paths[p]
        Y, chunks[p], freq = EdfAnalyzer.Read(pathProperties[0], pathProperties[1])
        cleanSpace()

    if should_zscore_before_ica:
        print("Zscore")
        signals[A] = zscore(signals[A], axis=1)
        signals[B] = zscore(signals[B], axis=1)

    appendedSignal = np.append(signals[A], signals[B], axis=1)

    return appendedSignal, len(signals[A][0])

def shared_ica(data_path, current_session, should_zscore_before_ica = False):
    appendedSignal, point_of_seperation = get_appended_signal_and_point_of_seperation(data_path, current_session, should_zscore_before_ica)
    print("ICA")
    w, x = EdfAnalyzer.ICA(appendedSignal, num_channels)
    # not important now    np.savetxt(getICACsvFileNameWithPath(data_path, current_session), w, delimiter=",")
    return w, x, point_of_seperation

def plot_heatmap(image_path, w):
    get_heatmap(image_path, num_channels, w, False, False, "heatmaps_zscore_check_shared")

if __name__== "__main__":
    data_path = r"C:\Liron\DataEmg\Try"
    current_session = "04122022_1545"
    w, x, point_of_seperation = shared_ica(data_path, current_session, should_zscore_before_ica=True)
    plot_heatmap(get_participant_photo_path(data_path, current_session, A), w[:point_of_seperation])