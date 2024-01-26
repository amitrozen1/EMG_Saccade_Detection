import os
from General.init import *
from scipy.stats import zscore
from EDF.EdfAnalyzer import EdfAnalyzer
from EDF.PlotHeatmap import get_heatmap, default_heatmap_with_channels_removal
import numpy as np
from General.PlotFunctions import plotChunkData
from General.HelperFunctions import remove_rows_from_matrix, replace_rows_with_zeros, replace_rows_with_ones, getPathsFromSessionFolder

from ComponentAnalysis.ComponentExtractor import GetCorrugatorComponent, GetNoseComponent, GetSideEyeComponent, GetZygomaticusComponent

def MeetingWithYael():
    folder = r"C:\Liron\DataEmg\MeetingWithYael"
    edf_path = os.path.join(folder, "07122023_1000_A_SD_filtered.edf")
    correction_value = -0.2

    Y, chunks, freq = EdfAnalyzer.Read(edf_path, correction_value)#, downsample_window_in_ms)
    # Y = zscore(Y, axis=1)

    list_of_faulty_channels = [1,3, 4, 7, 8, 11, 13, 15, 16]
    list_of_faulty_channels = [i-1 for i in list_of_faulty_channels]

    Y = remove_rows_from_matrix(Y, list_of_faulty_channels)


    w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(list_of_faulty_channels))

    # independentComponents = EdfAnalyzer.window_rms(independentComponents, 200, freq)
    # independentComponents = EdfAnalyzer.resample_by_taking_one_every_n(independentComponents, downsample_window_in_ms, freq)

    for state in ["smile", "angry", "blink"]:
        callibrationTicks = EdfAnalyzer.getCallibrationTicks(chunks, state)
        s = callibrationTicks[0]
        e = callibrationTicks[-1]
        callibrationTicks = np.array(callibrationTicks) - s
        plotChunkData(independentComponents[:, s:e],  state, callibrationTicks, num_components=independentComponents.shape[0])

    #get_heatmap(image_path, num_components, w, default_pic=True)
    default_heatmap_with_channels_removal(w, list_of_faulty_channels)

def check_without_12():
    session = "15122022_1430"
    p = A

    path, correction_value = getPathsFromSessionFolder(os.path.join(data_path, session))[p]
    Y, chunks, freq = EdfAnalyzer.Read(path, correction_value)
    
    list_of_faulty_channels = []
    list_of_faulty_channels = [i-1 for i in list_of_faulty_channels]

    Y = remove_rows_from_matrix(Y, list_of_faulty_channels)

    w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(list_of_faulty_channels))


    #print components names:
    print("Nose: ", GetNoseComponent(w, list_of_faulty_channels) + 1)
    print("SideEye: ", GetSideEyeComponent(w, list_of_faulty_channels) + 1)
    print("Zygomaticus: ", GetZygomaticusComponent(w, list_of_faulty_channels) + 1)
    print("Corrugator: ", GetCorrugatorComponent(w, list_of_faulty_channels) + 1)

    for state in ["smile", "angry", "blink"]:
        callibrationTicks = EdfAnalyzer.getCallibrationTicks(chunks, state)
        s = callibrationTicks[0]
        e = callibrationTicks[-1]
        callibrationTicks = np.array(callibrationTicks) - s
        plotChunkData(independentComponents[:, s:e],  state, callibrationTicks, num_components=independentComponents.shape[0])

    default_heatmap_with_channels_removal(w, list_of_faulty_channels)

def temporal_component():
    session = "23112022_0830"
    p = A
    list_of_faulty_channels = [9, 15, 16]

    rms_window_in_ms = 100
    downsample_window_in_ms = 20

    path, correction_value = getPathsFromSessionFolder(os.path.join(data_path, session))[p]
    Y, chunks, freq = EdfAnalyzer.Read(path, correction_value, downsample_window_in_ms)
    
    list_of_faulty_channels = [i-1 for i in list_of_faulty_channels]

    Y = remove_rows_from_matrix(Y, list_of_faulty_channels)

    Y = zscore(Y, axis=1)

    w, independentComponents = EdfAnalyzer.ICA(Y, 16-len(list_of_faulty_channels))

    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_window_in_ms, freq)
    independentComponents = EdfAnalyzer.resample_by_taking_one_every_n(independentComponents, downsample_window_in_ms, freq)

    special_comps_getters = [GetNoseComponent, GetSideEyeComponent, GetZygomaticusComponent, GetCorrugatorComponent]
    special_comps_index = [getter(w, list_of_faulty_channels) for getter in special_comps_getters]
    #take only these components:
    special_max = np.max(independentComponents[special_comps_index, :], axis=0)

    #stack an extra channel that is the max of the channels
    independentComponents = np.row_stack((independentComponents, np.max(independentComponents, axis=0)))
    independentComponents = np.row_stack((independentComponents, special_max))

    subtitles = ["component " + str(i) for i in range(1, independentComponents.shape[0] - 1)]
    subtitles.append("max component")
    subtitles.append("special_max")

    for state in ["smile", "angry", "blink"]:
        callibrationTicks = EdfAnalyzer.getCallibrationTicks(chunks, state)
        s = callibrationTicks[0]
        e = callibrationTicks[-1]
        callibrationTicks = np.array(callibrationTicks) - s
        plotChunkData(independentComponents[:, s:e],  state, callibrationTicks, num_components=independentComponents.shape[0], subtitles=subtitles)

temporal_component()