import os
from General.HelperFunctions import GetSessions, cleanSpace, reshape_num_channels, RemoveRowFromMatrix
from EDF.ComponentValidator import GetNoiseComponentsBySort
import numpy as np
from EDF.SharedIcaRunner import LoadComponents
from scipy.stats import zscore
from matplotlib.colors import Normalize
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from General import HtmlTablePlot
import pandas as pd
from PsychData.AnalyzeQuestionnaire import GetStatsRowsFromSession, GetStatsHeaderWithouSession
from PsychData.LoadChoicesJson import loadChoiceFromSession
from sklearn.metrics import classification_report
from tqdm import tqdm
from General.StatisticsFunctions import cross_rqa
from EDF.BlankRemover import EmgBlankRemover
from General.CorrolationMethods import get_peak_xcorr, dtai_dist_dtw_pearson, dtai_dist_dtw_pearson_over_seconds, dtai_dist_dtw_pearson_over_seconds_blank_removed
import cupy as xp

data_path = r"C:\Liron\DataEmg\Try"

num_channels = 16
A = 0
B = 1
downsampleWindowInMs = 10 # 20MS == 50HZ.
orig_freq = 4000

freq = orig_freq // (4*downsampleWindowInMs)
windowForCorrInMs = 1000
numOfPixelsForXCorrWindow = windowForCorrInMs//downsampleWindowInMs
numOfPixelsForXCorrStep = 50 // downsampleWindowInMs # 100ms


def GetElectrodeFromComponent(data_path, session, component):
    w_path = os.path.join(data_path, session, "weightsAfterSharedICA.csv")
    w = np.loadtxt(w_path, delimiter=",")
    components_to_electrodes = np.argmax(np.absolute(np.linalg.inv(w)), axis=0)[::-1]
    return components_to_electrodes[int(component)]


if ( __name__ == "__main__"):
    sessions = GetSessions(data_path)

    electrodes_in_all_particiapants = []
    for current_session in sessions:
        print(current_session)
        # independentComponents, chunks = LoadComponents(data_path, current_session, downsampleWindowInMs)
        # print("Zscore")
        # for p in [A, B]:
        #     independentComponents[p] = zscore(independentComponents[p], axis=1)
        # cleanSpace()
        # print("finish z")

        noise_components = GetNoiseComponentsBySort(data_path, current_session)

        not_noise_electrodes = []
        for i in range(0, num_channels):
            if i in noise_components:
                continue
            not_noise_electrodes.append(GetElectrodeFromComponent(data_path, current_session, i))
        
        # check if new electrodes repeat in all participants
        if ( len(electrodes_in_all_particiapants) == 0):
            electrodes_in_all_particiapants += not_noise_electrodes
        else:
            for electrode in electrodes_in_all_particiapants:
                if (electrode not in not_noise_electrodes):
                    electrodes_in_all_particiapants.remove(electrode)
                    

        # not_noise_electrodes = sorted(not_noise_electrodes)
        # print(not_noise_electrodes)
        # print(len(not_noise_electrodes))
    print(electrodes_in_all_particiapants)
    # only electrode in everyone is 11. maybe a more sophisticated way?


        #todo - compare results with smile weight electrodes (where ICA was done alone and not with partenrt)