import os
import numpy as np
from EDF.ComponentValidator import GetNoiseComponentsBySort
from General.HelperFunctions import GetSessions, GetNoiseComponentsByVar, GetNoiseComponentsByPeak
import seaborn as sns
from scipy.signal import find_peaks

num_channels = 16
A = 0
B = 1
downsampleWindowInMs = 20 # downsamples to 100HZ
orig_freq = 4000

data_path = r"C:\Liron\DataEmg\Done"

freq = orig_freq // (4*downsampleWindowInMs)
windowForCorrInMs = 250
numOfPixelsForXCorrWindow = windowForCorrInMs//downsampleWindowInMs
numOfPixelsForXCorrStep = 100 // downsampleWindowInMs # 100ms


# def GetNoiseComponents(data_path, current_session):
#     w_path = os.path.join(data_path, current_session, "weightsAfterSharedICA.csv")
#     w = np.loadtxt(w_path, delimiter=",")
#     new_w = np.absolute(np.linalg.inv(w))
#     log_w = np.log(np.transpose(new_w))
#     components_to_remove = []

#     for i in range(0,16):
#         if ( len(find_peaks(log_w[i], prominence=0.5)[0]) == 0):
#             components_to_remove += [i]

#     return components_to_remove


def GetComponentByElectrodeNumber(data_path, session, electrode):
    w_path = os.path.join(data_path, session, "weightsAfterSharedICA.csv")
    w = np.loadtxt(w_path, delimiter=",")
    components_to_electrodes = np.argmax(np.absolute(np.linalg.inv(w)), axis=0) #[::-1]
    componentsArr = np.where(components_to_electrodes == electrode)[0]
    
    if ( len(componentsArr) == 0):
        return "Error", "Error"
    return componentsArr[0], electrode


for session in GetSessions(data_path):
    print(f"Session : {session}, ByVar : {GetNoiseComponentsByVar(data_path, session)} , ByPeak: {GetNoiseComponentsByPeak(data_path, session, 0.5)}, BySort: {GetNoiseComponentsBySort(data_path, session)}")

# session = "20122022_1415"

# w_path = os.path.join(data_path, session, "weightsAfterSharedICA.csv")
# w = np.loadtxt(w_path, delimiter=",")
# new_w = np.absolute(np.linalg.inv(w))
# log_w = np.log(np.transpose(new_w))

# components_to_remove = []
# for i in range(0,16):
#     print(sorted(log_w[i]))
#     if ( len(find_peaks(sorted(log_w[i]), prominence=1)[0]) == 0):
#         components_to_remove += [i]

# print(components_to_remove)

