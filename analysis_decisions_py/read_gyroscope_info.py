from General.init import *
from EDF.EdfAnalyzer import EdfAnalyzer
from General.HelperFunctions import GetSessions, getPathsFromSessionFolder
import os
from General.PlotFunctions import plotChunkData
import numpy as np
from scipy.stats import zscore

def main():
    downsample_window_in_ms = 20
    rms_window_in_ms = 100

    session = GetSessions(data_path)[1]
    p = A
    path, correction = getPathsFromSessionFolder(os.path.join(data_path, session))[p]
    # folder = r"C:\Liron\DataEmg\MeetingWithYael"
    # path = os.path.join(folder, "07122023_1000_A_SD_filtered.edf")
    # correction = -0.2

    emgData, chunks, freq, aclData, gyroFreq = EdfAnalyzer.ReadWithInertial(path, correction, downsample_window_in_ms)#, just_accelorometer=True)

    aclData = aclData[0:3]
    # #Z:
    aclData = zscore(aclData, axis=1)
    aclData = EdfAnalyzer.window_rms(aclData, rms_window_in_ms, gyroFreq)
    aclData = EdfAnalyzer.resample_by_taking_one_every_n(aclData, downsample_window_in_ms, gyroFreq)
    
    combined = np.sqrt(np.sum(np.square(aclData), axis=0))
    aclData = np.vstack((aclData, combined))

    for state in ["smile", "angry", "blink"]:
        callibrationTicks = EdfAnalyzer.getCallibrationTicks(chunks, state)
        #subtitles = ["gyro_x", "gyro_y", "gyro_z","accelerometer_x", "accelerometer_y", "accelerometer_z", "combined"]
        subtitles = ["accelerometer_x", "accelerometer_y", "accelerometer_z", "combined"]
        s = callibrationTicks[0]
        e = callibrationTicks[-1]

        #take each pair of ticks and check out the combined mean:
        for i in range(0, len(callibrationTicks)-1, 2):
            c = combined[callibrationTicks[i]:callibrationTicks[i+1]]
            print(f"mean of combined between {i} and {i+1}: {np.mean(c)} median: {np.median(c)} std: {np.std(c)}")

        callibrationTicks = np.array(callibrationTicks) - s
        plotChunkData(aclData[:, s:e], state, callibrationTicks, subtitles,  num_components=aclData.shape[0])
        # plotChunkData(emgComponents[:, s:e],  state, callibrationTicks, num_components=emgComponents.shape[0])
