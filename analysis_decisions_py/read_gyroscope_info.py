from EDF.EdfAnalyzer import EdfAnalyzer
from General.HelperFunctions import GetSessions, getPathsFromSessionFolder
from General.init import *
import os
from General.PlotFunctions import plotChunkData
import numpy as np
from scipy.stats import zscore

downsample_window_in_ms = 10

# session = GetSessions(data_path)[10]
# p = A
# path, correction = getPathsFromSessionFolder(os.path.join(data_path, session))[p]
folder = r"C:\Liron\DataEmg\MeetingWithYael"
path = os.path.join(folder, "07122023_1000_A_SD_filtered.edf")
correction = -0.2

emgData, chunks, freq, gyroData, gyroFreq = EdfAnalyzer.ReadWithGyro(path, correction, downsample_window_in_ms)

# #Z:
gyroData = zscore(gyroData, axis=1)
combined = np.sqrt(np.sum(np.square(gyroData), axis=0))
gyroData = np.vstack((gyroData, combined))
gyroData = EdfAnalyzer.window_rms(gyroData, 10, gyroFreq)


gyroData = EdfAnalyzer.resample_by_taking_one_every_n(gyroData, downsample_window_in_ms, gyroFreq)


#trim at the end of blinking:
# end = EdfAnalyzer.getCallibrationTicks(chunks, "blink")[-1]*4*downsample_window_in_ms
# emgData = emgData[:, :end]

# w, emgComponents = EdfAnalyzer.ICA(emgData, 16)
# emgComponents = EdfAnalyzer.window_rms(emgComponents, 200, freq)
#taking different rms and downsampling window:
# emgComponents = EdfAnalyzer.resample_by_taking_one_every_n(emgComponents, downsample_window_in_ms, freq)
#emgComponents = EdfAnalyzer.window_rms_downsample(emgComponents, downsample_window_in_ms, freq)

#emgComponents = EdfAnalyzer.window_rms_downsample(emgData, downsample_window_in_ms, freq)


#gyroData = np.vstack((gyroData, combined))


for state in ["smile", "angry", "blink"]:
    callibrationTicks = EdfAnalyzer.getCallibrationTicks(chunks, state)
    subtitles = ["gyro_x", "gyro_y", "gyro_z", "combined"]
    s = callibrationTicks[0]
    e = callibrationTicks[-1]
    callibrationTicks = np.array(callibrationTicks) - s
    plotChunkData(gyroData[:, s:e], state, callibrationTicks, subtitles,  num_components=gyroData.shape[0])
    # plotChunkData(emgComponents[:, s:e],  state, callibrationTicks, num_components=emgComponents.shape[0])
