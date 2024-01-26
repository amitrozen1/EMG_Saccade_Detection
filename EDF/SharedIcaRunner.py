import os
from General.HelperFunctions import GetSessions, getPathsFromSessionFolder, cleanSpace, getICACsvFileNameWithPath, getWeightsFileNameWithPath 
import pyedflib
from EDF.EdfAnalyzer import EdfAnalyzer
import numpy as np
from tqdm import tqdm
from scipy.stats import zscore


num_channels = 16
A = 0
B = 1


def SingleIca(data_path, session, p, doRms = False, downSampleWindowInMs = 1):
    freq = 0
    paths = getPathsFromSessionFolder(os.path.join(data_path, session))
    pathProperties = paths[p]

    Y, chunks, freq = EdfAnalyzer.Read(pathProperties[0], pathProperties[1], 
                                       downsample_window_in_ms=downSampleWindowInMs)
    w, x = EdfAnalyzer.ICA(Y, num_channels)
    del Y

    if ( doRms ):
        x = EdfAnalyzer.window_rms_downsample(x, downSampleWindowInMs)

    return chunks, w, x, freq

def SaveSessionIcaToDisk(current_session, data_path):
    print(current_session)

    signals = []
    freq = 0
    independentComponents = [[], []]

    paths = getPathsFromSessionFolder(os.path.join(data_path, current_session))
    print("Load Files")
    
    if (os.path.exists(getICACsvFileNameWithPath(data_path, current_session)) ):
        return
    
    for p in [A,B]:
        pathProperties = paths[p]
        Y, chunks, freq = EdfAnalyzer.Read(pathProperties[0], pathProperties[1])
        signals.append(Y)
        cleanSpace()

    appendedSignal = np.append(signals[A], signals[B], axis=1)
    print("ICA")
    # ICA 
    w, x = EdfAnalyzer.ICA(appendedSignal, num_channels) # W*Y = X

    # seperate components :
    splitValue = len(signals[A][0])
    independentComponents[A] = x[:, :splitValue]
    independentComponents[B] = x[:, splitValue:]

    np.savetxt(getWeightsFileNameWithPath(data_path, current_session), w, delimiter=",")
    
    del w
    del x
    del Y
    del signals

    cleanSpace()
        
    np.savez(getICACsvFileNameWithPath(data_path, current_session), independentComponents[A], independentComponents[B])
        
    del independentComponents
    cleanSpace()


def LoadComponents(data_path, current_session, downsampleWindowInMs=1, doRms=True):
    chunks = []
    independentComponents = [[], []]

    paths = getPathsFromSessionFolder(os.path.join(data_path, current_session))
    print("Load Files")
    
    if (not os.path.exists(getICACsvFileNameWithPath(data_path, current_session)) ):
        SaveSessionIcaToDisk(current_session, data_path)
    
    loaded = np.load(getICACsvFileNameWithPath(data_path, current_session))
    independentComponents[A] = loaded["arr_0"]
    independentComponents[B] = loaded["arr_1"]
    del loaded
    cleanSpace()
    
    for p in [A,B]:
        pathProperties = paths[p]
        f = pyedflib.EdfReader(pathProperties[0])
        if ( doRms ):
            chunks.append(EdfAnalyzer.getAnnotationChunks(f, pathProperties[1], downsampleWindowInMs)[0])
        else:
            chunks.append(EdfAnalyzer.getAnnotationChunks(f, pathProperties[1])[0])
        
        f.close()
        cleanSpace()

    if ( doRms ):
        print("RMS")
        for p in [A, B]:
            independentComponents[p] = EdfAnalyzer.window_rms_downsample(independentComponents[p], downsampleWindowInMs)
        cleanSpace()
        print("finishRMS")

    return independentComponents, chunks

def LoadComponentsWithRegularRms(data_path, current_session):
    chunks = []
    independentComponents = [[], []]

    paths = getPathsFromSessionFolder(os.path.join(data_path, current_session))
    print("Load Files")
    
    if (not os.path.exists(getICACsvFileNameWithPath(data_path, current_session)) ):
        SaveSessionIcaToDisk(current_session, data_path)
    
    loaded = np.load(getICACsvFileNameWithPath(data_path, current_session))
    independentComponents[A] = loaded["arr_0"]
    independentComponents[B] = loaded["arr_1"]
    del loaded
    cleanSpace()
    
    for p in [A,B]:
        pathProperties = paths[p]
        f = pyedflib.EdfReader(pathProperties[0])
        chunks.append(EdfAnalyzer.getAnnotationChunks(f, pathProperties[1])[0])
        
        f.close()
        cleanSpace()

    print("RMS")
    for p in [A, B]:
        independentComponents[p] = EdfAnalyzer.window_rms(independentComponents[p], 30)
    cleanSpace()
    print("finishRMS")

    return independentComponents, chunks

def LoadComponentsWithZscoreBefore(data_path, current_session, doZscroeBeforeIca = True):
    paths = getPathsFromSessionFolder(os.path.join(data_path, current_session))
    
    signals = [[],[]]
    chunks = [[],[]]

    print("Load Files")
    for p in [A,B]:
        pathProperties = paths[p]
        signals[p], chunks[p], freq = EdfAnalyzer.Read(pathProperties[0], pathProperties[1])
        cleanSpace()

    if ( doZscroeBeforeIca ):
        print("Zscore")
        signals[A] = zscore(signals[A], axis=1)
        signals[B] = zscore(signals[B], axis=1)

    appendedSignal = np.append(signals[A], signals[B], axis=1)

    print("ICA")
    w, x = EdfAnalyzer.ICA(appendedSignal, num_channels)

    # seperate components :
    splitValue = len(signals[A][0])
    signals[A] = x[:, :splitValue]
    signals[B] = x[:, splitValue:]

    return chunks, w, signals




if __name__ == "__main__":
    data_path = r"C:\Liron\DataEmg\Try"
    sessions = GetSessions(data_path)
    for current_session in tqdm(sessions):
        SaveSessionIcaToDisk(current_session, data_path)

    print("Finished")
    exit()



    