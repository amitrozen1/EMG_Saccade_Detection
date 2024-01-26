from SingleComponentAnalysis import GetFacialComponentsElcrotdesAndIndexes
import os
from General.HelperFunctions import GetSessions, cleanSpace, getPathsFromSessionFolder, particiapnt_to_char
from EDF.SharedIcaRunner import LoadComponents, LoadComponentsWithRegularRms, LoadComponentsWithZscoreBefore
from scipy.stats import zscore
from General.PlotFunctions import plotDataFromBothChunks
import matplotlib.pyplot as plt
import numpy as np
import pyedflib
from EDF.EdfAnalyzer import EdfAnalyzer
from picard import picard
from scipy.stats import kurtosis


A=0
B=1


def getTicks(chunks, keys):
    ticks = [[],[]]
    for p in [A,B]:
        for key in keys:
            ticks[p] += [int(chunks[p][key].Start.Time), int(chunks[p][key].End.Time)]
    return ticks

def plotState(chunks, independentComponents, state, componentsToPlot, saveas="", importance_order=None):
    keys = [f"{state}__{i}" for i in range(0,3)]
    ticks = getTicks(chunks,keys)
    s = [ticks[A][0], ticks[B][0]]
    e = [ticks[A][-1], ticks[B][-1]]

    X1 = independentComponents[A][:, s[A]:e[A]]
    X2 = independentComponents[B][:, s[B]:e[B]]
    plotDataFromBothChunks(X1, X2, ticks[A], ticks[B], 
                            title=f"{state}:{componentsToPlot}", 
                            num_components=16, saveAt=saveas, importance_order=importance_order) #, num_components=len(componentsToPlot) )
    plt.clf()


def create_edf_from_component_matrix(matrix, freq, path, filename):
    f = pyedflib.EdfWriter(os.path.join(path, filename), n_channels=16, file_type=pyedflib.FILETYPE_EDFPLUS)
    channel_info = []
    for i in range(16):
        channel_info.append({'label': f'ch{i}', 'dimension': 'uV', 'sample_rate': freq, 'transducer': 'transducer'})
    f.setSignalHeaders(channel_info)
    f.writeSamples(matrix)
    f.close()

def compute_component_kurtosis(independentComponents):
    kurt =  [kurtosis(independentComponents[i], axis=1) for i in range(len(independentComponents))]
    importance_order = [np.argsort(kurt[i])[::-1] for i in range(len(kurt))]
    return kurt, importance_order



if __name__  == "__main__":
    orig_freq = 4000
    data_path = r"C:\Liron\DataEmg\try"
    csv_path = os.path.join(data_path, "four_components_mean_and_choices_NoZscore.csv")

    sessions = GetSessions(data_path) # last one doesn't have nose component

    session = sessions[2]
    print(session)
    save_plots_dir = os.path.join(data_path, f"figures\{session}")

    zscore_timing = "After" # After

    signle_or_seperate_ica = "single" # "single" # "seperate"

    independentComponents = [[],[]]
    chunks = [[],[]]
    w = [[],[]]

    if ( signle_or_seperate_ica == "single"):
        chunks, w, independentComponents = LoadComponentsWithZscoreBefore(data_path, session, zscore_timing == "Before")
    #create_edf_from_component_matrix(independentComponents[A], orig_freq, save_plots_dir, f"{session}_A.edf")
    #create_edf_from_component_matrix(independentComponents[B], orig_freq, save_plots_dir, f"{session}_B.edf")
    else:
        for p in [A,B]:
            paths = getPathsFromSessionFolder(os.path.join(data_path, session))
            pathProperties = paths[p]
            Y, chunks[p], freq = EdfAnalyzer.Read(pathProperties[0], pathProperties[1])
            #w[p], independentComponents[p] = EdfAnalyzer.ICA(Y, 16)
            K, W, independentComponents[p] = picard(Y, n_components=16, ortho=True, extended=False)
            w[p] = np.dot(K, W)

    indexes = []
    if ( signle_or_seperate_ica == "single"):        
        indexesAndElectrodes = GetFacialComponentsElcrotdesAndIndexes(w)
        indexes = str([x[0] for x in indexesAndElectrodes])
    else:
        for p in [A,B]:
            indexesAndElectrodes = GetFacialComponentsElcrotdesAndIndexes(w[p])
            indexes += f"{particiapnt_to_char(p)} : {[x[0] for x in indexesAndElectrodes]}; "

    comoponentKurtosis, importance_order = compute_component_kurtosis(independentComponents)
    print(comoponentKurtosis)

    cleanSpace()
    if ( zscore_timing != "Before" ):
        print("Zscore")
        for p in [A, B]:
            independentComponents[p] = zscore(independentComponents[p], axis=1)
        cleanSpace()


    
    if not os.path.exists(save_plots_dir):
        os.makedirs(save_plots_dir)
    
    for state in ["smile", "angry", "blink"]:
        save_at = os.path.join(save_plots_dir, f"{signle_or_seperate_ica}_ICA_{state}_zscore_{zscore_timing}.png")
        plotState(chunks, independentComponents, state, indexes, importance_order=importance_order)


# The code above is for shared ICA
# let's try to do it for regular ICA. the compoents would not be aligned, but would they be more clear? and more suitable with W ? 