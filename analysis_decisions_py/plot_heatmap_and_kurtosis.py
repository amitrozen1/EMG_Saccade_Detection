import os
from General.init import *
from General.HelperFunctions import GetSessions, getPathsFromSessionFolder
from scipy.stats import zscore
from EDF.EdfAnalyzer import EdfAnalyzer
from scipy.stats import kurtosis
from EDF.PlotHeatmap import get_heatmap, get_participant_photo_path
import numpy as np


if __name__ == "__main__":
    sessions = GetSessions(data_path)
    session = sessions[5]
    chunks = [[],[]]
    w = [[],[]]
    independentComponents = [[],[]]


    for p in [A,B]:
        paths = getPathsFromSessionFolder(os.path.join(data_path, session))
        pathProperties = paths[p]
        Y, chunks[p], freq = EdfAnalyzer.Read(pathProperties[0], pathProperties[1])
        Y = zscore(Y, axis=1)
        w[p], independentComponents[p] = EdfAnalyzer.ICA(Y, 16)
    
    #plot heatmaps with kurtosis values:
    kurt =  [kurtosis(independentComponents[i], axis=1) for i in range(len(independentComponents))]
    vars = [np.var(independentComponents[i], axis=1) for i in range(len(independentComponents))]
    means = [np.mean(independentComponents[i], axis=1) for i in range(len(independentComponents))]
    extra_headers = [[f"v-{v:.2f}, m-{m:.2f}, k-{k:.2f}" for v,m,k in zip(vars_p, means_p, kurt_p)] for vars_p, means_p, kurt_p in zip(vars, means, kurt)]
    #extra_headers = [[f"k-{k:.2f}" for k in kurt_p] for kurt_p in kurt]

    #using same picture for all:
    p_photo = get_participant_photo_path(data_path, sessions[0], A)
    
    for p in [A,B]:
        get_heatmap("", num_components, w[p], extra_headers=extra_headers[p], default_pic=True)
