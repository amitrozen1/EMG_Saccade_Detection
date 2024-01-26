from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np
from General.PlotFunctions import savePlot, clearPlot
from EDF.PlotHeatmap import get_heatmap
from General.HelperFunctions import particiapnt_to_char
import os
from scipy.stats import zscore


def check_parameter_importance(svc, X_test, y_test, feature_names, savePath, participant=None):
    perm_importance = permutation_importance(svc, X_test, y_test)
    features = np.array(feature_names)
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.xlabel("Permutation Importance")
    
    parameters_file = os.path.join(savePath,f"param_importance.jpg")
    if ( participant != None):
        parameters_file = os.path.join(savePath,f"param_importance_{particiapnt_to_char(participant)}.jpg")
        
    savePlot(parameters_file)
    clearPlot()
    return perm_importance.importances_mean


def GetElectrodeByComponentMappingByW(w):
    components_to_electrodes = np.argmax(np.absolute(np.linalg.inv(w)), axis=0)
    return components_to_electrodes

def GetElectrodeByComponentMappingByPath(weightPath):
    w = np.loadtxt(weightPath, delimiter=",")
    return GetElectrodeByComponentMappingByW(w)

def GetHeatmapForSession(session, p):
    data_path = r"C:\Liron\DataEmg\Try"
    weights_file = f"{session}\smile\svc\weights_{particiapnt_to_char(p)}.csv"
    weights_file = os.path.join(data_path, weights_file)
    w = np.loadtxt(weights_file, delimiter=",")
    num_components=16
    order = np.arange(num_components)
    get_heatmap(os.path.join(data_path, session, f"{particiapnt_to_char(p)}.jpeg"), num_components, w, order)

if __name__ ==  "__main__":
    A = 0
    B = 1
    #GetHeatmapForSession("20122022_1215", B)
    session = "20122022_1215"
    for p in [A,B]:
        data_path = r"C:\Liron\DataEmg\Try"
        weights_file = f"{session}\smile\svc\weights_{particiapnt_to_char(p)}.csv"
        weights_file = os.path.join(data_path, weights_file)
        print (f"{particiapnt_to_char(p)} : ")
        print(np.array(GetElectrodeByComponentMappingByPath(weights_file)) + 1)






