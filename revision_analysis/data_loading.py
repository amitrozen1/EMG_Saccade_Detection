import os
import numpy as np
from General.init import *
from General.HelperFunctions import getPathsFromSessionFolder
from EDF.EdfAnalyzer import EdfAnalyzer
from ComponentAnalysis.ComponentExtractor import open_w_if_exisis, open_whiten_if_exisis
from scipy import signal
from Hilas_component_classification.classifying_ica_components import wavelet_denoising, whiten, center
from revision_analysis.clustering_by_hila.classifying_ica_components import classify_components
from analysis_decisions_py.original_actress_readings import export_actress_chunked_data_with_dwt

def load_single_data(session, p, downsampleWindowInMs, rms_size_ms):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(get_data_path(), session))
    Y, chunks, freq = EdfAnalyzer.Read(path_and_corrections[p][0], path_and_corrections[p][1], downsampleWindowInMs)

    Y = wavelet_denoising(Y, freq, 'db15')
    Y, _ = center(Y)

    whiten_mat = open_whiten_if_exisis(session, p, dwt=True)
    if (whiten_mat is None):
        Y = whiten(Y)
    else:
        Y = np.dot(whiten_mat, Y)

    w = open_w_if_exisis(session, p, dwt=True)
    
    if ( w is None):
        w, independentComponents = EdfAnalyzer.ICA(Y, 16, whiten=False)
    else:
        independentComponents = np.matmul(w,Y)

    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    hz = int(1000/downsampleWindowInMs)
    independentComponents = signal.resample(independentComponents, int(independentComponents.shape[1] / freq * hz), axis=1)

    return w, independentComponents, chunks


def load_dyad_data(session, downsampleWindowInMs, rms_size_ms):
    both_components = [[],[]]
    both_chunks = [[],[]]
    for p in [A,B]:
        w, independentComponents, chunks = load_single_data(session, p, downsampleWindowInMs, rms_size_ms)
        components_dictionary = dict()
        component_to_cluster_class = classify_components(w)
        for i, cluster in enumerate(component_to_cluster_class):
            if cluster != 16:
                components_dictionary[cluster] = independentComponents[i]
        dicionary_to_arr = [components_dictionary.get(i, np.nan) for i in range(16)]

        both_components[p] = dicionary_to_arr
        both_chunks[p] = chunks
    
    return both_components, both_chunks

def load_actress_data(rms_size_ms, ds_factor):
    independentComponents, chunks_as_dict = export_actress_chunked_data_with_dwt(ds_factor, rms_size_ms)
    component_to_cluster_class = [0,3,5,2,1,8,16,11,15,14,4,6,7,10,13,9]
    components_dictionary = dict()
    for i, cluster in enumerate(component_to_cluster_class):
        if cluster != 16:
            components_dictionary[cluster] = independentComponents[i]
    dictionary_to_arr = [components_dictionary.get(i, np.nan) for i in range(16)]
    
    
    for key in chunks_as_dict:
        chunk = chunks_as_dict[key]
        chunk.data = [[] for i in range(16)]
        for i in range(16):
            if dictionary_to_arr[i] is not np.nan:
                chunk.data[i] = dictionary_to_arr[i][chunk.start:chunk.end]
            else:
                chunk.data[i] = np.nan
    
    return chunks_as_dict

def validate_classification(session):
    session = "04122022_1545"
    p = B
    downsampleWindowInMs = 5
    rms_size_ms = 50
    #w, independentComponents, chunks = load_single_data(session, p, downsampleWindowInMs, rms_size_ms)
    
    #todo: check how to do it with the actress - had different electordes!!!!

    w=np.load(r"C:\Liron\neuroscience\PHD\First year experiment - EMG\StoriesFromActress\EDF\w_dwt_db15.npy")
    print(classify_components(w))

def check_actress_smile_corr():
    zygomaticus = 5
    cheeck_raiser = 9
    talking = 1

    zygomaticus_arr = []
    cheeck_raiser_arr = []
    talking_arr = []

    actress_chunks = load_actress_data(50, 5)
    #concatenate
    zygomaticus_arr = np.hstack([chunk.data[zygomaticus] for chunk in actress_chunks.values()])
    cheeck_raiser_arr = np.hstack([chunk.data[cheeck_raiser] for chunk in actress_chunks.values()])
    talking_arr = np.hstack([chunk.data[talking] for chunk in actress_chunks.values()])

    from scipy.stats import pearsonr


    #plot all data of first story with 15 subplots and cluster titles:
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(15, 1, figsize=(10, 30))
    curr_ax = 0
    for i in range(16):
        vector = list(actress_chunks.values())[0].data[i]
        if vector is np.nan:
            continue
        axs[curr_ax].plot(vector)
        axs[curr_ax].set_title("cluster " + str(i+1))
        curr_ax += 1

    plt.tight_layout()
    plt.show()

    #pearson correlation:
    print("zygomaticus")
    print(pearsonr(zygomaticus_arr, talking_arr))
    print("cheeck_raiser")
    print(pearsonr(cheeck_raiser_arr, talking_arr))