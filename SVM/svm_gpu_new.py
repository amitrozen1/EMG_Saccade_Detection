import os
from General.HelperFunctions import GetSessions, cleanSpace
import numpy as np
from scipy.stats import zscore
from General.init import *
from tqdm import tqdm
from General.HelperFunctions import getPathsFromSessionFolder
from EDF.EdfAnalyzer import EdfAnalyzer
from ComponentAnalysis.ComponentExtractor import open_w_if_exisis, open_whiten_if_exisis, load_special_components_if_exists, get_special_component_indexes, GetLipCornerDownPuller, GetComponentByElecrodesNumber
from scipy import signal
from Hilas_component_classification.classifying_ica_components import wavelet_denoising, whiten, center
import cupy as xp
from sklearn.metrics import classification_report
import pickle as pkl

#todo - figure out if I can convert the svm classifier to numpy and save it to disk.

# to run in wsl : conda activate rapids-22.10

default_downsample_window_in_ms = 5
default_rms_size_ms = 50

def load_single(session, p, rms_size_ms=default_rms_size_ms, downsampleWindowInMs=default_downsample_window_in_ms):
    path_and_corrections = getPathsFromSessionFolder(os.path.join(data_path, session))

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
        #save_new_w(session, p, w)
    else:
        independentComponents = np.matmul(w,Y)

    independentComponents = EdfAnalyzer.window_rms(independentComponents, rms_size_ms, freq)
    hz = int(1000/downsampleWindowInMs)
    independentComponents = signal.resample(independentComponents, int(independentComponents.shape[1] / freq * hz), axis=1)

    #select components:
    special_comps_index = load_special_components_if_exists(data_path, session, p, dwt=True)
    if (special_comps_index is None):
        special_comps_index = get_special_component_indexes(w)
    
    special_comps_index += [GetComponentByElecrodesNumber(w, [], [2])]
    special_comps_index += [GetComponentByElecrodesNumber(w, [], [5])]
    special_comps_index += [GetLipCornerDownPuller(w, [])]

    independentComponents = independentComponents[special_comps_index, :]
    independentComponents = zscore(independentComponents, axis=1)

    return independentComponents, chunks

def convert_to_xp_training_and_testing_set(independentComponents, chunks, state_to_predict="smile"):
    #create an array in the shape of independentComponents:
    x = xp.empty((independentComponents.shape[0], 0))
    y = xp.empty((1,))
    all_states = ["smile", "angry", "blink"]
    for state in all_states:
        for i in range(3):
            current_start = int(chunks[f"{state}__{i}"].Start.Time)
            current_end =   int(chunks[f"{state}__{i}"].End.Time)
            x = xp.hstack((x, independentComponents[:, current_start:current_end]))
            y = xp.hstack((y, int(state == state_to_predict)*np.ones((current_end - current_start,))))

    #convert y to dichotomus variable:
    y = xp.array(y, dtype=int)
    ratio = 0.8
    split = int(x.shape[1]*ratio)
    #randomly split:
    indices = xp.random.permutation(x.shape[1])
    training_idx, test_idx = indices[:split], indices[split:]
    x_train, x_test = x[:, training_idx], x[:, test_idx]
    y_train, y_test = y[training_idx], y[test_idx]

    return x_train, x_test, y_train, y_test


def get_filename(session, p, classication_res = False, svm_type="SVC"):
    p = "A" if p == A else "B"
    path = os.path.join(data_path, session, "SVM_2024")
    if not os.path.exists(path):
        os.makedirs(path)
    if classication_res:
        return os.path.join(path, f"{p}_smile-{svm_type}_rms_{default_rms_size_ms}_ds_{default_downsample_window_in_ms}_classification_res.txt")
    return os.path.join(path, f"{p}_smile-{svm_type}_rms_{default_rms_size_ms}_ds_{default_downsample_window_in_ms}.pkl")

def save_data(session, p, data, svm_type="SVC"):
    filename = get_filename(session, p, svm_type=svm_type)
    with open(filename, "wb") as file:
        pkl.dump(data, file)

def save_classification_res(session, p, res):
    filename = get_filename(session, p, classication_res=True)
    with open(filename, "w") as file:
        file.write(res)

def run(session, p, svm_type="SVC"):
    independentComponents, chunks = load_single(session, p)
    x_train, x_test, y_train, y_test = convert_to_xp_training_and_testing_set(independentComponents, chunks)
    from cuml.svm import SVC, SVR
    if svm_type == "SVC":
        svm_classifier = SVC(kernel= 'rbf')
    else:
        svm_classifier = SVR(kernel= 'rbf')
    svm_classifier.fit(x_train.T, y_train)

    #classification report:
    y_pred = svm_classifier.predict(x_test.T).get()
    y_test = y_test.get()
    
    if ( svm_type == "SVC"):
        class_res = classification_report(y_test, y_pred)
        save_classification_res(session, p, class_res)
        print(class_res)

    for chunk in chunks:
        chunk_start = int(chunks[chunk].Start.Time)
        chunk_end =   int(chunks[chunk].End.Time)
        chunk_data = xp.array(independentComponents[:, chunk_start:chunk_end])
        #predict:
        y_pred = svm_classifier.predict(chunk_data.T).get()
        print(f"chunk {chunk} length : {(chunk_end-chunk_start)//(1000//default_downsample_window_in_ms)} prediction %: {np.mean(y_pred)}")
        chunks[chunk].data = y_pred
    
    save_data(session, p, chunks, svm_type=svm_type)

    del independentComponents, x_train, x_test, y_train, y_test, svm_classifier
    cleanSpace()

def load_data(session, p, svm_type="SVC"):
    filename = get_filename(session, p, svm_type=svm_type)
    with open(filename, "rb") as file:
        return pkl.load(file)