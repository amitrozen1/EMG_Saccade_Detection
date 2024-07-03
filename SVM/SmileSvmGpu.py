import sys
sys.path.insert(0, "/mnt/c/Liron/Code/LIRON_EMG_DataAnalysis")

import pyedflib
import numpy as np
from scipy.stats import zscore
#from cuml.svm import LinearSVC
from cuml.svm import SVC, SVR
from cuml.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cupy as xp
import os
from cupy import savetxt
from EDF.EdfAnalyzer import EdfAnalyzer
import gc
import pandas as pd
import json
from General.HelperFunctions import toFilePath, getPathsFromSessionFolder, log
from SVM.InvestigateSvm import check_parameter_importance, GetElectrodeByComponentMappingByW
from SVM.InnerClasses import SvmComponentWeight
from General.HelperFunctions import remove_rows_from_matrix, get_review_data
from General.Experiment_CSV import read_experiment_csv


def cleanSpace():
    gc.collect()
    xp.get_default_memory_pool().free_all_blocks()

def getSvmTypeString(isSvc, downsample_window_in_ms):
    if ( isSvc):
        start = "svc"
    else:
        start = "svr"
    return f"{start}_{1000//downsample_window_in_ms}HZ"

def getParticiapntString(particiapnt):
    if( particiapnt == 0):
        return "A"
    return "B"

def getPathsFromSessionFolder(sessionId):
    files = [x for x in os.listdir(sessionId) if x.endswith(".edf")]
    fileA = ""
    fileB = ""
    if ( "_a_" in files[0].lower()):
        fileA = files[0]
        fileB = files[1]
    else:
        fileA = files[1]
        fileB = files[0]
    
    p = os.path.join(".", sessionId, "corrections.txt")
    with open(p, "r") as f:
        corrections = json.load(f)
    
    return [(toFilePath(sessionId, fileA), corrections["A"]),(toFilePath(sessionId, fileB), corrections["B"])]

def saveParameterImportance(importanceVec, w, savePath, p, num_components=16):
    componentToElectrode = GetElectrodeByComponentMappingByW(w)
    componentWeights = [SvmComponentWeight(i+1, int(componentToElectrode[i] + 1), float(importanceVec[i])) for i in range(num_components)]
    string_data = SvmComponentWeight.JsonDumps(componentWeights)
    with open(os.path.join(savePath, f"components_relevance_{getParticiapntString(p)}.json"), 'w') as f:
        f.write(string_data)
    

num_components = 16
num_channels = 16
states = ["smile", "angry", "blink"]
A=0
B=1

downsample_window_in_ms = 20

def PredictState(sessionId, workingDir = ".", isICA = True, RmsWindowSizeInMs = 100, stateToAnalyze="smile"):
    if ( workingDir!= "."):
        os.chdir(workingDir)
    
    userChoiceDict = read_experiment_csv(sessionId)
    paths = getPathsFromSessionFolder(sessionId)
    elecs_to_remove, is_dark = get_review_data(sessionId, workingDir)

    for participant in [A,B]:
        path, correction = paths[participant]

        # ------------------- READ FILE ------------------- #
        log(f"reading {path}, correction: {correction}")
        Y, anotations_chunks, freq = EdfAnalyzer.Read(path, correction, downsample_window_in_ms)
        Y = remove_rows_from_matrix(Y, elecs_to_remove[participant])
        num_components = num_channels - len(elecs_to_remove[participant])
        cleanSpace()

        # ------------------- INIT ------------------- #
        w = []
        x = []

        tagged_x = None
        tagged_y = xp.array([])
        
        # ------------------- Components ------------------- #
        if ( isICA ):
            log("ICA")
            w, x = EdfAnalyzer.ICA(Y, num_components)
            log("End ICA")
        else:
            log("PCA")
            w,x = EdfAnalyzer.PCA(xp.array(Y), num_components)
            x = x.get() # convert to numpy for zscore
            log("End PCA")
        
        del Y
        cleanSpace()

        # ------------------- RMS ------------------- #
        log("RMS")
        x = EdfAnalyzer.window_rms(x, window_size_in_ms = RmsWindowSizeInMs)
        cleanSpace()

        #------------------- Downsample ------------------- #
        log("Downsample")
        x = EdfAnalyzer.resample_by_averaging_every_n(x, downsample_window_in_ms, freq)
        cleanSpace()
        
        # ------------------- zscore ------------------- #
        log("zscore")
        x = xp.asarray(zscore(x,1))

        # ------------------- divide to sets ------------------- #
        for state in states:
            log(f"start state {state}")
            ticks = EdfAnalyzer.getCallibrationTicks(anotations_chunks, state)
            for tickIndex in range(int(len(ticks)/2)):
                start = int(ticks[2*tickIndex])
                end =   int(ticks[2*tickIndex+1])

                m = x[:, start:end]
                
                if ( type(tagged_x) == type(None)):
                    tagged_x = m
                else:
                    tagged_x = xp.append(tagged_x, m, axis=1)
                
                if ( state == stateToAnalyze): 
                    tagged_y = xp.append(tagged_y, xp.ones((1, m.shape[1])))
                else:
                    tagged_y = xp.append(tagged_y, xp.zeros((1, m.shape[1])))

        tagged_x = xp.transpose(tagged_x)
        cleanSpace()
        x = xp.transpose(x)
        cleanSpace()

        X_train, X_test, y_train, y_test = train_test_split(tagged_x, tagged_y, test_size = 0.2, random_state=42)

        # ------------------- SVM - train ------------------- #
        log("start train SVM")
        svm_classifier = None
        svmText = ""

        for isSVC in [True, False]:
            if ( isSVC ):
                svm_classifier = SVC(kernel= 'rbf')
            else:
                svm_classifier = SVR(kernel= 'rbf')
            
            svm_classifier.fit(X_train, y_train)
            
            log("end svm training")

            if ( isSVC ):
                y_pred = svm_classifier.predict(X_test)
                svmText = str(confusion_matrix(y_test.get(),y_pred.get()))
                svmText += "\r\n"
                svmText += classification_report(y_test.get(),y_pred.get())
                print(svmText)
                y_pred = []
            cleanSpace()

            # ------------------- SVM - predict ------------------- #
            log("compute SVM over all the record")

            stateAccrossEdf = svm_classifier.predict(x)
            cleanSpace()

            # ------------------- Print results ------------------- #
            printed_choices = []
            summaries = []
            print(anotations_chunks.keys())
            log(path)
            for chunk_id in anotations_chunks:
                chunk_summary = dict()
                chunk_summary["key"] = chunk_id
                if ( "." in chunk_id ):
                    ucIdx = chunk_id.split('.')[1]
                    if ( ucIdx in userChoiceDict):
                        uc = userChoiceDict[ucIdx]
                        chunk_summary.update(uc.__dict__)
                        if ( ucIdx not in printed_choices):
                            print(uc)
                            printed_choices.append(ucIdx)

                chunk = anotations_chunks[chunk_id]
                start = int(chunk.Start.Time)
                end = int(chunk.End.Time)
                mean_occurences = xp.mean(stateAccrossEdf[start:end])
                chunk_summary["start"] = start
                chunk_summary["end"] = end
                chunk_summary["mean_occurences"] = float(mean_occurences)
                log(f"Id : {chunk_id} ; {stateToAnalyze}s : {mean_occurences}")
                summaries.append(chunk_summary)
            
            # ------------------- Save Files ------------------- #
            resultsPath = os.path.dirname(path)
            resultsPath = os.path.join(resultsPath, stateToAnalyze)
            resultsPath = os.path.join(resultsPath, getSvmTypeString(isSVC, downsample_window_in_ms))
            if ( not os.path.exists(resultsPath)):
                os.makedirs(resultsPath)
            
            
            savetxt(os.path.join(resultsPath, f"{getParticiapntString(participant)}.csv"), X = stateAccrossEdf, delimiter=",")
            with(open(os.path.join(resultsPath, f"{getParticiapntString(participant)}.list"), 'w') as jsonfile):
                json.dump(summaries, jsonfile)
            if ( isSVC ):
                with (open(os.path.join(resultsPath, f"SVC_Classification_{getParticiapntString(participant)}.txt"), 'w') as f):
                    f.write(svmText)
                
                savetxt(os.path.join(resultsPath, f"weights_{getParticiapntString(participant)}.csv"), X=w, delimiter=",")
                
                feature_names = [f"comp_{x+1}" for x in np.arange(16)]
                importance_vec = check_parameter_importance(svm_classifier,X_test.get(), y_test.get(), feature_names,resultsPath, participant)
                saveParameterImportance(importance_vec, w, resultsPath, participant, num_components)
            del stateAccrossEdf
            cleanSpace()
        del x
        del X_train
        del y_train
        del X_test 
        del y_test
        cleanSpace()

