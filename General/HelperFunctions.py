import os
from datetime import datetime
import gc
import cupy as xp
import json
import numpy as np
from scipy.signal import find_peaks
import re



def log(string):
    print(f"{datetime.now().strftime('%H:%M:%S')} : {string}")

def toFilePath(sessionFolder, fname):
    return os.path.join(sessionFolder, fname)

def getPathsFromSessionFolder(sessionFolder, correctionA = 0.4, correctionB = 0.4):
    files = [x for x in os.listdir(sessionFolder) if x.endswith(".edf")]
    fileA = ""
    fileB = ""
    if ( "_a_" in files[0].lower()):
        fileA = files[0]
        fileB = files[1]
    else:
        fileA = files[1]
        fileB = files[0]
    
    return [(toFilePath(sessionFolder, fileA), correctionA),(toFilePath(sessionFolder, fileB), correctionB)]

def getFirstCsvFile(path):
    for x in os.listdir(path):
        if x.endswith(".csv"):
            return toFilePath(path, x)

def isMyAnnotation(annotation):
    parts = [".ogg", ".png", "smile_", "angry_", "blink_"]
    s = str(annotation).lower()
    for part in parts:
        if ( part in s):
            return True
    return False

def cleanSpace():
    gc.collect()
    xp.get_default_memory_pool().free_all_blocks()


def getICACsvFileName():
    return "ICafterSharedIca.npz"

def getICACsvFileNameWithPath(data_path, session):
    return os.path.join(os.path.join(data_path, session) ,getICACsvFileName())

def getWeightsFileNameWithPath(data_path, session):
    pathOfSession = os.path.join(data_path, session)
    return os.path.join(pathOfSession, "weightsAfterSharedICA.csv")

def isSession(s):
    #return "202" in s
    # Regular expression pattern to match the format ddmmyyyy_hhmm
    pattern = r'\d{8}_\d{4}'

    # Using fullmatch to ensure the entire string conforms to the pattern
    return bool(re.fullmatch(pattern, s))

def GetSessions(path):
    return [ s for s in os.listdir(path) if isSession(s) ]

def getPathsFromSessionFolder(fullFolder, edf_type="sd"):
    files = [x for x in os.listdir(fullFolder) if x.endswith(".edf") and edf_type in x.lower()]
    fileA = ""
    fileB = ""
    if ( "_a_" in files[0].lower()):
        fileA = files[0]
        fileB = files[1]
    else:
        fileA = files[1]
        fileB = files[0]
    
    p = os.path.join(fullFolder, "corrections.txt")
    with open(p, "r") as f:
        corrections = json.load(f)
    
    return [(toFilePath(fullFolder, fileA), corrections["A"]),(toFilePath(fullFolder, fileB), corrections["B"])]

def reshape_num_channels(mat, num_channels=16):
    return np.reshape(mat,(num_channels, len(mat)//num_channels) )

A=0
B=1

def particiapnt_to_char(p):
    if p == A:
        return "A"
    return "B"

def GetNoiseComponentsByVar(data_path, current_session, prominenece = 0.5):
    w_path = os.path.join(data_path, current_session, "weightsAfterSharedICA.csv")
    w = np.loadtxt(w_path, delimiter=",")
    new_w = np.absolute(np.linalg.inv(w))
    log_w = np.log(np.transpose(new_w))
    components_to_remove = []

    for i in range(0,16):
        if ( np.var(log_w[i]) < 0.1 ):
            components_to_remove += [i]

    return components_to_remove

def GetNoiseComponentsByPeak(data_path, current_session, prominenece = 0.5):
    w_path = os.path.join(data_path, current_session, "weightsAfterSharedICA.csv")
    w = np.loadtxt(w_path, delimiter=",")
    new_w = np.absolute(np.linalg.inv(w))
    log_w = np.log(np.transpose(new_w))
    components_to_remove = []

    for i in range(0,16):
        if ( len(find_peaks(log_w[i], prominence=prominenece)[0]) == 0):
            components_to_remove += [i]

    return components_to_remove

def RemoveRowFromMatrix(a, row):
    return np.append(a[0:row, :], a[row+1:, :], axis=0)

def remove_rows_from_matrix(mat, rows_to_delete):
    rows_to_delete = np.array(rows_to_delete)
    if (rows_to_delete.shape[0] == 0):
        return mat
    return np.delete(mat, rows_to_delete, axis=0)

def replace_rows_with_zeros(mat, rows_to_delete):
    for row in rows_to_delete:
        mat[row] = np.zeros(mat.shape[1])
    return mat

def replace_rows_with_ones(mat, rows_to_delete):
    for row in rows_to_delete:
        mat[row] = np.ones(mat.shape[1])
    return mat

def stack_rows_with_pad(a, b):
    a = np.array(a)
    b = np.array(b)
    if a.size == 0:
        return b
    
    a_len = a.shape[1] if len(a.shape) > 1 else len(a)
    b_len = b.shape[1] if len(b.shape) > 1 else len(b)

    diff = a_len - b_len
    if diff > 0:
        if ( len(b.shape) == 1 or b.shape[1] == 0):
            b = np.append(b,[0]*abs(diff))
        else:
            b = np.append(b, np.zeros((b.shape[0], abs(diff))), axis=1)
    elif diff < 0:
        if ( len(a.shape) == 1 or a.shape[1] == 0):
            a = np.append(a,[0]*abs(diff))
        else:
            a = np.append(a, np.zeros((a.shape[0], abs(diff))), axis=1)
    
    return np.row_stack((a,b))