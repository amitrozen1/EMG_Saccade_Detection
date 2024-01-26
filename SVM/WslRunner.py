from SmileSvmGpu import PredictState
import os
from General.HelperFunctions import cleanSpace, log
import traceback

import sys
sys.path.insert(0, "/mnt/c/Liron/Code/LIRON_EMG_DataAnalysis")

data_path = r'/mnt/c/Liron/DataEmg/ForLater'


def isSession(s):
    return "202" in s

def GetSessions(path = data_path):
    return [ s for s in os.listdir(path) if isSession(s) ]

def runSingleSession(session, dataPath = data_path):
    # in the paper - The sound of smile: Auditory biofeedback of facial EMG activity
    # https://doi.org/10.1016/j.displa.2016.09.002
    # They used window of 50MS for sonification and 150MS for the SVM classification.
    PredictState(session, dataPath, True, 50, "smile" )


if ( __name__=="__main__"):
    runAll = True
    if ( runAll ):
        sessions = GetSessions(data_path)
        for session in sessions:
            try:
                runSingleSession(session)
                cleanSpace()
            except Exception as e:
                log(str(e))
                traceback.print_exc()
    else:
        data_path = r"/mnt/c/Liron/DataEmg/new_batch"
        session = "20122022_1215"
        runSingleSession(session, data_path)