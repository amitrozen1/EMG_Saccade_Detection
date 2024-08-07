import sys
sys.path.insert(0, "/mnt/c/Liron/Code/emg_data_analysis")

from SmileSvmGpu import PredictState
import os
from General.HelperFunctions import cleanSpace, log
import traceback
from General.HelperFunctions import isSession

data_path = r'/mnt/c/Liron/DataEmg/Try'

def GetSessions(path = data_path):
    return [ s for s in os.listdir(path) if isSession(s) ]

def runSingleSession(session, dataPath = data_path):
    # in the paper - The sound of smile: Auditory biofeedback of facial EMG activity
    # https://doi.org/10.1016/j.displa.2016.09.002
    # They used window of 50MS for sonification and 150MS for the SVM classification.
    PredictState(session, dataPath, True, 100, "smile" )


if ( __name__=="__main__"):
    
    trySingle = -1
    # 15 is the best dyad.
    sessions = ["18122022_1115"] #GetSessions(data_path)

    if ( trySingle != -1):
        runSingleSession(sessions[trySingle])
        cleanSpace()
        exit(0)
    
    for session in sessions:
        try:
            runSingleSession(session)
            cleanSpace()
        except Exception as e:
            log(str(e))
            traceback.print_exc()