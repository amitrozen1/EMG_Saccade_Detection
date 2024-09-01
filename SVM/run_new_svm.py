from SVM.svm_gpu_new import run
from General.HelperFunctions import GetSessions
from General.init import *
from tqdm import tqdm

def run_svc(svm_type="SVC"):
    sessions = GetSessions(data_path)[6:]
    for session in tqdm(sessions):
        print("running session: " + session)
        print("---------> A")
        run(session, A, svm_type)
        print("---------> B")
        run(session, B, svm_type)

if __name__ == '__main__':
    run_svc("SVR")