from General.HelperFunctions import GetSessions, cleanSpace, reshape_num_channels
from tqdm import tqdm
import os
import shutil


def copy_folder(location, new_location, new_folder_name):
    new_folder = os.path.join(new_location, new_folder_name)
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
    for file in os.listdir(location):
        file_path = os.path.join(location, file)
        if os.path.isfile(file_path):
            shutil.copy(file_path, new_folder)


if __name__=="__main__":

    data_folder = r"C:\Liron\DataEmg\Try"
    svmType = "svc"
    stateToAnalyze = "smile"
    experiment_part = "listen" #"listen"

    freq = 100

    sessions = GetSessions(data_folder)

    for session in tqdm(sessions):
        svm_folder = os.path.join(data_folder, session, stateToAnalyze, svmType)
        ml_task_folder = os.path.join(svm_folder, f"MLTask_no_z_{experiment_part}ing_{freq}HZ_CORRECT_RMS")
        #output = fr"C:\Liron\neuroscience\PHD\courses\psychological-ML-MatanBenShachar\Smiling_And_Frowning\{stateToAnalyze}\{experiment_part}_100HZ"
        output = fr"C:\Liron\neuroscience\PHD\courses\psychological-ML-MatanBenShachar\smiling_correct_rms\{stateToAnalyze}\{experiment_part}_50HZ"
        #create folder if not exists:
        if not os.path.exists(output):
            os.makedirs(output)
        copy_folder(ml_task_folder, output, session)