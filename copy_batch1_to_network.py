from General.HelperFunctions import GetSessions
import os
from tqdm import tqdm

def copy(source, dest):
    return os.system('copy "%s" "%s"' % (source, dest))

def ends_with_txt(f):
    return f.endswith("txt")

def experiment_file(f):
    return f.endswith("csv") and "TwoPeopleEmg" in f

def valid_edf(f):
    return f.endswith("filtered.edf") or f.endswith("filterd.edf")

def copy_files_that_meets_condition(path_to_folder, condition, copy_to_path):
    files = [f for f in os.listdir(path_to_folder) if condition(f)]
    for file in files:
        full_path = os.path.join(path_to_folder, file)
        new_path = os.path.join(copy_to_path, file)
        os.makedirs(copy_to_path, exist_ok=True)
        copy(full_path, new_path)

def copy_entire_folder(path_to_folder, copy_to_path):
    os.makedirs(copy_to_path, exist_ok=True)
    os.system(f'xcopy "{path_to_folder}" "{copy_to_path}" /e')

def handle_session(path_to_folder, dest_path, session):
    path_to_session = os.path.join(path_to_folder, session)
    # cp corrections and psychData
    copy_to_path = os.path.join(dest_path, session)
    copy_files_that_meets_condition(path_to_session, ends_with_txt, copy_to_path)
    copy_files_that_meets_condition(path_to_session, experiment_file, copy_to_path)
    w_folder = os.path.join(path_to_session, "DWT_all_comps_db6")
    copy_entire_folder(w_folder, os.path.join(copy_to_path, "DWT_all_comps_db6"))
    # copy files that ends with "filtered.edf" accounted for typos.
    copy_files_that_meets_condition(path_to_session, valid_edf, copy_to_path)


origin_path = r"C:\Liron\DataEmg\try"
dest_path = r"Z:\Backup\Liron\DATA_FROM_EXPERIMENT\batch_1_filtered"
os.makedirs(dest_path, exist_ok=True)
sessions = GetSessions(origin_path)


for session in tqdm(sessions):
    handle_session(origin_path, dest_path, session)