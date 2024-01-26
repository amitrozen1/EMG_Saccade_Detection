from General.HelperFunctions import GetSessions
import os
from tqdm import tqdm

def copy(source, dest):
    return os.system('copy "%s" "%s"' % (source, dest))

def copy_file_that_ends_with(path_to_folder, file_end, copy_to_path):
    files = [f for f in os.listdir(path_to_folder) if f.endswith(file_end)]
    for file in files:
        full_path = os.path.join(path_to_folder, file)
        new_path = os.path.join(copy_to_path, file)
        os.makedirs(copy_to_path, exist_ok=True)
        copy(full_path, new_path)


def handle_session(path_to_folder, dest_path, session):
    path_to_session = os.path.join(path_to_folder, session)
    # cp corrections and psychData
    copy_to_path = os.path.join(dest_path, session)
    exts = ["csv", "txt"]
    for ext in exts:
        copy_file_that_ends_with(path_to_session, ext, copy_to_path)
    
    for p in ["A", "B"]:
        path_to_session_p = os.path.join(path_to_session, p)
        # copy files that ends with "filtered.edf" accounted for typos.
        for ext in ["filtered.edf", "filterd.edf", "jpg", "jpeg", "png"]:
            copy_file_that_ends_with(path_to_session_p, ext, copy_to_path)


origin_path = r"Z:\Backup\Liron\DATA_FROM_EXPERIMENT\Batch_2"
origin_path_2 = os.path.join(origin_path, "just_added")
dest_path = r"C:\Liron\DataEmg\new_batch"

sessions = GetSessions(origin_path)


sessions_2 = GetSessions(origin_path_2)
sessions.extend(sessions_2)
existing_sessions = GetSessions(dest_path)

for session in tqdm(sessions_2):
    if session not in existing_sessions:
        handle_session(origin_path_2, dest_path, session)