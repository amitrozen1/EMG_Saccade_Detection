from General.HelperFunctions import GetSessions
from tqdm import tqdm
import os

def copy(source, dest):
    return os.system('xcopy "%s" "%s" /E /I' % (source, dest))

def GetDirsToCopy(sessoinPath):
    filesToCopy = []
    filesAndDirs = os.listdir(sessoinPath)
    for name in filesAndDirs:
        p = os.path.join(sessoinPath, name)
        if "." not in name:
            #directory
            filesToCopy.append(p)
    return filesToCopy

def CopyAllFolders(directory, saveAt):
    session = str.split(directory, "\\")[-1]
    dirAtDest = os.path.join(saveAt, session)
    if( not os.path.exists(dirAtDest) ):
        os.makedirs(dirAtDest)
        folders = GetDirsToCopy(directory)
        for full_folder in folders:
            folder = str.split(full_folder, "\\")[-1]
            copy(full_folder, os.path.join(dirAtDest, folder))

data_path = r"C:\Liron\DataEmg\Try"
saveAt = r"Z:\Backup\Liron\ForYuval\data"

sessions = GetSessions(data_path)
existing_sessions = GetSessions(saveAt)

for session in tqdm(sessions):
    if session not in existing_sessions:
        directory = os.path.join(data_path, session)
        CopyAllFolders(directory, saveAt)
