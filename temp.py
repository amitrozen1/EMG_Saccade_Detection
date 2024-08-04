
import pandas as pd
from tqdm import tqdm
from General.HelperFunctions import getPathsFromSessionFolder, GetSessions
import os
from General.init import *
from ComponentAnalysis.ComponentExtractor import load_special_components_if_exists


df = pd.DataFrame()

def list_to_str(l):
    l = [int(x)+1 for x in l]
    return ",".join(map(str, l))

sessions = GetSessions(data_path)
for session in tqdm(sessions):
    new_row = {"session_name": session, "isEnter": 1, "include": 1, "filtered_A": "", "filtered_B": ""}
    for p, pchar in zip([A, B], ["A", "B"]):
        chosen_electrodes = load_special_components_if_exists(data_path,session, p, dwt=True)
        new_row["NoF_"+pchar] = list_to_str(chosen_electrodes)
    df = df.append(new_row, ignore_index=True)

df.columns = ["session_name", "isEnter", "include", "filtered_A","filtered_B", "NoF_A", "NoF_B"]
df.to_excel(os.path.join(data_path, "batch1_review_after_db6_dwt.xlsx"), index=False)
