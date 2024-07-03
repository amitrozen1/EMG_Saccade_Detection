import pandas as pd
from General.init import *
import os
import numpy as np

class SessionComponentDef:
    def __init__(self, session_name="", w_path=None, bad_electrode=None, chosen_comps=None):
        self.session_name = session_name
        self.w_path = w_path if w_path is not None else []
        self.bad_electrode = bad_electrode if bad_electrode is not None else []
        self.chosen_comps = chosen_comps if chosen_comps is not None else []
    
    def __str__(self):
        return f"{self.session_name}, {self.w_path}, {self.bad_electrode}, {self.chosen_comps}\n"


def convert_string_to_list_of_ints(s):
    if pd.isna(s):
        return []
    s = str(s).replace(" ", "")
    return list(map(int, s.split(",")))


if current_batch == 1:
    df = pd.read_excel("C:\\Liron\\DataEmg\\Try\\batch1_review_after_dwt.xlsx")
else:
    df = pd.read_excel("C:\\Liron\\DataEmg\\new_batch\\batch1_review_after_db6_dwt.xlsx")

sessions = dict()

excluded_for_stats = []

for index, row in df.iterrows():
    if (row["isEnter"] == 1):
        if ( row["include"] == 0):
            excluded_for_stats.append(row["session_name"])
        current_session = row["session_name"]
        #if not empty:
        session_def = SessionComponentDef(current_session)
        for p in ["A", "B"]:
            if not pd.isna(row["filtered_"+p]):
                session_def.w_path.append(fr"IC_w_and_photos\{p}\w.npy")
                session_def.bad_electrode.append(convert_string_to_list_of_ints(row["bad_electrodes_"+p]))
                session_def.chosen_comps.append(convert_string_to_list_of_ints(row["filtered_"+p]))
            else:
                session_def.w_path.append(fr"DWT_all_comps\{p}\w.npy") #(fr"IC_w_and_photos_all_comps\{p}\w.npy")
                session_def.bad_electrode.append([])
                session_def.chosen_comps.append(convert_string_to_list_of_ints(row["NoF_"+p]))

        sessions[current_session] = session_def

def GetSessionsBatch2():
    return list(sessions.keys())

def GetExcludeForStatsSessionBatch2():
    return excluded_for_stats

def get_w_path(session, p):
    return fr"{data_path}\{session}\{sessions[session].w_path[p]}"

def open_w_if_exists_batch2(session, p):
    w_path = get_w_path(session, p)
    if os.path.exists(w_path):
        return np.load(w_path)
    return None

def get_bad_electrode(session, p):
    bad_elecs = sessions[session].bad_electrode[p]
    bad_elecs = [x-1 for x in bad_elecs]
    return bad_elecs

def get_chosen_comps(session, p):
    comps =  sessions[session].chosen_comps[p]
    comps = [x-1 for x in comps]
    return comps

def is_no_review_folder(session, p):
    return "all_comps" in sessions[session].w_path[p]
