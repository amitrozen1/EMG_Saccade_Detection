import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

#add to path parent of current folder:
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ComponentAnalysis.ReviewComponents_batch2 import GetExcludeForStatsSessionBatch2



def add_before_extention(path, add):
    path = path.split('.')
    return path[0] + add + '.' + path[1]


arctan = False # True if we want to use arctan on the values before the wilcoxon test
batch2 = False

muscles = ["Nose", "Eye", "Zygomaticus", "Corrugator", "NegComp", "PosComp"]
muscle_colls = [f"mean_{muscle}_{p}" for muscle in muscles for p in ["A", "B"]]
sync_base = muscles #["chosen_max_comp", "all_comps"] #["chosen_comps", "all_max_comp", "chosen_max_comp"]
sync_colls = [f"{prefix}{ending}" for prefix in sync_base for ending in ['_mean_corr', '_mean_lag', '_max_corr']]
colls =  muscle_colls #sync_colls + muscle_colls


results_base = "RMS_50_downsample_5_window_3_maxlag_3_after_pre_reg_dwt_db15_corrEachComp_scipy_resample_noWeightedCorr_2direct_allow_0.5_delay_noz_noartrem_with_var_withValenceComps"

# Load the data from the uploaded Excel file
#file_path = r"C:\Liron\DataEmg\Try\results_2024\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan.csv"
#first batch new:
#file_path = r"C:\Liron\DataEmg\Try\results_2024\RMS_100_downsample_50_window_3_maxlag_3_after_pre_reg_allcomps_2dmaxver.csv"
file_path = r"C:\Liron\DataEmg\Try\results_2024\RMS_50_downsample_5_window_3_maxlag_3_after_pre_reg_dwt_db15_corrEachComp_scipy_resample_noWeightedCorr_2direct_allow_0.5_delay_noz_noartrem_with_var_withValenceComps.csv"
#second batch new:
if ( batch2 ):
    file_path = r"C:\Liron\DataEmg\new_batch\results_2024\RMS_100_downsample_50_window_3_maxlag_3_after_prereg_au6_allcomps_2dmaxver.csv"
    file_path = r"C:\Liron\DataEmg\new_batch\results_2024\RMS_100_downsample_50_window_3_maxlag_3_after_prereg_dwt_2dmaxver.csv"

data = pd.read_csv(file_path)

data = data[data['action'] == 'Reading']

data['story'] = data['key'].str.extract('Reading_(\d+).png')[0].astype(int)
# data['story'] = data['key'].str.extract('Listening_(\d+).ogg')[0].astype(int)
data['ListenerChoice'] = data.apply(lambda row: row['isChoiceB'] if row['whoseReading'] == 1 else row['isChoiceA'], axis=1)
data['ReaderChoice'] = data.apply(lambda row: row['isChoiceB'] if row['whoseReading'] == 2 else row['isChoiceA'], axis=1)

data['story_prob'] = (data.groupby('story')['isChoiceA'].transform('mean') + data.groupby('story')['isChoiceB'].transform('mean'))/2
data['sum_choice'] = data.apply(lambda row: row['isChoiceA'] + row['isChoiceB'], axis=1)


if False:
    fatigue_df = pd.DataFrame()
    fatigue_df_index = 0
    for index, row in enumerate(data.itertuples()):
        couples_index = index % 8
        if couples_index == 7:
            fatigue_df_index += 2

        if ( couples_index == 0 or couples_index == 2):
            fill_index = fatigue_df_index
            if ( couples_index == 2):
                fill_index = fatigue_df_index +1
            for prefix in sync_base:
                for ending in ['_mean_corr', '_mean_lag', '_max_corr']:
                    fatigue_df.at[fill_index, f"first_reading_{prefix}{ending}"] = getattr(row, f'{prefix}{ending}')
        
        if ( couples_index == 4 or couples_index == 6):
            fill_index = fatigue_df_index
            if ( couples_index == 6):
                fill_index = fatigue_df_index +1
            for prefix in sync_base:
                for ending in ['_mean_corr', '_mean_lag', '_max_corr']:
                    fatigue_df.at[fill_index, f"second_reading_{prefix}{ending}"] = getattr(row, f'{prefix}{ending}')
        
    
    for prefix in sync_base:
        for ending in ['_mean_corr', '_mean_lag', '_max_corr']:
            print(f'{prefix}{ending}')
            a = fatigue_df[f'first_reading_{prefix}{ending}']
            b = fatigue_df[f'second_reading_{prefix}{ending}']
            if ( arctan ):
                a = np.arctan(a)
                b = np.arctan(b)

            print(wilcoxon(a, b, alternative='greater'))

if True:

    data_odd = data.iloc[::2, :].reset_index(drop=True)
    data_even = data.iloc[1::2, :].reset_index(drop=True)


    # Initialize an empty DataFrame to hold the new structure
    new_df = pd.DataFrame()
        
    # Iterate over the rows of data_odd and data_even simultaneously using zip
    for index, (odd_row, even_row) in enumerate(zip(data_odd.itertuples(), data_even.itertuples())):
        # Extract and store values for each of the specified columns, appending "_1" or "_2" as appropriate

        # if batch2 and odd_row.session in GetExcludeForStatsSessionBatch2():
        #     continue

        new_df.at[index, 'chose_same'] = int(odd_row.isChoiceA == odd_row.isChoiceB)
        new_df.at[index, 'story_prob_div'] = odd_row.story_prob / even_row.story_prob
        new_df.at[index, 'story_prob_1'] = odd_row.story_prob
        new_df.at[index, 'story_prob_2'] = even_row.story_prob
        new_df.at[index, 'isOther'] = odd_row.isOther
        new_df.at[index, 'session'] = odd_row.session
        new_df.at[index, 'a_choice'] = even_row.isChoiceA

        if ( odd_row.ListenerChoice == 1):
            new_df.at[index, 'ListenerChoice'] = int(0)
        else:
            new_df.at[index, 'ListenerChoice'] = int(1)

        if ( odd_row.ReaderChoice == 1):
            new_df.at[index, 'ReaderChoice'] = int(0)
        else:
            new_df.at[index, 'ReaderChoice'] = int(1)

        for col in sync_colls:
            new_df.at[index, f'{col}_1'] = getattr(odd_row, col)
            new_df.at[index, f'{col}_2'] = getattr(even_row, col)

            if ( new_df.at[index, "ListenerChoice"] == 0):
                new_df.at[index, f'selected_{col}'] = getattr(odd_row, col)
                new_df.at[index, f'not_selected_{col}'] = getattr(even_row, col)
            else:
                new_df.at[index, f'selected_{col}'] = getattr(even_row, col)
                new_df.at[index, f'not_selected_{col}'] = getattr(odd_row, col)

        for col in colls:
            new_df.at[index, f'{col}_1'] = getattr(odd_row, col)
            new_df.at[index, f'{col}_2'] = getattr(even_row, col)
            
            if ( new_df.at[index, "ListenerChoice"] == 0):
                new_df.at[index, f'selected_{col}'] = getattr(odd_row, col)
                new_df.at[index, f'not_selected_{col}'] = getattr(even_row, col)
            else:
                new_df.at[index, f'selected_{col}'] = getattr(even_row, col)
                new_df.at[index, f'not_selected_{col}'] = getattr(odd_row, col)
            
            
        ReaderIndex = index % 4
        for col in [f"mean_{muscle}" for muscle in muscles]:
            if ( ReaderIndex == 0 or ReaderIndex == 2): # A is reading!
                #story order:
                new_df.at[index, f'Reader_{col}_1'] = getattr(odd_row, f"{col}_A")
                new_df.at[index, f'Reader_{col}_2'] = getattr(even_row, f"{col}_A")
                new_df.at[index, f'Listener_{col}_1'] = getattr(odd_row, f"{col}_B")
                new_df.at[index, f'Listener_{col}_2'] = getattr(even_row, f"{col}_B")
            else: # B is Reading
                new_df.at[index, f'Reader_{col}_1'] = getattr(odd_row, f"{col}_B")
                new_df.at[index, f'Reader_{col}_2'] = getattr(even_row, f"{col}_B")
                new_df.at[index, f'Listener_{col}_1'] = getattr(odd_row, f"{col}_A")
                new_df.at[index, f'Listener_{col}_2'] = getattr(even_row, f"{col}_A")

            #choice order:
            chosen = new_df.at[index, "ListenerChoice"].astype(int) + 1
            unchosen = (chosen % 2) + 1

            new_df.at[index, f'selected_Reader_{col}'] = new_df.at[index, f'Reader_{col}_{chosen}']
            new_df.at[index, f'not_selected_Reader_{col}'] = new_df.at[index, f'Reader_{col}_{unchosen}']
            new_df.at[index, f'selected_Listener_{col}'] = new_df.at[index, f'Listener_{col}_{chosen}']
            new_df.at[index, f'not_selected_Listener_{col}'] = new_df.at[index, f'Listener_{col}_{unchosen}']


                # if ( new_df.at[index, f'{prefix}{ending}_2'] != 0):
                #     new_df.at[index, f'{prefix}{ending}_div'] = new_df.at[index, f'{prefix}{ending}_1'] / new_df.at[index, f'{prefix}{ending}_2']

    # for col in muscle_colls:
    #     print(col)
    #     a = new_df[f'selected_{col}']
    #     b = new_df[f'not_selected_{col}']
    #     if ( arctan ):
    #         a = np.arctan(a)
    #         b = np.arctan(b)

    #     print(wilcoxon(a, b, alternative='greater'))

    alternative = 'greater'


    if (True): #muscles:
        dict = {}
        dict['headers'] = ["Reader_s1_v_s2", "Listener_s1_v_s2", "Reader_chosen_unchosen", "Listener_chosen_unchosen"]
        for muscle in muscles:
            dict[muscle] = []
        
        print("story1 vs story2")

        for muscle in muscles:
            col = f"mean_{muscle}"
            a = new_df[f'Reader_{col}_1']
            b = new_df[f'Reader_{col}_2']

            if ( arctan ):
                a = np.arctan(a)
                b = np.arctan(b)
            
            # print(f"Reader_{col}")
            # print(wilcoxon(a, b, alternative=alternative))
            dict[muscle].append(wilcoxon(a, b, alternative=alternative).pvalue)

        for muscle in muscles:
            col = f"mean_{muscle}"
            a = new_df[f'Listener_{col}_1']
            b = new_df[f'Listener_{col}_2']

            if ( arctan ):
                a = np.arctan(a)
                b = np.arctan(b)
            
            # print(f"Listener_{col}")
            # print(wilcoxon(a, b, alternative=alternative))
            dict[muscle].append(wilcoxon(a, b, alternative=alternative).pvalue)



        print("chosen unchosen")

        for p_type in ["Reader", "Listener"]:
            for muscle in muscles:
                col = f"mean_{muscle}"
                a = new_df[f'selected_{p_type}_{col}']
                b = new_df[f'not_selected_{p_type}_{col}']

                if ( arctan ):
                    a = np.arctan(a)
                    b = np.arctan(b)
                
                # print(f"{p_type}_{col}")
                # print(wilcoxon(a, b, alternative=alternative))
                dict[muscle].append(wilcoxon(a, b, alternative=alternative).pvalue)
        #convert dict to dataframe
        df = pd.DataFrame.from_dict(dict)

        #save df and color in yellow significant p-value:
        df = df.style.applymap(lambda x: 'background-color : yellow' if type(x)!= str and x < 0.05 else '')
        df.to_excel(fr".\reading_sync_tries\muscles_{alternative}_{results_base}.xlsx", engine='openpyxl', index=False)

    if (True): #sync
        dict = {}
        dict['headers'] = ["story1_v_story2", "chosen_unchosen"]
        for col in sync_colls:
            dict[col] = []

        print("story1 vs story2")

        for col in sync_colls:
            a = new_df[f'{col}_1']
            b = new_df[f'{col}_2']

            if ( arctan ):
                a = np.arctan(a)
                b = np.arctan(b)
            
            # print(col)
            # print(wilcoxon(a, b, alternative=alternative))
            dict[col].append(wilcoxon(a, b, alternative=alternative).pvalue)

        print("chosen unchosen")


        for col in sync_colls:
            a = new_df[f'selected_{col}']
            b = new_df[f'not_selected_{col}']

            if ( arctan ):
                a = np.arctan(a)
                b = np.arctan(b)
            
            #print(f"{col}")
            #print(wilcoxon(a, b, alternative=alternative))
            dict[col].append(wilcoxon(a, b, alternative=alternative).pvalue)
        #convert dict to dataframe
        df = pd.DataFrame.from_dict(dict)
        df = df.style.applymap(lambda x: 'background-color : yellow' if type(x)!= str and x < 0.05 else '')
        df.to_excel(fr".\reading_sync_tries\sync_{alternative}_{results_base}.xlsx", engine='openpyxl', index=False)