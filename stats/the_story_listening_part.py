import pandas as pd
import numpy as np
from scipy.stats import wilcoxon

def add_before_extention(path, add):
    path = path.split('.')
    return path[0] + add + '.' + path[1]

def get_first_ordered_stories():
    pairs = [[6,12], [3,11], [5,10], [8,16]]
    #first values from pairs:
    first_values = [pair[0] for pair in pairs]
    return first_values

def is_original_ordered(first_story):
    return first_story in get_first_ordered_stories()


# Load the data from the uploaded Excel file
#file_path = r"C:\Liron\DataEmg\Try\results_2024\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_no_arctan_nooverlap_with_actress_compare_no_zyg.csv"
#file_path = r"C:\Liron\DataEmg\Try\results_2024\RMS_300_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_no_arctan_nooverlap_with_actress_compare_no_zyg_hardremove.csv"

#best:
#file_path = r"C:\Liron\DataEmg\Try\results_2024\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_no_arctan_nooverlap_with_actress_compare_no_zyg_hardremove.csv"

#file_path = r"C:\Liron\DataEmg\Try\results_2024\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_no_arctan_nooverlap_with_actress_compare_no_zyg_for_sync_hardremove.csv"

#file_path = r"C:\Liron\DataEmg\new_batch\results_2024\RMS_100_downsample_50_window_3_maxlag_3_after_prereg_au6_allcomps_with_actress.csv"
file_path = r"C:\Liron\DataEmg\new_batch\results_2024\RMS_100_downsample_50_window_3_maxlag_3_after_prereg_au6_allcomps_with_actress_2d.csv"
data = pd.read_csv(file_path)

data = data[data['action'] == 'Listening']

data['story'] = data['key'].str.extract('Listening_(\d+).ogg')[0].astype(int)
data['pair_id'] = data['key'].str.extract('Listening_(\d+).ogg_(\d+)')[1].astype(int)

muscles = ["Nose", "Eye", "Zygomaticus", "Corrugator"]

data['story_prob'] = (data.groupby('story')['isChoiceA'].transform('mean') + data.groupby('story')['isChoiceB'].transform('mean'))/2
data['sum_choice'] = data.apply(lambda row: row['isChoiceA'] + row['isChoiceB'], axis=1)

#add a pairwise ["a","b"] column each even row is a and odd row is b
data['order'] = ['a' if i % 2 == 0 else 'b' for i in range(len(data))]

sync_colls = [f"{prefix}{ending}" for prefix in ["chosen_comps", "all_comps"] for ending in ['_mean_corr', '_mean_lag', '_max_corr']]


data_odd = data.iloc[::2, :].reset_index(drop=True)
data_even = data.iloc[1::2, :].reset_index(drop=True)

# Initialize an empty DataFrame to hold the new structure
new_df = pd.DataFrame()


# Iterate over the rows of data_odd and data_even simultaneously using zip
for index, (odd_row, even_row) in enumerate(zip(data_odd.itertuples(), data_even.itertuples())):
    # Initialize a new row as a dictionary
    new_row_A = {}
    new_row_B = {}

    for new_row in [new_row_A, new_row_B]:
    # Extract and store values for each of the specified columns, appending "_1" or "_2" as appropriate
        new_row['chose_same'] = int(odd_row.isChoiceA == odd_row.isChoiceB)
        new_row['session'] = odd_row.session
        new_row['order_type'] = 'reg' if is_original_ordered(odd_row.story) else 'reverse'
        new_row['story_1'] = odd_row.story
        new_row['story_2'] = even_row.story
        new_row['pair_id'] = odd_row.pair_id
        new_row['is_other'] = int(odd_row.isOther)
        for part in muscles:
            new_row[f"mean_actress_{part}_1"] = getattr(odd_row, f"mean_actress_{part}")
            new_row[f"mean_actress_{part}_2"] = getattr(even_row, f"mean_actress_{part}")
    
    new_row_A['choice'] = even_row.isChoiceA
    new_row_B['choice'] = even_row.isChoiceB

    # Add additional columns based on your dynamic column names stored in 'colls'
    
    for new_row, p in zip([new_row_A, new_row_B], ["A", "B"]):
        for part in muscles + ["chosen_max"]:
            new_row[f"mean_{part}_1"] = getattr(odd_row, f"mean_{part}_{p}")
            new_row[f"mean_{part}_2"] = getattr(even_row, f"mean_{part}_{p}")

            if ( new_row['choice'] == 0):
                new_row[f"mean_{part}_selected"] = getattr(odd_row, f"mean_{part}_{p}")
                new_row[f"mean_{part}_not_selected"] = getattr(even_row, f"mean_{part}_{p}")
            else:
                new_row[f"mean_{part}_selected"] = getattr(even_row, f"mean_{part}_{p}")
                new_row[f"mean_{part}_not_selected"] = getattr(odd_row, f"mean_{part}_{p}")
        
        for col in sync_colls:
            new_row[f'{col}_1'] = getattr(odd_row, col)
            new_row[f'{col}_2'] = getattr(even_row, col)

            if ( new_row['choice'] == 0):
                new_row[f'{col}_selected'] = getattr(odd_row, col)
                new_row[f'{col}_not_selected'] = getattr(even_row, col)
            else:
                new_row[f'{col}_selected'] = getattr(even_row, col)
                new_row[f'{col}_not_selected'] = getattr(odd_row, col)
        
        for col in ['mean_corr', 'mean_lag', 'max_corr']:
            actress_prefix = "chosen_actress_2d"
            new_row[f'chosen_actress_{col}_1'] = getattr(odd_row, f'{actress_prefix}_{p}_{col}')
            new_row[f'chosen_actress_{col}_2'] = getattr(even_row, f'{actress_prefix}_{p}_{col}')

            if (new_row['choice'] == 0):
                new_row[f'chosen_actress_{col}_selected'] = getattr(odd_row, f'{actress_prefix}_{p}_{col}')
                new_row[f'chosen_actress_{col}_not_selected'] = getattr(even_row, f'{actress_prefix}_{p}_{col}')
            else:
                new_row[f'chosen_actress_{col}_selected'] = getattr(even_row, f'{actress_prefix}_{p}_{col}')
                new_row[f'chosen_actress_{col}_not_selected'] = getattr(odd_row, f'{actress_prefix}_{p}_{col}')

    # Append the new row dictionary as a DataFrame row to new_df
    #new_df = new_df.append(new_row, ignore_index=True)
    # new_df = new_df.append(new_row_A, ignore_index=True)
    # new_df = new_df.append(new_row_B, ignore_index=True)

    new_df = pd.concat([new_df, pd.DataFrame.from_records([new_row_A, new_row_B])])

arctan = False
#compare story 1 vs story 2:

# new_df = new_df[new_df['chose_same'] == 1]
# print(len(new_df))


#only where is other == 0:
df_self = new_df[new_df['is_other'] == 0]
df_other = new_df[new_df['is_other'] == 1]

print("\nmuscles story1 vs story2")
if True:
    for col in [f"mean_{muscle}" for muscle in muscles]:
        a_self = df_self[f'{col}_1']
        b_self = df_self[f'{col}_2']

        a_other = df_other[f'{col}_1']
        b_other = df_other[f'{col}_2']

        a = new_df[f'{col}_1']
        b = new_df[f'{col}_2']
        
        if ( arctan ):
            a_self = np.arctan(a_self)
            b_self = np.arctan(b_self)
            a_other = np.arctan(a_other)
            b_other = np.arctan(b_other)
        
        print(f"{col}_self")
        print(wilcoxon(a_self, b_self, alternative='two-sided'))

        print(f"{col}_other")
        print(wilcoxon(a_other, b_other, alternative='two-sided'))

        print(f"{col}_all")
        print(wilcoxon(a, b, alternative='two-sided'))

if True:
    print ("\ncompare to actress - story1 vs story2")
    for col in ['mean_corr', 'mean_lag', 'max_corr']:
        a = new_df[f'chosen_actress_{col}_1']
        b = new_df[f'chosen_actress_{col}_2']
        
        a_self = df_self[f'chosen_actress_{col}_1']
        b_self = df_self[f'chosen_actress_{col}_2']

        a_other = df_other[f'chosen_actress_{col}_1']
        b_other = df_other[f'chosen_actress_{col}_2']
        
        if ( arctan ):
            a_self = np.arctan(a_self)
            b_self = np.arctan(b_self)
            a_other = np.arctan(a_other)
            b_other = np.arctan(b_other)
        
        print(f"chosen_actress_{col}_self")
        print(wilcoxon(a_self, b_self, alternative='two-sided'))

        print(f"chosen_actress_{col}_other")
        print(wilcoxon(a_other, b_other, alternative='two-sided'))

        print(f"chosen_actress_{col}_all")
        print(wilcoxon(a, b, alternative='two-sided'))  


if True:
    print("\nmuscles selected vs un-selected:")
    for col in [f"mean_{muscle}" for muscle in muscles]:
        a = new_df[f'{col}_selected']
        b = new_df[f'{col}_not_selected']

        a_self = df_self[f'{col}_selected']
        b_self = df_self[f'{col}_not_selected']

        a_other = df_other[f'{col}_selected']
        b_other = df_other[f'{col}_not_selected']

        if ( arctan ):
            a = np.arctan(a)
            b = np.arctan(b)
        
        print(f"{col}_self")
        print(wilcoxon(a_self, b_self, alternative='two-sided'))

        print(f"{col}_other")
        print(wilcoxon(a_other, b_other, alternative='two-sided'))

        print(f"{col}_all")
        print(wilcoxon(a, b, alternative='two-sided'))

#take only odd rows:
# new_df = df_self.iloc[::2, :].reset_index(drop=True)

if True:
    print("\nsynchrony selected vs un-selected compare:")
    for col in sync_colls:
        a = new_df[f'{col}_selected']
        b = new_df[f'{col}_not_selected']

        a_self = df_self[f'{col}_selected']
        b_self = df_self[f'{col}_not_selected']

        a_other = df_other[f'{col}_selected']
        b_other = df_other[f'{col}_not_selected']

        if ( arctan ):
            a = np.arctan(a)
            b = np.arctan(b)
        
        print(f"{col}_self")
        print(wilcoxon(a_self, b_self, alternative='two-sided'))

        print(f"{col}_other")
        print(wilcoxon(a_other, b_other, alternative='two-sided'))

        print(f"{col}_all")
        print(wilcoxon(a, b, alternative='two-sided'))

    if True:
        print("\ncompare to actress : selected vs un-selected")

        for col in ['mean_corr', 'mean_lag', 'max_corr']:
            a = new_df[f'chosen_actress_{col}_selected']
            b = new_df[f'chosen_actress_{col}_not_selected']

            a_self = df_self[f'chosen_actress_{col}_selected']
            b_self = df_self[f'chosen_actress_{col}_not_selected']

            a_other = df_other[f'chosen_actress_{col}_selected']
            b_other = df_other[f'chosen_actress_{col}_not_selected']

            if ( arctan ):
                a = np.arctan(a)
                b = np.arctan(b)
            
            print(f"chosen_actress_{col}_self")
            print(wilcoxon(a_self, b_self, alternative='two-sided'))

            print(f"chosen_actress_{col}_other")
            print(wilcoxon(a_other, b_other, alternative='two-sided'))

            print(f"chosen_actress_{col}_all")
            print(wilcoxon(a, b, alternative='two-sided'))
    
#Todo: try with the corrEachComp also ? 

