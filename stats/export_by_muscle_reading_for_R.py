import pandas as pd

def add_before_extention(path, add):
    #split to the last instance of "."
    parts = path.split('.')
    ext = parts[-1]
    before_ext = ''.join(parts[:-1])
    return before_ext + add + '.' + ext

def get_first_ordered_stories():
    pairs = [[1,14], [9,13], [7,15], [2,4]]
    #first values from pairs:
    first_values = [pair[0] for pair in pairs]
    return first_values

def is_original_ordered(first_story):
    return first_story in get_first_ordered_stories()


# Load the data from the uploaded Excel file
file_path = r"c:\Liron\DataEmg\new_batch\results_2024\RMS_50_downsample_5_window_3_maxlag_3_after_pre_reg_dwt_db15_corrEachComp_scipy_resample_noWeightedCorr_1direct_allow_05_delay_noz_noartrem_with_var_Neg2Pos5Puller4_batch2_all_with_ok_with_complist.csv"

data = pd.read_csv(file_path)

data = data[data['action'] == 'Reading']

data['story'] = data['key'].str.extract('Reading_(\d+).png')[0].astype(int)
data['pair_id'] = data['key'].str.extract('Reading_(\d+).png_(\d+)')[1].astype(int)

# data['story'] = data['key'].str.extract('Listening_(\d+).ogg')[0].astype(int)
data['ListenerChoice'] = data.apply(lambda row: row['isChoiceB'] if row['whoseReading'] == 1 else row['isChoiceA'], axis=1)
data['ReaderChoice'] = data.apply(lambda row: row['isChoiceB'] if row['whoseReading'] == 2 else row['isChoiceA'], axis=1)
data['ListenerRT'] = data.apply(lambda row: row['rtChoiceB'] if row['whoseReading'] == 1 else row['rtChoiceA'], axis=1)

muscles = ["Nose", "Eye", "Zygomaticus", "Corrugator", "NegComp", "PosComp", "LipCornerDownPuller"]
muscle_actions = ["mean", "max", "mean_second_half"]

# for muscle in muscles:
#     for col in muscle_actions:
#         col = f"{col}_{muscle}"
#         data[f"Reader_{col}_{muscle}"] = data.apply(lambda row: row[f"{col}_B"] if row['whoseReading'] == 2 else row[f"{col}_A"], axis=1)
#         data[f"Listener_{col}_{muscle}"] = data.apply(lambda row: row[f"{col}_B"] if row['whoseReading'] == 1 else row[f"{col}_A"], axis=1)

data['story_prob'] = (data.groupby('story')['isChoiceA'].transform('mean') + data.groupby('story')['isChoiceB'].transform('mean'))/2
data['sum_choice'] = data.apply(lambda row: row['isChoiceA'] + row['isChoiceB'], axis=1)

#add a pairwise ["a","b"] column each even row is a and odd row is b
data['order'] = ['a' if i % 2 == 0 else 'b' for i in range(len(data))]


muscle_colls = [f"mean_{muscle}_{p}" for muscle in muscles for p in ["A", "B"]]
sync_colls = [f"{muscle}_{ending}" for muscle in muscles for ending in ['mean_corr', 'mean_lag', 'max_corr', 'mean_second_half_corr', 'mean_second_half_lag', 'lags_under_500', 'lags_under_1000', 'var_corr', 'var_lag']]
whole_sync_colls = [f"{prefix}_{muscle}_{ending}" for prefix in ["whole", "whole_second_half"] for muscle in muscles for ending in ['corr', 'lag']]

data_odd = data.iloc[::2, :].reset_index(drop=True)
data_even = data.iloc[1::2, :].reset_index(drop=True)

# Initialize an empty DataFrame to hold the new structure
new_df = pd.DataFrame()

# Iterate over the rows of data_odd and data_even simultaneously using zip
for index, (odd_row, even_row) in enumerate(zip(data_odd.itertuples(), data_even.itertuples())):
    # Extract and store values for each of the specified columns, appending "_1" or "_2" as appropriate

    new_df.at[index, 'chose_same'] = int(odd_row.isChoiceA == odd_row.isChoiceB)
    new_df.at[index, 'session'] = odd_row.session
    new_df.at[index, 'whoseReading'] = odd_row.whoseReading
    new_df.at[index, 'order_type'] = 'reg' if is_original_ordered(odd_row.story) else 'reverse'
    new_df.at[index, 'pair_id'] = odd_row.pair_id

    if ( odd_row.ListenerChoice == 1):
        new_df.at[index, 'ListenerChoice'] = int(0)
    else:
        new_df.at[index, 'ListenerChoice'] = int(1)

    if hasattr(odd_row,"ok_electrodes"):
        new_df.at[index, 'ok_electrodes'] = odd_row.ok_electrodes

    if ( odd_row.ReaderChoice == 1):
        new_df.at[index, 'ReaderChoice'] = int(0)
    else:
        new_df.at[index, 'ReaderChoice'] = int(1)

    new_df.at[index, 'ListenerRT'] = odd_row.ListenerRT

    for col in sync_colls + whole_sync_colls:
        #check if oddrow continas the column:
        if hasattr(odd_row, col):
            new_df.at[index, f'{col}_1'] = getattr(odd_row, col)
            new_df.at[index, f'{col}_2'] = getattr(even_row, col)
        
    ReaderIndex = index % 4
    for col in [f"{action}_{muscle}" for muscle in muscles for action in muscle_actions]:
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
            

new_df.to_csv(add_before_extention(file_path, "_readnig_for_R"), index = False)