import pandas as pd


def add_before_extention(path, add):
    path = path.split('.')
    return path[0] + add + '.' + path[1]

def change_to_ordered_choice(listener_choice, oreder):
    if listener_choice == 1 and oreder == 'a':
        return 'a'
    elif listener_choice == 1 and oreder == 'b':
        return 'b'
    elif listener_choice == 0 and oreder == 'a':
        return 'b'
    elif listener_choice == 0 and oreder == 'b':
        return 'a'

def get_first_ordered_stories():
    pairs = [[1,14], [9,13], [7,15], [2,4]]
    #first values from pairs:
    first_values = [pair[0] for pair in pairs]
    return first_values

def is_original_ordered(first_story):
    return first_story in get_first_ordered_stories()

def change_to_paired_ordered_choice(listener_choice, story):
    first_values = get_first_ordered_stories()
    if ( story in first_values):
        if listener_choice == 1:
            return 'a'
        else:
            return 'b'
    else:
        if listener_choice == 1:
            return 'b'
        else:
            return 'a'

# Load the data from the uploaded Excel file
file_path = r"C:\Liron\DataEmg\Try\results_2024\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap_with_mean_of_corr_comp_VAR.csv"
data = pd.read_csv(file_path)

data = data[data['action'] == 'Reading']

data['story'] = data['key'].str.extract('Reading_(\d+).png')[0].astype(int)
data['pair_id'] = data['key'].str.extract('Reading_(\d+).png_(\d+)')[1].astype(int)

# data['story'] = data['key'].str.extract('Listening_(\d+).ogg')[0].astype(int)
data['ListenerChoice'] = data.apply(lambda row: row['isChoiceB'] if row['whoseReading'] == 1 else row['isChoiceA'], axis=1)
data['ReaderChoice'] = data.apply(lambda row: row['isChoiceB'] if row['whoseReading'] == 2 else row['isChoiceA'], axis=1)

muscles = ["Nose", "Eye", "Zygomaticus", "Corrugator", "chosen_max", "sum"]

for muscle in muscles:
    col = f"mean_{muscle}"
    data[f"Reader_{muscle}"] = data.apply(lambda row: row[f"{col}_B"] if row['whoseReading'] == 2 else row[f"{col}_A"], axis=1)
    data[f"Listener_{muscle}"] = data.apply(lambda row: row[f"{col}_B"] if row['whoseReading'] == 1 else row[f"{col}_A"], axis=1)

data['story_prob'] = (data.groupby('story')['isChoiceA'].transform('mean') + data.groupby('story')['isChoiceB'].transform('mean'))/2
data['sum_choice'] = data.apply(lambda row: row['isChoiceA'] + row['isChoiceB'], axis=1)

#add a pairwise ["a","b"] column each even row is a and odd row is b
data['order'] = ['a' if i % 2 == 0 else 'b' for i in range(len(data))]


muscle_colls = [f"mean_{muscle}_{p}" for muscle in muscles for p in ["A", "B"]]
sync_colls = [f"{prefix}{ending}" for prefix in ["chosen_max_comps","chosen_sum_comps"] for ending in ['_mean_corr', '_mean_lag', '_max_corr', '_var_corr', '_var_lag']]
#[f"{prefix}{ending}" for prefix in ['chosen_max_comps', 'chosen_sum_comps'] for ending in ['_mean_corr', '_mean_lag', '_max_corr']]
colls =  sync_colls + muscle_colls

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

    if ( odd_row.ReaderChoice == 1):
        new_df.at[index, 'ReaderChoice'] = int(0)
    else:
        new_df.at[index, 'ReaderChoice'] = int(1)

    for col in colls:
        new_df.at[index, f'{col}_1'] = getattr(odd_row, col)
        new_df.at[index, f'{col}_2'] = getattr(even_row, col)
        
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
            

new_df.to_csv(add_before_extention(file_path, "_readnig_for_R_paired_ordered"), index = False)