import pandas as pd

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
#file_path = r"C:\Liron\DataEmg\Try\results_2024\RMS_100_downsample_50_window_3_maxlag_3_remove-artifacts_nooverlap_actress-compare_no-zyg-for-sync_hardremove_correctlag3_lagCount_actress_lag.csv"
file_path = r"C:\Liron\DataEmg\new_batch\results_2024\RMS_100_downsample_50_window_3_maxlag_3_after_prereg_au6_allcomps_with_actress.csv"
data = pd.read_csv(file_path)

data = data[data['action'] == 'Listening']

data['story'] = data['key'].str.extract('Listening_(\d+).ogg')[0].astype(int)
data['pair_id'] = data['key'].str.extract('Listening_(\d+).ogg_(\d+)')[1].astype(int)

muscles = ["Nose", "Eye", "Zygomaticus", "Corrugator"]

data['story_prob'] = (data.groupby('story')['isChoiceA'].transform('mean') + data.groupby('story')['isChoiceB'].transform('mean'))/2
data['sum_choice'] = data.apply(lambda row: row['isChoiceA'] + row['isChoiceB'], axis=1)

#add a pairwise ["a","b"] column each even row is a and odd row is b
data['order'] = ['a' if i % 2 == 0 else 'b' for i in range(len(data))]

sync_colls = [f"{prefix}{ending}" for prefix in ["chosen_comps"] for ending in ['_mean_corr', '_mean_lag', '_max_corr']]

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

    # Add additional columns based on your dynamic column names stored in 'colls'
    
    for new_row, p in zip([new_row_A, new_row_B], ["A", "B"]):
        for part in muscles + ["chosen_max"]:
            new_row[f"mean_{part}_1"] = getattr(odd_row, f"mean_{part}_{p}")
            new_row[f"mean_{part}_2"] = getattr(even_row, f"mean_{part}_{p}")
        
        for col in sync_colls + ["lags_under_500"]:
            new_row[f'{col}_1'] = getattr(odd_row, col)
            new_row[f'{col}_2'] = getattr(even_row, col)
        
        for col in ['mean_corr', 'mean_lag', 'max_corr']:
            new_row[f'chosen_actress_{col}_1'] = getattr(odd_row, f'chosen_actress_{p}_{col}')
            new_row[f'chosen_actress_{col}_2'] = getattr(even_row, f'chosen_actress_{p}_{col}')

        actress_underlag = "actress_lags_under_500"
        new_row[actress_underlag + "_1"] = getattr(odd_row, actress_underlag + "_" + p)
        new_row[actress_underlag + "_2"] = getattr(even_row, actress_underlag + "_" + p)


    new_row_A['choice'] = even_row.isChoiceA
    new_row_B['choice'] = even_row.isChoiceB
    # Append the new row dictionary as a DataFrame row to new_df
    #new_df = new_df.append(new_row, ignore_index=True)
    new_df = new_df.append(new_row_A, ignore_index=True)
    new_df = new_df.append(new_row_B, ignore_index=True)

new_df.to_csv(add_before_extention(file_path, "_listening_for_R_split_for_particiapnt"), index = False)