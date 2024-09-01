import pandas as pd
#add parent folder to path:
import sys
sys.path.append(r"C:\Liron\Code\emg_data_analysis")
from Analysis.cross_corr_comp_with_decisions import corrolation_each_comp_with_actress_headers, add_mean_component_headers, corr_lag_measurments_n_headers
from ComponentAnalysis.ComponentExtractor import get_special_component_names

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
import os
folder = r"C:\Liron\DataEmg\try\results_2024\smiles"
file_path = os.path.join(folder, "batch1_svr.csv")

data = pd.read_csv(file_path)

data = data[data['action'] == 'Listening']

data['story'] = data['key'].str.extract('Listening_(\d+).ogg')[0].astype(int)
data['pair_id'] = data['key'].str.extract('Listening_(\d+).ogg_(\d+)')[1].astype(int)

data['story_prob'] = (data.groupby('story')['isChoiceA'].transform('mean') + data.groupby('story')['isChoiceB'].transform('mean'))/2
data['sum_choice'] = data.apply(lambda row: row['isChoiceA'] + row['isChoiceB'], axis=1)

#add a pairwise ["a","b"] column each even row is a and odd row is b
data['order'] = ['a' if i % 2 == 0 else 'b' for i in range(len(data))]

colls_for_A_B = ["mean_corr", "mean_lag", "mean_smile_A", "mean_smile_B"]

data_odd = data.iloc[::2, :].reset_index(drop=True)
data_even = data.iloc[1::2, :].reset_index(drop=True)

# Initialize an empty DataFrame to hold the new structure
AB_df = pd.DataFrame()

# Iterate over the rows of data_odd and data_even simultaneously using zip
for index, (odd_row, even_row) in enumerate(zip(data_odd.itertuples(), data_even.itertuples())):
    AB_df.at[index, 'chose_same'] = int(odd_row.isChoiceA == odd_row.isChoiceB)
    AB_df.at[index, 'session'] = odd_row.session
    AB_df.at[index, 'order_type'] = 'reg' if is_original_ordered(odd_row.story) else 'reverse'
    AB_df.at[index, 'pair_id'] = odd_row.pair_id
    AB_df.at[index, "is_other"] = odd_row.isOther

    AB_df.at[index, 'choice_a'] = even_row.isChoiceA
    AB_df.at[index, 'choice_b'] = even_row.isChoiceB

    for col in colls_for_A_B:
        AB_df.at[index, col + "_1"] = getattr(odd_row, col)
        AB_df.at[index, col + "_2"] = getattr(even_row, col)


AB_df.to_csv(add_before_extention(file_path, "_listening_AB_for_R"), index = False)

