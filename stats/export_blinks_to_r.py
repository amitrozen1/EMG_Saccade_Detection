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
file_path = r"C:\Liron\DataEmg\Try\results_2024\blinks\reading_blinks_batch1.csv"

data = pd.read_csv(file_path)

data['story'] = data['key'].str.extract('Reading_(\d+).png')[0].astype(int)
data['pair_id'] = data['key'].str.extract('Reading_(\d+).png_(\d+)')[1].astype(int)

# data['story'] = data['key'].str.extract('Listening_(\d+).ogg')[0].astype(int)
data['ListenerChoice'] = data.apply(lambda row: row['isChoiceB'] if row['whoseReading'] == 1 else row['isChoiceA'], axis=1)
data['ReaderChoice'] = data.apply(lambda row: row['isChoiceB'] if row['whoseReading'] == 2 else row['isChoiceA'], axis=1)
data['ListenerRT'] = data.apply(lambda row: row['rtChoiceB'] if row['whoseReading'] == 1 else row['rtChoiceA'], axis=1)

data['story_prob'] = (data.groupby('story')['isChoiceA'].transform('mean') + data.groupby('story')['isChoiceB'].transform('mean'))/2
data['sum_choice'] = data.apply(lambda row: row['isChoiceA'] + row['isChoiceB'], axis=1)

#add a pairwise ["a","b"] column each even row is a and odd row is b
data['order'] = ['a' if i % 2 == 0 else 'b' for i in range(len(data))]

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

    for col in ["plv_score", "rated_score"]:
        #check if oddrow continas the column:
        if hasattr(odd_row, col):
            new_df.at[index, f'{col}_1'] = getattr(odd_row, col)
            new_df.at[index, f'{col}_2'] = getattr(even_row, col)
        
    ReaderIndex = index % 4
    for col in ["sum_blinks"]:
        if ( ReaderIndex == 0 or ReaderIndex == 2): # A is reading!
            #story order:
            new_df.at[index, f'Reader_{col}_1'] = getattr(odd_row, f"{col}_a")
            new_df.at[index, f'Reader_{col}_2'] = getattr(even_row, f"{col}_a")
            new_df.at[index, f'Listener_{col}_1'] = getattr(odd_row, f"{col}_b")
            new_df.at[index, f'Listener_{col}_2'] = getattr(even_row, f"{col}_b")
        else: # B is Reading
            new_df.at[index, f'Reader_{col}_1'] = getattr(odd_row, f"{col}_b")
            new_df.at[index, f'Reader_{col}_2'] = getattr(even_row, f"{col}_b")
            new_df.at[index, f'Listener_{col}_1'] = getattr(odd_row, f"{col}_a")
            new_df.at[index, f'Listener_{col}_2'] = getattr(even_row, f"{col}_a")
            

new_df.to_csv(add_before_extention(file_path, "_readnig_for_R"), index = False)