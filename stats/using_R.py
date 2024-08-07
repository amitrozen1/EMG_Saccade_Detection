import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
import pandas as pd


def add_before_extention(path, add):
    path = path.split('.')
    return path[0] + add + '.' + path[1]

# Load the data from the uploaded Excel file
file_path = r"C:\Liron\DataEmg\Try\results_2024\RMS_100_downsample_20_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan_nooverlap.csv"
#file_path = r"C:\Liron\DataEmg\Try\results_2024\RMS_100_downsample_50_window_3_maxlag_3_after_decision_and_remove_artifacts_with_callibration_remove_zyg_for_read_no_arctan.csv"
data = pd.read_csv(file_path)

data = data[data['action'] == 'Reading']

# # # # Normalize the correlation values in each session:
# for prefix in ['chosen_comps', 'all_max_comp', 'chosen_max_comp']:
#     data[f'{prefix}_mean_corr'] = data.groupby('session')[f'{prefix}_mean_corr'].transform(lambda x: (x - x.mean()) / x.std())
#     data[f'{prefix}_mean_lag'] = data.groupby('session')[f'{prefix}_mean_lag'].transform(lambda x: (x - x.mean()) / x.std())
#     data[f'{prefix}_max_corr'] = data.groupby('session')[f'{prefix}_max_corr'].transform(lambda x: (x - x.mean()) / x.std())



data['story'] = data['key'].str.extract('Reading_(\d+).png')[0].astype(int)
# data['story'] = data['key'].str.extract('Listening_(\d+).ogg')[0].astype(int)
data['ListenerChoice'] = data.apply(lambda row: row['isChoiceB'] if row['whoseReading'] == 1 else row['isChoiceA'], axis=1)
data['ReaderChoice'] = data.apply(lambda row: row['isChoiceB'] if row['whoseReading'] == 2 else row['isChoiceA'], axis=1)

data['story_prob'] = (data.groupby('story')['isChoiceA'].transform('mean') + data.groupby('story')['isChoiceB'].transform('mean'))/2
data['sum_choice'] = data.apply(lambda row: row['isChoiceA'] + row['isChoiceB'], axis=1)

data.to_csv(add_before_extention(file_path, "_readnig_for_R"), index = False)

#keep in data only rows where story_prob == 0.5
#data = data[data['story_prob'] == 0.5]

data_odd = data.iloc[::2, :].reset_index(drop=True)
data_even = data.iloc[1::2, :].reset_index(drop=True)

# Initialize an empty DataFrame to hold the new structure
new_df = pd.DataFrame()

# Iterate over the rows of data_odd and data_even simultaneously using zip
for index, (odd_row, even_row) in enumerate(zip(data_odd.itertuples(), data_even.itertuples())):
    # Extract and store values for each of the specified columns, appending "_1" or "_2" as appropriate

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


    colls = [f"{prefix}{ending}" for prefix in ['chosen_comps', 'all_max_comp', 'chosen_max_comp'] for ending in ['_mean_corr', '_mean_lag', '_max_corr']]
    colls += [f"mean_{muscle}_{p}" for muscle in ["Nose", "Eye", "Zygomaticus", "Corrugator"] for p in ["A", "B"]]

    for col in colls:
        new_df.at[index, f'{col}_1'] = getattr(odd_row, col)
        new_df.at[index, f'{col}_2'] = getattr(even_row, col)
        
        if ( new_df.at[index, "ListenerChoice"] == 0):
            new_df.at[index, f'selected_{col}'] = getattr(odd_row, col)
            new_df.at[index, f'not_selected_{col}'] = getattr(even_row, col)
        else:
            new_df.at[index, f'selected_{col}'] = getattr(even_row, col)
            new_df.at[index, f'not_selected_{col}'] = getattr(odd_row, col)
        
            # if ( new_df.at[index, f'{prefix}{ending}_2'] != 0):
            #     new_df.at[index, f'{prefix}{ending}_div'] = new_df.at[index, f'{prefix}{ending}_1'] / new_df.at[index, f'{prefix}{ending}_2']


# new_df = new_df[new_df['chose_same'] == 1]

new_df.to_csv(add_before_extention(file_path, '_reading_for_R_2S1R'), index=False)

#wilcoxon test:
from scipy.stats import wilcoxon, PermutationMethod
import numpy as np

for prefix in ['chosen_comps', 'chosen_max_comp']:
    for ending in ['_mean_corr', '_max_corr']:
        print(f'{prefix}{ending}')
        # a = np.arctan(new_df[f'selected_{prefix}{ending}'])
        # b = np.arctan(new_df[f'not_selected_{prefix}{ending}'])
        # print(wilcoxon(a, b, alternative='greater'))
        print(wilcoxon(new_df[f'selected_{prefix}{ending}'], 
                         new_df[f'not_selected_{prefix}{ending}'], alternative='greater'))

#new_df.to_csv(add_before_extention(file_path, '_check_wilcox_2S1R'), index=False)


exit()

import statsmodels.api as sm

prefix = 'chosen_max_comp'
#prefix = 'all_max_comp'
# Define the dependent and independent variables
# X = new_df[[f'{prefix}_mean_corr_div', f'{prefix}_mean_lag_div', f'{prefix}_max_corr_div', 'story_prob_div']]  # Independent variables
X = new_df[[f"selected_{prefix}_mean_corr", f"not_selected_{prefix}_mean_corr"]] # Independent variables
X = sm.add_constant(X)  # Adds a constant term to the predictor
Y = new_df['ListenerChoice']  # Dependent variable

model = sm.GLM(Y, X, family=sm.families.Binomial())
result = model.fit()

# Display the results
print(result.summary())

exit()









# Sample data: Replace this with your own dataset
# Assuming 'Y' is the binary dependent variable and 'X1', 'X2' are independent variables

prefix = 'chosen_comps'
#prefix = 'all_max_comp'
# Define the dependent and independent variables
# X = new_df[[f'{prefix}_mean_corr_div', f'{prefix}_mean_lag_div', f'{prefix}_max_corr_div', 'story_prob_div']]  # Independent variables
X = new_df[[f'{prefix}_mean_corr_1', f'{prefix}_mean_corr_2',
            f'{prefix}_mean_lag_1', f'{prefix}_mean_lag_2',
            'story_prob_1', 'story_prob_2']]  # Independent variables
X = sm.add_constant(X)  # Adds a constant term to the predictor
Y = new_df['ListenerChoice']  # Dependent variable

# X = data[['chosen_comps_mean_corr', 'chosen_comps_mean_lag' , 'isOther']]  # Independent variables
# X = sm.add_constant(X)  # Adds a constant term to the predictor
# Y = data['sum_choice']  # Dependent variable


# Fit the GLM model
#model = sm.GLM(Y, X, family=sm.families.Binomial(link=sm.families.links.logit()))
new_df['session'] = new_df['session'].astype('category')
model = sm.GLM(Y, X, family=sm.families.Binomial(), groups=new_df['session'])
#model = sm.MixedLM(Y, X, groups=new_df['session'])
result = model.fit()

# Display the results
print(result.summary())

#save the df for R:
# new_df.to_csv(add_before_extention(file_path, '_for_R_listening'), index=False)
# data.to_csv(add_before_extention(file_path, '_normalized'), index=False)
