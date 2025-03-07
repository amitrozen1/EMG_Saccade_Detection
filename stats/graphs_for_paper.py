import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import numpy as np

path_1 = r"C://Liron//DataEmg//Try//results_2024//RMS_50_downsample_5_window_3_maxlag_3_after_pre_reg_dwt_db15_corrEachComp_scipy_resample_noWeightedCorr_1direct_allow_05_delay_noz_noartrem_with_var_neg2Pos5_with4_readnig_for_R.csv"
path_2 = r"C://Liron//DataEmg//new_batch//results_2024//RMS_50_downsample_5_window_3_maxlag_3_after_pre_reg_dwt_db15_corrEachComp_scipy_resample_noWeightedCorr_1direct_allow_05_delay_noz_noartrem_with_var_Neg2Pos5Puller4_batch2_new_list_readnig_for_R.csv"

df1 = pd.read_csv(path_1)
df2 = pd.read_csv(path_2)

#bind:
df = pd.concat([df1, df2])

#create a new column of chosen story value vs unchosen value:
negative_comps = ["NegComp", "LipCornerDownPuller"]
positive_comps = ["Corrugator", "Eye"]

# #synch:
# for comp in negative_comps + positive_comps:
#     df[f'chosen_{comp}_sync'] = df.apply(lambda row: row[f'{comp}_mean_corr_{int(row["ListenerChoice"]+1)}'], axis=1)
#     df[f'unchosen_{comp}_sync'] = df.apply(lambda row: row[f'{comp}_mean_corr_{int(2-row["ListenerChoice"])}'], axis=1)


# #mean negative sync values:
# df['chosen_negative_sync'] = df.apply(lambda row: np.mean([row[f'chosen_{comp}_sync'] for comp in negative_comps]), axis=1)
# df['unchosen_negative_sync'] = df.apply(lambda row: np.mean([row[f'unchosen_{comp}_sync'] for comp in negative_comps]), axis=1)


# #mean positive sync values:
# df['chosen_positive_sync'] = df.apply(lambda row: np.mean([row[f'chosen_{comp}_sync'] for comp in positive_comps]), axis=1)
# df['unchosen_positive_sync'] = df.apply(lambda row: np.mean([row[f'unchosen_{comp}_sync'] for comp in positive_comps]), axis=1)

# print(np.mean(df['chosen_negative_sync'] - df['unchosen_negative_sync']))

# #positive graph. y-axis is chosen value, x-axis is unchosen value:
# import matplotlib.pyplot as plt

# plt.scatter(df['unchosen_positive_sync'], df['chosen_positive_sync'])
# plt.xlabel('unchosen positive sync')
# plt.ylabel('chosen positive sync')
# plt.title('positive sync values')
# #draw y=x
# # % over y=x:
# percentage = np.mean(df['chosen_positive_sync'] > df['unchosen_positive_sync'])
# plt.legend([f'% over y=x: {percentage}'])
# plt.plot([0, 1], [0, 1], color='red')

# #compute median distance from y=x:
# median_distance = np.median(np.abs(df['chosen_positive_sync'] - df['unchosen_positive_sync']))
# q3 = np.percentile(np.abs(df['chosen_positive_sync'] - df['unchosen_positive_sync']), 75)
# #mark in yellow that are above median distance:
# plt.scatter(df['unchosen_positive_sync'][np.abs(df['chosen_positive_sync'] - df['unchosen_positive_sync']) > q3], df['chosen_positive_sync'][np.abs(df['chosen_positive_sync'] - df['unchosen_positive_sync']) > q3], color='yellow')


# plt.show()
# plt.cla()

# #negative graph. y-axis is chosen value, x-axis is unchosen value:

# plt.scatter(df['unchosen_negative_sync'], df['chosen_negative_sync'])
# plt.xlabel('unchosen negative sync')
# plt.ylabel('chosen negative sync')
# plt.title('negative sync values')
# #draw y=x
# plt.plot([0, 1], [0, 1], color='red')

# # % over y=x:
# percentage = np.mean(df['chosen_negative_sync'] > df['unchosen_negative_sync'])
# plt.legend([f'% over y=x: {percentage}'])

# median_distance = np.median(np.abs(df['chosen_negative_sync'] - df['unchosen_negative_sync']))
# #3/4 quartile distance:
# q3 = np.percentile(np.abs(df['chosen_negative_sync'] - df['unchosen_negative_sync']), 75)

# #mark in yellow that are above median distance:
# plt.scatter(df['unchosen_negative_sync'][np.abs(df['chosen_negative_sync'] - df['unchosen_negative_sync']) > q3], df['chosen_negative_sync'][np.abs(df['chosen_negative_sync'] - df['unchosen_negative_sync']) > q3], color='yellow')

# plt.show()


#mean negative sync values:
for i in [1,2]:
    df[f'positive_sync_{i}'] = df.apply(lambda row: np.mean([row[f'{comp}_mean_corr_{i}'] for comp in positive_comps]), axis=1)
    df[f'negative_sync_{i}'] = df.apply(lambda row: np.mean([row[f'{comp}_mean_corr_{i}'] for comp in negative_comps]), axis=1)


import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Define color mapping for ListenerChoice
colors = {0: 'blue', 1: 'orange'}

# Scatter plot with custom colors
sc = plt.scatter(df['positive_sync_1'], df['positive_sync_2'], c=df['ListenerChoice'].map(colors))

# Add labels and title
plt.xlabel('positive sync 1')
plt.ylabel('positive sync 2')
plt.title('positive sync values')

# Create custom legend
blue_patch = mpatches.Patch(color='blue', label='chose story 1')
orange_patch = mpatches.Patch(color='orange', label='chose story 2')
plt.legend(handles=[blue_patch, orange_patch], loc='upper left')

# Add line for comparison
#plt.plot([0.25, 0.5], [0.25, 0.5], color='red')

#fit a line:
X = df['positive_sync_1'].values.reshape(-1, 1)
y = df['positive_sync_2'].values
reg = LinearRegression().fit(X, y)
plt.plot(X, reg.predict(X), color='red')


# Display the plot
plt.show()
plt.cla()

#plot negative:

# Scatter plot with custom colors

sc = plt.scatter(df['negative_sync_1'], df['negative_sync_2'], c=df['ListenerChoice'].map(colors))

# Add labels and title
plt.xlabel('negative sync 1')
plt.ylabel('negative sync 2')

plt.title('negative sync values')

# Create custom legend
blue_patch = mpatches.Patch(color='blue', label='chose story 1')
orange_patch = mpatches.Patch(color='orange', label='chose story 2')
plt.legend(handles=[blue_patch, orange_patch], loc='upper left')

#plt.plot([0.25, 0.5], [0.25, 0.5], color='red')
#fit a line:


X = df['negative_sync_1'].values.reshape(-1, 1)
y = df['negative_sync_2'].values
reg = LinearRegression().fit(X, y)
plt.plot(X, reg.predict(X), color='red')

# Display the plot
plt.show()
plt.cla()


#plot x as positive1 and y as negative2:
sc = plt.scatter(df['positive_sync_1'], df['negative_sync_1'], c=df['ListenerChoice'].map(colors))

# Add labels and title
plt.xlabel('positive sync 1')
plt.ylabel('negative sync 1')

# Create custom legend
blue_patch = mpatches.Patch(color='blue', label='chose story 1')
orange_patch = mpatches.Patch(color='orange', label='chose story 2')

plt.legend(handles=[blue_patch, orange_patch], loc='upper left')

#plt.plot([0.25, 0.45], [0.45, 0.25], color='red')
#fit a line:
from sklearn.linear_model import LinearRegression
import numpy as np

X = df['positive_sync_1'].values.reshape(-1, 1)
y = df['negative_sync_1'].values
reg = LinearRegression().fit(X, y)
plt.plot(X, reg.predict(X), color='red')

# Display the plot
plt.show()
plt.cla()


#plot x as negative1 and y as positive2:
sc = plt.scatter(df['positive_sync_2'], df['negative_sync_2'], c=df['ListenerChoice'].map(colors))

# Add labels and title
plt.xlabel('positive sync 2')
plt.ylabel('negative sync 2')

# Create custom legend
blue_patch = mpatches.Patch(color='blue', label='chose story 1')
orange_patch = mpatches.Patch(color='orange', label='chose story 2')
plt.legend(handles=[blue_patch, orange_patch], loc='upper left')

#plt.plot([0.25, 0.45], [0.45, 0.25], color='red')

#fit a line:
X = df['positive_sync_2'].values.reshape(-1, 1)
y = df['negative_sync_2'].values

reg = LinearRegression().fit(X, y)
plt.plot(X, reg.predict(X), color='red')



# Display the plot
plt.show()
plt.cla()