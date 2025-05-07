import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, ttest_1samp


# Define the directory and filename pattern components
directory = r"C:\Liron\DataEmg\revision_analysis"
windows = [1000, 2000, 3000, 4000, 5000]
lags = [1000, 2000, 3000, 4000, 5000]
file_suffixes = ["_tInc5_combined_for_R.csv", "_combined_for_R.csv"]

var_type = "corr" #corr

# Function to calculate mean(cluster_i_corr_S2 - cluster_i_corr_S1) for i in 0 to 15
def process_file(filepath):
    try:
        df = pd.read_csv(filepath)
        diffs = []
        for i in range(16):
            col_s1 = f'cluster_{i}_{var_type}_S1'
            col_s2 = f'cluster_{i}_{var_type}_S2'
            if col_s1 in df.columns and col_s2 in df.columns:
                diff = df[col_s1] - df[col_s2] / (df[col_s1] + df[col_s2])
                diffs.append(diff.mean(skipna=True))
        return np.nanmean(diffs) if diffs else np.nan
    except Exception as e:
        return np.nan

def process_file_t(filepath):
    try:
        df = pd.read_csv(filepath)
        t_values = []
        for i in range(16):
            col_s1 = f'cluster_{i}_{var_type}_S1'
            col_s2 = f'cluster_{i}_{var_type}_S2'

            diffs = []
            if col_s1 in df.columns and col_s2 in df.columns:
                # Drop NaN values for paired t-test
                valid_data = df[[col_s1, col_s2]].dropna()
                if not valid_data.empty:
                    # t_stat, _ = ttest_rel(valid_data[col_s1], valid_data[col_s2], alternative='two-sided')
                    # t_values.append(np.abs(t_stat))
                    diffs = np.abs((valid_data[col_s1] - valid_data[col_s2])/(valid_data[col_s1] + valid_data[col_s2]))
                    #check t value of diffs > 0:
                    
                    t_stat, p_value = ttest_1samp(diffs, 0, alternative='greater')
                    #t_values.extend(diffs.tolist())
                    t_values.append(t_stat)
        return np.nanmean(t_values) if t_values else np.nan
    except Exception as e:
        return np.nan
    
import pandas as pd
import numpy as np

def get_var_across_all(filepath, var_type='corr'):
    try:
        df = pd.read_csv(filepath)
        values = []

        for i in range(16):
            col_s1 = f'cluster_{i}_{var_type}_S1'
            col_s2 = f'cluster_{i}_{var_type}_S2'

            if col_s1 in df.columns:
                values.extend(df[col_s1].dropna().tolist())
            if col_s2 in df.columns:
                values.extend(df[col_s2].dropna().tolist())

        return np.nanvar(values) if values else np.nan
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return np.nan


def process_file_mean_abs_diff(filepath, var_type='corr'):
    try:
        df = pd.read_csv(filepath)
        mean_abs_diffs = []

        for index, row in df.iterrows():
            diffs = []
            for i in range(16):
                col_s1 = f'cluster_{i}_{var_type}_S1'
                col_s2 = f'cluster_{i}_{var_type}_S2'

                if pd.notna(row.get(col_s1)) and pd.notna(row.get(col_s2)):
                    diff = abs(row[col_s1] - row[col_s2])
                    diffs.append(diff)

            if diffs:
                mean_abs_diffs.append(np.mean(diffs))
            else:
                mean_abs_diffs.append(np.nan)

        t_stat, p_value = ttest_1samp(mean_abs_diffs, 0)
        return t_stat
    except Exception as e:
        print(f"Error processing file {filepath}: {e}")
        return []


# Initialize result matrix
result_matrix = np.full((len(lags), len(windows)), np.nan)

# Process all combinations of windows and lags
for i, lag in enumerate(lags):
    for j, window in enumerate(windows):
        found = False
        for suffix in file_suffixes:
            filename = f"reading_rms_50_ds_5_window_{window}_max_lag_{lag}_no_pre500{suffix}"
            filepath = os.path.join(directory, filename)
            if os.path.exists(filepath):
                result_matrix[i, j] = get_var_across_all(filepath)
                found = True
                break
        if not found:
            print (f"File not found: {filepath}")
            result_matrix[i, j] = np.nan  # Ensure explicitly marked as missing

# Create a heatmap
fig, ax = plt.subplots(figsize=(8, 6))
cax = ax.imshow(result_matrix, interpolation='nearest', cmap='viridis', aspect='auto')

# Add text annotations for non-NaN values
# Calculate the order of values, where the highest is 1
sorted_indices = np.argsort(-result_matrix, axis=None)  # Flatten and sort in descending order
order_matrix = np.empty_like(result_matrix, dtype=int)
order_matrix.flat[sorted_indices] = np.arange(1, len(sorted_indices) + 1)

# Add text annotations for the order
for i in range(len(lags)):
    for j in range(len(windows)):
        if not np.isnan(result_matrix[i, j]):
            ax.text(j, i, f'{order_matrix[i, j]}', ha='center', va='center', color='white')

# Add colorbar and labels
ax.set_xticks(np.arange(len(windows)))
ax.set_yticks(np.arange(len(lags)))
ax.set_xticklabels(windows)
ax.set_yticklabels(lags)
ax.set_xlabel("Window")
ax.set_ylabel("Lag")
plt.title("Mean(cluster_i_corr_S1 - cluster_i_corr_S2)")

# Add text for missing values
for i in range(len(lags)):
    for j in range(len(windows)):
        if np.isnan(result_matrix[i, j]):
            ax.text(j, i, 'NA', ha='center', va='center', color='white')

fig.colorbar(cax)
plt.tight_layout()
plt.show()
