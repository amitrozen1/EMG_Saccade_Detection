import pandas as pd
from revision_analysis.compute_cross_corr import get_headers_reading, get_headers_listening, get_headers_both_listeners, get_headers_reading_combinations, get_headers_reading_GC
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.decomposition import PCA, NMF
import numpy as np

feature_names = [f"cluster_{i}_corr" for i in range(16)]

def create_minus_coll(row, header):
    #if either is nan return nan
    if pd.isna(row[header + "_S1"]) or pd.isna(row[header + "_S2"]):
        return float('nan')
    return (row[header + "_S2"] - row[header + "_S1"])

def get_listener_choice(second_row):
    if getattr(second_row, "whoseReading") == 1:
        return getattr(second_row, "isChoiceB")
    return getattr(second_row, "isChoiceA")

def get_reader_choice(second_row):
    if getattr(second_row, "whoseReading") == 1:
        return getattr(second_row, "isChoiceA")
    return getattr(second_row, "isChoiceB")


saved_H = None
reading_combined = None

def add_reader_listener_mean(row, header):
    whose_reading = row["whoseReading"]
    if whose_reading == 1:
        return row

def export_reading_for_combinations_to_r(csv_path):
    data = pd.read_csv(csv_path)
    data_odd = data.iloc[::2, :].reset_index(drop=True)
    data_even = data.iloc[1::2, :].reset_index(drop=True)

    new_df = pd.DataFrame()

    combinations = ["interest", "happiness", "disgust", "sadness"]

    for index, (odd_row, even_row) in enumerate(zip(data_odd.itertuples(), data_even.itertuples())):
        #story 1 in odd_row
        #story 2 in even_row

        new_row = {}
        new_row["session"] = getattr(odd_row, "session")
        new_row["story_1"] = getattr(odd_row, "key").split("_")[1].split(".")[0]
        new_row["story_2"] = getattr(even_row, "key").split("_")[1].split(".")[0]
        new_row["listener_choice"] = get_listener_choice(even_row)
        new_row["reader_choice"] = get_reader_choice(even_row)

        for combi in combinations:
            new_row[f"{combi}_corr_S1"] = getattr(odd_row, f"{combi}_corr")
            new_row[f"{combi}_corr_S2"] = getattr(even_row, f"{combi}_corr")
            if str(getattr(odd_row, "whoseReading")) == "1": #A is reading
                new_row[f"{combi}_mean_listener_S2"] = getattr(even_row, f"{combi}_mean_B")
                new_row[f"{combi}_mean_reader_S2"] = getattr(even_row, f"{combi}_mean_A")
                new_row[f"{combi}_mean_listener_S1"] = getattr(odd_row, f"{combi}_mean_A")
                new_row[f"{combi}_mean_reader_S1"] = getattr(odd_row, f"{combi}_mean_B")
            else:
                new_row[f"{combi}_mean_listener_S2"] = getattr(even_row, f"{combi}_mean_A")
                new_row[f"{combi}_mean_reader_S2"] = getattr(even_row, f"{combi}_mean_B")
                new_row[f"{combi}_mean_listener_S1"] = getattr(odd_row, f"{combi}_mean_B")
                new_row[f"{combi}_mean_reader_S1"] = getattr(odd_row, f"{combi}_mean_A")
        
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
    
    new_df.to_csv(csv_path.replace(".csv", "_for_R.csv"), index = False)

def export_reading_to_r(csv_path):
    data = pd.read_csv(csv_path)
    data_odd = data.iloc[::2, :].reset_index(drop=True)
    data_even = data.iloc[1::2, :].reset_index(drop=True)

    new_df = pd.DataFrame()

    for index, (odd_row, even_row) in enumerate(zip(data_odd.itertuples(), data_even.itertuples())):
        #story 1 in odd_row
        #story 2 in even_row

        new_row = {}
        new_row["session"] = getattr(odd_row, "session")
        new_row["story_1"] = getattr(odd_row, "key").split("_")[1].split(".")[0]
        new_row["story_2"] = getattr(even_row, "key").split("_")[1].split(".")[0]
        new_row["listener_choice"] = get_listener_choice(even_row)
        new_row["reader_choice"] = get_reader_choice(even_row)


        for header in [x for x in get_headers_reading() if "optimal" not in x]:
            new_row[header + "_S1"] = getattr(odd_row, header)
            new_row[header + "_S2"] = getattr(even_row, header)
            new_row["minus_" + header] = create_minus_coll(new_row, header)
        
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

    #arctan corr:
    # for i in range(16):
    # #     new_df[f"cluster_{i}_corr_S1"] = np.arctan(new_df[f"cluster_{i}_corr_S1"])
    # #     new_df[f"cluster_{i}_corr_S2"] = np.arctan(new_df[f"cluster_{i}_corr_S2"])

    # combined_data = np.vstack([new_df[[f"cluster_{i}_corr_S1" for i in range(16)]].values, 
    #                         new_df[[f"cluster_{i}_corr_S2" for i in range(16)]].values])
    # # combined_data = new_df[[f"cluster_{i}_corr_S1" for i in range(16)]].values

    # combined_mean_data = np.vstack([new_df[[f"cluster_{i}_mean_comp_listener_S1" for i in range(16)]].values,
    #                                 new_df[[f"cluster_{i}_mean_comp_listener_S2" for i in range(16)]].values])

    # imputer = SimpleImputer(strategy='median')
    # imputer = SimpleImputer(strategy='mean')
    # imputer = SimpleImputer(strategy='constant', fill_value=0)
    

    # combined_data = imputer.fit_transform(combined_data.T).T
    # combined_mean_data = imputer.fit_transform(combined_mean_data.T).T
    
    # num_comps = 2


    # from sklearn.feature_selection import VarianceThreshold

    # selector = VarianceThreshold(threshold=0.001)
    # X_reduced = selector.fit_transform(combined_data)

    # #what was removed:
    # global feature_names
    # removed_features = [feature_names[i] for i in range(16) if i not in selector.get_support(indices=True)]
    # print("Removed features: ", removed_features)

    # feature_names = [feature_names[i] for i in range(16) if i in selector.get_support(indices=True)]

    
    # #seabron pairplot:
    # import seaborn as sns
    # sns.pairplot(pd.DataFrame(combined_data, columns=feature_names), plot_kws=dict(marker="+", linewidth=1), diag_kws=dict(fill=False))
    # plt.show()
    # import matplotlib.pyplot as plt

    # plt.matshow(pd.DataFrame(X_reduced, columns=feature_names).corr())
    # plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90)
    # plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
    # plt.colorbar()
    # plt.show()

    # exit()

    # #pca:
    # from sklearn.decomposition import PCA
    # pca = PCA(n_components=num_comps)
    # pca.fit(X_reduced)
    # loadings = pca.components_

    # print("actual PCA")
    # loadings_df = pd.DataFrame(loadings.T, index=feature_names, columns=[f"PC{i+1}" for i in range(num_comps)])
    # print("PCA Loadings (Coefficients):\n", loadings_df)

    #factor analyssi
    # from sklearn.decomposition import FactorAnalysis
    # fa = FactorAnalysis(n_components=num_comps, svd_method='lapack', max_iter=10000)
    # fa.fit(combined_data)
    # print(fa.components_)

    # perplexities = []
    # for num_comps in range(1, 16):
    #     nmf = NMF(n_components=num_comps, init='random', random_state=42)
    #     W = nmf.fit_transform(combined_data)
    #     H = nmf.components_
    #     # Assuming you have some validation set to compute perplexity
    #     # Or use the NMF model's built-in ability to estimate the perplexity
    #     perplexity = nmf.reconstruction_err_  # You can use this as a proxy
    #     perplexities.append(perplexity)

    # plt.plot(range(1, 16), perplexities)
    # plt.xlabel("Number of components")
    # plt.ylabel("Perplexity")
    # plt.title("Perplexity vs Number of Components")
    # plt.show()

 
    
    # best_reconstruction_err = np.inf
    # best_model = None

    # for random_seed in range(50):
    #     nmf = NMF(
    #         n_components=2,
    #         init='random',
    #         random_state=random_seed)
    #     W = nmf.fit_transform(combined_data)
    #     #reconstruction_err = nmf.reconstruction_err_
        
    #     H = nmf.components_

    #     #features by importance:
    #     importance = np.array([x[::-1] for x in np.argsort(H, axis=1)])
    #     for i in range(num_comps):
    #         H[i][importance[i][7:]] = 0
    #     nmf.components_ = H

    #     #computing reconstruction error:
    #     reconstructed_data = np.dot(W, H)
    #     reconstruction_err = np.sum((combined_data - reconstructed_data) ** 2)

    #     if reconstruction_err < best_reconstruction_err:
    #         best_reconstruction_err = reconstruction_err
    #         best_model = nmf
    #         best_W = W
    #         best_H = nmf.components_

    # print(f"Best random seed: {best_model.random_state}")
    # print(f"Best reconstruction error: {best_reconstruction_err}")

    #exit()
    
    # nmf = NMF(n_components=num_comps, init='random', random_state=42, max_iter=200)
    # nmf = NMF(n_components=num_comps, init='nndsvda', beta_loss='kullback-leibler', solver='mu')

    
    # nmf = NMF(
    # n_components=num_comps, # e.g., 2-4 for interpretability
    # init='nndsvda',                       # Recommended initialization for non-negative data
    # solver='cd',                          # Coordinate descent is reliable and stable    
    # max_iter=1000,                         # Higher iterations ensure better convergence
    # )

    # W = nmf.fit_transform(combined_data)
    # H = nmf.components_
    # loadings = H

    # #print H
    # print("before H")
    # loadings_df = pd.DataFrame(H.T, index=feature_names, columns=[f"PC{i+1}" for i in range(len(loadings))])
    # print("PCA Loadings (Coefficients):\n", loadings_df)

    # #inverse H
    # #try_inv = np.sum((W - np.matmul(combined_data, np.linalg.pinv(H)))**2)

    # H_inv = np.linalg.pinv(H).T

    # #before H_inv:
    # print("before H_inv")
    # loadings_df = pd.DataFrame(H_inv.T, index=feature_names, columns=[f"PC{i+1}" for i in range(len(loadings))])
    # print("PCA Loadings (Coefficients):\n", loadings_df)

    # #features by importance:
    # importance = np.array([x[::-1] for x in np.argsort(H, axis=1)])
    # for i in range(num_comps):
    #     H[i][importance[i][5:]] = 0
    
    # nmf.components_ = H
    # #H_inv = H_inv.T
    # H_inv = np.linalg.pinv(H)


    # orig_h = H.copy()
    # explained_variances = []

    # for comps_num in range(1, 16):
    #     H = orig_h.copy()
    #     #features by importance:
    #     importance = np.array([x[::-1] for x in np.argsort(H, axis=1)])
    #     for i in range(num_comps):
    #         H[i][importance[i][comps_num:]] = 0
    #     #nmf.components_ = H

    #     reconstructed_data = np.dot(W, H)

    #     # Compute explained variance (pseudo R²)
    #     total_variance = np.sum(combined_data ** 2)
    #     reconstruction_error = np.sum((combined_data - reconstructed_data) ** 2)
    #     explained_variance_ratio = 1 - (reconstruction_error / total_variance)
    #     explained_variances.append(explained_variance_ratio)
    #     print(f"Explained Variance (R²) with {comps_num} components: {explained_variance_ratio}")

    # plt.plot(range(1, 16), explained_variances)
    # plt.xlabel("Number of components")
    # plt.ylabel("Explained Variance (R²) ratio")
    # plt.title("Explained Variance (R²) ratio vs Number of Components")
    # plt.show()


    # print("after H")
    # #  Create a DataFrame for readability
    # loadings_df = pd.DataFrame(H.T, index=feature_names, columns=[f"PC{i+1}" for i in range(len(loadings))])
    # print("PCA Loadings (Coefficients):\n", loadings_df)


    # print("after H_inv")
    # #  Create a DataFrame for readability
    # loadings_df = pd.DataFrame(H_inv, index=feature_names, columns=[f"PC{i+1}" for i in range(len(loadings))])
    # print("PCA Loadings (Coefficients):\n", loadings_df)


    # Apply the same PCA transformation to S1 and S2
    # S1_data = new_df[[f"cluster_{i}_corr_S1" for i in range(16)]].fillna(0).values
    #S1_data = new_df[[f"cluster_{i}_mean_comp_listener_S1" for i in range(16)]].fillna(0).values
    # S1_data = new_df[[f"cluster_{i}_corr_S1" for i in range(16)]].values
    # S1_data = imputer.fit_transform(S1_data.T).T
    # #remove col 14:
    # S1_data = np.delete(S1_data, 14, axis=1)
    # S1_transformed = pca.transform(S1_data)
    #S1_transformed = nmf.transform(S1_data)
    #S1_transformed = np.dot(S1_data, H_inv)
    #S1_transformed = fa.transform(S1_data)
    # S2_data = new_df[[f"cluster_{i}_corr_S2" for i in range(16)]].fillna(0).values
    #S2_data = new_df[[f"cluster_{i}_mean_comp_listener_S2" for i in range(16)]].fillna(0).values
    # S2_data = new_df[[f"cluster_{i}_corr_S2" for i in range(16)]].values
    # S2_data = np.delete(S2_data, 14, axis=1)
    # S2_data = imputer.fit_transform(S2_data.T).T
    # S2_transformed = pca.transform(S2_data)
    #S2_transformed = nmf.transform(S2_data)
    #S2_transformed = np.dot(S2_data, H_inv)
    #S2_transformed = fa.transform(S2_data)

    # for i in range(num_comps):
    #     new_df[f"PCA_{i+1}_corr_S1"] = S1_transformed[:, i]
    #     new_df[f"PCA_{i+1}_corr_S2"] = S2_transformed[:, i]

    #     # Create minus columns:
    #     new_df[f"minus_PCA_{i+1}_corr"] = (new_df[f"PCA_{i+1}_corr_S2"] - new_df[f"PCA_{i+1}_corr_S1"])
    new_df.to_csv(csv_path.replace(".csv", "_for_R.csv"), index = False)


def export_listening_to_r(csv_path):
    data = pd.read_csv(csv_path)
    new_df = pd.DataFrame()

    #data = data[data["isOther"] == 0]
    data = data.reset_index(drop=True)

    for p in ["A","B"]:
        df_p = data[data["participant"] == p]
        df_p = df_p.reset_index(drop=True)

        data_odd = df_p.iloc[::2, :].reset_index(drop=True)
        data_even = df_p.iloc[1::2, :].reset_index(drop=True)
        for index, (odd_row, even_row) in enumerate(zip(data_odd.itertuples(), data_even.itertuples())):
        #story 1 in odd_row
        #story 2 in even_row

            new_row = {}
            new_row["session"] = getattr(odd_row, "session")
            new_row["story_1"] = getattr(odd_row, "key").split("_")[1].split(".")[0]
            new_row["story_2"] = getattr(even_row, "key").split("_")[1].split(".")[0]
            new_row["listener_choice"] = getattr(even_row, "choice") #get_listener_choice(even_row)
            new_row["is_other"] = getattr(even_row, "isOther")
            new_row["p"] = getattr(even_row, "participant")

            for header in get_headers_listening():
                new_row[header + "_S1"] = getattr(odd_row, header)
                new_row[header + "_S2"] = getattr(even_row, header)
                new_row["minus_" + header] = create_minus_coll(new_row, header)
            
            new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

    index_range = range(16)
    #remove 13 and 14
    index_range = [i for i in index_range if i not in [12, 13]]

    combined_data = np.vstack([new_df[[f"cluster_{i}_corr_S1" for i in index_range]].values, 
                            new_df[[f"cluster_{i}_corr_S2" for i in index_range]].values])
    
    num_comps = 2
    #nmf = NMF(n_components=num_comps, init='random', random_state=42)
    nmf = NMF(
        n_components=num_comps, # e.g., 2-4 for interpretability
        init='nndsvda',                       # Recommended initialization for non-negative data
        solver='cd',                          # Coordinate descent is reliable and stable
        beta_loss = 'frobenius',        
        max_iter=1000,                         # Higher iterations ensure better convergence
        )
    #impute using mean within subject clusters:
    imputer = SimpleImputer(strategy='mean')
    combined_data = imputer.fit_transform(combined_data.T).T
    
    
    W = nmf.fit_transform(combined_data)
    H = nmf.components_
    loadings = H #pca.components_

    # Create a DataFrame for readability
    feature_names = [f"cluster_{i}_corr" for i in index_range]  # Adjust if needed
    loadings_df = pd.DataFrame(loadings.T, index=feature_names, columns=[f"PC{i+1}" for i in range(len(loadings))])
    print("PCA Loadings (Coefficients):\n", loadings_df)


    #features by importance:
    importance = np.array([x[::-1] for x in np.argsort(H, axis=1)])
    for i in range(num_comps):
        H[i][importance[i][6:]] = 0
    nmf.components_ = H

    # Create a DataFrame for readability
    print("after")
    loadings_df = pd.DataFrame(H.T, index=feature_names, columns=[f"PC{i+1}" for i in range(len(loadings))])
    print("PCA Loadings (Coefficients):\n", loadings_df)

    S1_data = new_df[[f"cluster_{i}_corr_S1" for i in index_range]].fillna(0).values
    S1_transformed = nmf.transform(S1_data)
    S2_data = new_df[[f"cluster_{i}_corr_S2" for i in index_range]].fillna(0).values
    S2_transformed = nmf.transform(S2_data)

    for i in range(num_comps):
        new_df[f"PCA_{i+1}_corr_S1"] = S1_transformed[:, i]
        new_df[f"PCA_{i+1}_corr_S2"] = S2_transformed[:, i]
        # Create minus columns:
        new_df[f"minus_PCA_{i+1}_corr"] = (new_df[f"PCA_{i+1}_corr_S2"] - new_df[f"PCA_{i+1}_corr_S1"]) / (new_df[f"PCA_{i+1}_corr_S2"] + new_df[f"PCA_{i+1}_corr_S1"])

    new_df.to_csv(csv_path.replace(".csv", "_remove_12_13_for_R.csv"), index = False)


def export_both_listeners_to_r(csv_path):
    data = pd.read_csv(csv_path)
    data_odd = data.iloc[::2, :].reset_index(drop=True)
    data_even = data.iloc[1::2, :].reset_index(drop=True)

    new_df = pd.DataFrame()

    for index, (odd_row, even_row) in enumerate(zip(data_odd.itertuples(), data_even.itertuples())):
        #story 1 in odd_row
        #story 2 in even_row

        new_row = {}
        new_row["session"] = getattr(odd_row, "session")
        new_row["story_1"] = getattr(odd_row, "key").split("_")[1].split(".")[0]
        new_row["story_2"] = getattr(even_row, "key").split("_")[1].split(".")[0]
        new_row["shared_choice"] = int(getattr(even_row, "isChoiceA")) + int(getattr(even_row, "isChoiceB"))
        new_row["is_other"] = getattr(even_row, "isOther")

        for header in get_headers_both_listeners():
            new_row[header + "_S1"] = getattr(odd_row, header)
            new_row[header + "_S2"] = getattr(even_row, header)
            new_row["minus_" + header] = create_minus_coll(new_row, header)
        
        new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)

    # Replace all nans with 0
    # Perform PCA on S1
    num_comps = 2
    pca = PCA(n_components=4)
    nmf = NMF(n_components=num_comps, init='random', random_state=42)

    combined_data = np.vstack([new_df[[f"cluster_{i}_corr_S1" for i in range(16)]].values, 
                            new_df[[f"cluster_{i}_corr_S2" for i in range(16)]].values])

    #impute using mean within subject clusters:
    imputer = SimpleImputer(strategy='mean')
    combined_data = imputer.fit_transform(combined_data.T).T

    W = nmf.fit_transform(combined_data)
    H = nmf.components_
    loadings = H #pca.components_

    # Create a DataFrame for readability
    feature_names = [f"cluster_{i}_corr" for i in range(16)]  # Adjust if needed
    loadings_df = pd.DataFrame(loadings.T, index=feature_names, columns=[f"PC{i+1}" for i in range(len(loadings))])
    print("PCA Loadings (Coefficients):\n", loadings_df)


    #features by importance:
    importance = np.array([x[::-1] for x in np.argsort(H, axis=1)])
    for i in range(num_comps):
        H[i][importance[i][7:]] = 0
    nmf.components_ = H

    print("after")
     # Create a DataFrame for readability
    loadings_df = pd.DataFrame(H.T, index=feature_names, columns=[f"PC{i+1}" for i in range(len(loadings))])
    print("PCA Loadings (Coefficients):\n", loadings_df)


    # Apply the same PCA transformation to S1 and S2
    S1_data = new_df[[f"cluster_{i}_corr_S1" for i in range(16)]].fillna(0).values
    S1_transformed = nmf.transform(S1_data)
    S2_data = new_df[[f"cluster_{i}_corr_S2" for i in range(16)]].fillna(0).values
    S2_transformed = nmf.transform(S2_data)

    for i in range(num_comps):
        new_df[f"PCA_{i+1}_corr_S1"] = S1_transformed[:, i]
        new_df[f"PCA_{i+1}_corr_S2"] = S2_transformed[:, i]
        # Create minus columns:
        new_df[f"minus_PCA_{i+1}_corr"] = (new_df[f"PCA_{i+1}_corr_S2"] - new_df[f"PCA_{i+1}_corr_S1"]) / (new_df[f"PCA_{i+1}_corr_S2"] + new_df[f"PCA_{i+1}_corr_S1"])
    new_df.to_csv(csv_path.replace(".csv", "withNMF2_top7features_MeanAcrossClustersImpute_for_R.csv"), index = False)
