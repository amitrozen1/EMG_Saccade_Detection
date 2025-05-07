from revision_analysis.data_loading import load_dyad_data
import numpy as np
from General.init import *
import matplotlib.pyplot as plt
from Analysis.analysis_runner import get_ok_sessions
from tqdm import tqdm
import os
import pickle
import pandas as pd
from General.CorrolationMethods import  get_max_xcorr_pearson_1d


def get_cor_file(session, p):
    p = "A" if p == A else "B"
    return os.path.join(data_path, session, f"{session}_{p}_comps_corr.npy")

def get_chunks_filenmae(session, p):
    p = "A" if p == A else "B"
    return os.path.join(data_path, session, f"{session}_{p}_chunks_rms_50_ds_5.pkl")

def load_chunks(session, p):
    with open(get_chunks_filenmae(session, p), 'rb') as f:
        return pickle.load(f)

def save_chunks(session, p, chunks):
    #pickle save
    with open(get_chunks_filenmae(session, p), 'wb') as f:
        pickle.dump(chunks, f)

def save_data():
    rms = 50
    ds = 5

    for session in tqdm(get_ok_sessions()):

        titles_A = list(range(16))
        titles_B = list(range(16))

        both_components, both_chunks = load_dyad_data(session, ds, rms)
        dataA = dict()
        dataB = dict()
        for chunks, comp, current_dict, title_list in zip(both_chunks, both_components, [dataA, dataB], [titles_A,titles_B]):
            for key in chunks:
                chunk = chunks[key]
                s = int(chunk.Start.Time)
                e = int(chunk.End.Time)
                chunk.Data = [np.nan for i in range(16)]
                for i in range(16):
                    if comp[i] is not np.nan:
                        chunk.Data[i] = comp[i][s:e]
        
        save_chunks(session, A, both_chunks[A])
        save_chunks(session, B, both_chunks[B])

def my_columnwise_zscore(data, length):
    for i in range(len(data)):
        if np.isnan(data[i]).any():
            #nan values in length arr:
            data[i] = [np.nan for i in range(length)]
    
    return (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)

def argmax_n(data, n):
    arr = []
    for i in range(16):
        if not np.isnan(data[i]).any():
            arr.append(np.mean(data[i]))
        else:
            arr.append(0)
    #nanargsort:
    return np.argsort(arr)[::-1][:n]





def all_pca():
    all_vectors = [[] for _ in range(16)]
    for session in tqdm(get_ok_sessions()):
        for p in [A, B]:
            chunks = load_chunks(session, p)
            for key in chunks:
                for i in range(16):
                    if not np.isnan(chunks[key].Data[i]).any():
                        all_vectors[i] += list(chunks[key].Data[i])
                    else:
                        all_vectors[i] += [np.nan] * (int(chunks[key].End.Time) - int(chunks[key].Start.Time))
    
    # Pad lists to the same length
    # max_length = max(len(v) for v in all_vectors)
    # all_vectors = [v + [np.nan] * (max_length - len(v)) for v in all_vectors]
    
    # Impute nan values
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(strategy='median')
    all_vectors = imputer.fit_transform(all_vectors.T).T

    #replace all values < 0 with 0:
    # all_vectors = np.where(all_vectors < 0, 0, all_vectors)
    
    #use factor analysis:
    from sklearn.decomposition import FactorAnalysis
    fa = FactorAnalysis(n_components=2)
    fa.fit(all_vectors.T)
    H = fa.components_

    #pca two components:
    # from sklearn.decomposition import PCA
    # from sklearn.decomposition import NMF
    # from sklearn.decomposition import FastICA
    # pca = PCA(n_components=3)
    # pca.fit(np.array(all_vectors).T)
    #print(pca.explained_variance_ratio_)
    #H = pca.mixing_
    # H = pca.components_
    featues = [f"cluster_{i}" for i in range(1,17)]

    try:
        loadings_df = pd.DataFrame(H, index=featues, columns=[f"PC{i+1}" for i in range(2)])
        print("PCA Loadings (Coefficients):\n", loadings_df)
    except:
        pass
    
    try:
        loadings_df = pd.DataFrame(H.T, index=featues, columns=[f"PC{i+1}" for i in range(2)])
        print("PCA Loadings (Coefficients):\n", loadings_df)
    except:
        pass


def localizer_running():
    var_counter = [0] * 16
    vectors_var = [0] * 16
    smile_data = [0] * 16
    angry_data = [0] * 16
    blink_data = [0] * 16
    for session in tqdm(get_ok_sessions()):
        for p in [A,B]:
            chunks = load_chunks(session, p)
            for key in chunks:
                for i in range(16):
                    if not np.isnan(chunks[key].Data[i]).any():
                        vectors_var[i] += (np.var(chunks[key].Data[i])/np.mean(chunks[key].Data[i]))
                        var_counter[i] += 1

                if ("smile" in key.lower()):
                    indexes = argmax_n(chunks[key].Data, 4)
                    for i in indexes:
                        smile_data[i] += 1
                        # chunks[key].Data = my_columnwise_zscore(chunks[key].Data, int(chunks[key].End.Time) - int(chunks[key].Start.Time))
                        # for i in range(16):
                        #     if (not np.isnan(chunks[key].Data[i]).any()):
                        #         smile_data[i] += np.mean(chunks[key].Data[i])
                elif("angry" in key.lower()):
                    indexes = argmax_n(chunks[key].Data, 4)
                    for i in indexes:
                        angry_data[i] += 1
                        # chunks[key].Data = my_columnwise_zscore(chunks[key].Data, int(chunks[key].End.Time) - int(chunks[key].Start.Time))
                        # for i in range(16):
                        #     if (not np.isnan(chunks[key].Data[i]).any()):
                        #         angry_data[i] += np.mean(chunks[key].Data[i])
                elif ("blink" in key.lower() and "0" not in key):
                    indexes = argmax_n(chunks[key].Data, 4)
                    for i in indexes:
                        blink_data[i] += 1
                        # chunks[key].Data = my_columnwise_zscore(chunks[key].Data, int(chunks[key].End.Time) - int(chunks[key].Start.Time))
                        # for i in range(16):
                        #     if (not np.isnan(chunks[key].Data[i]).any()):
                        #         blink_data[i] += np.mean(chunks[key].Data[i])
    
    for i in range(16):
        smile_data[i] /= 2*len(get_ok_sessions())
        angry_data[i] /= 2*len(get_ok_sessions())
        blink_data[i] /= 2*len(get_ok_sessions())
    mean_blink_angry = [(blink_data[i] + angry_data[i])/2 for i in range(16)]

    vectors_var = [vectors_var[i]/var_counter[i] for i in range(16)]
    print(np.argsort(vectors_var) + 1)

    # # Scatter plot
    plt.scatter(smile_data, mean_blink_angry)
    for i in range(16):
        plt.text(smile_data[i], mean_blink_angry[i], f"{i+1}")
    
    plt.xlabel("Smile mean")
    plt.ylabel("mean_blink_angry")

    # Plot y=x across the entire graph with dashed line
    x = np.linspace(min(smile_data), max(smile_data), 100)
    plt.plot(x, x, color='red', linestyle='--')
    plt.show()

    #plotly scatter plot:

    import plotly.express as px
    import pandas as pd

    df = pd.DataFrame({
        "Smile": smile_data,
        "Angry": angry_data,
        "Blink": blink_data,
        "Component": [f"{i+1}" for i in range(16)]
    })

    fig = px.scatter_3d(df, x='Smile', y='Angry', z='Blink', text='Component')
    fig.show()
    



def main():
    rms = 50
    ds = 5

    #init a 16x16xN matrix of N lists:
    all_corrs = np.zeros((16,16,0)).tolist()

    for session in tqdm(get_ok_sessions()):

        titles_A = list(range(16))
        titles_B = list(range(16))

        both_components, both_chunks = load_dyad_data(session, ds, rms)
        dataA = dict()
        dataB = dict()
        for chunk, comp, current_dict, title_list in zip(both_chunks, both_components, [dataA, dataB], [titles_A,titles_B]):
            s_key = list(chunk.keys())[0]
            e_key = list(chunk.keys())[-1]

            s = int(chunk[s_key].Start.Time)
            e = int(chunk[e_key].End.Time)

            for i in range(16):
                if comp[i] is not np.nan:
                    current_dict[i] = comp[i][s:e]
                else:
                    title_list.remove(i)

        p_corrs = np.zeros((16,16))
        for i in titles_A:
            for j in range(i+1):
                if j in titles_A:
                    r = np.corrcoef(dataA[i], dataA[j])[0,1]
                    all_corrs[i][j].append(r)
                    p_corrs[i][j] = r
        np.save(get_cor_file(session, A), p_corrs)

        p_corrs = np.zeros((16,16))
        
        for i in titles_B:
            for j in range(i+1):
                if j in titles_B:
                    r = np.corrcoef(dataB[i], dataB[j])[0,1]
                    all_corrs[i][j].append(r)
                    p_corrs[i][j] = r
        np.save(get_cor_file(session, B), p_corrs)

    np.save("corrs.npy", all_corrs)

    #replace corrs with avg:
    for i in range(16):
        for j in range(16):
            if len(all_corrs[i][j]) > 0:
                all_corrs[i][j] = np.mean(all_corrs[i][j])
            else:
                all_corrs[i][j] = np.nan
    
    #show matrix heatmap:
    plt.matshow(all_corrs)
    plt.colorbar()
    #titles:
    plt.xticks(ticks=np.arange(16), labels=[f"{i+1}" for i in range(16)], rotation=90)
    plt.yticks(ticks=np.arange(16), labels=[f"{i+1}" for i in range(16)])
    plt.show()


        # for i in range(16):        #vstack dataA:
        #     combined_data = np.vstack([dataA[i] for i in titles_A]).T
        #     feature_names = [f"{i+1}" for i in titles_A]
        # plt.matshow(pd.DataFrame(combined_data, columns=feature_names).corr())
        # plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90)
        # plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
        # plt.colorbar()
        # plt.show()

        #B:
        # combined_data = np.vstack([dataB[i] for i in titles_B]).T
        # feature_names = [f"{i+1}" for i in titles_B]

        # plt.matshow(pd.DataFrame(combined_data, columns=feature_names).corr())
        # plt.xticks(ticks=np.arange(len(feature_names)), labels=feature_names, rotation=90)
        # plt.yticks(ticks=np.arange(len(feature_names)), labels=feature_names)
        # plt.colorbar()
        # plt.show()


def plot_just_corrs_graph():
    all_corrs = np.load("corrs.npy", allow_pickle=True)
    for i in range(16):
        for j in range(16):
            if len(all_corrs[i][j]) > 0:
                all_corrs[i][j] = np.mean(all_corrs[i][j])
            else:
                all_corrs[i][j] = np.nan
            if i==j:
                all_corrs[i][j] = np.nan
    
    # Convert all_corrs to a numpy array
    all_corrs = np.array(all_corrs, dtype=float)
    #show matrix heatmap:
    plt.matshow(all_corrs)
    plt.colorbar()
    #titles:
    plt.xticks(ticks=np.arange(16), labels=[f"{i+1}" for i in range(16)], rotation=90)
    plt.yticks(ticks=np.arange(16), labels=[f"{i+1}" for i in range(16)])
    plt.show()

def get_max_len(both_chunks, key):
    chunks_a = both_chunks[A][key].Data
    chunks_b = both_chunks[B][key].Data
    len_a = 0
    len_b = 0
    for i in range(16):
        if chunks_a[i] is not np.nan:
            len_a = len(chunks_a[i])
        if chunks_b[i] is not np.nan:
            len_b = len(chunks_b[i])
        if len_a != 0 and len_b != 0:
            break
    return min(len_a, len_b)


def compute_sync_score_for_dyad(session, window_size, max_lag_ms, ds_facotr):
    # Load the components for both participants
    titles_A = list(range(16))
    titles_B = list(range(16))
    n=16
    numOfPixelsForXCorrWindow = window_size//ds_facotr
    numOfPixelsForXCorrStep = numOfPixelsForXCorrWindow


    both_chunks = [[],[]]
    for p in [A, B]:
        both_chunks[p] = load_chunks(session, p)

    reading_corrs = []
    listening_corrs = []

    reading_order = [A, A, B, B, A, A, B, B]
    for key in both_chunks[A]:
        if "reading" in key.lower():
            current_reader = reading_order.pop(0)
        elif "listening" in key.lower():
            current_reader = A
        else:
            continue

        max_len = get_max_len(both_chunks, key)

        for s in range(0, max_len - numOfPixelsForXCorrWindow , numOfPixelsForXCorrStep):
            e = s + numOfPixelsForXCorrWindow
            direction = "one-direction"
            # all_corr_vecs_within_s = [[] for i in range(n)]
            for i in range(n):
                mat1 = both_chunks[A][key].Data[i]
                mat2 = both_chunks[B][key].Data[i]
                
                if np.isnan(mat1).any() or np.isnan(mat2).any():
                    continue

                if ( current_reader == A):
                    #todo - fix this is I return to reader's assumption.
                    # should I have one best lag in a window for all comps? if so - return all corrs arr and determine max here.
                    max_corr, max_ind = get_max_xcorr_pearson_1d(mat2, mat1, max_lag_ms,
                                                        ds_facotr, s, e, direction)
                else:
                    max_corr, max_ind = get_max_xcorr_pearson_1d(mat1, mat2, max_lag_ms,
                                                        ds_facotr, s, e, direction)
                if "reading" in key.lower():
                    reading_corrs.append(max_corr)
                elif "listening" in key.lower():
                    listening_corrs.append(max_corr)
    return np.mean(reading_corrs), np.mean(listening_corrs)

output_path = r"C:\Liron\DataEmg\revision_analysis\reading_vs_listening_best_windows"
windows = [3000,4000,5000]
lags = [3000,4000,5000]
for window in windows:
    for lag in lags:
        print(f"window: {window}, lag: {lag}")
        reading_corrs = []
        listening_corrs = []
        for session in tqdm(get_ok_sessions()):
            reading, listening = compute_sync_score_for_dyad(session, window, lag, 5)
            reading_corrs.append(reading)
            listening_corrs.append(listening)
            print("session: ", session, "reading: ", reading, "listening: ", listening)
        np.save(os.path.join(output_path, f"corrs_{window}_{lag}.npy"), [reading_corrs, listening_corrs])

