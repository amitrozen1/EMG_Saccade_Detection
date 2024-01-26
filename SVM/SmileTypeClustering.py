import numpy as np
import os
from General.HelperFunctions import GetSessions, particiapnt_to_char
from SVM.InnerClasses import SvmComponentWeight, LoadSvmComponentsArrFromFile
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
from scipy.stats import zscore

A = 0
B = 1

def to_particiapnt_string(session, p):
    return f"{session}_{particiapnt_to_char(p)}"

# def min_max_norm(mat):
#     mat = (mat - np.min(mat)) / (np.max(mat) - np.min(mat))
#     return mat


# No meaning for min_max here, because when a component is negative, it means that using permutation importance
# and shuffleling the component's value - actually imporved the accuracy of the model. and thus, it's not
# relevant for the actual classification.

def zscore_component_relevance(svm_comp_weight_list):
    fixed_relevance = zscore(np.array([x.relevance for x in svm_comp_weight_list]))
    for i in range(len(svm_comp_weight_list)):
        svm_comp_weight_list[i].relevance = fixed_relevance[i]
    #return svm_comp_weight_list

def zscore_relevvance_score_across_all_list():
    pass


def openElectrodeRelevanceFile(data_path, session, p):
    file_path = os.path.join(data_path, session, "smile", "svc",
                              f"components_relevance_{particiapnt_to_char(p)}.json")
    comp_weight_arr = LoadSvmComponentsArrFromFile(file_path)
    #zscore_component_relevance(comp_weight_arr)
    comp_weight_arr.sort(key = lambda x: x.relevance, reverse = True)
    return comp_weight_arr

def split_weight_comps(list_of_comps, num_of_comps):
    electrodes = []
    relevances = []
    for i in range(num_of_comps):
        electrodes.append(list_of_comps[i].electrode)
        relevances.append(list_of_comps[i].relevance)
    return electrodes, relevances

def get_relevence_statistics(data_path, sessions):
    all_relevances = []
    for session in sessions:
        for p in [A, B]:
            comp_weight_arr = openElectrodeRelevanceFile(data_path, session, p)
            for comp in comp_weight_arr:
                all_relevances.append(comp.relevance)
    return np.mean(all_relevances), np.std(all_relevances)

def build_representation_vector(data_path, session, p, num_of_comps, m, s):
    vec = np.array([0]*16, dtype=np.float32)
    relevanceVec = openElectrodeRelevanceFile(data_path, session, p)
    elecs, weights = split_weight_comps(relevanceVec, num_of_comps)
    weights = (np.array(weights) - m) / s
    print ( to_particiapnt_string(session, p ) )
    print ( elecs )
    for i in range(num_of_comps):
        if ( vec[elecs[i]-1] == 0 ):
            vec[elecs[i]-1] = weights[i]
    return vec

def plot_hist_from_list(data):
    # Use matplotlib to plot a histogram of the data
    plt.hist(data, bins=range(1, max(data) + 1),align='left', rwidth=0.8)

    # Set the title and labels for the histogram
    plt.title('Histogram of Data')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.xticks(range(1, max(data) + 1))
    # Show the histogram
    plt.show()

def plot_histogram_of_comps(data_path, sessions):
    elec_list = []
    for session in sessions:
        for p in [A,B]:
            arr = openElectrodeRelevanceFile(data_path, session, p)
            e, _ = split_weight_comps(arr, 4)
            elec_list += e
            
    plot_hist_from_list(elec_list)

def find_optimal_k(data_path, num_of_representation_comps = 5):
    sessions = GetSessions(data_path)
    representations = dict()
    m,s = get_relevence_statistics(data_path, sessions)
    for session in sessions:
        for p in [A,B]:
            representations[to_particiapnt_string(session, p)] = build_representation_vector(data_path,
                                                                                             session, 
                                                                                             p, 
                                                                                             num_of_representation_comps, 
                                                                                             m, 
                                                                                             s)            

    vec_list = []
    for seesion_p in representations:
        vec_list.append(representations[seesion_p])
    
    pca = PCA(n_components=3)
    proj_data = pca.fit_transform(vec_list)
    
    wcss = []
    for K in range(2, 10):
        kmeans = KMeans(n_clusters=K, random_state=0)
        kmeans.fit(vec_list)
        print ( f"K = {K}, score = {kmeans.score(vec_list)}" )
        wcss.append(kmeans.inertia_)
    
    plt.plot(range(2, 10), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


def ClusterSmiles(data_path, K=4, num_of_representation_comps = 5, plot_fig = False):
    sessions = GetSessions(data_path)
    m,s = get_relevence_statistics(data_path, sessions)
    representations = dict()
    for session in sessions:
        for p in [A,B]:
            representations[to_particiapnt_string(session, p)] = build_representation_vector(data_path,
                                                                                             session, 
                                                                                             p, 
                                                                                             num_of_representation_comps, 
                                                                                             m, s)            

    vec_list = []
    for seesion_p in representations:
        vec_list.append(representations[seesion_p])
    
    pca = PCA(n_components=3)
    proj_data = pca.fit_transform(vec_list)


    kmeans = KMeans(n_clusters=K, random_state=0).fit(vec_list)
    
    labels = {}
    for i, key in enumerate(representations):
        labels[key] = (kmeans.labels_[i], representations[key])
    if ( plot_fig ):
        fig = px.scatter_3d(x=proj_data[:,0], y=proj_data[:,1], z=proj_data[:,2], color=kmeans.labels_)
        fig.show()
    
    return labels

if __name__ == "__main__":
    find_optimal_k(r"C:\liron\dataEmg\Try", 8)
    exit()
    labels_and_representations = ClusterSmiles(r"C:\liron\dataEmg\Try", 3, 8, True)
    clusters = dict()
    for key in labels_and_representations:
        label = labels_and_representations[key][0]
        rep = np.array(labels_and_representations[key][1])
        if ( label not in clusters ):
            clusters[label] = [[], []]
        clusters[label][0].append(rep)
        clusters[label][1].append(key)
    
    for key in clusters:
        print(f"cluster {key}")
        print(f"cluster particiapnts: {clusters[key][1]}")
        cluster_mean_rep = np.mean(clusters[key][0], axis=0)
        print(f"cluster most promiment electrodes: {np.argsort(cluster_mean_rep)[::-1][:8].tolist()}")
        print(f"cluster mean representation: {cluster_mean_rep}")


