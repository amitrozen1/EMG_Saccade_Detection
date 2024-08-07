from Analysis.blinking_for_shoval import *
from Analysis.blink_detection import *
from Analysis.blink_correlation import *
from Analysis.mark_blinks import *
from General.init import *
from Analysis.blink_t_test import *

import pickle as pkl

def extract_blinks_from_chunck(chunks, blink_component, participant_blink, A, height):
    """
    Extracts the blinks from a chunk - from all 8 periods of 'reading'
    chunk - a dictionary with the chunk's data (start time, end time, data)
    blink_component - the component of the data that is the blinking component
    participant_blink - one blink of the participant to create a filter from
    returns a dict with lists of the blinks in the different chunks - spike trains
    """
    participant_blink_filter = create_filter(participant_blink, 4)
    blink_in_chunks = {}
    Flag = True
    for key in chunks.keys():
        if 'Reading' in key:
            start_time = chunks[key].Start.Time
            end_time = chunks[key].End.Time 
            data = blink_component[int(start_time):int(end_time)]
            blink_spike_train, height1 = blink_peaks_detect(participant_blink_filter, data, height, Flag)
            blink_in_chunks[key] = blink_spike_train
            Flag = False

    return blink_in_chunks, height1

def extract_blink_and_data_from_participants(data_path, session):
    """
    Extracts one blink from the participant's data
    returns one blink of each participant, the blink component and the chuncks
    """

    vectorA, chunksA, compA = extract_blinking_component(data_path, session, A, 20, 100)
    print("finished A")
    vectorB, chunksB, compB = extract_blinking_component(data_path, session, B, 20, 100)
    print("finished B")

    s1 = chunksA["blink__0"].Start.Time
    e1 = chunksA["blink__0"].End.Time
    signal_A_for_identify = vectorA[int(s1):int(e1)]

    s2 = chunksB["blink__0"].Start.Time
    e2 = chunksB["blink__0"].End.Time
    signal_B_for_identify = vectorB[int(s2):int(e2)]
    
    one_blink_A = mark_blink_out_of_signal(signal_A_for_identify)
    one_blink_B = mark_blink_out_of_signal(signal_B_for_identify)

    return one_blink_A, one_blink_B, vectorA, vectorB, chunksA, chunksB, compA, compB

def extract_blinks_2_participants_main(data_path):
    """
    Extracts the blinks from the data of two participants
    returns two dicts containing the blinks of the each participant during the reading chunks
    """
    sessions = GetSessions(data_path)
    session = sessions[0]
    one_blink_A, one_blink_B, vectorA, vectorB, chunksA, chunksB, compA, compB = extract_blink_and_data_from_participants(data_path, session)
    blinks_in_chunks_A = extract_blinks_from_chunck(chunksA, vectorA, one_blink_A)
    blinks_in_chunks_B = extract_blinks_from_chunck(chunksB, vectorB, one_blink_B)
    
    return blinks_in_chunks_A, blinks_in_chunks_B

def get_output_folder():
    """
    Returns the path to the output folder
    """
    folder = 'General/blinkingGraphsAndData'
    if os.path.exists(folder) == False:
        os.makedirs(folder)
    return folder


def save_data_to_file(data, file_name):
    """
    Saves the data to a file under the folder 'blinkingGraphsAndData'
    """
    directory = get_output_folder()
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'wb') as file:
        pkl.dump(data, file)
     
def save_graphs_to_file(graph, file_name):
    """
    Saves the graph to a file under the folder 'blinkingGraphsAndData'
    """
    directory = get_output_folder()
    file_path = os.path.join(directory, file_name)
    graph.savefig(file_path)

def plot_plv_scores(plv_scores, session):
    """
    Plots the plv scores of the blinks of the participants
    """
    plt.plot(plv_scores)
    plt.title('PLV scores of session ' + session)
    plt.xlabel('Chunk')
    plt.ylabel('PLV score')
    return plt

def plot_rated_blinks_scores(rated_blinks_scores, session):
    """
    Plots the rated blinks scores of the blinks of the participants
    """
    plt.plot(rated_blinks_scores)
    plt.title('Rated blinks scores of session ' + session)
    plt.xlabel('Chunk')
    plt.ylabel('Rated blinks score')
    return plt

def save_blink_as_a_graph(blink, file_name, A_or_B=None):
    """
    Saves the blink as a graph
    """
    fig, ax = plt.subplots(figsize=(9,5))
    plt.rcParams['font.size'] = '16'
    if A_or_B == 'A':
        ax.set_xlabel("one blink A", fontsize=16)
    elif A_or_B == 'B':
        ax.set_xlabel("one blink B", fontsize=16)
    else:
        ax.set_xlabel("one blink", fontsize=16)
    ax.plot(blink)
    save_graphs_to_file(fig, file_name)

def run_over_all_sessions(data_path, PLV=True, rate_based=True):
    """
    Runs over all the sessions in the data and extracts the blinks of the participants
    for each session calculates the correlation between the blinks of the participants
    saves the data (blinks and vectors) and different correlation graphs (plv and rate based) to files under the folder 'blinkingGraphsAndData'
    """
    sessions = GetSessions(data_path)[0:1]
    heightA = 0.25
    heightB = 0.25
    for session in sessions:
        one_blink_A, one_blink_B, vectorA, vectorB, chunksA, chunksB, compA, compB = extract_blink_and_data_from_participants(data_path, session)
        save_blink_as_a_graph(one_blink_A, 'one_blink_A_session_' + session + '.png', 'A')
        save_blink_as_a_graph(one_blink_B, 'one_blink_B_session_' + session + '.png', 'B')
        
        blinks_in_chunks_A, heightA = extract_blinks_from_chunck(chunksA, vectorA, one_blink_A, 'A', heightA)
        blinks_in_chunks_B, heightB = extract_blinks_from_chunck(chunksB, vectorB, one_blink_B, 'B', heightB)
        save_data_to_file({'vectorA': vectorA, 'vectorB': vectorB, 'blinks_in_chunks_A': blinks_in_chunks_A, 'blinks_in_chunks_B': blinks_in_chunks_B, 'one_blink_A': one_blink_A, 'one_blink_B': one_blink_B, 'height_A': heightA, 'height_B': heightB, 'comp_A': compA, 'comp_B': compB}, 'data_session_' + session + '.pkl')

        if PLV:
            plv_scores = []
            for key in blinks_in_chunks_A.keys():
                a_blinks = blinks_in_chunks_A[key]
                print("a_blinks: ", a_blinks)
                b_blinks = blinks_in_chunks_B[key]
                print("b_blinks: ", b_blinks)
                plv_score = blink_correlation_phase_based(a_blinks, b_blinks)
                plv_scores.append(plv_score)

            save_data_to_file({'plv_scores': plv_scores}, 'plv_scores_session_' + session + '.pkl')
            save_graphs_to_file(plot_plv_scores(plv_scores, session), 'plv_' + session + '.png')
            
        if rate_based:
            rated_blinks_scores = []
            for key in blinks_in_chunks_A.keys():
                a_blinks = blinks_in_chunks_A[key]
                b_blinks = blinks_in_chunks_B[key]
                a_blinks_duration = find_blink_duration(vectorA, a_blinks)
                b_blinks_duration = find_blink_duration(vectorB, b_blinks)
                a_rated_blinks = rate_blinks(a_blinks_duration)
                b_rated_blinks = rate_blinks(b_blinks_duration)
                rated_blinks_score = blink_correlation_rate_based(a_rated_blinks, b_rated_blinks)
                rated_blinks_scores.append(rated_blinks_score)
        
            save_data_to_file({'rated_blinks_scores': rated_blinks_scores}, 'rated_blinks_session_' + session + '.pkl')
            save_graphs_to_file(plot_rated_blinks_scores(rated_blinks_scores, session), 'rated_blinks_' + session + '.png')

def read_data_from_file(session, plv_score=False, rated_blinks=False):
    """
    Reads the data of the session from the files under the folder 'blinkingGraphsAndData'
    param session - the session of the data the user wants to read
    param plv_score - if True, reads the plv scores of the session
    param rated_blinks - if True, reads the rated blinks scores of the session
    """
    directory = get_output_folder()
    file_name = 'data_session_' + session + '.pkl'
    file_path = os.path.join(directory, file_name)
    plv_data = None
    rated_blinks_data = None
    directory = get_output_folder()
    with open(file_path, 'rb') as file:
        data = pkl.load(file)
    if plv_score:
        path1 = os.path.join(directory, 'plv_scores_session_' + session + '.pkl')
        with open(path1, 'rb') as file:
            plv_data = pkl.load(file)
    if rated_blinks:
        path2 = os.path.join(directory, 'rated_blinks_session_' + session + '.pkl')
        with open(path2, 'rb') as file:
            rated_blinks_data = pkl.load(file)
    return data, plv_data, rated_blinks_data

def create_asynchronized_group_PLV_and_rate_based_scores(data_path):
    """
    creates a group of asynchronized sessions
    from each two following sessions calculates the PLV score and rate_based scores of the two A participants
    returns a list of PLV or rate-based scores of the asynchronized sessions
    """
    sessions = GetSessions(data_path)
    plv_scores_asynchronized = []
    rate_based_scores_asynchronized = []
    for i in range(len(sessions) - 1):
        data_1, plv_data1, rated_blinks_data1 = read_data_from_file(sessions[i])
        data_2, plv_data2, rated_blinks_data2 = read_data_from_file(sessions[i+1])
        A_1_blinks = data_1['blinks_in_chunks_A']
        A_2_blinks = data_2['blinks_in_chunks_A']
        plv_scores = []
        rate_based_scores = []

        for key in A_1_blinks.keys():
            a1_blinks = A_1_blinks[key]
            a2_blinks = A_2_blinks[key]
            plv_score = blink_correlation_phase_based(a1_blinks, a2_blinks)
            plv_scores.append(plv_score)
            sigA1 = data_1['vectorA']
            sigA2 = data_2['vectorA']
            a1_blinks_duration = find_blink_duration(sigA1, a1_blinks)
            a2_blinks_duration = find_blink_duration(sigA2, a2_blinks)
            a1_rated_blinks = rate_blinks(a1_blinks_duration)
            a2_rated_blinks = rate_blinks(a2_blinks_duration)
            rated_blinks_score = blink_correlation_rate_based(a1_rated_blinks, a2_rated_blinks)
            rate_based_scores.append(rated_blinks_score)
        
        plv_scores_asynchronized.append(plv_scores)
        rate_based_scores_asynchronized.append(rate_based_scores)

    return plv_scores_asynchronized, rate_based_scores_asynchronized 

def main_all_sessions(data_path, PLV=True, rate_based=True):
    """
    Runs over all the sessions in the data and extracts the blinks of the participants
    for each session calculates the correlation between the blinks of the participants
    saves the data (blinks and vectors) and different correlation graphs (plv and rate based) to files under the folder 'blinkingGraphsAndData'
    compares the percision of the two correlation algorithms
    """
    
    # once (at the beginning)
    # don't forget to comment out the line below after the first run
    run_over_all_sessions(data_path, PLV, rate_based) 

    # only after 'run_over_all_sessions' was executed, don't forget to comment out the line above
    sessions = GetSessions(data_path)
    plv_scores_all = []
    rated_blinks_scores_all = []
    for session in sessions:
        data, plv_scores, rated_blinks_scores = read_data_from_file(session, plv_score=True, rated_blinks=True)
        plv_scores_list = plv_scores['plv_scores']
        plv_scores_all.append(plv_scores_list)
        rated_blinks_scores_list = rated_blinks_scores['rated_blinks_scores']
        rated_blinks_scores_all.append(rated_blinks_scores_list)

    plv_scores_asynchronized, rate_based_scores_asynchronized = create_asynchronized_group_PLV_and_rate_based_scores(data_path)

    plv_scores_all_ttest = []
    rated_blinks_scores_all_ttest = []
    plv_scores_asynchronized_ttest = []
    rate_based_scores_asynchronized_ttest = []

    # change each list of lists to one long list
    for i in range(len(sessions)):
        for j in range(len(plv_scores_all[i])):
            plv_scores_all_ttest.append(plv_scores_all[i][j])
        for j in range(len(rated_blinks_scores_all[i])):
            rated_blinks_scores_all_ttest.append(rated_blinks_scores_all[i][j])
    for i in range(len(sessions) - 1):
        for j in range(len(plv_scores_asynchronized[i])):
            plv_scores_asynchronized_ttest.append(plv_scores_asynchronized[i][j])
        for j in range(len(rate_based_scores_asynchronized[i])):
            rate_based_scores_asynchronized_ttest.append(rate_based_scores_asynchronized[i][j])
   
    t_score_plv, t_score_rate_based = PLV_vs_rate_based_ttest(plv_scores_all_ttest, rated_blinks_scores_all_ttest, plv_scores_asynchronized_ttest, rate_based_scores_asynchronized_ttest)
    save_data_to_file({'t_score_plv': t_score_plv, 't_score_rate_based': t_score_rate_based}, 't_scores.pkl')

def main():
    # blinks_in_chunks_A, blinks_in_chunks_B = extract_blinks_2_participants_main(data_path)
    # sessions = GetSessions(data_path)
    # session = sessions[0]
    # one_blink_A, one_blink_B, vectorA, vectorB, chunksA, chunksB = extract_blink_and_data_from_participants(data_path, session)
    # print("blinks in chunks A: ", blinks_in_chunks_A)
    # print("blinks in chunks B: ", blinks_in_chunks_B)
    # print("A chunks", chunksA)
    # print("B chunks", chunksB)

    # # saving the data to a file
    # with open('chunks_A_B.pkl', 'wb') as file:
    #     pkl.dump({'chunksA': chunksA, 'chunksB': chunksB, 'one_blink_A': one_blink_A, 'one_blink_B': one_blink_B, 'vectorA': vectorA, 'vectorB': vectorB}, file)
    
    # Getting back the objects:
    with open('chunks_A_B.pkl', 'rb') as file:
        data = pkl.load(file)
        chunksA = data['chunksA']
        chunksB = data['chunksB']
        one_blink_A = data['one_blink_A']
        one_blink_B = data['one_blink_B']
        vectorA = data['vectorA']
        vectorB = data['vectorB']
    

    s = chunksA["blink__0"].Start.Time
    e = chunksA["blink__0"].End.Time

    signal_A_for_identify = vectorA[int(s):int(e)]

    s = chunksB["blink__0"].Start.Time
    e = chunksB["blink__0"].End.Time

    signal_B_for_identify = vectorB[int(s):int(e)]

    one_blink_A = mark_blink_out_of_signal(signal_A_for_identify)
    one_blink_B = mark_blink_out_of_signal(signal_B_for_identify)

    heightA = 0.25
    heightB = 0.25

    blink_in_chunks_A, heightA = extract_blinks_from_chunck(chunksA, vectorA, one_blink_A, 'A', heightA)
    blink_in_chunks_B, heightB = extract_blinks_from_chunck(chunksB, vectorB, one_blink_B, 'B', heightB)

    # Saving the data to another file
    with open('blinks_in_chunks_A_B.pkl', 'wb') as file:
        pkl.dump({'blink_in_chunks_A': blink_in_chunks_A, 'blink_in_chunks_B': blink_in_chunks_B}, file)

    # Getting back the objects:
    with open('blinks_in_chunks_A_B.pkl', 'rb') as file:
        data = pkl.load(file)
        blink_in_chunks_A = data['blink_in_chunks_A']
        blink_in_chunks_B = data['blink_in_chunks_B']

    save_blink_as_a_graph(one_blink_A, 'one_blink_A.png', 'A')
    save_blink_as_a_graph(one_blink_B, 'one_blink_B.png', 'B')

    plv_scores = []
    for key in blink_in_chunks_A.keys():
        a_blinks = blink_in_chunks_A[key]
        print("a_blinks: ", a_blinks)
        b_blinks = blink_in_chunks_B[key]
        print("b_blinks: ", b_blinks)

        plv_score = blink_correlation_phase_based(a_blinks, b_blinks)
        plv_scores.append(plv_score)

    print("PLV scores: ", plv_scores)

    rated_blinks_scores = []
    for key in blink_in_chunks_A.keys():
        a_blinks = blink_in_chunks_A[key]
        b_blinks = blink_in_chunks_B[key]
        
        a_blinks_duration = find_blink_duration(vectorA, a_blinks)
        b_blinks_duration = find_blink_duration(vectorB, b_blinks)
        a_rated_blinks = rate_blinks(a_blinks_duration)
        b_rated_blinks = rate_blinks(b_blinks_duration)
        rated_blinks_score = blink_correlation_rate_based(a_rated_blinks, b_rated_blinks)
        rated_blinks_scores.append(rated_blinks_score)

    print("Rated blinks scores: ", rated_blinks_scores)

def main_1():
   sessions = GetSessions(data_path)
   for session in sessions:
        directory = get_output_folder()
        name = 'data_session_' + session + '.pkl'
        name = os.path.join(directory, name)
        with open(name, 'rb') as file:
            data = pkl.load(file)
            vectorA = data['vectorA']
            vectorB = data['vectorB']
            blinks_in_chunks_A = data['blinks_in_chunks_A']
            blinks_in_chunks_B = data['blinks_in_chunks_B']

        plv_scores = []
        for key in blinks_in_chunks_A.keys():
            a_blinks = blinks_in_chunks_A[key]
            print("a_blinks: ", a_blinks)
            b_blinks = blinks_in_chunks_B[key]
            print("b_blinks: ", b_blinks)
            plv_score = blink_correlation_phase_based(a_blinks, b_blinks)
            plv_scores.append(plv_score)

        save_data_to_file({'plv_scores': plv_scores}, 'plv_scores_session_' + session + '.pkl')
        save_graphs_to_file(plot_plv_scores(plv_scores, session), 'plv_' + session + '.png')
            