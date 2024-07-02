from Analysis.blinking_for_shoval import *
from Analysis.blink_detection import *
from Analysis.blink_correlation import *
from Analysis.mark_blinks import *
from General.init import *
from Analysis.blink_t_test import *

import pickle as pkl



def extract_blinks_from_chunck(chunks, blink_component, participant_blink, A):
    """
    Extracts the blinks from a chunk - from all 8 periods of 'reading'
    chunk - a dictionary with the chunk's data (start time, end time, data)
    blink_component - the component of the data that is the blinking component
    participant_blink - one blink of the participant to create a filter from
    returns a dict with lists of the blinks in the different chunks - spike trains
    """
    participant_blink_filter = create_filter(participant_blink, 4)
    blink_in_chunks = {}
    if A=='A':
        height = 0.2
    else:
        height = 0.28
    for key in chunks.keys():
        if 'Reading' in key:
            start_time = chunks[key].Start.Time
            end_time = chunks[key].End.Time 
            data = blink_component[int(start_time):int(end_time)]
            blink_spike_train = blink_peaks_detect(participant_blink_filter, data, height=height)
            blink_in_chunks[key] = blink_spike_train

    return blink_in_chunks

def extract_blink_and_data_from_participants(data_path, session):
    """
    Extracts one blink from the participant's data
    returns one blink of each participant, the blink component and the chuncks
    """

    vectorA, chunksA = extract_blinking_component(data_path, session, A, 20, 100)
    print("finished A")
    vectorB, chunksB = extract_blinking_component(data_path, session, B, 20, 100)
    print("finished B")

    one_blink_A = mark_blink_out_of_signal(vectorA)
    one_blink_B = mark_blink_out_of_signal(vectorB)

    return one_blink_A, one_blink_B, vectorA, vectorB, chunksA, chunksB

def extract_blinks_2_participants_main(data_path):
    """
    Extracts the blinks from the data of two participants
    returns two dicts containing the blinks of the each participant during the reading chunks
    """
    sessions = GetSessions(data_path)
    session = sessions[0]
    one_blink_A, one_blink_B, vectorA, vectorB, chunksA, chunksB = extract_blink_and_data_from_participants(data_path, session)
    blinks_in_chunks_A = extract_blinks_from_chunck(chunksA, vectorA, one_blink_A)
    blinks_in_chunks_B = extract_blinks_from_chunck(chunksB, vectorB, one_blink_B)
    
    return blinks_in_chunks_A, blinks_in_chunks_B

def save_data_to_file(data, file_name):
    """
    Saves the data to a file under the folder 'blinkingGraphsAndData'
    """
    directory = 'General/blinkingGraphsAndData'
    file_path = os.path.join(directory, file_name)
    with open(file_path, 'wb') as file:
        pkl.dump(data, file)
     
def save_graphs_to_file(graph, file_name):
    """
    Saves the graph to a file under the folder 'blinkingGraphsAndData'
    """
    directory = 'General/blinkingGraphsAndData'
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
    plt.figure(figsize=(9,5))
    plt.rcParams['font.size'] = '16'
    if A_or_B == 'A':
        plt.xlabel("one blink A", fontsize=16)
    elif A_or_B == 'B':
        plt.xlabel("one blink B", fontsize=16)
    else:
        plt.xlabel("one blink", fontsize=16)
    save_graphs_to_file(plt.plot(blink), file_name)

def run_over_all_sessions(data_path, PLV=True, rate_based=True):
    """
    Runs over all the sessions in the data and extracts the blinks of the participants
    for each session calculates the correlation between the blinks of the participants
    saves the data (blinks and vectors) and different correlation graphs (plv and rate based) to files under the folder 'blinkingGraphsAndData'
    """
    sessions = GetSessions(data_path)
    for session in sessions:
        one_blink_A, one_blink_B, vectorA, vectorB, chunksA, chunksB = extract_blink_and_data_from_participants(data_path, session)
        save_blink_as_a_graph(one_blink_A, 'one_blink_A_session_' + session + '.png', 'A')
        save_blink_as_a_graph(one_blink_B, 'one_blink_B_session_' + session + '.png', 'B')
        
        blinks_in_chunks_A = extract_blinks_from_chunck(chunksA, vectorA, one_blink_A, 'A')
        blinks_in_chunks_B = extract_blinks_from_chunck(chunksB, vectorB, one_blink_B, 'B')
        save_data_to_file({'vectorA': vectorA, 'vectorB': vectorB, 'blinks_in_chunks_A': blinks_in_chunks_A, 'blinks_in_chunks_B': blinks_in_chunks_B}, 'data_session_' + session + '.pkl')
        
        if PLV:
            plv_scores = []
            for key in blinks_in_chunks_A.keys():
                a_blinks = blinks_in_chunks_A[key]
                b_blinks = blinks_in_chunks_B[key]
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
    directory = 'General/blinkingGraphsAndData'
    file_name = 'data_session_' + session + '.pkl'
    file_path = os.path.join(directory, file_name)
    plv_data = None
    rated_blinks_data = None
    with open(file_path, 'rb') as file:
        data = pkl.load(file)
    if plv_score:
        with open('plv_scores_session_' + session + '.pkl', 'rb') as file:
            plv_data = pkl.load(file)
    if rated_blinks:
        with open('rated_blinks_session_' + session + '.pkl', 'rb') as file:
            rated_blinks_data = pkl.load(file)
    return data, plv_data, rated_blinks_data

def main_all_sessions(data_path, actual_sync, PLV=True, rate_based=True):
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
        plv_scores_all.append(plv_scores)
        rated_blinks_scores_all.append(rated_blinks_scores)
    
    # for all session see which rate synchronization algorithm did better
    t_score_plv, t_score_rate_based = calculate_t_scores(plv_scores_all, rated_blinks_scores_all, actual_sync)
    print("t_score_plv: ", t_score_plv)
    print("t_score_rate_based: ", t_score_rate_based)
    
    if t_score_plv > t_score_rate_based:
        print("The PLV algorithm is better")
    else:
        print("The rate-based algorithm is better")


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

    # one_blink_A = mark_blink_out_of_signal(vectorA)
    # one_blink_B = mark_blink_out_of_signal(vectorB)

    # blink_in_chunks_A = extract_blinks_from_chunck(chunksA, vectorA, one_blink_A, 'A')
    # blink_in_chunks_B = extract_blinks_from_chunck(chunksB, vectorB, one_blink_B, 'B')

    # # Saving the data to another file
    # with open('blinks_in_chunks_A_B.pkl', 'wb') as file:
    #     pkl.dump({'blink_in_chunks_A': blink_in_chunks_A, 'blink_in_chunks_B': blink_in_chunks_B}, file)

    # Getting back the objects:
    with open('blinks_in_chunks_A_B.pkl', 'rb') as file:
        data = pkl.load(file)
        blink_in_chunks_A = data['blink_in_chunks_A']
        blink_in_chunks_B = data['blink_in_chunks_B']

    plv_scores = []
    for key in blink_in_chunks_A.keys():
        a_blinks = blink_in_chunks_A[key]
        b_blinks = blink_in_chunks_B[key]

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
