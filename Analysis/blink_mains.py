from Analysis.blinking_for_shoval import *
from Analysis.blink_detection import *
from Analysis.blink_correlation import *
from Analysis.mark_blinks import *
from General.init import *

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

    # # Saving objects to a file
    # with open('chunks_A_B.pkl', 'wb') as file:
    #     js.dump({'blinks_in_chunks_A': blinks_in_chunks_A, 'blinks_in_chunks_B': blinks_in_chunks_B}, file) 
    
    # # Getting back the objects:
    # with open('chunks_A_B.pkl', 'rb') as file:
    #     data = pkl.load(file)
    #     blinks_in_chunks_A = data['blinks_in_chunks_A']
    #     blinks_in_chunks_B = data['blinks_in_chunks_B']

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

