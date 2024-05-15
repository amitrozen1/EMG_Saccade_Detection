from Analysis.blinking_for_shoval import *
from Analysis.blink_detection import *
from Analysis.blink_correlation import *
from Analysis.mark_blinks import *
from General.init import *


def extract_blinks_from_chunck(chunks, blink_component, participant_blink):
    """
    Extracts the blinks from a chunk - from all 8 periods of 'reading'
    chunk - a dictionary with the chunk's data (start time, end time, data)
    blink_component - the component of the data that is the blinking component
    participant_blink - one blink of the participant to create a filter from
    returns a dict with lists of the blinks in the different chunks - spike trains
    """
    participant_blink_filter = create_filter(participant_blink, 4)
    blink_in_chunks = {}
    for key in chunks.keys():
        if 'reading' in key:
            start_time = chunks[key].Start.Time
            end_time = chunks[key].End.Time 
            data = blink_component[int(start_time):int(end_time)]
            blink_spike_train = blink_peaks_detect(participant_blink_filter, data)
            blink_in_chunks[key] = blink_spike_train

    return blink_in_chunks

def extract_blink_and_data_from_participants(data_path, session):
    """
    Extracts one blink from the participant's data
    returns one blink of each participant, the blink component and the chuncks
    """

    vectorA, chunksA = extract_blinking_component(data_path, session, A, 20, 100)
    vectorB, chunksB = extract_blinking_component(data_path, session, B, 20, 100)

    one_blink_A = mark_blink_out_of_signal(vectorA)
    one_blink_B = mark_blink_out_of_signal(vectorB)

    return one_blink_A, one_blink_B, vectorA, vectorB, chunksA, chunksB

def extract_blinks_2_participants_main():
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