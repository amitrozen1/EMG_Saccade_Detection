from General.HelperFunctions import GetSessions

batch1_dir = "C:\\Liron\\DataEmg\\Try"
batch2_dir = "C:\\Liron\\DataEmg\\new_batch"

batch1_sessions = GetSessions(batch1_dir)
batch2_sessions = GetSessions(batch2_dir)

#create csv for the batches where each item in the list is in new row
with open('batch1.csv', 'w') as f:
    for item in batch1_sessions:
        f.write("%s\n" % item)

with open('batch2.csv', 'w') as f:
    for item in batch2_sessions:
        f.write("%s\n" % item)

