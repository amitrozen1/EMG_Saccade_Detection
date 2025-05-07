from revision_analysis.data_loading import load_actress_data
import os
import numpy as np
from scipy import signal
from scipy.stats import pearsonr


is_binary = True

def get_au_title(au, type):
    if ( type == "regression"):
        return au + "r"
    return au + "c"

def get_smile_from_video_openface(story):
    openFaceDir = r"C:\Liron\neuroscience\PHD\First year experiment - EMG\StoriesFromActress\video\tablet_with_emg\GoodTakes\OpenFace"
    smileTitles = ["AU06_", "AU12_"]
    fullFile = os.path.join(openFaceDir, f"{story}.csv")
    OpenFaceData = np.genfromtxt(fullFile, dtype=float, delimiter=',', names=True)
    auType = "regression"
    if ( is_binary == True ):
        auType = "choice"
    data = [OpenFaceData[get_au_title(x, auType)] for x in smileTitles]
    return np.mean(data, axis=0)

def get_smile_from_video_openface_with_ds(story, ds_factor):
    smile_data = get_smile_from_video_openface(story)
    hz = int(1000/ds_factor)
    freq = 30
    smile_data = signal.resample(smile_data, int(len(smile_data) / freq * hz))
    return smile_data

def main():
    ds_factor = 50
    actress_data = load_actress_data(50, ds_factor)

    for story in actress_data:
        smile_data = get_smile_from_video_openface_with_ds(story, ds_factor)
        
        max_len = min(len(smile_data), len(actress_data[story].data[5]))
        smile_data = smile_data[:max_len]

        actress_mean_5_9_smile = np.mean(np.vstack([actress_data[story].data[5], actress_data[story].data[9]]), axis=0)
        actress_mean_7_9_smile = np.mean(np.vstack([actress_data[story].data[7], actress_data[story].data[9]]), axis=0)
        actress_mean_9_10_smile = np.mean(np.vstack([actress_data[story].data[9], actress_data[story].data[10]]), axis=0)

        print(story)
        print("5_9_mean_smile", pearsonr(actress_mean_5_9_smile, smile_data))
        print("7_9_mean_smile", pearsonr(actress_mean_7_9_smile, smile_data))
        print("9_10_mean_smile", pearsonr(actress_mean_9_10_smile, smile_data))
        print("zygomaticus ", pearsonr(actress_data[story].data[5], smile_data))
        print("cheek ", pearsonr(actress_data[story].data[9], smile_data))
        # print("nose ", pearsonr(actress_data[story].data[8], smile_data))
        # print("talk", pearsonr(actress_data[story].data[0], smile_data))
        # print("maseter", pearsonr(actress_data[story].data[2], smile_data))

        #plot all in on graph:
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(6, 1, figsize=(10, 30))
        axs[0].plot(actress_data[story].data[5])
        axs[0].set_title("zygomaticus")
        axs[1].plot(actress_data[story].data[9])
        axs[1].set_title("cheek")
        axs[2].plot(actress_mean_5_9_smile, color='green')
        axs[2].set_title("actress_mean_5_9_smile")
        axs[3].plot(actress_mean_7_9_smile, color='green')
        axs[3].set_title("actress_mean_7_9_smile")
        axs[4].plot(actress_mean_9_10_smile, color='green')
        axs[4].set_title("actress_mean_9_10_smile")

        #in red:
        axs[5].plot(smile_data, color='red')
        axs[5].set_title("smile")

        plt.show()
        plt.cla()