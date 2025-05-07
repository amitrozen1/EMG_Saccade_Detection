
import os
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

is_binary = False

def get_expression_titles(exp):
    if exp == "smile":
        return ["AU06_", "AU12_"]
    elif exp == "angry":
        return ["AU04_"]
    return []


def get_au_title(au, type):
    if ( type == "regression"):
        return au + "r"
    return au + "c"

def get_expression_from_video_openface(openFaceDir, vid_name, expression = "smile"):
    titles = get_expression_titles(expression)
    fullFile = os.path.join(openFaceDir, f"{vid_name}.csv")
    OpenFaceData = np.genfromtxt(fullFile, dtype=float, delimiter=',', names=True)
    auType = "regression"
    if ( is_binary == True ):
        auType = "choice"
    data = np.array([OpenFaceData[get_au_title(x, auType)] for x in titles])
    return np.mean(data, axis=0)

def get_expression_from_video_openface_with_ds(openFaceDir, vid_name, ds_factor, expession = "smile"):
    smile_data = get_expression_from_video_openface(openFaceDir, vid_name, expession)
    hz = int(1000/ds_factor)
    freq = 30
    smile_data = signal.resample(smile_data, int(len(smile_data) / freq * hz))
    return smile_data

def get_actor_smile_array():
    open_face_path = r"C:\Liron\neuroscience\PHD\First year experiment - EMG\expressions_stims_Amit\OpenFace"
    vid_name = "Happy1"
    expression = "smile"
    return get_expression_from_video_openface(open_face_path, vid_name, expression)

def get_actor_angry_array():
    open_face_path = r"C:\Liron\neuroscience\PHD\First year experiment - EMG\expressions_stims_Amit\OpenFace"
    vid_name = "Angry1"
    expression = "angry"
    return get_expression_from_video_openface(open_face_path, vid_name, expression)

if __name__ == "__main__":
    a = get_actor_smile_array()
    plt.title("Actor smile")
    plt.plot(a)
    plt.show()
    plt.close()
    plt.cla()

    a = get_actor_angry_array()
    plt.title("Actor angry")
    plt.plot(a)
    plt.show()
    plt.close()
