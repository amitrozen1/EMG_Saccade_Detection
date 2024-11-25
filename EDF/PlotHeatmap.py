from matplotlib import image as mpimg, pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from numpy.linalg import inv
import cv2  # for image loading+user electrode location selection, install 'opencv-python'
from General.HelperFunctions import GetSessions, getWeightsFileNameWithPath, particiapnt_to_char, getPathsFromSessionFolder
import os
import pyedflib
from EDF.EdfAnalyzer import EdfAnalyzer
from General.PlotFunctions import savePlot, clearPlot
from scipy.stats import zscore
from General.init import *


# electrode coordinates (empty if need to be obtained)
x_coor = []
y_coor = []

def reset_points():
    global x_coor
    global y_coor
    x_coor = []
    y_coor = []


def image_load(image_path):
    # load the image, write the path where the image is saved (if there is no image uncomment these two lines)
    global img
    img = cv2.imread(image_path, 1)  # for electrode location selection
    image = mpimg.imread(image_path)  # for heatmap

    # image dimensions
    height = img.shape[0]
    width = img.shape[1]

    # new_height = int((height/width)*600)
    # new_width = int((height/width)*800)
    # # image resize
    # img = cv2.resize(img, (new_height, new_width))
    # image.resize((new_height, new_width))
    # #image = cv2.resize(image, (new_height, new_width))
    
    return image, height, width


def click_event(event, x, y, flags, params):
    global x_coor
    global y_coor
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        x_coor.append(x)
        y_coor.append(y)

        # displaying the coordinates
        # on the image window
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt = '(' + str(x) + ',' + str(y) + ')'
        cv2.putText(img, f"_{len(x_coor)}" , (x, y), font,
                    0.5, (255, 0, 0), 2)
        cv2.imshow('image', img)

    return x_coor, y_coor


def get_location():
    # displaying the image
    cv2.imshow('image', img)

    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('image', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()


def create_folder_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_index_after_removal(index, electrodes_to_remove):
    for i, electrode in enumerate(electrodes_to_remove):
        if index >= electrode:
            index += 1
    return index

def insert_zeros_in_removed_places(vector, electrodes_to_remove):
    for electrode in electrodes_to_remove:
        vector = np.insert(vector, electrode, 0)
    return vector

def default_heatmap_with_channels_removal(W, electrodes_to_remove, save_different_figures=False, save_at="heatmaps", extra_headers=None, selected=[], save_whole_heatmap=False):
    #reset points
    reset_points()
    number_of_channels = W.shape[1]
    order = np.arange(number_of_channels)

    should_print_only_selected = len(selected) > 0

    image_path = r"C:\Liron\DataEmg\Try\04122022_1545\A.jpeg"
    x_coor = [324, 209, 179, 251, 247, 178, 233, 302, 362, 251, 189, 178, 225, 306, 302, 404]
    y_coor = [611, 621, 550, 592, 544, 504, 491, 498, 444, 461, 448, 387, 283, 274, 212, 221]

    # for i, electrode in enumerate(electrodes_to_remove):
    #     x_coor.pop(electrode-i)
    #     y_coor.pop(electrode-i)

    image, height, width = image_load(image_path)

    # calculations for the heatmap
    inverse = np.absolute(inv(W))

    new_w = np.zeros((16,number_of_channels))
    for i in range(number_of_channels):
        new_w[:,i] = insert_zeros_in_removed_places(inverse[:,i], electrodes_to_remove)

    #elecs = np.argmax(inverse, axis=0) + 1
    elecs = np.argmax(new_w, axis=0) + 1

    grid_y, grid_x = np.mgrid[1:height + 1, 1:width + 1]

    points = np.column_stack((x_coor, y_coor))

    f_interpolate = []
    for i in range(number_of_channels):
        current = new_w[:, i]
        #insert_zeros_in_removed_places(inverse[:, i], electrodes_to_remove)
        f_interpolate.append(griddata(points, current, (grid_x, grid_y), method='linear'))

    if not save_different_figures:

        if should_print_only_selected:
            ncols = int(len(selected) / 2) + (len(selected) % 2)
            fig, axs = plt.subplots(2, ncols, figsize=(number_of_channels, ncols))
        else:
        # plot heatmap - don't delete
            ncols = int(number_of_channels / 2) + (number_of_channels % 2)
            fig, axs = plt.subplots(2, ncols, figsize=(number_of_channels, ncols))
        
        axs = axs.ravel()
        
        plot_counter = 0
        for i in range(number_of_channels):
            if should_print_only_selected and i not in selected:
                continue
            axs[plot_counter].imshow(image)
            axs[plot_counter].pcolormesh(f_interpolate[order[i]], cmap='jet', alpha=0.5)
            if extra_headers is not None:
                axs[plot_counter].set_title(f"IC_{i+1}-e_{elecs[i]}\n{extra_headers[i]}")
            else:
                axs[plot_counter].set_title(f"IC_{i+1}-e_{elecs[i]}")
            axs[plot_counter].axis('off')
            plot_counter += 1

        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        if ( save_whole_heatmap ):
            savePlot(os.path.join(save_at, f"whole_heatmap{saved_images_extention}"))
            clearPlot()
        else:
            plt.show()
    else:
        create_folder_if_not_exists(save_at)
        # for i in range(number_of_channels):
        #     if ( should_print_only_selected and i not in selected):
        #         continue
        loop_arr = range(number_of_channels)
        if should_print_only_selected:
            loop_arr = selected
        header_count=0
        for i in loop_arr:
            filename = f"IC_{i + 1}-e_{elecs[i]}{saved_images_extention}"
            if ( should_print_only_selected):
                # i index inside selected
                filename = f"{extra_headers[header_count]}{saved_images_extention}" #header_count was selected.index(i)
                header_count += 1
            plt.figure(dpi = 400)
            #hide x and y axis:
            plt.axis('off')
            plt.imshow(image)
            plt.pcolormesh(f_interpolate[order[i]], cmap='jet', alpha=0.5)
            plt.title(f"IC_{i + 1}-e_{elecs[i]}")
            savePlot(os.path.join(save_at, filename))
            clearPlot()


def get_heatmap(image_path, number_of_channels, W, reverse_order=False, save_different_figures = False, save_at = "heatmaps", extra_headers = None, default_pic=False, electrodes_to_remove=[]):
    #reset points
    reset_points()
    number_of_channels = W.shape[1]
    order = np.arange(number_of_channels)
    if reverse_order:
        order = order[::-1]
    
    global x_coor, y_coor

    if default_pic:
        image_path = r"C:\Liron\DataEmg\Try\04122022_1545\A.jpeg"
        x_coor = [324, 209, 179, 251, 247, 178, 233, 302, 362, 251, 189, 178, 225, 306, 302, 404]
        y_coor = [611, 621, 550, 592, 544, 504, 491, 498, 444, 461, 448, 387, 283, 274, 212, 221]
        image, height, width = image_load(image_path)
    else:
        image, height, width = image_load(image_path)
        get_location()
        print(x_coor)
        print(y_coor)

    # calculations for the heatmap
    inverse = np.absolute(inv(W))

    new_w = np.zeros((16,number_of_channels))
    for i in range(number_of_channels):
        new_w[:,i] = insert_zeros_in_removed_places(inverse[:,i], electrodes_to_remove)


    elecs = np.argmax(new_w, axis=0) + 1

    grid_y, grid_x = np.mgrid[1:height + 1, 1:width + 1]

    points = np.column_stack((x_coor, y_coor))

    f_interpolate = []
    for i in range(number_of_channels):
        f_interpolate.append(griddata(points, new_w[:, i], (grid_x, grid_y), method='linear'))

    if not save_different_figures:
        # plot heatmap - don't delete
        ncols = int(number_of_channels / 2) + (number_of_channels % 2)
        fig, axs = plt.subplots(2, ncols, figsize=(number_of_channels, ncols))
        axs = axs.ravel()
        # plt.show the image
        
        for i in range(number_of_channels):
            axs[i].imshow(image)
            axs[i].pcolormesh(f_interpolate[order[i]], cmap='jet', alpha=0.5)
            if extra_headers is not None:
                axs[i].set_title(f"IC_{i+1}-e_{elecs[i]}\n{extra_headers[i]}")
            else:
                axs[i].set_title(f"IC_{i+1}-e_{elecs[i]}")
            axs[i].axis('off')

        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()
    else:
        create_folder_if_not_exists(os.path.join(os.path.dirname(image_path), save_at))
        for i in range(number_of_channels):
            plt.figure(dpi = 1200)
            plt.imshow(image)
            plt.pcolormesh(f_interpolate[order[i]], cmap='jet', alpha=0.5)
            plt.title(f"IC_{i + 1}-e_{elecs[i]}")
            savePlot(os.path.join(os.path.dirname(image_path), save_at, f"IC_{i + 1}-e_{elecs[i]}.png"))
            clearPlot()

def get_participant_photo_path(data_path, current_session, participant):
    return os.path.join(data_path, current_session, f"{particiapnt_to_char(participant)}.jpeg")

A=0
B=1
num_channels = 16
generate_weight_from_scratch = True
participant = A
if(__name__ == "__main__"):
    data_path = r"C:\Liron\DataEmg\Try"
    num_components=16
    #sessions = GetSessions(data_path)
    current_session = "04122022_1545"
    w = []

    do_zscore_before_ica = True

    if not generate_weight_from_scratch:
        w = np.loadtxt(getWeightsFileNameWithPath(data_path, current_session), delimiter=",")
    else:
        paths = getPathsFromSessionFolder(os.path.join(data_path, current_session))
        pathProperties = paths[participant]
        Y, chunks, freq = EdfAnalyzer.Read(pathProperties[0], pathProperties[1])

        if do_zscore_before_ica:
            Y = zscore(Y, axis=1)

        w, x = EdfAnalyzer.ICA(Y, num_channels)
        np.savetxt(os.path.join(data_path, current_session, f"single_w_{particiapnt_to_char(participant)}.csv"), w, delimiter=",")

    get_heatmap(get_participant_photo_path(data_path, current_session, participant), num_components, w, False, False, "heatmaps_zscore_check_single")
