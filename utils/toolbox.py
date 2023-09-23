# torch
import torch
# classic libraries
import numpy as np
import pandas as pd
import matplotlib
import cv2

matplotlib.use('agg')  # need it on the server (no GUI)
import matplotlib.pyplot as plt
import random, os, sys, json


def plot_stat_training(df, folder_name):
    ''' Statistics over epochs. Saves a loss/jaccard progression plot during training.
        df: dataframe Containing epoch number and loss/jaccard for both training and test sets.
        folder_name: Destination folder
    '''
    # init
    nbEpochs = len(df) - 1
    # plot
    plt.figure(1);
    plt.clf()
    plt.ioff()
    plt.plot(df['epoch'], df['loss_train'], '-o', label='loss train')
    plt.plot(df['epoch'], df['loss_test'], '-o', label='loss test')
    plt.plot(df['epoch'], df['jaccard_train'], '-o', label='jaccard train')
    plt.plot(df['epoch'], df['jaccard_test'], '-o', label='jaccard test')
    plt.grid(b=True, which='major')
    plt.xlabel(r'epoch')
    plt.ylabel(r'loss')
    plt.legend(loc=0)
    plt.axis([-.5, nbEpochs + .5, 0, 1.01])
    plt.draw()
    plt.savefig(folder_name + '/stat_epochs.pdf')
    plt.close()


def plot_save_proba(s, nameFile):
    ''' Saves a colored probability plot of the predicted output.
        s: HxW numpy array containing values between 0 and 1
        nameFile: Desired path and name of output file
    '''
    plt.figure(2);
    plt.clf()
    plt.ioff()
    plt.imshow(s)
    plt.colorbar()
    plt.draw()
    plt.savefig(nameFile)
    plt.close()


def plot_save_mask(s, nameFile):
    ''' Saves a black and white mask of the predicted output.
        s: HxW numpy array containing values between 0 and 1 (same as plot_save_proba)
        nameFile: Desired path and name of output file
    '''
    plt.figure(2);
    plt.clf()
    plt.ioff()
    plt.imshow(s > .5, cmap="gray")
    plt.draw()
    plt.savefig(nameFile)
    plt.close()


def compute_jaccard_batch(mask1, mask2):
    ''' Computes the Jaccard Index between two mini-batch of images
        mask1,mask2: Boolean NumPy arrays of the same shape (NxHxW) where N is the batch size
        Returns: Jaccard index (computed as intersection/union and always between 0 and 1)
    '''
    num = np.minimum(mask1, mask2)
    num_sum = np.sum(np.sum(num, axis=2), axis=1)
    den = np.maximum(mask1, mask2)
    den_sum = np.sum(np.sum(den, axis=2), axis=1)
    return num_sum / den_sum


def compute_jaccard_individual(mask1, mask2):
    ''' Computes the Jaccard index between two individual masks
        mask1, mask2: Boolean NumPy arrays of the same shape (HxW)
        Returns: Jaccard index (computed as intersection/union and always between 0 and 1)
    '''
    num = np.minimum(mask1, mask2)
    num_sum = np.sum(num)
    den = np.maximum(mask1, mask2)
    den_sum = np.sum(den)
    return num_sum / den_sum


def save_df_network(myModel, cfg, df):
    ''' Saves the trained model, hyperparameters, and training stats
        myModel: Model used for training
        cfg: Hyperparameter dictionary set in main_network.py
        df: Statistics dataframe with loss/jaccard for train and test sets
    '''
    # 1) create folder
    os.makedirs(cfg['folder_result'] + '/Report_' + cfg['str_time'])
    # 2) save nework
    torch.save(myModel.state_dict(), cfg['folder_result'] + '/Report_' + cfg['str_time'] + '/myNetwork.pth')
    # 3) save parameters
    with open(cfg['folder_result'] + '/Report_' + cfg['str_time'] + '/parameters.json', 'w') as jsonFile:
        json.dump(cfg, jsonFile, indent=2)
    # 4) save stat training (and a plot)
    df.to_csv(cfg['folder_result'] + '/Report_' + cfg['str_time'] + '/stat_epochs.csv', index=False)
    plot_stat_training(df, cfg['folder_result'] + '/Report_' + cfg['str_time'])


def sigmoid(x, a, b, c):
    return a * 1 / (1 + np.exp(-b * (x - c)))


def only_keep_one_cc(box_proba):
    ''' Keeps the largest connected component on the boxNet prediction
        box_proba: CxHxW NumPy array containing values from 0 to 1
        Returns: CxHxW NumPy array with values from 0 to 1 keeping only the largest connected component 
    '''
    img_tp = (box_proba[1, :, :] > .6).astype('uint8')  # threshold slightly above .5
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img_tp, connectivity=4)
    max_label, max_size = max([(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)], key=lambda x: x[1])
    mask_tp = (output == max_label)
    box_proba_1cc = box_proba * mask_tp
    return box_proba_1cc


def compute_box(box_proba):
    ''' compute the box from the prediction of box_net
        box_proba: CxHxW NumPy array with values from 0 to 1
        Returns: Dictionary containing the box mask's center coordinates, corner coordinates, and radius
    '''
    # init
    _, l, w = box_proba.shape
    mask_coords = np.argwhere(box_proba[1, :, :] > 0.5)
    mask_y_coords = mask_coords[:, 0]
    mask_x_coords = mask_coords[:, 1]
    # stat
    x_mean = int(np.mean(mask_x_coords))
    y_mean = int(np.mean(mask_y_coords))
    x_var = np.var(mask_x_coords)
    y_var = np.var(mask_y_coords)
    radius = round(np.sqrt(1.5 * (x_var + y_var)))
    # deduce the box
    x_left_corner = max(x_mean - radius, 0)
    y_left_corner = max(y_mean - radius, 0)
    x_right_corner = min(x_mean + radius, w)
    y_right_corner = min(y_mean + radius, l)
    # top_left_corner = (x_mean - radius, y_mean - radius)
    # bottom_right_corner = (x_mean + radius, y_mean + radius)
    # saved
    the_box = {'x_mean': x_mean,
               'y_mean': y_mean,
               'radius': radius,
               'x_left_corner': x_left_corner,
               'y_left_corner': y_left_corner,
               'x_right_corner': x_right_corner,
               'y_right_corner': y_right_corner
               }
    return the_box


def compute_axes(mask, scale=9/400):
    '''computes the major and minor axes of a slime mold mask.
       mask: boolean 2d numpy array with shape HxW
       scale: scale factor of centimeters to pixels. Default: 9 cm per 400 pixels
       return: minor axis and major axis length in cm
    '''
    y_coords, x_coords = np.where(mask == 1)
    covariance = np.cov(x_coords, y_coords)
    eigenvalues, _ = np.linalg.eig(covariance)
    major_axis_length = 4 * np.sqrt(np.max(eigenvalues)) * scale
    minor_axis_length = 4 * np.sqrt(np.min(eigenvalues)) * scale
    return minor_axis_length, major_axis_length


def compute_area_perimeter(mask, scale=9/400): # set scale factor of centimeters to pixels. Default: 9cm per 400 pixels
    '''computes the area and perimeter of a slime mold mask.
       mask: boolean 2d numpy array with shape HxW
       scale: scale factor of centimeters to pixels. Default: 9cm per 400 pixels
       return: area and perimeter in cm^2 and cm
    '''
    contours,hierarchy = cv2.findContours(mask, 1, 2)
    len_contour = [len(contour_k) for contour_k in contours]
    idx_contour = np.argmax(len_contour)
    cnt = contours[idx_contour]
    area = cv2.contourArea(cnt) * scale**2
    perimeter = cv2.arcLength(cnt,True) * scale
    return area, perimeter

