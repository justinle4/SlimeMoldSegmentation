# torch
import torch
import torch.nn as nn
from torchvision.transforms import *
import torch.nn.functional as F
# classic
import numpy as np
import cv2
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd
import ntpath, os, shutil, glob, datetime, time
import json
# personal
from utils.slime_loader import *
from utils.models import *
from utils.toolbox import *
from utils.draw_output import *


#  Hyper-parameters
#------------------
path_folders = "datasets/inference_images/"
path_results = "results_ML/Box_Slime_Inference/"
boxnet_weights = "models/boxNet_v4.pth"
slimenet_weights = "models/slimeNet_v4.pth"
std_tol = np.inf                  # # Adjust desired slime mold standard deviation tolerance. Default = 0.12.
                                  # # Predictions with a standard deviation higher than the tolerance will be discarded. 
                                  # # Set tolerance to np.inf to prevent any discarding.

# A.1) Data
directories = os.listdir(path_folders)
transform_tensor = ToTensor()
transform_slimeNet_forTensor = Compose([
                               Resize(400),
                               CenterCrop(400)])
# Load in a circle shaped mask to turn all pixels outside the petri dish into zeros
with open('resources/petri_template.png', 'rb') as f:
    petri_mask = Image.open(f).convert('P')
petri_mask = transform_tensor(petri_mask)

# A.2) Model
# Initialize models and load in weights. Must wrap model with DataParallel if pretrained model was trained with DataParallel.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# boxNet
boxnet = FCN_Resnet50().to(device)
boxnet = nn.DataParallel(boxnet, device_ids=[0, 1])
boxnet.load_state_dict(torch.load(boxnet_weights, map_location=device))
# slimeNet
slimenet = FCN_Resnet50().to(device)
slimenet = nn.DataParallel(slimenet, device_ids=[0,1])
slimenet.load_state_dict(torch.load(slimenet_weights, map_location=device))
# A.3) saving
df_combined = pd.DataFrame(columns=('Image Name', 'X-Center', 'Y-Center', 'Radius', 'Slime Mold Std', 'Predicted Jaccard', 'Area', 'Perimeter', 'Minor Axis Length',
                               'Major Axis Length', 'Eccentricity'))
str_time = datetime.datetime.now().replace(microsecond=0).isoformat(sep='_').replace(':', 'h', 1).replace(':', 'm', 1)

#-----------------------------------------------#
#--                loop images                --#
#-----------------------------------------------#

t0 = time.time()
for directory in directories:
    # One directory at a time
    df = pd.DataFrame(columns=('Image Name', 'X-Center', 'Y-Center', 'Radius', 'Slime Mold Std', 'Predicted Jaccard', 'Area', 'Perimeter', 'Minor Axis Length', 
                               'Major Axis Length', 'Eccentricity'))
    saved_folder = path_results + 'Results_' + str_time + '/' + directory
    os.makedirs(saved_folder)
    filenames_tp = sorted(set(glob.glob(path_folders + directory + "/*.jpg")))
    filenames = [f[:-12] for f in filenames_tp]
    N = len(filenames)
    print(f"Starting new directory: {directory}")
    for idx in range(N):
        # B.1) load image
        filename = filenames[idx]
        with open(filename + '_resized.jpg', 'rb') as f:
            image = Image.open(f).convert('RGB')
        filename_cut = ntpath.basename(filename)
        # B.2) apply models
        X = transform_tensor(image).unsqueeze(dim=0)
        # B.2a) boxnet
        #-------------
        box_pred = boxnet(X.to(device))
        box_proba = F.softmax(box_pred, dim=1)
        box_proba = box_proba.squeeze(0).cpu().detach().numpy()
        box_proba_1cc = only_keep_one_cc(box_proba)
        # compute the box
        the_box = compute_box(box_proba_1cc)
        cropped_X = X[:, :,
                      the_box['y_left_corner']:the_box['y_right_corner'],
                      the_box['x_left_corner']:the_box['x_right_corner']]
        _,_,cropped_length,cropped_width = cropped_X.shape
        # B.2b) slimenet
        #---------------
        square_X = transform_slimeNet_forTensor(cropped_X)
        # keep all pixels within a 400 radius from the center and turn the rest to zeros
        square_X *= petri_mask  # multiply with petri mask (identity mapping inside petri dish, zero mapping outside)
        yhat = slimenet(square_X.to(device))
        yhat_proba = (F.softmax(yhat, dim=1)).squeeze(0)
        pred_mask = yhat_proba[0, :, :] > 0.5
        transform_interpolate = Resize((cropped_length, cropped_width))
        pred_mask_original = transform_interpolate(pred_mask.unsqueeze(0).unsqueeze(0)).squeeze(0).squeeze(0)
        # B.2c) deduce mask
        #------------------
        # computing slime mold standard deviation - Use regression function to predict Jaccard.
        yhat_proba_list = yhat_proba[0, :, :].cpu().detach().numpy().flatten()
        threshold = 0.5
        Z = yhat_proba_list >= threshold
        background_proba, slime_proba = yhat_proba_list[Z == 0], yhat_proba_list[Z == 1]
        slime_std = np.std(slime_proba)
        # discard prediction if standard deviation is too high
        if slime_std > std_tol:
            print(f"Discarding image {idx+1}; standard deviation exceeds tolerance.")
            continue
        # create a mask with the same shape as the original data
        _,_,original_length,original_width = X.shape
        slime_mold_coordinates = np.argwhere(pred_mask_original.cpu().detach().numpy() > 0.5)
        pred_mask_resized = np.zeros((original_length, original_width))
        slime_y_coords = slime_mold_coordinates[:, 0].tolist()
        slime_x_coords = slime_mold_coordinates[:, 1].tolist()
        # populate original size mask with the slime mold coordinates
        for x1, y1 in zip(slime_x_coords, slime_y_coords):
            x1_new = x1 + the_box['x_left_corner']
            y1_new = y1 + the_box['y_left_corner']
            if (y1_new < original_length) and (x1_new < original_width):
                pred_mask_resized[y1_new][x1_new] = 1
        # B.2d) Statistics
        # use regression function to produce predicted jaccard. parameters: [a, b, c] = [1.02829613, -25.37520733, 0.17050771]
        pred_jac = sigmoid(slime_std, 1.02829613, -25.37520733, 0.17050771)

        # Slime mold geometric information deduced from mask
        pred_mask_numpy = pred_mask.cpu().detach().numpy().astype('uint8')
        area, perimeter = compute_area_perimeter(pred_mask_numpy)
        minor_axis, major_axis = compute_axes(pred_mask_numpy)
        eccentricity = np.sqrt(1 - minor_axis**2 / major_axis**2)
        df.loc[idx] = [filename_cut, the_box['x_mean'], the_box['y_mean'], the_box['radius'], slime_std, pred_jac, area, perimeter, minor_axis, major_axis, eccentricity]

        # B.2e) Saving image
        mask_PIL = Image.fromarray(pred_mask_resized.astype('uint8') * 255)
        draw_prediction(image, mask_PIL, the_box['x_mean'], the_box['y_mean'], the_box['radius'], pred_jac, saved_folder, filename_cut) 
                                        
        print(f"Image {idx + 1} done!")
    # Done folder
    df.to_csv(saved_folder + "/summary.csv")
    df_combined = pd.concat([df_combined, df], ignore_index=True)

# Done all folders
df_combined.to_csv(path_results + '/Results_' + str_time + "/combined_summary.csv")
seconds_elapsed = time.time() - t0
time_inference = '{:.0f} minutes, {:.0f} seconds'.format(seconds_elapsed // 60, seconds_elapsed % 60)
print(f"--- Inference Complete in {time_inference} ---")
