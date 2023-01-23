import cv2
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


from numpy.fft import fft, fftfreq, ifft

from scipy import ndimage as nd
from scipy.fft import fft, ifft
from scipy import fftpack


from scipy.signal import iirnotch




def balanced_hist_thresholding(hist):

    # Lower value in distribution
    i_s = np.min(np.where(hist[0]>0))
    # Higher value in distribution
    i_e = np.max(np.where(hist[0]>0))
    # Middle of histogram
    i_m = (i_s + i_e)//2

    # Left side weight
    w_l = np.sum(hist[0][i_e:i_m+1])
    # Right side weight
    w_r = np.sum(hist[0][i_m+1:i_e+1])
    
    # Until starting point not equal to endpoint
    while (i_s != i_e):
        # If right side is heavier
        if (w_r > w_l):
            # Remove the end weight
            w_r -= hist[0][i_e]
            i_e -= 1
            # Adjust the center position and recompute the weights
            if ((i_s+i_e)//2) < i_m:
                w_l -= hist[0][i_m]
                w_r += hist[0][i_m]
                i_m -= 1
        else:
            # If left side is heavier, remove the starting weight
            w_l -= hist[0][i_s]
            i_s += 1
            # Adjust the center position and recompute the weights
            if ((i_s+i_e)//2) >= i_m:
                w_l += hist[0][i_m+1]
                w_r -= hist[0][i_m+1]
                i_m += 1
    return i_m

def crop_character_horizontally(data):
    """data is a boolean array where True stands for white, and False stands for black."""
    # First, we look for the limits of the area where the black pixels are located.
    looking_for_first_non_empty_col = True
    idx_first_non_empty_col = 0
    idx_last_non_empty_col = 0
    for i, col in enumerate(data.T):
        if not col.all():
            if looking_for_first_non_empty_col:
                idx_first_non_empty_col = i
                looking_for_first_non_empty_col = False
            else:
                idx_last_non_empty_col = i
    # Then, we crop the picture.
    data = data[:, idx_first_non_empty_col:idx_last_non_empty_col + 1]
    return data

def crop_character_vertically(data):
    """data is a boolean array where True stands for white, and False stands for black."""
    # First, we look for the limits of the area where the black pixels are located.
    looking_for_first_non_empty_line = True
    idx_first_non_empty_line = 0
    idx_last_non_empty_line = 0
    for i, line in enumerate(data):
        if not line.all():
            if looking_for_first_non_empty_line:
                idx_first_non_empty_line = i
                looking_for_first_non_empty_line = False
            else:
                idx_last_non_empty_line = i
    # Then, we crop the picture.
    data = data[idx_first_non_empty_line:idx_last_non_empty_line + 1, :]
    return data


def extract_characters(equation):
    # gray scale
    equation = cv2.cvtColor(equation, cv2.COLOR_BGR2GRAY)

    # Bilateral filtering for threshold computation
    blur = cv2.bilateralFilter(equation, 9, 80, 80) # Warning ! The performance is HIGHLY dependent on the parameters of this filter

    # find threshold using Balanced Histogram Thresholding on the blured image
    histogram = plt.hist(blur.ravel(),256,[0,256])
    plt.clf()
    thresh = balanced_hist_thresholding(histogram)


    # Apply the threshold on the not blured image (better results)
    equation[equation > thresh] = 255
    equation[equation < thresh] = 0


    # Detect characters
    intensity_dist = np.sum(equation, axis=0)
    mask = intensity_dist == np.max(intensity_dist)
    integer_mask = np.arange(len(mask))[mask]
    splits = np.hsplit(equation, integer_mask)

    # We only keep splits with more than one pixel of width:
    splits = [split for split in splits if split.shape[1] > 1]

    if len(splits) == 0:
        print("No character could be detected on the image.")
        return

    # contains the list of arrays representing each character in this equation
    cropped_splits = [crop_character_vertically(split) for split in splits]

    if len(cropped_splits)==1:
        plt.imshow(cropped_splits[0], cmap="gray")
        plt.show()
    else:
        fig, ax = plt.subplots(1, len(cropped_splits), figsize=(15, 15))
        for i, split in enumerate(cropped_splits):
            #tmp_img = Image.fromarray(split)
            ax[i].imshow(split, cmap="gray")
        plt.show()

    return(cropped_splits)
