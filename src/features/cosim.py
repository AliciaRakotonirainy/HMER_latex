import cv2
import numpy as np
import pandas as pd
from matplotlib import rcParams

from src.utils.constants import *

rcParams['text.usetex'] = True

def compute_mean_filters(img, save_path = None):
    """ 
    Compute the mean of the image on 100 windows. 
    If save_path is provided, saves this averaged image at this path.
    """

    assert img.dtype == float, f"Got type {img.dtype}, but expected float"

    img = ready_for_similarity(img)

    

    H, W = img.shape

    N_LINES = 10
    N_COLS = 10
    h = H // N_LINES # height of every filter
    w = W // N_COLS # width of every filter

    features = []
    for i in range(N_LINES):
        for j in range(N_COLS):
            features.append(np.mean(img[i * h:(i + 1) * h, j * w:(j + 1) * w]))

    if save_path is not None:
        features = pd.DataFrame(features)
        features.to_csv(save_path, header=False)

    features = np.array(features)
    return features

def cosine_sim(features1, features2):
    # we do not need to resize the images to the same size since we are doing 
    # an average on the same number of windows, regardless of the initial size of the image
    
    if np.linalg.norm(features1) * np.linalg.norm(features2) == 0:
        cos_sim = 0
    else:
        cos_sim = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
    return cos_sim


def ready_for_similarity(data):
    """Preprocess array to be ready for the similarity computation."""
    final_data = cv2.resize(final_data, STANDARD_SHAPE)
    final_data = np.zeros(shape=data.shape, dtype=float)
    mask = data > 128
    final_data[mask] = -1
    final_data[np.invert(mask)] = 1
    return final_data

