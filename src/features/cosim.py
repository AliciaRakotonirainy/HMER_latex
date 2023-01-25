import cv2
import numpy as np
from matplotlib import rcParams

rcParams['text.usetex'] = True


def score(data1, data2):
    assert data1.dtype == float, f"Got type {data1.dtype}, but expected float"
    assert data2.dtype == float, f"Got type {data2.dtype}, but expected float"
    
    data2 = cv2.resize(data2, (data1.shape[1], data1.shape[0]))

    H, W = data1.shape
    # Feature extraction.
    N_LINES = 5
    N_COLS = 5
    h = H // N_LINES # height of every filter
    w = H // N_COLS # width of every filter
    features1 = []
    features2 = []
    for i in range(N_LINES):
        for j in range(N_COLS):
            features1.append(np.mean(data1[i * h:(i + 1) * h, j * w:(j + 1) * w]))
            features2.append(np.mean(data2[i * h:(i + 1) * h, j * w:(j + 1) * w]))
    features1 = np.array(features1)
    features2 = np.array(features2)
    
    if np.linalg.norm(features1) * np.linalg.norm(features2) == 0:
        cos_sim = 0
    else:
        cos_sim = np.dot(features1, features2) / (np.linalg.norm(features1) * np.linalg.norm(features2))
        print(cos_sim)
    return cos_sim

