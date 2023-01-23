import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import similaritymeasures
import pandas as pd
rcParams['text.usetex'] = True

SYMBOLS_DIR = "isolated_symbols/"
DATA_DIR = "HMER_latex/data/"

def compute_angles_and_distances(equation, display=False):
    # Binarize the image
    ret, mask = cv2.threshold(equation, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract largest blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour = np.squeeze(largest_contour)

    # compute centroid
    M = cv2.moments(equation)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])

    if display:
        # print character + contour + centroid
        plt.imshow(equation, cmap="gray")
        plt.scatter(largest_contour[:,0], largest_contour[:,1], s = 1)
        plt.scatter(cX, cY, c="red")
        plt.show()

    ## compute distance from centroid
    x = largest_contour[:, 0]
    y = largest_contour[:, 1]
    distances = np.sqrt((x - cX) ** 2 + (y - cY) ** 2)

    # scale distances so that the maximal distance is 1
    distances = distances / distances.max()

    # angles of each point in the contour
    angles = np.degrees(np.arctan2(y - cY, x - cX))

    if display:
        # plot distance as a function of contour
        plt.plot(angles, distances)
        plt.show()

def make_ref_if_not_exists(label):
    # Write the reference symbol on image
    name = label.replace("$","")
    name = name.replace("\\", "")
    ref_path = 'HMER_latex/data/references/' + name + '.png'
    if os.path.exists(ref_path):
        return ref_path
    else:
        fig, ax = plt.subplots(figsize=(1.1,1.1))
        ax.text(0,0, label, ha='center', size=70)
        ax.axis('off')
        fig.tight_layout()
        plt.savefig(ref_path)
        return ref_path

def make_outer_contour(angles, distances, step=3):
    for i in range(-180,180,step):
        idx_similar_angle = np.where((i <= angles) & (angles < i + step))[0]

        if len(idx_similar_angle) > 0:
            max_distance = np.max(distances[idx_similar_angle])
            distances[idx_similar_angle] = max_distance

    return(angles, distances)


def signature_features(equation, label, display=False):
    """
    equation : np.array() representing the preprocessed image (loaded with cv2.imread). 
                should be one character in grayscale
    """

    angles, distances = compute_angles_and_distances(equation)

    ref_path = make_ref_if_not_exists(label)
    ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    # rescale reference to the size of the character
    ref = cv2.resize(ref, (equation.shape[1],equation.shape[0]))

    angles_ref, distances_ref = compute_angles_and_distances(ref)

    if display:
        plt.scatter(angles_ref, distances_ref, s=1, c="blue", label = f"Signature of reference symbol : {label}")
        plt.scatter(angles, distances, s = 1, c="red", label="Signature of handwritten character")
        plt.legend()
        plt.show()

    ## Warning ! The plot (angles, distances) is an **arbitrary curve**, meaning that 1 angle can
    ## be associated to several distances

    angles = angles.reshape(-1, 1)
    distances = distances.reshape(-1,1)
    signature = np.concatenate((angles, distances), axis = 1)

    angles_ref = angles_ref.reshape(-1,1)
    distances_ref = distances_ref.reshape(-1,1)
    signature_ref = np.concatenate((angles_ref, distances_ref), axis = 1)

    # Compute the difference between reference signature and handwritten character signature
    # The higher the distance, the most different the 2 symbols are
    pcm = similaritymeasures.pcm(signature, signature_ref)
    df = similaritymeasures.frechet_dist(signature, signature_ref)

def main():
    signature_features = pd.DataFrame()
    for label in os.listdir(DATA_DIR + SYMBOLS_DIR):
        pcm, df = signature_features(equation, label)
        name = label.replace("$","")
        name = name.replace("\\", "")
        signature_features["PCM_" + name] = pcm
        signature_features["DF_" + name] = df











