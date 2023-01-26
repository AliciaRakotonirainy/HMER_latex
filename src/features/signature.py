import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import similaritymeasures
import pandas as pd

from src.preprocessing.preprocess import crop_character_horizontally, crop_character_vertically
from src.utils.pathtools import logger
import logging

from src.utils.constants import *

logger.setLevel(logging.INFO)

rcParams['text.usetex'] = True


def compute_angles_and_distances(character, save_path_angles = None, save_path_distances = None, display=False):
    # Binarize the image
    ret, mask = cv2.threshold(character, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Extract largest blob
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    largest_contour = np.squeeze(largest_contour)

    # compute centroid
    M = cv2.moments(character)
    if M["m00"] == 0:
        cX = 0
        cY = 0
    else:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])

    if display:
        # print character + contour + centroid
        plt.imshow(character, cmap="gray")
        plt.scatter(largest_contour[:,0], largest_contour[:,1], s = 1)
        plt.scatter(cX, cY, c="red")
        plt.title("Handwritten character with contour and centroid")
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
        plt.title("Distance of contour to centroid, as a function of angle")
        plt.show()

    if save_path_angles is not None:
        angles_df = pd.DataFrame(angles)
        distances_df = pd.DataFrame(distances)
        angles_df.to_csv(save_path_angles, header=False)
        distances_df.to_csv(save_path_distances, header=False)

    angles = angles.reshape(-1, 1)
    distances = distances.reshape(-1,1)

    return angles, distances

def make_ref_if_not_exists(label):

    # Find the name of the symbol 
    name = label.replace("$","")
    name = name.replace("\\", "")
    name = name.replace("!", "exclamation")
    name = name.replace("(", "paropen")
    name = name.replace(")", "parclose")
    name = name.replace("+", "plus")
    name = name.replace(",", "comma")
    name = name.replace("-", "minus")
    name = name.replace(".", "point")
    name = name.replace("/", "slash")
    name = name.replace("[", "hookopen")
    name = name.replace("]", "hookclose")
    name = name.replace("{", "accopen")
    name = name.replace("}", "accclose")
    name = name.replace("|", "bar")
    name = name.replace("=", "equals")
    name = name.replace(">", "more")
    name = name.replace("<", "less")

    # modify label name for visualization purpose
    if "sqrt" in label:
        label = "$\sqrt{}$"
    if "\\" in label:
        label = "$" + label + "$"


    ref_path = DATA_DIR + 'references/' + name + '.png'
    if os.path.exists(ref_path):
        return ref_path
    else:
        # Write the reference symbol on image
        fig, ax = plt.subplots(figsize=(1.2,1.2))
        ax.text(0,0, label, ha='center', size=70)
        ax.axis('off')
        fig.tight_layout()
        plt.savefig(ref_path)
        plt.close()

        ref = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        ref = crop_character_horizontally(ref)
        ref = crop_character_vertically(ref)
        cv2.imwrite(ref_path, ref)

        return ref_path

def make_outer_contour(angles, distances, step=3):
    """ 
    Not useful here
    """
    for i in range(-180,180,step):
        idx_similar_angle = np.where((i <= angles) & (angles < i + step))[0]

        if len(idx_similar_angle) > 0:
            max_distance = np.max(distances[idx_similar_angle])
            distances[idx_similar_angle] = max_distance

    return(angles, distances)

def compute_signature(angles, distances, angles_ref, distances_ref, display=False):

    if display:
        plt.scatter(angles_ref, distances_ref, s=1, c="blue", label = f"Signature of reference symbol : {label}")
        plt.scatter(angles, distances, s = 1, c="red", label="Signature of handwritten character")
        plt.title("Comparison of reference and handwritten character signatures")
        plt.legend()
        plt.show()

    signature = np.concatenate((angles, distances), axis = 1)
    signature_ref = np.concatenate((angles_ref, distances_ref), axis = 1)

    ## Warning ! The plot (angles, distances) is an **arbitrary curve**, meaning that 1 angle can
    ## be associated to several distances

    # Compute the difference between reference signature and handwritten character signature
    # The higher the distance, the most different the 2 symbols are
    area = similaritymeasures.area_between_two_curves(signature, signature_ref)

    return area



