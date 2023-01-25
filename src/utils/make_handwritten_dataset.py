import cv2
import os
from inkml2img import inkml2img
from tqdm import tqdm
import random
import xml.etree.ElementTree as ET

from src.utils.constants import *

def make_handwritten_dataset():
    """
    Convert all INKML files of the chosen datasets (CROHME_train_2011/ for instance)
    INKML files downloaded from https://www.kaggle.com/datasets/rtatman/handwritten-mathematical-expressions

    Source of inkml2img : https://github.com/vndee/offline-crohme

    """
    # Create the folder for images converted from INKML to PNG
    os.makedirs(DATA_DIR + DATASET[:-1] + "_PNG")

    # Create the folder for PNG images of equations with background
    os.makedirs(DATA_DIR + DATASET[:-1] + "_background")

    # Create the folder for the labels of the images
    os.makedirs(DATA_DIR + DATASET[:-1] + "_labels")

    all_inkml_files = os.listdir(DATA_DIR + DATASET)
    all_backgrounds_files = os.listdir(DATA_DIR + "background/")
    n_backgrounds = len(all_backgrounds_files)

    for eq_file in tqdm(all_inkml_files):
        for n_back in range(n_backgrounds):

            eq_file_png = DATA_DIR + DATASET[:-1] + "_PNG/" + eq_file[:-5] + "png"

            # Convert INKML file to PNG image
            inkml2img.inkml2img(DATA_DIR + DATASET + eq_file, eq_file_png)

            # Read PNG image
            equation = cv2.imread(eq_file_png)
            eq_height = equation.shape[0]
            eq_width = equation.shape[1]

            # Read background image
            background = cv2.imread(DATA_DIR + "background/background_" + str(n_back) + ".png")
            back_height = background.shape[0]
            back_width = background.shape[1]

            # Choose random location for the equation on the background
            i = random.randint(0, back_height - eq_height)
            j = random.randint(0, back_width - eq_width)

            # Place the equation at this random place in the background
            background[i:i+eq_height, j:j+eq_width][equation < 150] = 0.3*background[i:i+eq_height, j:j+eq_width][equation < 150] + 0.7*equation[equation < 150 ]

            # Write PNG image of equation on background
            cv2.imwrite(DATA_DIR + DATASET[:-1] + "_background/" + eq_file[:-6] + str(n_back) + ".png", background)

            # write label of the image as txt file
            tree = ET.parse(DATA_DIR + DATASET + eq_file)
            root = tree.getroot()
            doc_namespace = "{http://www.w3.org/2003/InkML}"
            for child in root:
                if (child.tag == doc_namespace + 'annotation') and (child.attrib == {'type': 'truth'}):
                    with open(DATA_DIR + DATASET[:-1] + "_labels/" + eq_file[:-6] + str(n_back) + "_label.txt", 'w') as f:
                        f.write(child.text)
                    continue





