import cv2
import os
from tqdm import tqdm
import pandas as pd
from random import randint
import numpy as np

from src.utils.constants import *


def make_handwritten_dataset(n_images, n_symb_per_eq, sep=15):
    print("Creation of handcrafted dataset...")
    all_isolated_symb = pd.read_pickle(TRAINSET_ISOLATED_SYMBOLS)
    n_isolated_symb = len(all_isolated_symb)
    all_backgrounds_files = os.listdir(DATA_DIR + "background/")
    n_backgrounds = len(all_backgrounds_files)

    symb_width = STANDARD_SHAPE[1]
    eq_width = n_symb_per_eq * symb_width + n_symb_per_eq * sep
    eq_height = STANDARD_SHAPE[0]

    equations_labels = []

    for n in tqdm(range(n_images)):
        indexes = [randint(0, n_isolated_symb) for _ in range(n_symb_per_eq)]
        for n_back in range(n_backgrounds):
            label = []

            background = cv2.imread(DATA_DIR + "background/background_" + str(n_back) + ".png")
            back_height = background.shape[0]
            back_width = background.shape[1]

            # Choose random location for the equation on the background
            i = randint(0, back_height - eq_height)
            j = randint(0, back_width - eq_width)

            # Place the equation at this random place in the background
            for n_symb in range(n_symb_per_eq):
                symbol = all_isolated_symb[indexes[n_symb]]["features"].reshape(STANDARD_SHAPE)*255
                symbol = cv2.cvtColor(symbol,cv2.COLOR_GRAY2RGB)
                background[i:i+eq_height,j + n_symb*sep + n_symb*symb_width:j + n_symb*sep + (n_symb+1)*symb_width, :][symbol < 150] = \
                            0.2*background[i:i+eq_height,j + n_symb*sep + n_symb*symb_width:j + n_symb*sep + (n_symb+1)*symb_width, :][symbol < 150] \
                            + 0.8*symbol[symbol < 150]
                # Write PNG image of equation on background
                cv2.imwrite(HANDCRAFTED_EQ_DIR + "eq" + str(n) + "_" + str(n_back) + ".png", background)
                label.append(np.where(all_isolated_symb[indexes[n_symb]]["label"] == 1)[0][0])
            equations_labels.append(label)
    
    equations_labels = pd.DataFrame(equations_labels)
    equations_labels.to_csv(HANDCRAFTED_EQ_DIR + "handcrafted_img_labels.csv")

    print("Handcrafted images created!")


def inkml_to_background():
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





