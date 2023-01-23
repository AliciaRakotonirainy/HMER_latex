import cv2
import os
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
from src.preprocessing.preprocess import *


DATA_DIR = "HMER_latex/data/"
DATASET = "CROHME_train_2011/"

all_inkml_files = os.listdir(DATA_DIR + DATASET)
all_backgrounds_files = os.listdir(DATA_DIR + "background/")
n_backgrounds = len(all_backgrounds_files)


for eq_file in tqdm(all_inkml_files[100:103]):
    for n_back in range(n_backgrounds):
        equation = cv2.imread(DATA_DIR + DATASET[:-1] + "_background/" + eq_file[:-4] + str(n_back) + ".png")
        cropped_splits = extract_characters(equation)



# find the discrete fourier transform of the image
