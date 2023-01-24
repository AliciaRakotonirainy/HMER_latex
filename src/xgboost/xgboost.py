import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm

from src.features.cosim import score
from src.features.signature import signature
from src.utils.pathtools import logger


REF_DIR = "HMER_latex/data/references/"
CLASSES_FILE = "HMER_latex/data/isolated_symbols/one_hot_classes.txt"

train = pd.read_pickle("HMER_latex/data/isolated_symbols/train.pickle")
train = train[:100]

train_features = []
y_train = []


for symbol in tqdm(train):
    img = symbol["features"].reshape(50,50)*255
    lab = symbol["label"]
    y_train.append(np.where(lab == 1)[0][0])

    cosim_features = []
    
    # for each possible label, compute the cosine similarity with the reference
    logger.info("Computing cosine sim with each reference label...")
    for ref_path in os.listdir(REF_DIR):
        ref = cv2.imread(REF_DIR + ref_path)
        cosim = score(np.array(img, dtype=float), np.array(ref, dtype=float))
        cosim_features.append(pd.Series([cosim], index= ["cosim_" + ref_path[:-4]] ))
    cosim_features = pd.concat(cosim_features)

    logger.info("Compute signature similarity with each reference label...")
    signature_feature = signature(img)

    train_features.append(pd.concat([cosim_features, signature_feature]))

train_features = pd.concat(train_features, axis=1) # features x samples

train_features.to_csv("train_features.csv")