import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm

from src.features.cosim import score
from src.features.signature import signature
from src.utils.pathtools import logger
from src.preprocessing.preprocess import extract_characters

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import xgboost

import logging
from src.utils.constants import *

logger.setLevel(logging.INFO)

class FinalClassifier(object):

    def __init__(self, test_size = TEST_SIZE_XGB):
        self.trainset_path = TRAINSET_ISOLATED_SYMBOLS
        self.train_features_path = TRAIN_FEATURES_ISOLATED_SYMBOLS

        self._trained = False

        self.load_features()
        self.split_train_test(test_size)
        self.init_classifier()
    
    def load_features(self):

        # load whole trainset of isolated symbols
        self.train = pd.read_pickle(self.trainset_path)

        # test only on 50 samples so that it's fast : REMOVE it when we have more computational resource
        self.train = self.train[:50]

        # compute the features (cosine sim + signature sim) for each symbol in the train dataset
        self.full_train_features = []
        self.y_train = []
        for symbol in tqdm(self.train):
            img = symbol["features"].reshape(50,50)*255
            lab = symbol["label"]
            self.y_train.append(np.where(lab == 1)[0][0])

            cosim_features = []
            # for each possible reference symbol, compute the cosine similarity with it
            for ref_path in os.listdir(REF_DIR):
                # read image of the reference symbol
                ref = cv2.imread(REF_DIR + ref_path)
                cosim = score(np.array(img, dtype=float), np.array(ref, dtype=float))
                
                # replace names that are not allowed in XGBoost feature names
                name = ref_path[:-4]
                name = name.replace("[", "hookopen")
                name = name.replace("]", "hookclose")
                name = name.replace("<", "less")
                cosim_features.append(pd.Series([cosim], index= ["cosim_" + name] ))
            cosim_features = pd.concat(cosim_features)

            # Compute signature similarity with each reference symbol
            signature_feature = signature(img)

            self.full_train_features.append(pd.concat([cosim_features, signature_feature]))

        # all features for all the isolated symbols of the train set
        self.full_train_features = pd.concat(self.full_train_features, axis=1) # features x samples
        self.full_train_features = self.full_train_features.transpose() # samples x features
        self.full_train_features.to_csv(self.train_features_path)

    def split_train_test(self, test_size):
        """Splits the full_train set into train and test.
        :param test_size: The proportion of sample in the test set.
        """
        logger.info('Splitting into train and test set')
        (
            self.train_features,
            self.test_features,
            self.train_labels,
            self.test_labels,
        ) = train_test_split(
            self.full_train_features,
            self.y_train,
            test_size=test_size
        )

    def init_classifier(self):
        """Initiates the XGBoost classifier.
        """
        logger.info('Starting XGB tuning')

        # warning : feature_names must be string, and may not contain [, ] or <
        self.dtrain = xgboost.DMatrix(self.train_features, label = self.train_labels)
        self.dtest = xgboost.DMatrix(self.test_features, label = self.test_labels)
        
        self.xgb_params = {**XGB_DEFAULT_PARAM_TO_SEARCH, **XGB_ADDITIONNAL_PARAM}

    
    def train_xgb(self):
        logger.info('Training XGBoost classifier')
        self.trained_model = xgboost.train(
            self.xgb_params,
            self.dtrain,
        )

        self._trained = True

    def eval(self):
        logger.info('Evaluating the model')
        test_predictions = self.trained_model.predict(self.dtest)
        test_predictions = test_predictions.round(0).astype(int)
        test_labels = np.array(self.test_labels)

        # Metrics
        self.evaluation_results = {
            'accuracy':accuracy_score(y_true=test_labels, y_pred=test_predictions),
            'recall':recall_score(y_true=test_labels, y_pred=test_predictions, average='weighted', zero_division=0.0),
            'precision':precision_score(y_true=test_labels, y_pred=test_predictions, average='weighted', zero_division=0.0),
            'f1':f1_score(y_true=test_labels, y_pred=test_predictions, average='weighted', zero_division=0.0),
        }
        for metric in self.evaluation_results:
            logger.info(f'XGBoost evaluation (train/test split on whole train): {metric}: {self.evaluation_results[metric]}')

    def predict(self, equation, eval_first = True):
        """Makes the prediction of all individual characters in ONE equation only.
        """
        if not self._trained:
            self.train_xgb()

        if eval_first:
            self.eval()

        # preprocess the equation image
        character_list = extract_characters(equation)

        # load file giving the correspondance between one_hot_encoding and label
        one_hot_classes = pd.read_table(CLASSES_FILE, header=None)

        # apply prediction on each character of this equation :
        characters_features = []
        for char in character_list:
            # compute the features of this character

            cosim_features = []
            # for each possible reference symbol, compute the cosine similarity
            for ref_path in os.listdir(REF_DIR):
                # read image of the reference symbol
                ref = cv2.imread(REF_DIR + ref_path)
                cosim = score(np.array(char, dtype=float), np.array(ref, dtype=float))
                
                # replace names that are not allowed in XGBoost feature names
                name = ref_path[:-4]
                name = name.replace("[", "hookopen")
                name = name.replace("]", "hookclose")
                name = name.replace("<", "less")
                cosim_features.append(pd.Series([cosim], index= ["cosim_" + name] ))
            cosim_features = pd.concat(cosim_features)

            # Compute signature similarity with each reference symbol
            signature_feature = signature(char)
            characters_features.append(pd.concat([cosim_features, signature_feature]))

        characters_features = pd.concat(characters_features, axis = 1).transpose()

        # prediction of all the characters in the equation
        predicted_idx_characters = self.trained_model.predict(xgboost.DMatrix(characters_features))
        predicted_idx_characters = predicted_idx_characters.round(0).astype(int)

        print("PREDICTED CHARACTERS :")
        predicted_characters = []
        for id in predicted_idx_characters:
            print(one_hot_classes.iloc[id,0])
            predicted_characters.append(one_hot_classes.iloc[id,0])

        return predicted_characters


def main():
    final_xgboost = FinalClassifier()

    # For now, we make the prediction only on 1 equation. We will have to do it for more 
    # if we want to compute the accuracy of the whole model

    # load unprocessed equation img
    equation = cv2.imread(EQUATION_BACKGROUND_DIR + "formulaire001-equation0464.png")

    prediction = final_xgboost.predict(equation)

if __name__ == '__main__':
    main()


