import pandas as pd
import os
import cv2
import numpy as np
from tqdm import tqdm
import json

from src.features.cosim import compute_mean_filters, cosine_sim
from src.features.signature import compute_angles_and_distances, compute_signature, make_ref_if_not_exists
from src.preprocessing.preprocess import extract_characters
from src.utils.make_handwritten_dataset import make_handwritten_dataset

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import xgboost

import logging
from src.utils.constants import *

import logging
from Levenshtein import distance as levenshtein_distance

logger = logging.root
logFormatter = logging.Formatter('{relativeCreated:12.0f}ms {levelname:5s} [{filename}] {message:s}', style='{')
logger.setLevel(logging.DEBUG)


class FinalClassifier(object):

    def __init__(self, test_size = TEST_SIZE_XGB):
        self.trainset_path = TRAINSET_ISOLATED_SYMBOLS
        self.train_features_path = TRAIN_FEATURES_ISOLATED_SYMBOLS
        self.train_labels_path = TRAIN_LABELS_ISOLATED_SYMBOLS

        self._trained = False
        self._handcrafted_images = True

        self.load_features()
        self.split_train_test(test_size)
        self.init_classifier()

    def make_dir_if_not_exists(self, path):
        exist = os.path.exists(path)
        if not exist:
            os.makedirs(path)


    def balance_trainset(self, n_per_class = 500):
        labels = pd.read_table(CLASSES_FILE, header=None)
        n_classes = len(self.train[0]["label"])

        balanced_train = []
        occurence_per_class = dict([(label, 0) for label in range(n_classes)])
        for sample in self.train:
            if occurence_per_class[np.where(sample["label"] == 1)[0][0]] < n_per_class:
                balanced_train.append(sample)
                occurence_per_class[np.where(sample["label"] == 1)[0][0]] += 1
        return balanced_train


    def load_features(self, force_features_computation=False):

        if os.path.exists(self.train_features_path) and not force_features_computation:
            print("Training set features already computed: loading features...")
            self.full_train_features = pd.read_csv(self.train_features_path, header=0, index_col=0)
            self.y_train = pd.read_csv(self.train_labels_path, header=0, index_col=0)
            print("Training set features loaded!")
            return

        # load whole trainset of isolated symbols
        self.train = pd.read_pickle(self.trainset_path)
        
        # make balanced dataset, with 100 occurences of each class
        self.train = self.balance_trainset()

        ## Preparing reference features and saving it

        # making reference images if they dont exist :
        labels = pd.read_table(DATA_DIR + ALL_CLASSES_FILE, sep=" ").columns
        for label in labels:
            make_ref_if_not_exists(label)

        # creates output directory for reference features
        self.make_dir_if_not_exists(OUTPUT_DIR + "reference/")

        # computes and saves reference features 
        for ref_path in os.listdir(REF_DIR):
            ref = cv2.imread(REF_DIR + ref_path, cv2.IMREAD_GRAYSCALE)
            compute_mean_filters(np.array(ref, dtype=float), save_path= OUTPUT_DIR + "reference/average_filters_features_" + ref_path[:-4])
            compute_angles_and_distances(ref, 
                                        save_path_angles = OUTPUT_DIR + "reference/angles_" + ref_path[:-4],
                                        save_path_distances = OUTPUT_DIR + "reference/distances_" + ref_path[:-4])

        self.full_train_features = []
        self.y_train = []
        for symbol in tqdm(self.train):
            img = symbol["features"].reshape(50,50)*255
            lab = symbol["label"]
            self.y_train.append(np.where(lab == 1)[0][0])

            # cosine sim
            mean_filters = compute_mean_filters(np.array(img, dtype=float))
            cosim_features = []
            # signature
            angles, distances = compute_angles_and_distances(img.astype("uint8"))
            signature_features = []
            
            for ref_path in os.listdir(REF_DIR):
                
                ## Cosine sim

                # read the averaged values on 100 windows for this reference image : 
                mean_filters_ref = pd.read_csv(OUTPUT_DIR + "reference/average_filters_features_" + ref_path[:-4], 
                                                header=None, index_col=0)
                # compute the cosine sim between symbol and reference
                cosim = cosine_sim(mean_filters, np.array(mean_filters_ref, dtype=float).flatten())
                cosim_features.append(pd.Series([cosim], index= ["cosim_" + ref_path[:-4]] ))

                ## Signature

                # read angles and corresponding distances for this reference image
                angles_ref = pd.read_csv(OUTPUT_DIR + "reference/angles_" + ref_path[:-4], index_col=0)
                distances_ref = pd.read_csv(OUTPUT_DIR + "reference/distances_" + ref_path[:-4], index_col=0)

                if angles is None:
                    signature = -1
                else:
                    signature = compute_signature(angles, distances, angles_ref, distances_ref)
                signature_features.append(pd.Series([signature], index=["AREA_" + ref_path[:-4]]))
            
            cosim_features = pd.concat(cosim_features)
            signature_features = pd.concat(signature_features, axis=0)

            self.full_train_features.append(pd.concat([cosim_features, signature_features]))

        # all features for all the isolated symbols of the train set
        self.full_train_features = pd.concat(self.full_train_features, axis=1) # features x samples
        self.full_train_features = self.full_train_features.transpose() # samples x features
        self.full_train_features.to_csv(self.train_features_path)
        pd.DataFrame(self.y_train).to_csv(self.train_labels_path)

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

    def xgb_tuning(self):
        print("Tuning XGB...")
        xgbc = xgboost.XGBClassifier(objective='multi:softmax', num_class=NUM_CLASS_XGB)
        clf = GridSearchCV(estimator=xgbc, 
            param_grid=XGB_PARAM_SEARCH,
            scoring='accuracy', 
            verbose=1
        )
        clf.fit(self.full_train_features, self.y_train)
        result = clf.best_params_
        print('End of XGB tuning')

        # Saving
        with open(XGB_BEST_PARAMS_FILE, "w") as f:
            json.dump(result, f)
        print(f'Tuning parameters stored at {XGB_BEST_PARAMS_FILE}')

        return result

    def init_classifier(self, tune_xgb = False):
        """Initiates the XGBoost classifier.
        """
        logger.info('Starting XGB tuning')

        # warning : feature_names must be string, and may not contain [, ] or <
        self.dtrain = xgboost.DMatrix(self.train_features, label = self.train_labels)
        self.dtest = xgboost.DMatrix(self.test_features, label = self.test_labels)

        if tune_xgb:
            searched_parameters = self.xgb_tuning()
        else:
            searched_parameters = XGB_DEFAULT_PARAM_TO_SEARCH
        
        self.xgb_params = {**searched_parameters, **XGB_ADDITIONNAL_PARAM}

    
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

    def predict(self, equation, one_hot_classes, eval_first = True):
        """Makes the prediction of all individual characters in ONE equation only.
        """
        if not self._trained:
            self.train_xgb()

        if eval_first:
            self.eval()

        # file giving the correspondance between one_hot_encoding and label
        self.one_hot_classes = one_hot_classes

        # preprocess the equation image and extract individual characters (that are appropriately cropped)
        character_list = extract_characters(equation)

        # apply prediction on each character of this equation :
        characters_features = []
        for char in tqdm(character_list):
            # cosine sim
            mean_filters = compute_mean_filters(np.array(char, dtype=float))
            cosim_features = []
            # signature
            angles, distances = compute_angles_and_distances(char.astype("uint8"))
            signature_features = []
            
            for ref_path in os.listdir(REF_DIR):
                
                ## Cosine sim

                # read the averaged values on 100 windows for this reference image : 
                mean_filters_ref = pd.read_csv(OUTPUT_DIR + "reference/average_filters_features_" + ref_path[:-4], 
                                                header=None, index_col=0)
                # compute the cosine sim between symbol and reference
                cosim = cosine_sim(mean_filters, np.array(mean_filters_ref, dtype=float).flatten())
                cosim_features.append(pd.Series([cosim], index= ["cosim_" + ref_path[:-4]] ))

                ## Signature

                # read angles and corresponding distances for this reference image
                angles_ref = pd.read_csv(OUTPUT_DIR + "reference/angles_" + ref_path[:-4], index_col=0)
                distances_ref = pd.read_csv(OUTPUT_DIR + "reference/distances_" + ref_path[:-4], index_col=0)

                if angles is None:
                    signature = -1
                else:
                    signature = compute_signature(angles, distances, angles_ref, distances_ref)
                signature_features.append(pd.Series([signature], index=["AREA_" + ref_path[:-4]]))
            
            cosim_features = pd.concat(cosim_features)
            signature_features = pd.concat(signature_features, axis=0)

            characters_features.append(pd.concat([cosim_features, signature_features], axis=0))

        characters_features = pd.concat(characters_features, axis = 1).transpose()

        # prediction of all the characters in the equation
        predicted_idx_characters = self.trained_model.predict(xgboost.DMatrix(characters_features))
        predicted_idx_characters = predicted_idx_characters.round(0).astype(int)

        predicted_characters = []
        for id in predicted_idx_characters:
            predicted_characters.append(one_hot_classes.iloc[id,0])

        return predicted_characters, predicted_idx_characters



    def pipeline_evaluation(self, n_images = 30, n_symb_per_eq = 9):
        if self._handcrafted_images == False:
            make_handwritten_dataset(n_images, n_symb_per_eq)
            self._handcrafted_images = True

        one_hot_classes = pd.read_table(CLASSES_FILE, header=None)

        all_handcrafted_files = os.listdir(HANDCRAFTED_EQ_DIR)
        all_handcrafted_labels_files = pd.read_csv(HANDCRAFTED_EQ_DIR + "handcrafted_img_labels.csv", header=0, index_col=0)

        non_correctly_extracted = 0
        correctly_extracted = 0 # we allow for +3 false characters
        correctly_classified_char = 0

        mean_edit_dist = 0

        print("Beginning pipeline evaluation...")
        for i, file in enumerate(all_handcrafted_files):
            if file[-4:] == ".png":
                equation = cv2.imread(HANDCRAFTED_EQ_DIR + file)
                label = all_handcrafted_labels_files.iloc[i,:]
                predicted_characters, predicted_idx_characters = self.predict(equation, one_hot_classes=one_hot_classes)
                
                # if we don't have extracted the same number of characters as in the generated image
                if len(predicted_idx_characters) > n_symb_per_eq + 3:
                    non_correctly_extracted += 1
                else:
                    correctly_extracted += 1
                    mean_edit_dist += levenshtein_distance(predicted_idx_characters, label)

                    for i_char in range(min(n_symb_per_eq, len(predicted_idx_characters))):
                        if predicted_idx_characters[i_char] == label[i_char]:
                             correctly_classified_char += 1
        correctly_classified_char = correctly_classified_char / (correctly_extracted*n_symb_per_eq)
        mean_edit_dist = mean_edit_dist/correctly_extracted
        print(f"Mean Levenshtein edit distance : {mean_edit_dist}")
        print(f"Percentage of correctly extracted equations : {correctly_extracted / (correctly_extracted + non_correctly_extracted)}")
        print(f"Percentage of correctly classified characters when correctly extracted : {correctly_classified_char}")
        


def main():
    final_xgboost = FinalClassifier()
    final_xgboost.pipeline_evaluation()

    # For now, we make the prediction only on 1 equation. We will have to do it for more 
    # if we want to compute the accuracy of the whole model

    

if __name__ == '__main__':
    main()


