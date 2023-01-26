## FOLDERS
DATA_DIR = "data/"
OUTPUT_DIR = "output/"
DATASET = "CROHME_train_2011/"
EQUATION_BACKGROUND_DIR = DATA_DIR + DATASET[:-1] + "_background/"
HANDCRAFTED_EQ_DIR = DATA_DIR + "handcrafted_images/"
REF_DIR = DATA_DIR + "references/"

## FILES
CLASSES_FILE = DATA_DIR + "isolated_symbols/one_hot_classes.txt"
TRAINSET_ISOLATED_SYMBOLS = DATA_DIR + "isolated_symbols/train.pickle"
TRAIN_FEATURES_ISOLATED_SYMBOLS = OUTPUT_DIR + "isolated_symbols/train_features.csv"
TRAIN_LABELS_ISOLATED_SYMBOLS = OUTPUT_DIR + "isolated_symbols/train_labels.csv"
ALL_CLASSES_FILE = "all_classes.txt"

## FEATURES
STANDARD_SHAPE = (50,50)

## XGB

NUM_CLASS_XGB = 17
TEST_SIZE_XGB = 0.3
XGB_BEST_PARAMS_FILE = OUTPUT_DIR + "xgb_best_params.json"
XGB_PARAM_SEARCH = {
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'colsample_bytree': [0.2, 0.5, 0.8, 1]
}
XGB_DEFAULT_PARAM_TO_SEARCH = {
    'max_depth': 4,
    'learning_rate': 0.1,
    'colsample_bytree': 0.8,
}
XGB_ADDITIONNAL_PARAM = {
    'objective': 'multi:softmax',
    'num_class': NUM_CLASS_XGB,
    'min_child_weight': 2,
    'eta': 0.3,
    'subsample': 0.5,
    'gamma': 1,
    'eval_metric': 'error',
}
