Data_Prefix='/home/albelt/Deng/WSIDataSet'
DATA_SET_NAME='TumorSlice'
data_subset = ['train', 'train-aug', 'validation', 'validation-aug', 'heatmap']

TRAIN_TUMOR_WSI_PATH = Data_Prefix+'/Train_Tumor'
TRAIN_NORMAL_WSI_PATH = Data_Prefix+'/Train_Normal'
TRAIN_TUMOR_MASK_PATH = Data_Prefix+'/Ground_Truth/Mask'


HEAT_MAP_RAW_PATCHES_DIR = Data_Prefix + '/heatmap/patches'
HEAT_MAP_WSIs_PATH  = Data_Prefix + '/heatmap/WSIs'
HEAT_MAP_TF_RECORDS_DIR = Data_Prefix +  '/heatmap/tf-records'
HEAT_MAP_DIR = Data_Prefix + '/heatmap/heatmaps'
HEATMAP_FEATURE_CSV_TRAIN_ALL= Data_Prefix + '/heatmap/features/heatmap_features_train_all.csv'
HEATMAP_FEATURE_CSV_TRAIN = Data_Prefix + '/heatmap/features/heatmap_features_train.csv'
HEATMAP_FEATURE_CSV_VALIDATION = Data_Prefix + '/heatmap/features/heatmap_features_validation.csv'
HEATMAP_FEATURE_CSV_TEST = Data_Prefix + '/heatmap/features/heatmap_features_test.csv'
HEATMAP_FEATURE_CSV_TEST_GROUNDTRUTH = Data_Prefix + '/heatmap/features/GT.csv'

HEATMAP_FEATURE_CSV_PREDICT = Data_Prefix + '/heatmap/features/predict.csv'

HEATMAP_FEATURE_JSON_TRAIN_ALL= Data_Prefix + '/heatmap/features/heatmap_features_train_all.json'
HEATMAP_FEATURE_JSON_TRAIN = Data_Prefix + '/heatmap/features/heatmap_features_train.json'
HEATMAP_FEATURE_JSON_VALIDATION = Data_Prefix + '/heatmap/features/heatmap_features_validation.json'
HEATMAP_FEATURE_JSON_TEST = Data_Prefix + '/heatmap/features/heatmap_features_test.json'
HEATMAP_FEATURE_JSON_TEST_GROUNDTRUTH = Data_Prefix + '/heatmap/features/GT.json'


PIXEL_BLACK = 0

EVAL_DIR=Data_Prefix+'/Testset'
EVAL_LOGS=Data_Prefix+'/eval-logs'



EXTRACTED_PATCHES_NORMAL_PATH = Data_Prefix+'/Extracted_Negative_Patches/'
EXTRACTED_PATCHES_POSITIVE_PATH = Data_Prefix+'/Extracted_Positive_Patches/'
EXTRACTED_PATCHES_MASK_POSITIVE_PATH = Data_Prefix+'/Extracted-Mask-tumor/'
PROCESSED_PATCHES_TUMOR_NEGATIVE_PATH = Data_Prefix+'/Extracted_Negative_but_Tumor_Patches/'



TRAIN_TF_RECORDS_DIR = Data_Prefix+'/tf-records/'
PREFIX_SHARD_VALIDATiION = 'validation'
PATCHES_VALIDATION_DIR = Data_Prefix+'/Validation-Set/'

#about tf-records
PREFIX_SHARD_TRAIN = 'train'
PATCHES_TRAIN_DIR = Data_Prefix

#TRAIN_DIR=RuningPath+'/img_datasets/Processed/training/model8/'
#FINE_TUNE_MODEL_CKPT_PATH=RuningPath+'/img_datasets/Processed/training/model5/'

N_TRAIN_SAMPLES = 500
N_VALIDATION_SAMPLES = 500
N_SAMPLES_PER_TRAIN_SHARD = 50
N_SAMPLES_PER_VALIDATION_SHARD = 50

NUM_POSITIVE_PATCHES_FROM_EACH_BBOX = 500
NUM_NEGATIVE_PATCHES_FROM_EACH_BBOX = 500

PATCH_SIZE = 256


PATCH_INDEX_NEGATIVE = 0
PATCH_INDEX_POSITIVE = 0  
PATCH_NORMAL_PREFIX = 'normal_'
PATCH_TUMOR_PREFIX = 'tumor_'

#augment after 1st train and refine
PATCHES_VALIDATION_AUG_NEGATIVE_PATH=Data_Prefix+'/Patches-validation-aug-negative/'
PATCHES_VALIDATION_AUG_POSITIVE_PATH=Data_Prefix+'/Patches-validation-aug-positive/'
PATCH_AUG_NORMAL_PREFIX = 'aug_false_normal_'
PATCH_AUG_TUMOR_PREFIX = 'aug_false_tumor_'

TRAIN_DIR=Data_Prefix+'/events_logs/'
FINE_TUNE_MODEL_CKPT_PATH=Data_Prefix+'/Fine-tune-models/'
EVAL_MODEL_CKPT_PATH=Data_Prefix+'/Eval-models/'
TRAIN_MODELS=Data_Prefix+'/Train-models/'
TRAIN_LOGS=Data_Prefix+'/train-logs/'


heatmap_feature_names = ['region_count', 'ratio_tumor_tissue', 'largest_tumor_area', 'longest_axis_largest_tumor',
                         'pixels_gt_90', 'avg_prediction', 'max_area', 'mean_area', 'area_variance', 'area_skew',
                         'area_kurt', 'max_perimeter', 'mean_perimeter', 'perimeter_variance', 'perimeter_skew',
                         'perimeter_kurt', 'max_eccentricity', 'mean_eccentricity', 'eccentricity_variance',
                         'eccentricity_skew', 'eccentricity_kurt', 'max_extent', 'mean_extent', 'extent_variance',
                         'extent_skew', 'extent_kurt', 'max_solidity', 'mean_solidity', 'solidity_variance',
                         'solidity_skew', 'solidity_kurt', 'label']

def get_filename_from_path(file_path):
    return file_path.split('/')[-1].split('.')[0]


def format_2f(number):
    return float("{0:.2f}".format(number))

TEST_WSI_DIR = Data_Prefix + '/Testset/wsis/'
TEST_PROB_DIR = Data_Prefix + '/Testset/heatmaps'
TEST_IMAGE_OPEN_DIR = Data_Prefix + '/Testset/image_opens'
TEST_RESULT = Data_Prefix + '/Testset/results'

PREDICT_VALIDATION = Data_Prefix + '/Prediction/predict_validation.json'
PROB_PREDICT_VALIDATION = Data_Prefix + '/Prediction/prob_predict_validation.json'
PREDICT_TRAIN = Data_Prefix + '/Prediction/predict_train.json'
PROB_PREDICT_TRAIN = Data_Prefix + '/Prediction/prob_predict_train.json'

FEATURES_TEST = Data_Prefix + '/heatmap/features/heatmap_features_test.json'

RF_MODEL_PARAMETERS = Data_Prefix + '/heatmap/RF_Parms/RFModelParameters.pickle'


