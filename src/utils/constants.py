
DATASET_DIR = '../dataset/'     # processed train/valid/test data directory
DATA_DIR = '../data/'           # raw data directory

# Preprocess/DataLoader
TRAIN_SUFFIX = '.train.tsv'             # train file suffix
VALIDATION_SUFFIX = '.validation.tsv'   # validation file suffix
TEST_SUFFIX = '.test.tsv'               # test file suffix
ALL_SUFFIX = '.all.tsv'                 # all data file
FEATURE_SUFFIX = '.features.txt'         # feature file
TEST_PKL_SUFFIX = '.test.pkl'         # prepared test data pickle file suffix
VALID_PKL_SUFFIX = '.validation.pkl'  # prepared validation data pickle file suffix

# Recommender system related
USER = 'uid'                   # user column name
ITEM = 'iid'                   # item column name
LABEL = 'label'                 # label column name

RANK_FILE_NAME = 'rank.csv'     # Trained model generated ranking list
SAMPLE_ID = 'sample_id'         # sample id for each record