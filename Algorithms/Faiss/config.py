''' ### PATHS ### '''
PATH_DATASETS = '../../Datasets/Vectors/'
PATH_INDEX = '../../Datasets/Indexes/faissIndex.faiss'
PATH_IMAGES = '../../Datasets/Images/images.txt'


''' ### INDEX CONFIG ### '''
DATASETS_USED_TO_TRAIN = 1         # Number of datasets used to train
D = 2048                           # d-sized vectors
FACTORY_STRING = "HNSW64"
SEARCH_PARAMS = None

''' ### UTILS ### '''
def DEBUG(elements):
    print(' '.join(map(str, elements)))
