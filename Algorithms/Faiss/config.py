''' ### PATHS ### '''
PATH_DATASETS = '../../Datasets/Vectors/'
PATH_INDEX = '../../Datasets/Indexes/faissIndex.faiss'
PATH_IMAGES = '../../Datasets/Images/images.txt'


''' ### INDEX CONFIG ### '''
DATASETS_USED_TO_TRAIN = 16         # Number of datasets used to train
D = 2048                           # d-sized vectors
FACTORY_STRING = "IMI2x1,PQ2048x4fs,RFlat"
# "HNSW64"
# "IMI2x1,PQ2048x4fs,RFlat"
# "IVF3000,PQ2048x4fs,RFlat"

''' ### UTILS ### '''
def DEBUG(elements):
    print(' '.join(map(str, elements)))
