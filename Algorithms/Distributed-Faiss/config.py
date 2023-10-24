''' ### PATHS ### '''
PATH_DATASETS = '../../Datasets/Vectors/' # Path of datasets folder
PATH_INDEX = '../../Datasets/Indexes/dfaiss' # Index folder
PATH_IMAGES = '../../Datasets/Images/images.txt'


''' ### INDEX CONFIG ### '''
DISTANCE = "dot"
INDEX_ID = "images"
TRAIN_RATIO = 1.0
INDEX_ALREADY_CREATED = False
INDEX_TYPE = None
INDEX_FACTORY = "IVF256,PQ2048x4fs,RFlat"


''' ### UTILS ### '''
def DEBUG(elements):
    print(' '.join(map(str, elements)))

def get_dataset_columns():
    # get number of columns
    '''
    datasets_name = listdir(PATH_DATASETS)
    l = len(np.loadtxt(PATH_DATASETS + datasets_name[0], delimiter=',')[0])
    print(l)
    '''
    return 2048
