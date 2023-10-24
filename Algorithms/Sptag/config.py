''' ### PATHS ### '''
PATH_DATASETS = '../../Datasets/Vectors/' # Path of datasets folder
PATH_INDEX = '../../Datasets/Indexes/sptagIndex.ann' # Index name
PATH_IMAGES = '../../Datasets/Images/images.txt'


''' ### INDEX CONFIG ### '''
DISTANCE = 'Cosine'
NUMBER_OF_THREADS = '32'
DATASETS_USED_TO_TRAIN = 16
SPANN = False


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
