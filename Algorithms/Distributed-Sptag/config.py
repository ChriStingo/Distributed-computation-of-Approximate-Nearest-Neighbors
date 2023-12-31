''' ### PATHS ### '''
PATH_DATASETS = '../../Datasets/Vectors/' # Path of datasets folder
PATH_INDEX = '../../Datasets/Indexes/sptagIndex4.ann' # Index name
PATH_IMAGES = '../../Datasets/Images/images.txt'


''' ### AGGREGATOR ### '''
ADDR_IP_AGGREGATOR = '127.0.0.1'
ADDR_PORT_AGGREGATOR = '8100'


''' ### INDEX CONFIG ### '''
DISTANCE = 'Cosine'
NUMBER_OF_THREADS = '8'
DATASETS_USED_TO_TRAIN = 1
START_METADATA_OFFSET = 0 # count all the vectors inserted in the previus indexes, needed for correct ids in distributed version
SPANN = False

''' ### SEARCH CONFIG ### '''
NEIGHBORS_NUMBER = 100


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
