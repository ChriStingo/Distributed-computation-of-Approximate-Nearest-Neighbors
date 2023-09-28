''' ### PATHS ### '''
PATH_DATASETS = '../../Datasets/Vectors/' # Path of datasets folder
PATH_INDEX = '../../Datasets/Indexes/' # Index folder
PATH_IMAGES = '../../Datasets/Images/images.txt'


''' ### INDEX CONFIG ### '''
D = 2048                    # d-sized vectors
NLIST = 100                 # 100 coarse divisions of data / number of Voronoi cells
M = 256			            # number of subquantizers / subvectors
NBITS = 4		            # number of bits needed to represent codes learnt for each subvector group
DISTANCE = "l2"
INDEX_ID = "images"
INDEX_ALREADY_CREATED = False
INDEX_TYPE = 'hnswsq'
INDEX_FACTORY = None


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
