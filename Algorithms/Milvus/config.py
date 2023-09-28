''' ### PATHS ### '''
PATH_DATASETS = '../../Datasets/Vectors/' # Path of datasets folder
PATH_IMAGES = '../../Datasets/Images/images.txt'


''' ### INDEX CONFIG ### '''
INDEX_PARAMS = {
        "metric_type":"L2",
        "index_type":"IVF_FLAT",
        "params":{"nlist":1024}
    }


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
