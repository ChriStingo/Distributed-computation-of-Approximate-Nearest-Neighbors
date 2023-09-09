from os import listdir
import numpy as np


PATH_DATASETS = '../../Datasets/Vectors/' # Path of datasets folder
PATH_INDEX = '../../Datasets/Indexes/annoyIndex.ann' # Index name
PATH_IMAGES = '../../Datasets/Images/images.txt'
DISTANCE = 'angular'  # Metric: "angular", "euclidean", "manhattan", "hamming", or "dot"
NUMBER_OF_TREES = 8 # Is provided during build time and affects the build time and the index size. A larger value will give more accurate results, but larger indexes

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