from os import listdir
import numpy as np


PATH_DATASETS = '../../Datasets/Vectors/' # Path of datasets folder
PATH_INDEX = '../../Datasets/Indexes/sptagIndex.ann' # Index name
PATH_IMAGES = '../../Datasets/Images/images.txt'
DISTANCE = 'euclidean'  # Metric: "angular", "euclidean", "manhattan", "hamming", or "dot"

def DEBUG(elements):
    print(' '.join(map(str, elements)))
