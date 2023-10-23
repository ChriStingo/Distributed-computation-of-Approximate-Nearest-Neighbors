import heapq
from os import listdir
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from mocks import MOCKED_QUERY_VECTOR_1 as MOCKED_QUERY_VECTOR

''' ### PATHS ### '''
PATH_DATASETS = '../../../Datasets/Vectors/' # Path of datasets folder
PATH_IMAGES = '../../../Datasets/Images/images.txt'

''' ### UTILS ### '''
def DEBUG(elements):
    print(' '.join(map(str, elements)))


dist = []
for dataset_name in sorted(listdir(PATH_DATASETS))[:1]:
    with np.load(PATH_DATASETS + dataset_name) as fp:
        DEBUG(['Loading', dataset_name])
        data = fp['arr_0']
        
        dist.extend(cosine_similarity([MOCKED_QUERY_VECTOR], data)[0])

zipped_list = heapq.nlargest(5, zip(dist, [i for i in range(len(dist))]), key=lambda x: x[0]) # Partial sort

images = open(PATH_IMAGES, 'r')
lines = images.readlines()
tmpLinks = []
for _, meta in zipped_list:
    tmpLinks.append(lines[meta])

for el in tmpLinks:
    print(el, end='')