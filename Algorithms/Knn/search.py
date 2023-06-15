import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from config import MOCKED_QUERY_VECTOR, PATH_IMAGES, PATH_INDEX


def load_knn_index():                          
    return joblib.load(PATH_INDEX , mmap_mode ='r')

def query_knn_index(knn_index, query_vector):
    distances, indices = knn_index.kneighbors(query_vector)
    return indices[0]

def get_images_by_id(id_list):
    images = open(PATH_IMAGES, 'r')
    lines = images.readlines()
    return [lines[i] for i in id_list]

def main():
    knn_index = load_knn_index()
    knn_result_id = query_knn_index(knn_index, MOCKED_QUERY_VECTOR)
    knn_result_images = get_images_by_id(knn_result_id)
    print(knn_result_id)
    print(''.join(knn_result_images))

if __name__ == "__main__":
    main()