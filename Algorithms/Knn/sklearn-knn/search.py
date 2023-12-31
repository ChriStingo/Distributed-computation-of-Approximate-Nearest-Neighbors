import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from mocks import MOCKED_QUERY_VECTOR_1 as MOCKED_QUERY_VECTOR
from chronometer import Chronometer

from config import PATH_IMAGES, PATH_INDEX


def load_knn_index():                          
    return joblib.load(PATH_INDEX , mmap_mode ='r')

def query_knn_index(knn_index, query_vector, chronometer: Chronometer):
    chronometer.begin_time_window()
    distances, indices = knn_index.kneighbors(query_vector)
    chronometer.end_time_window()
    return distances[0], indices[0]

def get_images_by_id(id_list):
    images = open(PATH_IMAGES, 'r')
    lines = images.readlines()
    return [lines[i] for i in id_list]

def main():
    chronometer = Chronometer()
    knn_index = load_knn_index()
    knn_result_distance, knn_result_id = query_knn_index(knn_index, np.asarray([MOCKED_QUERY_VECTOR]), chronometer)
    knn_result_images = get_images_by_id(knn_result_id)
    for index in range(len(knn_result_images)):
        print(knn_result_distance[index], '-', knn_result_images[index], end='')
    chronometer.get_total_time()

if __name__ == "__main__":
    main()