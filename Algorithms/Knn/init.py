import csv
import joblib
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier


from config import DEBUG, PATH_INDEX, PATH_DATASETS
from chronometer import Chronometer

def create_knn_index(nearest_neighbors):
    return KNeighborsClassifier(n_neighbors = nearest_neighbors, metric='cosine')
    
def fill_index(knn_index, chronometer: Chronometer):
    # Read each dataset in ${PATH_DATASETS} and insert its vectors in the index
    matrix = []
    for dataset_name in sorted(listdir(PATH_DATASETS)):
        DEBUG(['Loading', dataset_name])
        data = np.load(PATH_DATASETS + dataset_name)['arr_0']
            
        if len(matrix) == 0:
            matrix = data
        else: 
            matrix = np.concatenate((matrix, data))

    chronometer.begin_time_window()
    knn_index.fit(matrix, np.arange(0, len(matrix)))
    chronometer.end_time_window()

def build_and_save_knn_index(knn_index):
    joblib.dump(knn_index, PATH_INDEX)

def main():
    chronometer = Chronometer()
    knn_index = create_knn_index(100)
    fill_index(knn_index, chronometer)
    build_and_save_knn_index(knn_index)
    chronometer.get_total_time()

if __name__ == "__main__":
    main()