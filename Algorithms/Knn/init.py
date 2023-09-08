import csv
import joblib
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier


from config import DEBUG, PATH_INDEX, PATH_DATASETS
from chronometer import Chronometer

def create_knn_index(nearest_neighbors):
    return KNeighborsClassifier(nearest_neighbors, metric='cosine')
    
def fill_index(knn_index, chronometer: Chronometer):
    # Read each dataset in ${PATH_DATASETS} and insert its vectors in the index
    matrix = []
    for dataset_name in sorted(listdir(PATH_DATASETS)):
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            datareader = csv.reader(dataset)
            DEBUG(['Loading', dataset_name])
            
            for vector in datareader:
                matrix.append(np.array(vector, dtype=np.longdouble))

    chronometer.begin_time_window()
    knn_index.fit(matrix, np.arange(0, len(matrix)))
    chronometer.end_time_window()

def build_and_save_knn_index(knn_index):
    joblib.dump(knn_index, PATH_INDEX)

def main():
    chronometer = Chronometer()
    knn_index = create_knn_index(25)
    fill_index(knn_index, chronometer)
    build_and_save_knn_index(knn_index)
    chronometer.get_total_time()

if __name__ == "__main__":
    main()