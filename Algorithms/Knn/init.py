import csv
import joblib
import numpy as np
from os import listdir
from sklearn.neighbors import KNeighborsClassifier

from config import DEBUG, DISTANCE, PATH_INDEX, PATH_DATASETS, get_dataset_columns

def create_knn_index(nearest_neighbors):
    return KNeighborsClassifier(nearest_neighbors)
    
def fill_index(knn_index):
    # Read each dataset in the folder and insert its vectors in the index
    matrix = []
    for dataset_name in sorted(listdir(PATH_DATASETS)):
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            datareader = csv.reader(dataset)
            DEBUG(['Loading', dataset_name])
            
            # Add to index
            for vector in datareader:
                matrix.append(np.array(vector, dtype=np.longdouble))

    knn_index.fit(matrix, np.arange(0, len(matrix)))

def build_and_save_knn_index(knn_index):
    joblib.dump(knn_index, PATH_INDEX)

def main():
    knn_index = create_knn_index(5)
    fill_index(knn_index)
    build_and_save_knn_index(knn_index)

if __name__ == "__main__":
    main()