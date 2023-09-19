import csv
import faiss
import numpy as np
from os import listdir

from config import DATASETS_USED_TO_TRAIN, DEBUG, D, FACTORY_STRING, PATH_INDEX, PATH_DATASETS
from chronometer import Chronometer

def create_faiss_index():  
    return faiss.index_factory(D, FACTORY_STRING)

def train_index(faiss_index, chronometer: Chronometer):
    # Read ${DATASETS_USED_TO_TRAIN} datasets in ${PATH_DATASETS}, insert its vectors in the index and train it
    matrix = []
    for dataset_name in sorted(listdir(PATH_DATASETS))[:DATASETS_USED_TO_TRAIN]:
        try:
            with open(PATH_DATASETS + dataset_name, "r") as dataset:
                datareader = csv.reader(dataset)
                DEBUG(['Loading and training', dataset_name])
                
                for vector in datareader:
                    matrix.append(np.array(vector).astype(np.float32))
        except:
            print("Bad chars in file:", dataset_name)
    norm_matrix = np.asmatrix(matrix)

    faiss.normalize_L2(norm_matrix)
    chronometer.begin_time_window()
    faiss_index.train(norm_matrix)
    faiss_index.add(norm_matrix)
    chronometer.end_time_window()


def fill_index(faiss_index, chronometer: Chronometer):
    # Read the remaining dataset in ${PATH_DATASETS} and insert its vectors in the index
    for dataset_name in sorted(listdir(PATH_DATASETS))[DATASETS_USED_TO_TRAIN:]:
        try:
            with open(PATH_DATASETS + dataset_name, "r") as dataset:
                datareader = csv.reader(dataset)
                DEBUG(['Loading', dataset_name])
                
                for vector in datareader:
                    norm_vector = np.asmatrix(vector).astype(np.float32)
                    faiss.normalize_L2(norm_vector)
                    
                    chronometer.begin_time_window()
                    faiss_index.add(norm_vector)
                    chronometer.end_time_window()
        except:
            print("Bad chars in file:", dataset_name)

def build_and_save_faiss_index(faiss_index):
    faiss.write_index(faiss_index, PATH_INDEX)

def main():
    chronometer = Chronometer()
    faiss_index = create_faiss_index()
    train_index(faiss_index, chronometer)
    fill_index(faiss_index, chronometer)
    build_and_save_faiss_index(faiss_index)
    chronometer.get_total_time()

if __name__ == "__main__":
    main()