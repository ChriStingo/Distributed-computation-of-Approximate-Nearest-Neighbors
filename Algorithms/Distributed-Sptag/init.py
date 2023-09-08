import csv
import numpy as np
import SPTAG
from os import listdir

from config import DATASETS_IN_RAM, DEBUG, DISTANCE, NUMBER_OF_THREADS, PATH_INDEX, PATH_DATASETS, get_dataset_columns
from chronometer import Chronometer

def create_sptag_index():
    sptag_index = SPTAG.AnnIndex('BKT', 'Float', get_dataset_columns())
    sptag_index.SetBuildParam("NumberOfThreads", NUMBER_OF_THREADS, "Index")
    sptag_index.SetBuildParam("DistCalcMethod", DISTANCE, "Index") 
    return sptag_index
    
def train_index(sptag_index, chronometer: Chronometer):
    # Read ${DATASETS_IN_RAM} datasets in ${PATH_DATASETS}, insert its vectors in the index and train it
    matrix = []
    for dataset_name in sorted(listdir(PATH_DATASETS))[:DATASETS_IN_RAM]:
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            datareader = csv.reader(dataset)
            DEBUG(['Loading and training', dataset_name])
            
            for vector in datareader:
                matrix.append(np.array(vector).astype(np.float32))
                
    chronometer.begin_time_window()
    sptag_index.Add(np.asmatrix(matrix).astype(np.float32), len(matrix), False)
    sptag_index.Build(np.asmatrix(matrix).astype(np.float32), len(matrix), False)
    chronometer.end_time_window()


def fill_index(sptag_index, chronometer: Chronometer):
    # Read the remaining dataset in ${PATH_DATASETS} and insert its vectors in the index
    for dataset_name in sorted(listdir(PATH_DATASETS))[DATASETS_IN_RAM:]:
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            datareader = csv.reader(dataset)
            DEBUG(['Loading', dataset_name])
            
            matrix = []
            for vector in datareader:
                matrix.append(np.array(vector).astype(np.float32))
                
            chronometer.begin_time_window()
            sptag_index.Add(np.asmatrix(matrix).astype(np.float32), len(matrix), False)
            chronometer.end_time_window()

def build_and_save_sptag_index(sptag_index):
    sptag_index.Save(PATH_INDEX)

def main():
    chronometer = Chronometer()
    sptag_index = create_sptag_index()
    train_index(sptag_index, chronometer)
    fill_index(sptag_index, chronometer)
    build_and_save_sptag_index(sptag_index)
    chronometer.get_total_time()

if __name__ == "__main__":
    main()