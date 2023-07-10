import csv
import numpy as np
import SPTAG
from os import listdir

from config import DATASETS_IN_RAM, DEBUG, DISTANCE, NUMBER_OF_THREADS, PATH_INDEX, PATH_DATASETS, get_dataset_columns

def create_sptag_index():
    index_type = SPTAG.IndexAlgo.SPANN  # Tipo di indice SPANN
    distance_type = SPTAG.DistanceType.Cosine  # Tipo di distanza Cosine
    index_builder = SPTAG.IndexBuilder()
    index_builder.SetNumThreads(NUMBER_OF_THREADS)
    index_builder.SetIndexAlgorithm(index_type)
    index_builder.SetDistanceType(distance_type)
    sptag_index = index_builder.Build(float, get_dataset_columns())
    return sptag_index
    
def train_index(sptag_index):
     # Read each dataset in the folder and insert its vectors in the index
    matrix = []
    for dataset_name in sorted(listdir(PATH_DATASETS))[:DATASETS_IN_RAM]:
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            datareader = csv.reader(dataset)
            DEBUG(['Loading and training', dataset_name])
            
            # Add to index
            for vector in datareader:
                matrix.append(np.array(vector).astype(np.float32))
                
    sptag_index.Build(np.asmatrix(matrix).astype(np.float32), len(matrix), False)
    sptag_index.AddWithIds(np.asmatrix(matrix).astype(np.float32), np.arange(len(matrix)))

def fill_index(sptag_index):
    # Read each dataset in the folder and insert its vectors in the index
    for dataset_name in sorted(listdir(PATH_DATASETS))[0:]:
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            datareader = csv.reader(dataset)
            DEBUG(['Loading', dataset_name])
            
            # Add to index
            for vector in datareader:
                to_add = np.asmatrix(np.array(vector).astype(np.float32))
                sptag_index.AddWithIds(to_add, np.arange(len(to_add)))

def build_and_save_sptag_index(sptag_index):
    sptag_index.Build()
    sptag_index.SaveIndex(PATH_INDEX)

def main():
    sptag_index = create_sptag_index()
    #train_index(sptag_index)
    fill_index(sptag_index)
    build_and_save_sptag_index(sptag_index)

if __name__ == "__main__":
    main()