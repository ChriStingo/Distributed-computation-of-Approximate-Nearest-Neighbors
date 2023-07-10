import csv
import numpy as np
import SPTAG
from os import listdir

from config import DATASETS_IN_RAM, DEBUG, DISTANCE, NUMBER_OF_THREADS, PATH_INDEX, PATH_DATASETS, get_dataset_columns

def create_sptag_index():
    sptag_index = SPTAG.AnnIndex('SPANN', 'Float', get_dataset_columns())
    # Set the thread number to speed up the build procedure in parallel 
    sptag_index.SetBuildParam("NumberOfThreads", NUMBER_OF_THREADS, "Index")
    # Set the distance type. Currently SPTAG only support Cosine and L2 distances. Here Cosine distance is not the Cosine similarity. The smaller Cosine distance it is, the better.
    sptag_index.SetBuildParam("DistCalcMethod", DISTANCE, "Index") 
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
                matrix.append(np.array(vector).astype(np.longdouble))
                
    if sptag_index.Build(np.asmatrix(matrix), len(matrix), False):
        print("build e save")
        sptag_index.Save(PATH_INDEX)
    #sptag_index.Add(np.asmatrix(matrix), len(matrix), False)

def fill_index(sptag_index):
    # Read each dataset in the folder and insert its vectors in the index
    for dataset_name in sorted(listdir(PATH_DATASETS))[DATASETS_IN_RAM:]:
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            datareader = csv.reader(dataset)
            DEBUG(['Loading', dataset_name])
            
            # Add to index
            for vector in datareader:
                to_add = np.asmatrix(np.array(vector).astype(np.longdouble))
                sptag_index.Add(to_add, len(to_add), False)

def build_and_save_sptag_index(sptag_index):
    sptag_index.Save(PATH_INDEX)

def main():
    sptag_index = create_sptag_index()
    train_index(sptag_index)
    #fill_index(sptag_index)
    #build_and_save_sptag_index(sptag_index)

if __name__ == "__main__":
    main()