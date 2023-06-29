import csv
import numpy as np
import SPTAG
from os import listdir

from config import DEBUG, DISTANCE, NUMBER_OF_THREADS, PATH_INDEX, PATH_DATASETS, get_dataset_columns

def create_sptag_index():
    sptag_index = SPTAG.AnnIndex('SPANN', 'Float', get_dataset_columns())
    # Set the thread number to speed up the build procedure in parallel 
    sptag_index.SetBuildParam("NumberOfThreads", NUMBER_OF_THREADS, "Index")
    # Set the distance type. Currently SPTAG only support Cosine and L2 distances. Here Cosine distance is not the Cosine similarity. The smaller Cosine distance it is, the better.
    sptag_index.SetBuildParam("DistCalcMethod", DISTANCE, "Index") 
    
def fill_index(sptag_index):
    # Read each dataset in the folder and insert its vectors in the index
    matrix = []
    for dataset_name in listdir(PATH_DATASETS):
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            datareader = csv.reader(dataset)
            DEBUG(['Loading', dataset_name])
            
            # Add to index
            for vector in datareader:
                matrix.append(np.array(vector, dtype=np.longdouble))

    m = ''
    for i in range(len(matrix)):
        m += str(i) + '\n'

    sptag_index.BuildWithMetaData(np.asmatrix(matrix), m, len(matrix), False, False)
    sptag_index.AddWithMetaData(np.asmatrix(matrix), m, len(matrix), False, False)

def build_and_save_sptag_index(sptag_index):
    sptag_index.Save(PATH_INDEX)

def main():
    sptag_index = create_sptag_index()
    fill_index(sptag_index)
    build_and_save_sptag_index(sptag_index)

if __name__ == "__main__":
    main()