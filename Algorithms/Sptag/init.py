import csv
import numpy as np
import sptag
from os import listdir

from config import DEBUG, DISTANCE, PATH_INDEX, PATH_DATASETS, get_dataset_columns

def create_sptag_index():
    sptag_index = SPTAG.AnnIndex('BKT', 'Float', get_dataset_columns)
    # Set the thread number to speed up the build procedure in parallel 
    sptag_index.SetBuildParam("NumberOfThreads", '4', "Index")
    # Set the distance type. Currently SPTAG only support Cosine and L2 distances. Here Cosine distance is not the Cosine similarity. The smaller Cosine distance it is, the better.
    sptag_index.SetBuildParam("DistCalcMethod", 'L2', "Index") 
    
def fill_index(sptag_index):
    # Read each dataset in the folder and insert its vectors in the index
    offset = 0
    for dataset_name in listdir(PATH_DATASETS):
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            datareader = csv.reader(dataset)
            DEBUG(['Loading', dataset_name])
            
            # Add to index
            rows = 0
            for index, vector in enumerate(datareader):
                rows += 1
                sptag_index.BuildWithMetaData(np.array(vector, dtype=np.float32), str(offset + index), 1, False, False)
            offset += rows

def build_and_save_sptag_index(sptag_index):
    sptag_index.save(PATH_INDEX)

def main():
    sptag_index = create_sptag_index()
    fill_index(sptag_index)
    build_and_save_sptag_index(sptag_index)

if __name__ == "__main__":
    main()