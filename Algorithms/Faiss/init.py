import csv
import faiss
import numpy as np
from os import listdir

from config import DEBUG, DISTANCE, PATH_INDEX, PATH_DATASETS, get_dataset_columns

def create_faiss_index():
    faiss.IndexFlatL2(get_dataset_columns())
    return faiss.IndexFlatL2(get_dataset_columns())
    
def fill_index(faiss_index):
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
                faiss_index.add(np.asmatrix(np.array(vector, dtype=np.longdouble)))
            offset += rows

def build_and_save_faiss_index(faiss_index):
    faiss.write_index(faiss_index, PATH_INDEX)

def main():
    faiss_index = create_faiss_index()
    fill_index(faiss_index)
    build_and_save_faiss_index(faiss_index)

if __name__ == "__main__":
    main()