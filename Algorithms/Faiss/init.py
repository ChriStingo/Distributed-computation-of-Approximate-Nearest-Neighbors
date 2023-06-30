import csv
import faiss
import numpy as np
from os import listdir

from config import DATASETS_IN_RAM, DEBUG, M, D, NBITS, NLIST, PATH_INDEX, PATH_DATASETS, get_dataset_columns

def create_faiss_index():
    coarse_quantizer = faiss.IndexFlatIP(D)
    return faiss.IndexIVFPQ(coarse_quantizer, D, NLIST, M, NBITS)   

def train_index(faiss_index):
     # Read each dataset in the folder and insert its vectors in the index
    matrix = []
    for dataset_name in sorted(listdir(PATH_DATASETS))[:DATASETS_IN_RAM]:
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            datareader = csv.reader(dataset)
            DEBUG(['Loading and training', dataset_name])
            
            # Add to index
            for vector in datareader:
                matrix.append(np.array(vector).astype(np.float32))
    norm_matrix = np.asmatrix(matrix)
    faiss.normalize_L2(norm_matrix)
    faiss_index.train(norm_matrix)
    faiss_index.add(norm_matrix)

def fill_index(faiss_index):
    # Read each dataset in the folder and insert its vectors in the index
    for dataset_name in sorted(listdir(PATH_DATASETS))[DATASETS_IN_RAM:]:
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            datareader = csv.reader(dataset)
            DEBUG(['Loading', dataset_name])
            
            # Add to index
            for vector in datareader:
                norm_vector = np.asmatrix(vector).astype(np.float32)
                faiss.normalize_L2(norm_vector)
                faiss_index.add(norm_vector)

def build_and_save_faiss_index(faiss_index):
    faiss.write_index(faiss_index, PATH_INDEX)

def main():
    faiss_index = create_faiss_index()
    train_index(faiss_index)
    fill_index(faiss_index)
    build_and_save_faiss_index(faiss_index)

if __name__ == "__main__":
    main()