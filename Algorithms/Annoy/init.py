import csv
import numpy as np
from annoy import AnnoyIndex
from os import listdir

from config import DEBUG, DISTANCE, NUMBER_OF_TREES, PATH_INDEX, PATH_DATASETS, get_dataset_columns

def create_annoy_index():
    return AnnoyIndex(get_dataset_columns(), DISTANCE)
    
def fill_index(annoy_index):
    # Read each dataset in ${PATH_DATASETS} and insert its vectors in the index
    index = 0
    for dataset_name in sorted(listdir(PATH_DATASETS)):
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            datareader = csv.reader(dataset)
            DEBUG(['Loading', dataset_name])
            
            for vector in datareader:
                annoy_index.add_item(index, np.array(vector).astype(dtype=np.longdouble))
                index += 1

def build_and_save_annoy_index(annoy_index):
    annoy_index.build(NUMBER_OF_TREES)
    annoy_index.save(PATH_INDEX)

def main():
    annoy_index = create_annoy_index()
    fill_index(annoy_index)
    build_and_save_annoy_index(annoy_index)

if __name__ == "__main__":
    main()