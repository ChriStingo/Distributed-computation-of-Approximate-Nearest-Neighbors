from annoy import AnnoyIndex
import numpy as np

from config import DISTANCE, PATH_IMAGES, PATH_INDEX, get_dataset_columns

def load_annoy_index():                          
    columns_number = get_dataset_columns()
    annoy_index = AnnoyIndex(columns_number, DISTANCE)  
    annoy_index.load(PATH_INDEX) 
    return annoy_index

def query_annoy_index(annoy_index, query_vector, nearest_neighbors):
    # VECTOR: print(annoy_index.get_nns_by_vector(query_vector, nearest_neighbors))
    return annoy_index.get_nns_by_item(1, nearest_neighbors)

def get_images_by_id(id_list):
    images = open(PATH_IMAGES, 'r')
    lines = images.readlines()
    return [lines[i] for i in id_list]

def main():
    annoy_index = load_annoy_index()
    annoy_result_id = query_annoy_index(annoy_index, [], 5)
    annoy_result_images = get_images_by_id(annoy_result_id)
    print(annoy_result_id)
    print(''.join(annoy_result_images))

if __name__ == "__main__":
    main()