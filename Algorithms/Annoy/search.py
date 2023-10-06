from mocks import MOCKED_QUERY_VECTOR_1 as MOCKED_QUERY_VECTOR
from annoy import AnnoyIndex
import numpy as np
from chronometer import Chronometer

from config import DISTANCE, PATH_IMAGES, PATH_INDEX, get_dataset_columns

def load_annoy_index():                          
    columns_number = get_dataset_columns()
    annoy_index = AnnoyIndex(columns_number, DISTANCE)  
    annoy_index.load(PATH_INDEX) 
    return annoy_index

def query_annoy_index(annoy_index, query_vector, nearest_neighbors, chronometer: Chronometer):
    # VECTOR: print(annoy_index.get_nns_by_vector(query_vector, nearest_neighbors))
    chronometer.begin_time_window()
    result = annoy_index.get_nns_by_vector(MOCKED_QUERY_VECTOR, nearest_neighbors)
    chronometer.end_time_window()
    return result

def get_images_by_id(id_list):
    images = open(PATH_IMAGES, 'r')
    lines = images.readlines()
    return [lines[i] for i in id_list]

def main():
    chronometer = Chronometer()
    annoy_index = load_annoy_index()
    annoy_result_id = query_annoy_index(annoy_index, [], 100, chronometer)
    annoy_result_images = get_images_by_id(annoy_result_id)
    print(annoy_result_id)
    print(''.join(annoy_result_images))
    chronometer.get_total_time()

if __name__ == "__main__":
    main()