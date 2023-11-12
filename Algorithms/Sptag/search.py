from mocks import MOCKED_QUERY_VECTOR_1 as MOCKED_QUERY_VECTOR
import SPTAG
import numpy as np
from chronometer import Chronometer

from config import PATH_IMAGES, PATH_INDEX, NEIGHBORS_NUMBER

def load_sptag_index():                          
    return SPTAG.AnnIndex.Load(PATH_INDEX)

def query_sptag_index(sptag_index, query_vector, nearest_neighbors, chronometer: Chronometer):
    chronometer.begin_time_window()
    tmp = sptag_index.Search(query_vector, nearest_neighbors)
    chronometer.end_time_window()
    return tmp[0]

def get_images_by_id(id_list):
    images = open(PATH_IMAGES, 'r')
    lines = images.readlines()
    return [lines[i] for i in id_list]

def main():
    chronometer = Chronometer()
    sptag_index = load_sptag_index()
    sptag_result_id = query_sptag_index(sptag_index, np.array(MOCKED_QUERY_VECTOR).astype(np.float32), NEIGHBORS_NUMBER, chronometer)
    sptag_result_images = get_images_by_id(sptag_result_id)
    print(sptag_result_id)
    print(''.join(sptag_result_images))
    chronometer.get_total_time()

if __name__ == "__main__":
    main()