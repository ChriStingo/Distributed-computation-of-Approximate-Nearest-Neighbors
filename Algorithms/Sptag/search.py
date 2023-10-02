from mocks import MOCKED_QUERY_VECTOR_1
import SPTAG
import numpy as np
from chronometer import Chronometer

from config import PATH_IMAGES, PATH_INDEX

def load_sptag_index():                          
    return SPTAG.AnnIndex.Load(PATH_INDEX)

def query_sptag_index(sptag_index, query_vector, nearest_neighbors, chronometer: Chronometer):
    # [0]: nearest k vector ids
    # [1]: nearest k vector distances
    # [2]: nearest k vector metadatas
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
    sptag_result_id = query_sptag_index(sptag_index, np.array(MOCKED_QUERY_VECTOR_1).astype(np.float32), 100, chronometer)
    sptag_result_images = get_images_by_id(sptag_result_id)
    print(sptag_result_id)
    print(''.join(sptag_result_images))
    chronometer.get_total_time()

if __name__ == "__main__":
    main()