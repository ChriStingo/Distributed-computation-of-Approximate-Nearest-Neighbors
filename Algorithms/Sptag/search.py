import SPTAG
import numpy as np

from config import MOCKED_QUERY_VECTOR, PATH_IMAGES, PATH_INDEX

def load_sptag_index():                          
    return SPTAG.AnnIndex.Load(PATH_INDEX)

def query_sptag_index(sptag_index, query_vector, nearest_neighbors):
    # [0]: nearest k vector ids
    # [1]: nearest k vector distances
    # [2]: nearest k vector metadatas
    tmp = sptag_index.Search(query_vector, nearest_neighbors)
    return tmp[0]

def get_images_by_id(id_list):
    images = open(PATH_IMAGES, 'r')
    lines = images.readlines()
    return [lines[i] for i in id_list]

def main():
    sptag_index = load_sptag_index()
    sptag_result_id = query_sptag_index(sptag_index, np.array(MOCKED_QUERY_VECTOR).astype(np.float32), 5)
    sptag_result_images = get_images_by_id(sptag_result_id)
    print(sptag_result_id)
    print(''.join(sptag_result_images))

if __name__ == "__main__":
    main()