import faiss
import numpy as np
from mocks import MOCKED_QUERY_VECTOR_1 as MOCKED_QUERY_VECTOR
from chronometer import Chronometer
from config import PATH_IMAGES, PATH_INDEX, SEARCH_PARAMS

def load_faiss_index():                          
    return faiss.read_index(PATH_INDEX)

def query_faiss_index(faiss_index, query_vector, nearest_neighbors, chronometer: Chronometer):
    # Function for tuned parameter: faiss.ParameterSpace().set_index_parameter(faiss_index, "nprobe", 128)
    if SEARCH_PARAMS:
        faiss.ParameterSpace().set_index_parameters(faiss_index, SEARCH_PARAMS)
    chronometer.begin_time_window()
    d, i = faiss_index.search(np.asmatrix(query_vector).astype(np.float32), nearest_neighbors)
    chronometer.end_time_window()
    return i

def get_images_by_id(id_list):
    images = open(PATH_IMAGES, 'r')
    lines = images.readlines()
    return [lines[i] for i in id_list]

def main():
    chronometer = Chronometer()
    faiss_index = load_faiss_index()
    faiss_result_id = query_faiss_index(faiss_index, MOCKED_QUERY_VECTOR, 100, chronometer)[0]
    faiss_result_images = get_images_by_id(faiss_result_id)
    print(faiss_result_id)
    print(''.join(faiss_result_images))
    chronometer.get_total_time()

if __name__ == "__main__":
    main()