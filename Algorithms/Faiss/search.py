import faiss
import numpy as np

from config import MOCKED_QUERY_VECTOR, PATH_IMAGES, PATH_INDEX

def load_faiss_index():                          
    return faiss.read_index(PATH_INDEX)

def query_faiss_index(faiss_index, query_vector, nearest_neighbors):
    # MOCKED_QUERY_VECTOR already normalized with faiss.normalize_L2
    d, i = faiss_index.search(np.asmatrix(MOCKED_QUERY_VECTOR).astype(np.float32), nearest_neighbors)
    return i

def get_images_by_id(id_list):
    images = open(PATH_IMAGES, 'r')
    lines = images.readlines()
    return [lines[i] for i in id_list]

def main():
    faiss_index = load_faiss_index()
    faiss_result_id = query_faiss_index(faiss_index, [], 5)[0]
    faiss_result_images = get_images_by_id(faiss_result_id)
    print(faiss_result_id)
    print(''.join(faiss_result_images))

if __name__ == "__main__":
    main()