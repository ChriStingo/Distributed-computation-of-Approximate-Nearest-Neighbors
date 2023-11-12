from distributed_faiss.client import IndexClient
from distributed_faiss.index_cfg import IndexCfg
import numpy as np
import faiss
from mocks import MOCKED_QUERY_VECTOR_1 as MOCKED_QUERY_VECTOR
from chronometer import Chronometer
from config import INDEX_FACTORY, DISTANCE, INDEX_ID, PATH_IMAGES, PATH_INDEX, get_dataset_columns, TRAIN_RATIO, SEARCH_PARAMS, NEIGHBORS_NUMBER

def load_faiss_index():
    faiss_client = IndexClient("DISCOVERY_CONFIG.txt")
    idx_cfg = IndexCfg(
        faiss_factory=INDEX_FACTORY, 
        dim=get_dataset_columns(),
        train_ratio=TRAIN_RATIO,
        metric=DISTANCE,
        index_storage_dir=PATH_INDEX,
    )
    faiss_client.load_index(INDEX_ID, idx_cfg)
    return faiss_client

def query_faiss_index(faiss_index, query_vector, nearest_neighbors, chronometer: Chronometer):
    if SEARCH_PARAMS:
        faiss_index.set_index_parameters(INDEX_ID, SEARCH_PARAMS)
    chronometer.begin_time_window()
    scores, meta = faiss_index.search(query_vector, nearest_neighbors, INDEX_ID, return_embeddings=False)
    chronometer.end_time_window()
    return meta[0]

def get_images_by_id(id_list):
    images = open(PATH_IMAGES, 'r')
    lines = images.readlines()
    return [lines[i] for i in id_list]

def main():
    chronometer = Chronometer()
    faiss_index = load_faiss_index()
    faiss_result_id = query_faiss_index(faiss_index, np.asmatrix(MOCKED_QUERY_VECTOR).astype(np.float32), NEIGHBORS_NUMBER, chronometer)
    faiss_result_images = get_images_by_id(faiss_result_id)
    print(faiss_result_id)
    print(''.join(faiss_result_images))
    chronometer.get_total_time()

if __name__ == "__main__":
    main()