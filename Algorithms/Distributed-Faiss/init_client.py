import csv
import time
import numpy as np
from os import listdir
from distributed_faiss.client import IndexClient
from distributed_faiss.index_state import IndexState
from distributed_faiss.index_cfg import IndexCfg
import faiss
from config import DEBUG, INDEX_FACTORY, DISTANCE, PATH_DATASETS, INDEX_ID, PATH_INDEX, get_dataset_columns, TRAIN_RATIO
from chronometer import Chronometer

def create_faiss_index():
    faiss_index = IndexClient("DISCOVERY_CONFIG.txt")
    idx_cfg = IndexCfg(
        faiss_factory=INDEX_FACTORY, 
        dim=get_dataset_columns(),
        train_ratio=TRAIN_RATIO,
        metric=DISTANCE,
        index_storage_dir=PATH_INDEX,
    )
    faiss_index.create_index(INDEX_ID, idx_cfg)
    return faiss_index
    
def fill_index(faiss_index, chronometer: Chronometer):
    # Read datasets in ${PATH_DATASETS}, insert its vectors in the index
    idx = 0
    for dataset_name in sorted(listdir(PATH_DATASETS)):
        with np.load(PATH_DATASETS + dataset_name) as fp:
            DEBUG(['Loading and training', dataset_name])
            data = fp['arr_0']
            idx += 1

            norm_matrix = np.asmatrix(data).astype(np.float32)
            faiss.normalize_L2(norm_matrix)

            chronometer.begin_time_window()
            faiss_index.add_index_data(INDEX_ID, norm_matrix, [i for i in range(len(norm_matrix)*(idx-1), len(norm_matrix)*idx)], train_async_if_triggered=False)
            chronometer.end_time_window()                
    

def build_and_save_faiss_index(faiss_index, chronometer: Chronometer):
    chronometer.begin_time_window()
    faiss_index.sync_train(INDEX_ID)
    while faiss_index.get_state(INDEX_ID) != IndexState.TRAINED:
        # wait 1s to avoid extreme polling
        time.sleep(1)
    faiss_index.save_index(INDEX_ID)
    chronometer.end_time_window()

def main():
    chronometer = Chronometer()
    faiss_index = create_faiss_index()
    fill_index(faiss_index, chronometer)
    build_and_save_faiss_index(faiss_index, chronometer)
    chronometer.get_total_time()

if __name__ == "__main__":
    main()