import csv
import time
import numpy as np
from os import listdir
from distributed_faiss.client import IndexClient
from distributed_faiss.index_state import IndexState
from distributed_faiss.index_cfg import IndexCfg
from config import DEBUG, D, INDEX_FACTORY, INDEX_TYPE, M, DISTANCE, NBITS, PATH_DATASETS, INDEX_ID, PATH_INDEX
from chronometer import Chronometer

def create_faiss_index():
    faiss_index = IndexClient("DISCOVERY_CONFIG.txt")
    idx_cfg = IndexCfg(
        index_builder_type=INDEX_TYPE,
        faiss_factory=INDEX_FACTORY, 
        dim=D,
        train_data_ratio=1.0,
        centroids=M,
        metric=DISTANCE,
        nprobe=NBITS,
        index_storage_dir=PATH_INDEX,
        infer_centroids=True
    )
    faiss_index.create_index(INDEX_ID, idx_cfg)
    return faiss_index
    
def fill_index(faiss_index, chronometer: Chronometer):
    # Read datasets in ${PATH_DATASETS}, insert its vectors in the index
    idx = 0
    for dataset_name in sorted(listdir(PATH_DATASETS)):
        try:
            with open(PATH_DATASETS + dataset_name, "r") as dataset:
                matrix = []
                datareader = csv.reader(dataset)
                DEBUG(['Loading and training', dataset_name])
                idx += 1
                
                for vector in datareader:
                    matrix.append(np.array(vector).astype(np.float32))

                chronometer.begin_time_window()
                faiss_index.add_index_data(INDEX_ID, np.asmatrix(matrix).astype(np.float32), [i for i in range(len(matrix)*(idx-1), len(matrix)*idx)], train_async_if_triggered=False)
                chronometer.end_time_window()
        except:
            print("Bad chars in file:", dataset_name)                    
    

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