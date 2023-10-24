from mocks import MOCKED_QUERY_VECTOR_1 as MOCKED_QUERY_VECTOR
import SPTAG
import SPTAGClient
import numpy as np
import heapq
import time
from chronometer import Chronometer
from config import PATH_IMAGES, ADDR_IP_AGGREGATOR, ADDR_PORT_AGGREGATOR 

def load_sptag_index():
    index = SPTAGClient.AnnClient(ADDR_IP_AGGREGATOR, ADDR_PORT_AGGREGATOR)
    print("Trying connection with", ADDR_IP_AGGREGATOR, ADDR_PORT_AGGREGATOR)
    while not index.IsConnected():
        time.sleep(1)
    print("Index connected")
    index.SetTimeoutMilliseconds(18000)
    index.SetSearchParam("MaxCheck", '32768')
    return index

def query_sptag_index(sptag_index, query_vector, nearest_neighbors, chronometer: Chronometer):
    # [0]: nearest k vector ids
    # [1]: nearest k vector distances
    # [2]: nearest k vector metadatas
    chronometer.begin_time_window()
    tmp = sptag_index.Search(query_vector, nearest_neighbors, 'Float', True)
    chronometer.end_time_window()
    return tmp[1], tmp[2]

def get_images_by_id(distances, metadata, chronometer: Chronometer):
    images = open(PATH_IMAGES, 'r')
    lines = images.readlines()

    tmpDistances = []
    tmpMetadata = []
    for index, i in enumerate(metadata):
        try:
            tmpMetadata.append(int(i.decode()))
            tmpDistances.append(distances[index])
        except:
            print("Bad/Corrupted value", i)
            continue

    chronometer.begin_time_window()
    zipped_list = heapq.nsmallest(100, zip(tmpDistances, tmpMetadata), key=lambda x: x[0]) # Partial sort

    tmpLinks = []
    for _, meta in zipped_list:
        tmpLinks.append(lines[meta])
    chronometer.end_time_window()
    
    return tmpLinks

def main():
    chronometer = Chronometer()
    sptag_index = load_sptag_index()
    sptag_result_distances, sptag_result_metadata = query_sptag_index(sptag_index, np.array(MOCKED_QUERY_VECTOR).astype(np.float32), 100, chronometer)
    sptag_result_images = get_images_by_id(sptag_result_distances, sptag_result_metadata, chronometer)
    print(sptag_result_metadata)
    print(''.join(sptag_result_images))
    chronometer.get_total_time()

if __name__ == "__main__":
    main()
