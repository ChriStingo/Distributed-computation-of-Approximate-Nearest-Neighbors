import numpy as np
from pymilvus import Collection, connections
from chronometer import Chronometer

from config import MOCKED_QUERY_VECTOR, PATH_IMAGES

def load_milvus_index():                          
    connections.connect("default", host="localhost", port="19530")
    collection = Collection("images") 
    return collection

def query_milvus_index(collection, query_vector, nearest_neighbors, chronometer: Chronometer):
    collection.load()
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    chronometer.begin_time_window()
    results = collection.search(
        data=[MOCKED_QUERY_VECTOR], 
        anns_field="images", 
        param=search_params, 
        limit=nearest_neighbors, 
        expr=None,
        consistency_level="Strong"
    )
    chronometer.end_time_window()
    return results[0].ids

def get_images_by_id(id_list):
    images = open(PATH_IMAGES, 'r')
    lines = images.readlines()
    return [lines[i] for i in id_list]

def main():
    chronometer = Chronometer()
    collection = load_milvus_index()
    milvus_result_id = query_milvus_index(collection, [], 5)
    collection.release()
    milvus_result_images = get_images_by_id(milvus_result_id)
    print(milvus_result_id)
    print(''.join(milvus_result_images))
    chronometer.get_total_time()

if __name__ == "__main__":
    main()