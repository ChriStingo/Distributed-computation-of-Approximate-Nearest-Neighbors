import csv
from os import listdir
import numpy as np
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections

from config import DEBUG, INDEX_PARAMS, PATH_DATASETS, get_dataset_columns
from chronometer import Chronometer

def create_milvus_collection():
    connections.connect("default", host="localhost", port="19530")

    image_id = FieldSchema(
        name="image_id", 
        dtype=DataType.INT64, 
        is_primary=True, 
    )
    vector = FieldSchema(
        name="images", 
        dtype=DataType.FLOAT_VECTOR, 
        dim=get_dataset_columns()
    )
    schema = CollectionSchema(
        fields=[image_id, vector], 
        description="Images"
    )
    collection_name = "images"
    collection = Collection(
        name=collection_name, 
        schema=schema, 
        using='default', 
        shards_num=2,
        consistency_level="Strong"
        )
    return collection

def save_milvus_index(collection, chronometer: Chronometer):
    chronometer.begin_time_window()
    collection.create_index(
        field_name="images", 
        index_params=INDEX_PARAMS
    )
    collection.load()
    chronometer.end_time_window()
    
def fill_index(collection, chronometer: Chronometer):
    # Read each dataset in ${PATH_DATASETS} and insert its vectors in the index
    idx = 0
    for dataset_name in sorted(listdir(PATH_DATASETS)):
        try:
            with open(PATH_DATASETS + dataset_name, "r") as dataset:
                matrix = []
                datareader = csv.reader(dataset)
                DEBUG(['Loading and training', dataset_name])
                idx += 1

                for vector in datareader:
                    matrix.append(list(map(float, vector)))
        except:
            print("Bad chars in file:", dataset_name)            
        
        chronometer.begin_time_window()
        collection.insert([[i for i in range(len(matrix)*idx, len(matrix)*(idx+1))], matrix])
        chronometer.end_time_window()

def main():
    chronometer = Chronometer()
    collection = create_milvus_collection()
    fill_index(collection, chronometer)
    save_milvus_index(collection, chronometer)
    chronometer.get_total_time()

if __name__ == "__main__":
    main()