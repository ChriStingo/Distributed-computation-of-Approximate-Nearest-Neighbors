import csv
from os import listdir
import numpy as np
from pymilvus import CollectionSchema, FieldSchema, DataType, Collection, connections
from config import DEBUG, INDEX_PARAMS, PATH_DATASETS, get_dataset_columns

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

def save_milvus_index(collection):
    collection.create_index(
        field_name="images", 
        index_params=INDEX_PARAMS
    )
    collection.load()
    
def fill_index(collection):
    # Read each dataset in ${PATH_DATASETS} and insert its vectors in the index
    for idx, dataset_name in enumerate(sorted(listdir(PATH_DATASETS))[:2]):
        with open(PATH_DATASETS + dataset_name, "r") as dataset:
            matrix = []
            datareader = csv.reader(dataset)
            DEBUG(['Loading and training', dataset_name])
            
            for vector in datareader:
                matrix.append(list(map(float, vector)))
                
        collection.insert([[i for i in range(len(matrix)*idx, len(matrix)*(idx+1))], matrix])

def main():
    collection = create_milvus_collection()
    fill_index(collection)
    save_milvus_index(collection)

if __name__ == "__main__":
    main()