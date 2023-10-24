import csv
import numpy as np
import SPTAG
from os import listdir

from config import DATASETS_USED_TO_TRAIN, SPANN, DEBUG, DISTANCE, NUMBER_OF_THREADS, PATH_INDEX, PATH_DATASETS, get_dataset_columns
from chronometer import Chronometer

def create_sptag_index():
    if SPANN:
        sptag_index = SPTAG.AnnIndex('SPANN', 'Float', get_dataset_columns())
        sptag_index.SetBuildParam("IndexAlgoType", "BKT", "Base")
        sptag_index.SetBuildParam("IndexDirectory", PATH_INDEX, "Base")
        sptag_index.SetBuildParam("DistCalcMethod", DISTANCE, "Base")

        sptag_index.SetBuildParam("isExecute", "true", "SelectHead")
        sptag_index.SetBuildParam("NumberOfThreads", NUMBER_OF_THREADS, "SelectHead")
        sptag_index.SetBuildParam("Ratio", "1.0", "SelectHead") # sptag_index.SetBuildParam("Count", "200", "SelectHead")

        sptag_index.SetBuildParam("isExecute", "true", "BuildHead")
        sptag_index.SetBuildParam("RefineIterations", "3", "BuildHead")
        sptag_index.SetBuildParam("NumberOfThreads", NUMBER_OF_THREADS, "BuildHead")

        sptag_index.SetBuildParam("isExecute", "true", "BuildSSDIndex")
        sptag_index.SetBuildParam("BuildSsdIndex", "true", "BuildSSDIndex")
        sptag_index.SetBuildParam("PostingPageLimit", "12", "BuildSSDIndex")
        sptag_index.SetBuildParam("SearchPostingPageLimit", "12", "BuildSSDIndex")
        sptag_index.SetBuildParam("NumberOfThreads", NUMBER_OF_THREADS, "BuildSSDIndex")
        sptag_index.SetBuildParam("InternalResultNum", "32", "BuildSSDIndex")
        sptag_index.SetBuildParam("SearchInternalResultNum", "64", "BuildSSDIndex")
    else:
        sptag_index = SPTAG.AnnIndex('BKT', 'Float', get_dataset_columns())
        sptag_index.SetBuildParam("IndexDirectory", PATH_INDEX, "Index")
        sptag_index.SetBuildParam("DistCalcMethod", DISTANCE, "Index")
    return sptag_index
    
def train_index(sptag_index, chronometer: Chronometer):

    # Read ${DATASETS_USED_TO_TRAIN} datasets in ${PATH_DATASETS}, insert its vectors in the index and train it
    matrix = []
    for dataset_name in sorted(listdir(PATH_DATASETS))[:DATASETS_USED_TO_TRAIN]:
        with np.load(PATH_DATASETS + dataset_name) as fp:
            DEBUG(['Loading and training', dataset_name])
            data = fp['arr_0']
                
            if len(matrix) == 0:
                matrix = data
            else: 
                matrix = np.concatenate((matrix, data))

    chronometer.begin_time_window()
    sptag_index.Add(np.asmatrix(matrix).astype(np.float32), len(matrix), False)
    sptag_index.Build(np.asmatrix(matrix).astype(np.float32), len(matrix), False)
    chronometer.end_time_window()

def fill_index(sptag_index, chronometer: Chronometer):
    # Read the remaining dataset in ${PATH_DATASETS} and insert its vectors in the index
    for dataset_name in sorted(listdir(PATH_DATASETS))[DATASETS_USED_TO_TRAIN:]:
        with np.load(PATH_DATASETS + dataset_name) as fp:
            DEBUG(['Loading and training', dataset_name])
            data = fp['arr_0']
        
            chronometer.begin_time_window()
            sptag_index.Add(np.asmatrix(data).astype(np.float32), len(data), False)
            chronometer.end_time_window()

def build_and_save_sptag_index(sptag_index):
    sptag_index.Save(PATH_INDEX)

def main():
    chronometer = Chronometer()
    sptag_index = create_sptag_index()
    train_index(sptag_index, chronometer)
    fill_index(sptag_index, chronometer)
    build_and_save_sptag_index(sptag_index)
    chronometer.get_total_time()

if __name__ == "__main__":
    main()