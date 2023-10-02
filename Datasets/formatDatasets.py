import numpy as np
import csv
import sys

def read(path_to_csv):
    matrix = []
    with open(path_to_csv, "r") as dataset:
        datareader = csv.reader(dataset)
        
        for vector in datareader:
            matrix.append(np.array(vector, dtype=np.longdouble))

    return matrix

def save(path_to_csv, matrix):
    np.savez(path_to_csv, matrix)

def main():
    matrix = read(sys.argv[1])
    save(sys.argv[1], matrix)

if __name__ == "__main__":
    main()