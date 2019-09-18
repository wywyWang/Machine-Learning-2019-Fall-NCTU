import argparse
import os
import csv
import numpy as np

if __name__ == '__main__':
    #parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type = str  )
    parser.add_argument("--n", type = int  )
    parser.add_argument("--lambda", type = int  )
    args = parser.parse_args()

    point = []
    with open(args.input_file, 'r') as file:
        file_rows = csv.reader(file, delimiter = ',')
        for idx, row in enumerate(file_rows):
            point.append([row[0],row[1]])

    point = np.array(point)
    print(point[:,0])
