import argparse
import os
import csv
import numpy as np
np.set_printoptions(precision=5, suppress=True)

# py hw1.py --INPUT_FILE=testfile.txt --N=2 --LAMBDA=0

def matrix_A(x,base):
    A = np.zeros((len(x), base))
    print(np.shape(A))
    for value_idx in range(len(x)):
        for base_idx in range(base-1,-1,-1):
            A[value_idx][base-base_idx-1] = x[value_idx]**base_idx

    return A

def LU_decomposition(A):
    row_size,col_size = np.shape(A)
    print(row_size) 
    print(col_size)
    L = np.identity(row_size)
    for row in range(row_size):
        L_tmp = np.identity(row_size)
        for col in range(col_size):
            if col < row and A[row][col] != 0:
                L_tmp[row][col] = (-1) * A[row][col] / A[col][col]
                A = np.matmul(L_tmp,A)
                L[row][col] = (-1) * L_tmp[row][col]    #change back
                L_tmp = np.identity(row_size)

    return L,A

if __name__ == '__main__':
    #parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--INPUT_FILE", type = str  )
    parser.add_argument("--N", type = int  )
    parser.add_argument("--LAMBDA", type = int  )
    args = parser.parse_args()
    INPUT_FILE = args.INPUT_FILE
    N = args.N
    LAMBDA = args.LAMBDA
    point = []
    with open(INPUT_FILE, 'r') as file:
        file_rows = csv.reader(file, delimiter = ',')
        for idx, row in enumerate(file_rows):
            point.append([float(row[0]),float(row[1])])

    point = np.array(point)
    A = matrix_A(point[:,0],N)
    grim_matrix = np.matmul(A.T,A)
    lambda_I = LAMBDA*np.identity(np.shape(grim_matrix)[0])
    grim_matrix_add_lambda_I = np.add(grim_matrix,lambda_I)
    test_matrix = np.array([[3,-1,2],[6,-1,5],[-9,7,3]])
    L,U = LU_decomposition(grim_matrix_add_lambda_I)
    print(L)
    print(U)