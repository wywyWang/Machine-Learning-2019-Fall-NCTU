import argparse
import os
import csv
import numpy as np
import matplotlib.pyplot as plt 
np.set_printoptions(precision=5, suppress=True)

# py hw1.py --INPUT_FILE=testfile.txt --N=2 --LAMBDA=2

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
                L[row][col] = (-1) * L_tmp[row][col]    
                L_tmp = np.identity(row_size)

    # print(L)
    # print(A)
    return L,A

def substitution(L,U,b):
    # Ly = b,solve y
    y = [b_value for b_value in b]
    for row in range(len(L)):
        for col in range(row):
            y[row] -= y[col] * L[row][col]
        y[row] /= L[row][row]
    # print(y)

    # Ux = y, solve x
    x = [y_value for y_value in y]
    for row in range(len(U)-1,-1,-1):
        for col in range(len(U)-1,row,-1):
            x[row] -= x[col] * U[row][col]
        x[row] /= U[row][row]
    # print(x)
    return x

def print_result(A,x,y,ori_x,model):
    total_error = 0
    cal_b = np.matmul(A,x)
    print(model + ': ')
    
    # Print fitting line
    print("Fitting line : ")
    for row in range(np.shape(x)[0]):
        if row != np.shape(x)[0] - 1:
            if row != 0:                   #check first
                if x[row] >= 0:            #check value
                    print("+", end=' ')
                    print(str(x[row]) + ' X^' + str((np.shape(x)[0]-1) - row), end=' ')
                else:
                    print("-", end=' ')
                    print(str((-1) * x[row]) + ' X^' + str((np.shape(x)[0]-1) - row), end=' ')
            else:
                if x[row] >= 0:
                    print(str(x[row]) + ' X^' + str((np.shape(x)[0]-1) - row), end=' ')
                else:
                    print("-", end=' ')
                    print(str((-1) * x[row]) + ' X^' + str((np.shape(x)[0]-1) - row), end=' ')
        else:
            if x[row] >= 0:
                print("+", end=' ')
                print(str(x[row]))
            else:
                print("-", end=' ')
                print(str((-1) * x[row]))

    # Calculate total error
    for row in range(np.shape(cal_b)[0]):
        tmp_error = (cal_b[row] - y[row]) ** 2
        total_error += tmp_error
    print("Total error: ",total_error)

    fig = plt.figure()
    plt.title(model)
    plt.scatter(ori_x, y, c = 'red')
    plt.plot(ori_x,cal_b)
    # plt.xlim((min(ori_x), max(ori_x)))
    fig.savefig(model + '.png')

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

    #Rlse
    print("LSE:")
    ori_x = point[:,0]
    b = point[:,1]
    A = matrix_A(point[:,0],N)
    grim_matrix = np.matmul(A.T,A)
    lambda_I = LAMBDA*np.identity(np.shape(grim_matrix)[0])
    grim_matrix_add_lambda_I = np.add(grim_matrix,lambda_I)

    # test_matrix2 = np.array([[2,1,3,2],[0,3,-2,1],[2,1,-1,1],[2,1,2,2]])
    # test_b = np.array([1,3,2,4])
    # L,U = LU_decomposition(test_matrix2)
    # result = substitution(L,U,test_b)

    L,U = LU_decomposition(grim_matrix_add_lambda_I)
    
    A_transpose_b = np.matmul(A.T,b)
    result = substitution(L,U,A_transpose_b)
    print_result(A,result,b,ori_x,'LSE')