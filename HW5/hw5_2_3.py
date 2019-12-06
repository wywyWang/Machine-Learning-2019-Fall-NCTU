from svmutil import *
import numpy as np
import csv
from scipy.spatial.distance import cdist, pdist, squareform

def read_csv():
    with open('./data/X_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        x_train = list(csv_reader)
        x_train = [[float(y) for y in x] for x in x_train]

    with open('./data/Y_train.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        y_train_2d = list(csv_reader)
        y_train = [y for x in y_train_2d for y in x]
        y_train = [ int(x) for x in y_train ]

    with open('./data/X_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        x_test = list(csv_reader)
        x_test = [[float(y) for y in x] for x in x_test]

    with open('./data/Y_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        y_test_2d = list(csv_reader)
        y_test = [y for x in y_test_2d for y in x]
        y_test = [ int(x) for x in y_test ]

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def combine_kernel(x_train, x_test, best_pair):
    gamma = best_pair[1]
    train_linear = np.matmul(x_train, x_train.T)
    pairwise_sq_dists = squareform(pdist(x_train, 'sqeuclidean'))
    train_rbf = np.exp(-gamma * pairwise_sq_dists)
    x_train_kernel = np.hstack((np.arange(1, 5001).reshape((5000, 1)), np.add(train_linear, train_rbf)))

    test_linear = np.matmul(x_test, x_train.T)
    pairwise_sq_dists = cdist(x_test, x_train, 'sqeuclidean')
    test_rbf = np.exp(-gamma * pairwise_sq_dists)
    x_test_kernel = np.hstack((np.arange(1, 2501).reshape((2500, 1)), np.add(test_linear, test_rbf)))

    return x_train_kernel, x_test_kernel

def SVM(x_train, y_train, x_test, y_test, best_pair):
    prob  = svm_problem(y_train, x_train, isKernel=True)
    param = svm_parameter('-s 0 -t 4 -c {} -g {} -q'.format(best_pair[0], best_pair[1]))          
    model = svm_train(prob, param)
    prediction = svm_predict(y_test, x_test, model)

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_csv()
    best_pair = [16, 0.03125]                                         #from hw5_2_2.py 
    x_train_kernel, x_test_kernel = combine_kernel(x_train, x_test, best_pair)
    SVM(x_train_kernel, y_train, x_test_kernel, y_test, best_pair)