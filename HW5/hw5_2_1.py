from svmutil import *
import numpy as np
import csv

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

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_csv()

    print("kernel function : linear kernel")
    prob  = svm_problem(y_train, x_train)
    param = svm_parameter('-t 0 -q')          # 0 : linear kernel
    model = svm_train(prob, param)
    prediction_linear = svm_predict(y_test, x_test, model)
    
    print("kernel function : polynomial kernel")
    prob  = svm_problem(y_train, x_train)
    param = svm_parameter('-t 1 -q')          # 1 : polynomial kernel
    model = svm_train(prob, param)
    prediction_polynomial = svm_predict(y_test, x_test, model)

    print("kernel function : rbf kernel")
    prob  = svm_problem(y_train, x_train)
    param = svm_parameter('-t 2 -q')          # 2 : rbf kernel
    model = svm_train(prob, param)
    prediction_rbf = svm_predict(y_test, x_test, model)