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
        y_train = [int(x) for x in y_train]

    with open('./data/X_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        x_test = list(csv_reader)
        x_test = [[float(y) for y in x] for x in x_test]

    with open('./data/Y_test.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        y_test_2d = list(csv_reader)
        y_test = [y for x in y_test_2d for y in x]
        y_test = [int(x) for x in y_test]

    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

def GridSearch(x_train, y_train):
    C = [1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8, 16]
    G = [1/32, 1/16, 1/8, 1/4, 1/2, 1, 2, 4, 8]
    best_pair = (0, 0)
    best_acc = 0

    # 5-fold cross validation
    for cost in C:
        for gamma in G:
            prob  = svm_problem(y_train, x_train)
            param = svm_parameter('-s 0 -t 2 -v 5 -c {} -g {} -q'.format(cost, gamma))          # 2 : rbf kernel
            acc = svm_train(prob, param)

            if acc > best_acc:
                best_acc = acc
                best_pair = (cost, gamma)

    return best_pair, best_acc

def TestBest(x_train, y_train, x_test, y_test, best_pair):
    prob  = svm_problem(y_train, x_train)
    param = svm_parameter('-s 0 -t 2 -c {} -g {} -q'.format(best_pair[0], best_pair[1]))          # 2 : rbf kernel
    model = svm_train(prob, param)
    prediction = svm_predict(y_test, x_test, model)

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_csv()
    best_pair, best_acc = GridSearch(x_train, y_train)

    print("Best pair  = {}".format(best_pair))
    print("Best acc = {}".format(best_acc))

    TestBest(x_train, y_train, x_test, y_test, best_pair)