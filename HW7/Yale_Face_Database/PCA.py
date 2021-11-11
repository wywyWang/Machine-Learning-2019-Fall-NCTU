from PIL import Image
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import os
import re

SHAPE = (60, 60)
gamma = 1e-5


def read_input(dirname):
    trainfile = os.listdir(dirname)
    data = []
    target = []
    totalfile = []
    for file in trainfile:
        totalfile.append(file)
        filename = dirname + file
        number = int(re.sub(r'\D', "", file.split('.')[0]))
        target.append(number)
        img = Image.open(filename)
        img = img.resize(SHAPE, Image.ANTIALIAS)
        width, height = img.size
        pixel = np.array(img.getdata()).reshape((width*height))
        data.append(pixel)
    data = np.array(data)
    target = np.array(target).reshape(-1,1)
    totalfile = np.array(totalfile)
    return data, target, totalfile


def compute_eigen(A):
    eigen_values, eigen_vectors = np.linalg.eigh(A)
    print("eigen_values = {}".format(eigen_values.shape))
    idx = eigen_values.argsort()[::-1]                          # sort largest
    return eigen_vectors[:, idx][:, :25]


def visualization(dirname, totalfile, storedir, data):
    idx = 0
    for file in totalfile:
        filename = dirname + file
        storename = storedir + file
        img = Image.open(filename)
        img = img.resize(SHAPE, Image.ANTIALIAS)
        width, height = img.size
        pixel = img.load()
        pixel = data[idx].reshape(width, height).copy()
        img.save(storename + '.png')
        idx += 1


def draweigenface(storedir, eigen_vectors):
    title = "PCA Eigen-Face" + '_'
    eigen_vectors = eigen_vectors.T
    for i in range(0, 25):
        plt.clf()
        plt.suptitle(title + str(i))
        plt.imshow(eigen_vectors[i].reshape(SHAPE), plt.cm.gray)
        plt.savefig(storedir + title + str(i) + '.png')


def KNN(train_data, test_data, target):
    trainsize = train_data.shape[0]
    testsize = test_data.shape[0]
    result = np.zeros(testsize)
    for testidx in range(testsize):
        alldist = np.zeros(trainsize)
        for trainidx in range(trainsize):
            alldist[trainidx] = np.sqrt(np.sum((test_data[testidx] - train_data[trainidx]) ** 2))
        result[testidx] = target[np.argmin(alldist)]
    return result


def checkperformance(target_test, predict):
    correct = 0
    for i in range(len(target_test)):
        if target_test[i] == predict[i]:
            correct += 1
    print("Accuracy of PCA = {}  ({} / {})".format(correct / len(target_test), correct, len(target_test)))


def PCA(data):
    covariance = np.cov(data.T)
    eigen_vectors = compute_eigen(covariance)
    lower_dimension_data = np.matmul(data, eigen_vectors)
    return lower_dimension_data, eigen_vectors


def kernelPCA(data, method):
    lower_dimension_data = None
    if method == 'rbf':
        sq_dists = squareform(pdist(data), 'sqeuclidean')
        gram_matrix = np.exp(-gamma * sq_dists)
        N = gram_matrix.shape[0]
        one_n = np.ones((N, N)) / N
        K = gram_matrix - one_n.dot(gram_matrix) - gram_matrix.dot(one_n) + one_n.dot(gram_matrix).dot(one_n)
        eigen_vectors = compute_eigen(K)
        lower_dimension_data = np.matmul(gram_matrix, eigen_vectors)
    elif method == 'linear':
        gram_matrix = np.matmul(data, data.T)
        N = gram_matrix.shape[0]
        one_n = np.ones((N, N)) / N
        K = gram_matrix - one_n.dot(gram_matrix) - gram_matrix.dot(one_n) + one_n.dot(gram_matrix).dot(one_n)
        eigen_vectors = compute_eigen(K)
        lower_dimension_data = np.matmul(gram_matrix, eigen_vectors)
    return lower_dimension_data


if __name__ == '__main__':
    #PCA
    dirtrain = './Training/'
    storedir = './PCA_result/'
    data, target, totalfile = read_input(dirtrain)
    lower_dimension_data, eigen_vectors = PCA(data)
    print("data shape = {}".format(data.shape))
    print("eigen vector shape = {}".format(eigen_vectors.shape))
    print("lower_dimension_data shape: {}".format(lower_dimension_data.shape))
    reconstruct_data = np.matmul(lower_dimension_data, eigen_vectors.T)
    print("reconstruct_data shape: {}".format(reconstruct_data.shape))
    visualization(dirtrain, totalfile, storedir, reconstruct_data)
    draweigenface(storedir, eigen_vectors)

    #Face recognition
    dirtest = './Testing/'
    datatest, targettest, totalfiletest = read_input(dirtest)
    data = np.concatenate((data, datatest), axis=0)
    lower_dimension_data, eigen_vectors = PCA(data)
    lower_dimension_data_test = lower_dimension_data[totalfile.shape[0]:].copy()
    lower_dimension_data = lower_dimension_data[:totalfile.shape[0]].copy()
    print("lower_dimension_data shape: {}".format(lower_dimension_data.shape))
    print("lower_dimension_data_test shape: {}".format(lower_dimension_data_test.shape))
    predict = KNN(lower_dimension_data, lower_dimension_data_test, target)
    checkperformance(targettest, predict)

    print("=======================================================================")

    #Kernel PCA
    dirtrain = './Training/'
    storedir = './kernelPCA_result/'
    method = 'linear'
    data, target, totalfile = read_input(dirtrain)
    print("data shape = {}".format(data.shape))

    #Face recognition
    dirtest = './Testing/'
    datatest, targettest, totalfiletest = read_input(dirtest)
    data = np.concatenate((data, datatest), axis=0)
    lower_dimension_data = kernelPCA(data, method)
    lower_dimension_data_test = lower_dimension_data[totalfile.shape[0]:].copy()
    lower_dimension_data = lower_dimension_data[:totalfile.shape[0]].copy()
    print("lower_dimension_data shape: {}".format(lower_dimension_data.shape))
    print("lower_dimension_data_test shape: {}".format(lower_dimension_data_test.shape))
    predict = KNN(lower_dimension_data, lower_dimension_data_test, target)
    checkperformance(targettest, predict)