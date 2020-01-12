from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
import os
import re

SHAPE = (60, 60)


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
    return eigen_vectors[:,idx][:,:25]


def PCA(data):
    covariance = np.cov(data.transpose())
    eigen_vectors = compute_eigen(covariance)
    lower_dimension_data = np.matmul(data, eigen_vectors)
    return lower_dimension_data, eigen_vectors


def visualization(dirname, totalfile, data):
    idx = 0
    for file in totalfile:
        filename = dirname + file
        storename = './PCA_result/' + file
        img = Image.open(filename)
        img = img.resize(SHAPE, Image.ANTIALIAS)
        width, height = img.size
        pixel = img.load()
        pixel = data[idx].reshape(width, height).copy()
        img.save(storename + '.png')
        idx += 1


def draweigenface(dirname, totalfile, eigen_vectors):
    title = "Eigen-Face" + '_'
    eigen_vectors = eigen_vectors.T
    for i in range(0, 25):
        plt.clf()
        plt.suptitle(title + str(i))
        plt.imshow(eigen_vectors[i].reshape(SHAPE), plt.cm.gray)
        plt.show()
        plt.savefig('./PCA_result/' + title + str(i) + '.png')


def KNN(traindata, testdata, target):
    result = np.zeros(testdata.shape[0])
    for testidx in range(testdata.shape[0]):
        alldist = np.zeros(traindata.shape[0])
        for trainidx in range(traindata.shape[0]):
            alldist[trainidx] = np.sqrt(np.sum((testdata[testidx] - traindata[trainidx]) ** 2))
        result[testidx] = target[np.argmin(alldist)]
    return result


def checkperformance(targettest, predict):
    correct = 0
    for i in range(len(targettest)):
        if targettest[i] == predict[i]:
            correct += 1
    print("Accuracy of PCA = {}".format(correct / len(targettest)))


if __name__ == '__main__':
    dirtrain = './Training/'
    data, target, totalfile = read_input(dirtrain)
    lower_dimension_data, eigen_vectors = PCA(data)
    print("data shape = {}".format(data.shape))
    print("eigen vector shape = {}".format(eigen_vectors.shape))
    print("lower_dimension_data shape: {}".format(lower_dimension_data.shape))
    reconstruct_data = np.matmul(lower_dimension_data, eigen_vectors.T)
    print("reconstruct_data shape: {}".format(reconstruct_data.shape))
    visualization(dirtrain, totalfile, reconstruct_data)
    draweigenface(dirtrain, totalfile, eigen_vectors)

    dirtest = './Testing/'
    datatest, targettest, totalfiletest = read_input(dirtest)
    lower_dimension_data_test = np.matmul(datatest, eigen_vectors)
    print("lower_dimension_data_test shape: {}".format(lower_dimension_data_test.shape))
    predict = KNN(lower_dimension_data, lower_dimension_data_test, target)
    checkperformance(targettest, predict)