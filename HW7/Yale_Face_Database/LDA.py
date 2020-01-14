from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
import os
import re

SHAPE = (60, 60)
gamma = 1e-3
CLASS = 15
SUBJECT = 11


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


def compute_mean(data, target):
    classmean = np.zeros([CLASS, data.shape[1]])
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            classmean[target[i][0]-1][j] += data[i][j]              #target 1-based
    for i in range(CLASS):
        for j in range(data.shape[1]):
            classmean[i][j] /= SUBJECT
    allmean = np.mean(data, axis=0).reshape(-1, 1)
    return classmean, allmean


def compute_withinclass(data, target, classmean):
    withinclass = np.zeros([data.shape[1], data.shape[1]])
    for i in range(data.shape[0]):
        dist = np.subtract(data[i], classmean[target[i][0]-1]).reshape(data.shape[1], 1)
        withinclass += np.matmul(dist, dist.T)
    return withinclass


def compute_betweenclass(classmean, allmean):
    betweenclass = np.zeros([data.shape[1], data.shape[1]])
    for i in range(CLASS):
        dist = np.subtract(classmean[i], allmean[i]).reshape(data.shape[1], 1)
        betweenclass += np.matmul(dist, dist.T)
    betweenclass *= SUBJECT
    return betweenclass


def compute_eigen(A):
    eigenvalues, eigenvectors = np.linalg.eigh(A)
    idx = eigenvalues.argsort()[::-1]                          # sort largest
    return eigenvectors[:,idx][:,:25]


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
    title = "LDA Fisher-Face" + '_'
    eigen_vectors = eigen_vectors.T
    for i in range(0, 25):
        plt.clf()
        plt.suptitle(title + str(i))
        plt.imshow(eigen_vectors[i].reshape(SHAPE), plt.cm.gray)
        plt.savefig(storedir + title + str(i) + '.png')


def KNN(traindata, testdata, target):
    trainsize = traindata.shape[0]
    testsize = testdata.shape[0]
    result = np.zeros(testsize)
    for testidx in range(testsize):
        alldist = np.zeros(trainsize)
        for trainidx in range(trainsize):
            alldist[trainidx] = np.sqrt(np.sum((testdata[testidx] - traindata[trainidx]) ** 2))
        result[testidx] = target[np.argmin(alldist)]
    return result


def checkperformance(targettest, predict):
    correct = 0
    for i in range(len(targettest)):
        if targettest[i] == predict[i]:
            correct += 1
    print("Accuracy of LDA = {}  ({} / {})".format(correct / len(targettest), correct, len(targettest)))

def kernelLDA(data, target, method):
    gram_matrix = None
    if method == 'rbf':
        sq_dists = squareform(pdist(data), 'sqeuclidean')
        gram_matrix = np.exp(-gamma * sq_dists)
    elif method == 'linear':
        gram_matrix = np.matmul(data, data.T)

    M = np.zeros([data.shape[0], data.shape[0]])
    for i in range(CLASS):
        classM = gram_matrix[np.where(target == i+1)[0]].copy()
        classM = np.sum(classM, axis=0).reshape(-1, 1) / SUBJECT
        allM = gram_matrix[np.where(target == i+1)[0]].copy()
        allM = np.sum(allM, axis=0).reshape(-1, 1) / data.shape[0]
        dist = np.subtract(classM, allM)
        multiplydist = SUBJECT * np.matmul(dist, dist.T)
        M += multiplydist

    N = np.zeros([data.shape[0], data.shape[0]])
    I_minus_one = np.identity(SUBJECT) - (SUBJECT * np.ones((SUBJECT, SUBJECT)))
    for i in range(CLASS):
        Kj = gram_matrix[np.where(target == i+1)[0]].copy()
        multiply = np.matmul(Kj.T, np.matmul(I_minus_one, Kj))
        N += multiply

    eigenvectors = compute_eigen(np.matmul(np.linalg.pinv(N), M))
    lower_dimension_data = np.matmul(gram_matrix, eigenvectors)
    return lower_dimension_data

if __name__ == '__main__':
    #LDA
    dirtrain = './Training/'
    dirtest = './Testing/'
    storedir = './LDA_result/'
    data, target, totalfile = read_input(dirtrain)      
    datatest, targettest, totalfiletest = read_input(dirtest)
    data = np.concatenate((data, datatest), axis=0)                 #data : 165 x 3600, 
    target = np.concatenate((target, targettest), axis=0)           #target : 165 x 1
    classmean, allmean = compute_mean(data, target)                 #classmean : 15 x 3600, allmean : 3600 x 1
    withinclass = compute_withinclass(data, target, classmean)
    betweenclass = compute_betweenclass(classmean, allmean)
    eigenvectors = compute_eigen(np.matmul(np.linalg.pinv(withinclass), betweenclass))
    lower_dimension_data = np.matmul(data, eigenvectors)
    lower_dimension_data_train = lower_dimension_data[:totalfile.shape[0]].copy()
    lower_dimension_data_test = lower_dimension_data[totalfile.shape[0]:].copy()
    targettrain = target[:totalfile.shape[0]].copy()
    targettest = target[totalfile.shape[0]:].copy()
    reconstruct_data = np.matmul(lower_dimension_data_train, eigenvectors.T)
    visualization(dirtrain, totalfile, storedir, reconstruct_data)
    draweigenface(storedir, eigenvectors)
    predict = KNN(lower_dimension_data_train, lower_dimension_data_test, targettrain)
    checkperformance(targettest, predict)

    print("=======================================================================")

    #Kernel LDA
    dirtrain = './Training/'
    dirtest = './Testing/'
    storedir = './LDA_result/'
    method = 'linear'
    data, target, totalfile = read_input(dirtrain)      
    datatest, targettest, totalfiletest = read_input(dirtest)
    data = np.concatenate((data, datatest), axis=0)                 #data : 165 x 3600, 
    target = np.concatenate((target, targettest), axis=0)           #target : 165 x 1
    lower_dimension_data = kernelLDA(data, target, method)
    lower_dimension_data_train = lower_dimension_data[:totalfile.shape[0]].copy()
    lower_dimension_data_test = lower_dimension_data[totalfile.shape[0]:].copy()
    targettrain = target[:totalfile.shape[0]].copy()
    targettest = target[totalfile.shape[0]:].copy()
    predict = KNN(lower_dimension_data_train, lower_dimension_data_test, targettrain)
    checkperformance(targettest, predict)