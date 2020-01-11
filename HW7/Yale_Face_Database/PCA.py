from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt
import os
import re

SHAPE = (60, 60)
PICK = 10


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
        # print("data[idx].reshape(width, height)= {}".format(data[idx].reshape(width, height)))
        pixel = data[idx].reshape(width, height).copy()
        # for w in range(img.size[0]):
        #     for h in range(img.size[1]):
        #         pixel[h, w] = data[j, i]
        img.save(storename + '.png')
        idx += 1


if __name__ == '__main__':
    dirname = './Training/'
    data, target, totalfile = read_input(dirname)
    lower_dimension_data, eigen_vectors = PCA(data)
    print("data shape = {}".format(data.shape))
    print("lower_dimension_data shape: {}".format(lower_dimension_data.shape))
    reconstruct_data = np.matmul(lower_dimension_data, eigen_vectors.T)
    print("reconstruct_data shape: {}".format(reconstruct_data.shape))
    visualization(dirname, totalfile, reconstruct_data)