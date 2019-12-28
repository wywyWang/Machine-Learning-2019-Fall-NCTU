from PIL import Image
import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform
import matplotlib.pyplot as plt

num = 100
epochs = 15
K = 4
gamma_c = 1/(255*255)
gamma_s = 1/(100*100)

def read_input(filename):
    img = Image.open(filename)
    width, height = img.size
    pixel = np.array(img.getdata()).reshape((width*height, 3))

    coord = np.array([]).reshape(0, 2)
    for i in range(num):
        row_x = np.full(num, i)
        row_y = np.arange(num)
        row = np.array(list(zip(row_x, row_y))).reshape(1*num, 2)
        coord = np.vstack([coord, row])

    return pixel, coord

def initial(data, initial_method):
    C_x = np.random.randint(0, num, size=K)
    C_y = np.random.randint(0, num, size=K)
    C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
    mu = np.random.randn(K, 2)
    if initial_method == 'random':
        prev_classification = np.random.randint(K, size=data.shape[0])
        return C, mu, prev_classification
    elif initial_method == 'modK':
        prev_classification = []
        for i in range(data.shape[0]):
            prev_classification.append(i%K)
        prev_classification = np.asarray(prev_classification)
        return C, mu, prev_classification
    elif initial_method == 'equal-divide':
        prev_classification = []
        border = num * num / K
        for i in range(data.shape[0]):
            prev_classification.append(int(i / border))
        prev_classification = np.asarray(prev_classification)
        return C, mu, prev_classification       

def compute_kernel(color, coord):
    spatial_sq_dists = squareform(pdist(coord, 'sqeuclidean'))
    spatial_rbf = np.exp(-gamma_s * spatial_sq_dists)
    color_sq_dists = squareform(pdist(color, 'sqeuclidean'))
    color_rbf = np.exp(-gamma_c * color_sq_dists)
    kernel = spatial_rbf * color_rbf

    return kernel

def calculate_third_term(kernel_data, classification):
    cluster_sum = np.zeros(K,dtype=np.int)
    kernel_sum = np.zeros(K,dtype=np.float)
    for i in range(classification.shape[0]):
        cluster_sum[classification[i]] += 1
    for cluster in range(K):
        for p in range(kernel_data.shape[0]):
            for q in range(kernel_data.shape[0]):
                if classification[p] == cluster and classification[q] == cluster:
                    kernel_sum[cluster] += kernel_data[p][q]
    for cluster in range(K):
        if cluster_sum[cluster] == 0:
            cluster_sum[cluster] = 1
        kernel_sum[cluster] /= (cluster_sum[cluster] ** 2)
    
    return kernel_sum

def calculate_second_term(kernel_data, classification, dataidx, cluster):
    cluster_sum = 0
    kernel_sum = 0
    for i in range(classification.shape[0]):
        if classification[i] == cluster:
            cluster_sum += 1
    if cluster_sum == 0:
        cluster_sum = 1
    for i in range(kernel_data.shape[0]):
        if classification[i] == cluster:
            kernel_sum += kernel_data[dataidx][i]

    return (-2) * kernel_sum / cluster_sum

def classify(data, kernel_data, mu, classification):
    this_classification = np.zeros(data.shape[0], dtype=np.int)
    third_term = calculate_third_term(kernel_data, classification)
    for dataidx in range(data.shape[0]):
        distance = np.zeros(K, dtype=np.float32)
        for cluster in range(K):
            distance[cluster] = calculate_second_term(kernel_data, classification, dataidx, cluster) + third_term[cluster]
        this_classification[dataidx] = np.argmin(distance)
        # print("class = {}".format(this_classification[dataidx]))
    
    return this_classification

def calculate_error(classification, prev_classification):
    error = 0
    for i in range(classification.shape[0]):
        error += np.absolute(classification[i] - prev_classification[i])

    return error

def visualization(filename, storename, iteration, classification, initial_method):
    img = Image.open(filename)
    width, height = img.size
    pixel = img.load()
    color = [(0,0,0), (100, 0, 0), (0, 255, 0), (255,255,255)]
    for i in range(img.size[0]):
        for j in range(img.size[1]):
            pixel[j, i] = color[classification[i * num + j]]
    img.save(storename + '_' + initial_method + '_' + str(gamma_c) + '_' + str(gamma_s) + '_' + str(iteration) + '_'+ str(K) + '.png')

def Kernel_K_Means(filename, storename, data, coord):
    method = ['random', 'modK', 'equal-divide']
    for initial_method in method:
        C, mu, classification = initial(data, initial_method)
        kernel_data = compute_kernel(data, coord)
        iteration = 0
        error = -10000
        prev_error = -10001
        print("mu = {}".format(mu))
        print("classification shape = {}".format(classification.shape))

        while(iteration <= epochs):
            iteration += 1
            print("iteration = {}".format(iteration))
            prev_classification = classification
            visualization(filename, storename, iteration, classification, initial_method)
            classification = classify(data, kernel_data, mu, classification)
            error = calculate_error(classification, prev_classification)
            print("error = {}".format(error))

            if error == prev_error:
                break
            prev_error = error

if __name__ == '__main__':
    filename = 'data/image1.png'
    storename = 'visualization/image1'
    pixel1, coord1 = read_input(filename)
    print("pixel shape = {}".format(pixel1.shape))
    print("coord1 shape = {}".format(coord1.shape))
    Kernel_K_Means(filename, storename, pixel1, coord1)

    filename = 'data/image2.png'
    storename = 'visualization/image2'
    pixel2, coord2 = read_input(filename)
    print("pixel shape = {}".format(pixel2.shape))
    print("coord1 shape = {}".format(coord2.shape))
    Kernel_K_Means(filename, storename, pixel2, coord2)